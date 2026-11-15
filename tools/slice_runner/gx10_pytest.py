"""Restricted gx10 test runner for slice leads (Linux validation without giving out ssh).

  python tools/slice_runner/gx10_pytest.py [--args "tests/affinity_pipeline -q -x"]

Ships ONLY the project's test-relevant directories to ~/gemma4_affinity_sim/repo_lead on gx10 and runs pytest there
with the gx10 venv under hard limits: CPU cores 0-1, nice 10, wall-clock timeout, one run at a time. Anything the run
leaves behind is found by an environment marker (not by pattern matching) and killed by PID. It cannot run arbitrary
commands. gx10 is a SHARED host: do not raise the limits. Run it with the project venv python (stdlib only).
"""
import argparse
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HOST = os.environ.get("GX10_HOST", "<user>@<gx10-host>")   # ssh target of the shared host, set in the environment
REMOTE_DIR = "~/gemma4_affinity_sim/repo_lead"
REMOTE_PY = "~/gemma4_affinity_sim/.venv/bin/python"
SHIP = ["affinity_pipeline", "tests/__init__.py", "tests/affinity_pipeline", "tests/fixtures", "config/machines",
        "data/routing_corpus_v2/sessions", "data/routing_corpus_v2/turns", "scripts"]   # small golden data (~200 KB), read-only for tests
ALLOWED_FLAGS = {"-q", "-x", "-v", "-vv", "-rs", "-rf", "--tb=short", "--tb=line", "--tb=no", "-k", "-p"}
TIMEOUT_S = 240
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST]


def validate_args(argline: str) -> list[str]:
    toks = argline.split()
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("tests/") and re.fullmatch(r"tests/[\w./:\[\]-]+", t):
            out.append(t)
        elif t in ALLOWED_FLAGS:
            out.append(t)
            if t in ("-k", "-p"):
                i += 1
                if i >= len(toks) or not re.fullmatch(r"[\w .:\[\]()-]+", toks[i]):
                    raise SystemExit(f"bad value for {t}")
                out.append(toks[i])
        else:
            raise SystemExit(f"argument not allowed: {t!r} (allowed: tests/... paths and {sorted(ALLOWED_FLAGS)})")
        i += 1
    if not any(a.startswith("tests/") for a in out):
        raise SystemExit("give at least one tests/... path")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--args", default="tests/affinity_pipeline -q --tb=short -p no:cacheprovider")
    args = ap.parse_args()
    pytest_args = validate_args(args.args)
    if "no:cacheprovider" not in pytest_args:
        pytest_args += ["-p", "no:cacheprovider"]
    marker = f"GX10_LEAD_RUN={uuid.uuid4().hex}"
    ship = [p for p in SHIP if (REPO_ROOT / p).exists()]

    tar = subprocess.Popen(["tar", "cf", "-", *ship], cwd=REPO_ROOT, stdout=subprocess.PIPE)
    unpack = subprocess.run(
        SSH + [f"mkdir -p {REMOTE_DIR} && cd {REMOTE_DIR} && rm -rf affinity_pipeline tests/affinity_pipeline tests/fixtures "
               f"config/machines data/routing_corpus_v2/sessions data/routing_corpus_v2/turns scripts && tar xf - && touch tests/__init__.py"],
        stdin=tar.stdout, capture_output=True, text=True, timeout=120)
    tar.wait()
    if unpack.returncode != 0:
        print("ship failed:", unpack.stderr[-400:])
        return 2

    run_cmd = (f"cd {REMOTE_DIR} && env {marker} taskset -c 0-1 nice -n 10 timeout {TIMEOUT_S} {REMOTE_PY} -m pytest "
               + " ".join(pytest_args) + " 2>&1 | tail -n 60; echo \"pytest_exit=${PIPESTATUS[0]}\"")
    r = subprocess.run(SSH + ["bash", "-c", "'" + run_cmd.replace("'", "'\\''") + "'"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=TIMEOUT_S + 60)
    print(r.stdout[-6000:])
    if r.stderr.strip():
        print("[ssh stderr]", r.stderr[-300:])

    # leftovers: processes that inherited our marker (env-based, never pattern-based) are killed by PID
    reap = ("for p in /proc/[0-9]*; do if tr '\\0' '\\n' < $p/environ 2>/dev/null | grep -qx '" + marker + "'; then "
            "echo \"reaping leftover pid ${p#/proc/}: $(tr '\\0' ' ' < $p/cmdline | cut -c1-80)\"; kill ${p#/proc/}; fi; done; echo reap_done")
    rr = subprocess.run(SSH + ["bash", "-c", "'" + reap.replace("'", "'\\''") + "'"], capture_output=True, text=True, timeout=60)
    leftovers = [ln for ln in rr.stdout.splitlines() if ln.startswith("reaping")]
    print("LEFTOVER PROCESSES FROM THIS RUN:", len(leftovers))
    for ln in leftovers:
        print("  " + ln)
    m = re.search(r"pytest_exit=(\d+)", r.stdout)
    return int(m.group(1)) if m else 3


if __name__ == "__main__":
    sys.exit(main())
