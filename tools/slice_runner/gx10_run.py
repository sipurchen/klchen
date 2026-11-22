"""Restricted gx10 runner for the affinity pipeline (shared host: hard limits, whitelist, PID-exact cleanup).

  python tools/slice_runner/gx10_run.py sync
  python tools/slice_runner/gx10_run.py run segment  --args "planner_pilot"
  python tools/slice_runner/gx10_run.py run jobs     --args "planner_pilot coder_pilot"
  python tools/slice_runner/gx10_run.py run profile  --args "--dry-run"
  python tools/slice_runner/gx10_run.py run aggregate
  python tools/slice_runner/gx10_run.py run gen      --args "--profile gx10_sim --family planner --session pilot --dry-run"
  python tools/slice_runner/gx10_run.py fetch data/routing_corpus_v2/sessions/planner_pilot.spans.json [--to DIR]
  python tools/slice_runner/gx10_run.py status

Rules: CPU cores 0-4 only (25%), nice 10, wall-clock timeout, refuse to start when
the 1-minute load average is above MAX_LOAD1, one run at a time, everything the run leaves behind is found through an
environment marker and killed by PID (never pkill -f), writes only under ~/gemma4_affinity_sim/repo_run, never touches
docker / other users' processes / Ollama. Only the whitelisted scripts below can be run; arguments are validated.
Run with the project venv python (stdlib only).
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HOST = os.environ.get("GX10_HOST", "<user>@<gx10-host>")   # ssh target of the shared host, set in the environment
REMOTE_DIR = "~/gemma4_affinity_sim/repo_run"
REMOTE_PY = "~/gemma4_affinity_sim/.venv/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST]
CORES = "0-4"
MAX_LOAD1 = 6.0
DEFAULT_TIMEOUT_S = 1800
MAX_TIMEOUT_S = 14400

SHIP = ["affinity_pipeline", "config/machines", "config/tier2_layouts", "scripts", "data/routing_corpus_v2/sessions",
        "data/routing_corpus_v2/turns", "tools/slice_runner/remote",
        "data/rag_eval", "docs",
]
# name -> argv prefix (relative to REMOTE_DIR)
WHITELIST = {
    "segment": ["scripts/segment_chunk_types.py"],
    "jobs": ["scripts/build_imatrix_jobs.py"],
    "profile": ["scripts/profile_imatrix.py"],
    "aggregate": ["scripts/aggregate_expert_affinity.py"],
    "gen": ["-m", "affinity_pipeline.gen_session"],
    "counts": ["tools/slice_runner/remote/extract_counts.py"],
    "lorapb": ["tools/slice_runner/remote/lora_probe.py"],
    "layout": ["tools/slice_runner/remote/layout_probe.py"],
    "tier2": ["scripts/tier2_locality.py"],
    "xcheck": ["tools/slice_runner/remote/crosscheck_tier1.py"],
    "h2": ["tools/slice_runner/remote/h2_run.py"],
    "h2targets": ["scripts/h2_targets.py"],
    "replay": ["scripts/pool_replay.py"],
    "embedreplay": ["scripts/embed_pool_eval.py"],
    "fingerprint": ["scripts/routing_fingerprint_eval.py"],
    "embedchunks": ["scripts/embed_chunks.py"],
    "rageval": ["scripts/eval_rag_retrieval.py"],
    "aembed": ["scripts/a_embedding_pools.py"],      # E03 (Claude)
    "embedcost": ["scripts/embed_cost.py"],          # E04 (Claude)
    "rage2e": ["scripts/rag_e2e.py"],                # E08 (Claude)
    "e5diag": ["scripts/e5_diagnose.py"],            # E10 (Claude)
}
ARG_RE = re.compile(r"[\w./:=,@+-]+")   # no shell metacharacters, no spaces inside a token
FETCH_RE = re.compile(r"(data|logs)/[\w./-]+")


def validate_args(argline: str) -> list[str]:
    toks = shlex.split(argline) if argline.strip() else []
    for t in toks:
        if not ARG_RE.fullmatch(t) or ".." in t.split("/"):
            raise SystemExit(f"argument not allowed: {t!r}")
        if t.startswith("/") or t.startswith("~"):
            raise SystemExit(f"absolute paths are not allowed: {t!r}")
    return toks


def ssh(cmd: str, timeout: int = 120, stdin=None) -> subprocess.CompletedProcess:
    return subprocess.run(SSH + ["bash", "-c", shlex.quote(cmd)], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, stdin=stdin)


def load1() -> float:
    r = ssh("cut -d' ' -f1 /proc/loadavg", timeout=30)
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise SystemExit(f"cannot read gx10 load: {r.stdout!r} {r.stderr[-200:]!r}")


def reap(marker: str) -> list[str]:
    script = ("for p in /proc/[0-9]*; do if tr '\\0' '\\n' < $p/environ 2>/dev/null | grep -qx '" + marker + "'; then "
              "echo \"leftover pid ${p#/proc/}: $(tr '\\0' ' ' < $p/cmdline | cut -c1-100)\"; kill ${p#/proc/}; fi; done; echo reap_done")
    out = ssh(script, timeout=60).stdout
    return [ln for ln in out.splitlines() if ln.startswith("leftover")]


def _tar_bytes(ship: list[str]) -> bytes:
    """tar of the shipped paths; files under data/ are LF-normalised (the Windows working copy is CRLF through
    autocrlf, and tokenizing a CRLF body gives a different token count than the committed LF corpus)."""
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for top in ship:
            base = REPO_ROOT / top
            files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
            for f in files:
                data = f.read_bytes()
                if top.startswith("data/"):
                    data = data.replace(b"\r\n", b"\n")
                info = tarfile.TarInfo(f.relative_to(REPO_ROOT).as_posix())
                info.size, info.mode = len(data), 0o644
                tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def cmd_sync() -> int:
    ship = [p for p in SHIP if (REPO_ROOT / p).exists()]
    rm = " ".join(ship)
    r = subprocess.run(SSH + [f"mkdir -p {REMOTE_DIR} && cd {REMOTE_DIR} && rm -rf {rm} && tar xf -"], input=_tar_bytes(ship),
                       capture_output=True, timeout=180)
    r.stderr = r.stderr.decode("utf-8", "replace")
    if r.returncode != 0:
        print("sync failed:", r.stderr[-400:])
        return 2
    print("synced:", ", ".join(ship))
    return 0


def cmd_run(name: str, argline: str, timeout_s: int) -> int:
    if name not in WHITELIST:
        raise SystemExit(f"not whitelisted: {name} (allowed: {sorted(WHITELIST)})")
    if not 1 <= timeout_s <= MAX_TIMEOUT_S:
        raise SystemExit(f"timeout must be 1..{MAX_TIMEOUT_S}")
    args = validate_args(argline)
    l1 = load1()
    print(f"gx10 load1={l1}")
    if l1 > MAX_LOAD1:
        print(f"REFUSED: load1 {l1} > {MAX_LOAD1} (shared host)")
        return 4
    marker = f"GX10_RUN={uuid.uuid4().hex}"
    argv = " ".join(shlex.quote(a) for a in WHITELIST[name] + args)
    run = (f"cd {REMOTE_DIR} && env {marker} AFFINITY_PROFILE=gx10_sim PYTHONIOENCODING=utf-8 "
           f"taskset -c {CORES} nice -n 10 timeout {timeout_s} {REMOTE_PY} {argv} 2>&1 | tail -n 80; echo \"exit=${{PIPESTATUS[0]}}\"")
    try:
        r = ssh(run, timeout=timeout_s + 60)
        print(r.stdout[-8000:])
        if r.stderr.strip():
            print("[ssh stderr]", r.stderr[-300:])
        out = r.stdout
    finally:
        left = reap(marker)
        print("LEFTOVER PROCESSES FROM THIS RUN:", len(left))
        for ln in left:
            print("  " + ln)
    m = re.search(r"exit=(\d+)", out)
    return int(m.group(1)) if m else 3


def cmd_fetch(rel: str, to: str | None) -> int:
    if not FETCH_RE.fullmatch(rel) or ".." in rel.split("/"):
        raise SystemExit("fetch path must be data/... or logs/... (relative, no ..)")
    dest = Path(to) if to else REPO_ROOT / "tools" / "slice_runner" / "runs" / "gx10_fetch"
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(SSH + [f"cd {REMOTE_DIR} && tar cf - {shlex.quote(rel)}"], capture_output=True, timeout=3000)
    if r.returncode != 0 or not r.stdout:
        print("fetch failed:", r.stderr[-300:].decode("utf-8", "replace"))
        return 2
    u = subprocess.run(["tar", "xf", "-", "-C", str(dest)], input=r.stdout, capture_output=True)
    print(f"fetched {rel} -> {dest}" if u.returncode == 0 else f"untar failed: {u.stderr[-200:]!r}")
    return u.returncode


def cmd_status() -> int:
    print("load1:", load1())
    print(ssh("ps -eo pid,ni,pcpu,etime,args --sort=-pcpu | head -n 6 | cut -c1-120").stdout)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("sync")
    sub.add_parser("status")
    pr = sub.add_parser("run")
    pr.add_argument("name")
    pr.add_argument("--args", default="")
    pr.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    pf = sub.add_parser("fetch")
    pf.add_argument("path")
    pf.add_argument("--to")
    a = ap.parse_args()
    if a.action == "sync":
        return cmd_sync()
    if a.action == "status":
        return cmd_status()
    if a.action == "run":
        return cmd_run(a.name, a.args, a.timeout)
    return cmd_fetch(a.path, a.to)


if __name__ == "__main__":
    sys.exit(main())
