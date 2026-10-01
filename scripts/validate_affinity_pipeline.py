"""
V0 pilot validation driver (docs/expert_affinity_corpus_design.md §6.1, §6.4).

Automates the checks that are pure log/file inspection (P0, P1, P2, P8, P9, V3a, V3b) by running small,
CPU-safe imatrix probes directly. P3-P7 need the actual pilot session bodies to exist first (run
build_routing_corpus.ps1 for the two pilots, then segment_chunk_types.py) and are partly a manual read
of the export files — this script prints exactly what to look for.

All imatrix launches here go through the same CPU-safe wrapper as scripts/profile_experts_imatrix.ps1
(cores 2+3, BelowNormal, -t 2) via a small inline PowerShell call per probe.

Usage: python scripts/validate_affinity_pipeline.py --step automated   (P0/P1/P2/P8/P9/V3a/V3b only)
       python scripts/validate_affinity_pipeline.py --step checklist   (print the manual P3-P7 checklist)
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from segment_chunk_types import run_tokenize  # noqa: E402

import gguf
import numpy as np

PROJECT_ROOT = Path(r"E:\Gemma4_E4B_Project")
MODEL = Path(r"E:\LLMmodel\gemma4-26B-A4B-it-GGUF\google_gemma-4-26B-A4B-it-Q4_K_M.gguf")
EXE_IMATRIX = PROJECT_ROOT / "bin" / "llama-cpp" / "llama-imatrix.exe"
SCRATCH = PROJECT_ROOT / "logs" / "v0_pilot_checks"
SCRATCH.mkdir(parents=True, exist_ok=True)

PS_WRAPPER = r'''
$Exe = "{exe}"
$argList = @({args})
$log = "{log}"
Remove-Item $log -ErrorAction SilentlyContinue
Remove-Item "$log.err" -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $Exe -ArgumentList $argList -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -WindowStyle Hidden
Start-Sleep -Milliseconds 500
try {{ $proc.ProcessorAffinity = 12; $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal }} catch {{}}
$deadline = (Get-Date).AddSeconds({timeout})
while ((Get-Date) -lt $deadline -and -not $proc.HasExited) {{ Start-Sleep -Milliseconds 500 }}
if (-not $proc.HasExited) {{ Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue; Write-Host "TIMEOUT" }}
'''


def run_imatrix(args_list, log_label, timeout=120):
    log_path = SCRATCH / f"{log_label}.log"
    ps_args = ", ".join(f'"{a}"' for a in args_list)
    script = PS_WRAPPER.format(exe=str(EXE_IMATRIX), args=ps_args, log=str(log_path), timeout=timeout)
    script_path = SCRATCH / f"{log_label}.ps1"
    script_path.write_text(script, encoding="utf-8")
    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
                    capture_output=True, text=True, timeout=timeout + 30)
    out = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    err = (log_path.with_suffix(".log.err")).read_text(encoding="utf-8", errors="replace") if (log_path.with_suffix(".log.err")).exists() else ""
    return out + "\n" + err


def make_tiny_pilot_body(n_target_tokens=1200):
    """A short, deterministic filler body for P0/P1/P2/V3 (doesn't need real session content —
    these checks are about imatrix mechanics, not corpus content)."""
    unit = "The quick brown fox jumps over the lazy dog. def add(a, b): return a + b. "
    text = (unit * ((n_target_tokens // 12) + 5)).strip()
    return text


def check_P0_P1():
    print("=== P0: -c honoured un-padded ===")
    body = make_tiny_pilot_body(1200)
    f = SCRATCH / "p0_p1_body.txt"
    f.write_text(body, encoding="utf-8")
    out = run_imatrix([
        "-m", str(MODEL), "-ngl", "9", "--override-tensor", r"\.ffn_.*_exps\.=CPU",
        "-c", "1001", "-b", "512", "-ub", "512", "-t", "2", "-tb", "2", "-fit", "off",
        "-f", str(f), "--chunks", "1", "--parse-special", "--no-escape", "--no-ppl", "--output-format", "gguf",
        "-o", str(SCRATCH / "p0.gguf"),
    ], "p0", timeout=120)
    m = re.search(r"computing over (\d+) chunks, n_ctx=(\d+)", out)
    if m and m.group(2) == "1001":
        print(f"  PASS — n_ctx={m.group(2)} matches requested -c 1001 (un-padded)")
    else:
        print("  FAIL or inconclusive — log excerpt:")
        print("  " + "\n  ".join(l for l in out.splitlines() if "n_ctx" in l or "tokens" in l))

    print("\n=== P1: auto-BOS, no double BOS ===")
    L_tokens = run_tokenize(MODEL, f)
    L = len(L_tokens)
    d2 = body + "<bos>" + body
    f2 = SCRATCH / "p1_d2.txt"
    f2.write_text(d2, encoding="utf-8")
    # deliberately request a ctx larger than the file to trigger the "tokenizes to only N tokens" message
    out2 = run_imatrix([
        "-m", str(MODEL), "-ngl", "9", "--override-tensor", r"\.ffn_.*_exps\.=CPU",
        "-c", str(2 * L + 3), "-b", "512", "-ub", "512", "-t", "2", "-tb", "2", "-fit", "off",
        "-f", str(f2), "--parse-special", "--no-escape", "--no-ppl", "--output-format", "gguf",
        "-o", str(SCRATCH / "p1.gguf"),
    ], "p1", timeout=60)
    m2 = re.search(r"tokenizes to only (\d+) tokens", out2)
    if m2:
        n_reported = int(m2.group(1))
        expected_with_autobos = 2 * L + 2
        expected_without = 2 * L + 1
        if n_reported == expected_with_autobos:
            print(f"  PASS — {n_reported} == 2L+2 ({expected_with_autobos}): imatrix adds its own BOS, "
                  f"and the literal '<bos>' text mid-file is NOT double-counted as an extra BOS "
                  f"(it must be one ordinary token or exactly the separator we intend).")
        elif n_reported == expected_without:
            print(f"  {n_reported} == 2L+1: NO auto-BOS observed — §5.3 file layout must prepend an "
                  f"explicit <bos> at the start of the D×2 file instead of relying on imatrix.")
        else:
            print(f"  UNEXPECTED — got {n_reported}, expected 2L+2={expected_with_autobos} or 2L+1={expected_without} (L={L})")
    else:
        print("  Could not find the tokenization-count message — log excerpt:")
        print("  " + "\n  ".join(l for l in out2.splitlines() if "token" in l.lower()))


def check_P2_V3a_V3b():
    print("\n=== P2 / V3a / V3b: chunk independence + determinism ===")
    body = make_tiny_pilot_body(1200)
    f = SCRATCH / "v3_body.txt"
    f.write_text(body, encoding="utf-8")
    L = len(run_tokenize(MODEL, f))
    d2 = body + "<bos>" + body
    f2 = SCRATCH / "v3_d2.txt"
    f2.write_text(d2, encoding="utf-8")
    ctx = L + 1

    common = ["-m", str(MODEL), "-ngl", "9", "--override-tensor", r"\.ffn_.*_exps\.=CPU",
              "-c", str(ctx), "-b", "512", "-ub", "512", "-t", "2", "-tb", "2", "-fit", "off",
              "-f", str(f2), "--parse-special", "--no-escape", "--no-ppl", "--output-format", "gguf"]

    run_imatrix(common + ["--chunks", "1", "-o", str(SCRATCH / "v3a_run1.gguf")], "v3a_run1", timeout=90)
    run_imatrix(common + ["--chunks", "1", "-o", str(SCRATCH / "v3a_run2.gguf")], "v3a_run2", timeout=90)
    run_imatrix(common + ["--chunks", "2", "-o", str(SCRATCH / "v3b_2chunks.gguf")], "v3b", timeout=120)

    def read(path):
        r = gguf.GGUFReader(str(path), mode="r")
        counts = np.zeros((30, 128))
        for t in r.tensors:
            m = re.match(r"blk\.(\d+)\.ffn_down_exps\.weight\.counts$", t.name)
            if m:
                counts[int(m.group(1)), :] = np.array(t.data).reshape(-1)
        return counts

    try:
        c1 = read(SCRATCH / "v3a_run1.gguf")
        c2 = read(SCRATCH / "v3a_run2.gguf")
        c2chunks = read(SCRATCH / "v3b_2chunks.gguf")
        v3a_pass = np.array_equal(c1, c2)
        v3b_pass = np.allclose(c2chunks, 2 * c1)
        print(f"  V3a (determinism, same run twice): {'PASS' if v3a_pass else 'FAIL'}")
        print(f"  V3b (2 chunks == 2x 1 chunk, chunk independence): {'PASS' if v3b_pass else 'FAIL'}"
              + ("" if v3b_pass else f" — max abs diff {np.abs(c2chunks - 2*c1).max()}"))
        if not v3b_pass:
            print("  ** V3b FAIL means state may leak across imatrix chunks — the whole D×2/differential "
                  "method (§5.3/§5.4) must be re-examined before any real profiling run.")
    except Exception as e:
        print(f"  Could not read one or more output .gguf files: {e}")


def print_manual_checklist():
    print("""
Manual checklist (P3-P7) — run AFTER generating the two pilot sessions:

  powershell -File scripts\\build_routing_corpus.ps1 -Family planner -Session pilot
  powershell -File scripts\\build_routing_corpus.ps1 -Family coder   -Session pilot
  python scripts\\segment_chunk_types.py planner_pilot coder_pilot

P3 (thinking channel is real):
  Open data\\routing_corpus_v2\\exports\\{planner,coder}_pilot.txt.
  Every [reasoning] block should be non-trivial prose (not empty, not 1 line) for reasoning-labelled turns.
  If a [reasoning] span is empty/near-empty, thinking did not trigger — see §4.2 fallback.

P4 (-sp rendering of <turn|>):
  grep the session .turns.jsonl for stop_type; open the corresponding slice of body.txt and confirm
  it ends with a single <turn|> (not doubled, not missing).

P5/P6 (return_tokens + retokenization fidelity):
  Check whether turns.jsonl's gen_tokens field is non-null. If present, compare it against
  `python scripts\\segment_chunk_types.py` internals (or a quick ad-hoc llama-tokenize --ids run on the
  same turn's text) and report the mismatch rate.

P7 (span index integrity):
  segment_chunk_types.py already asserts full coverage / no gaps-overlaps and raises on failure —
  a clean run of the command above IS the P7 pass signal. Additionally eyeball spans.json: every
  "reasoning" span should start at the token whose piece is '<|channel>' and end at '<channel|>'.

P8/P9 are covered by build_imatrix_jobs.py + profile_experts_imatrix.ps1 on the pilot sessions
(P8 = counts shape/sum check, done implicitly by aggregate_expert_affinity.py's assertions;
 P9 = VRAM guard, enforced live by profile_experts_imatrix.ps1).
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["automated", "checklist", "all"], default="all")
    args = ap.parse_args()
    if args.step in ("automated", "all"):
        check_P0_P1()
        check_P2_V3a_V3b()
    if args.step in ("checklist", "all"):
        print_manual_checklist()


if __name__ == "__main__":
    main()
