#!/usr/bin/env python3
"""E04: cost of the embedder under the restricted-machine budget (CPU only, 1 or 2 cores, no GPU).

For one model: take N real chunk texts of the routing corpus, run llama-embedding once per text (short process: this is the
'on-demand' option of the architecture document) restricted to the first `cores` CPU cores of the process affinity, and report
wall time per text, the model-load share (a one-token text measures it), the warm estimate (cold - load), and the peak resident
set of the child processes. Criterion H4c (plan section 3): <= 1.0 s per chunk, <= 300 MiB RSS, GPU 0.
  python scripts/embed_cost.py --model bge-small-en-v1.5-q8_0.gguf --n 20 --out logs/embed_cost/bge.json
"""
import argparse
import json
import os
import subprocess
import tempfile
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import embed_chunks as ec  # noqa: E402
from affinity_pipeline.config import load_profile  # noqa: E402
from affinity_pipeline import embedding_io as eio  # noqa: E402


def timed_embed(binary: Path, model_path: Path, text: str, threads: int, ctx: int = 0) -> tuple[float, float]:
    """one llama-embedding process: (wall seconds, peak VmHWM MiB of that process, read from /proc while it runs).
    ru_maxrss of RUSAGE_CHILDREN is not used: it can include memory the parent held before exec."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(eio.truncate(text, 480))
        path = Path(handle.name)
    try:
        t0 = time.perf_counter()
        proc = subprocess.Popen([str(binary), "-m", str(model_path), "-f", str(path), "-t", str(threads), "-ngl", "0",
                                 "--embd-separator", eio.SEP, "--embd-output-format", "json", "--no-warmup",
                                 *(["-c", str(ctx), "-b", str(ctx), "-ub", str(ctx)] if ctx else [])],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        peak = 0.0
        while proc.poll() is None:
            try:
                for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        peak = max(peak, int(line.split()[1]) / 1024)
            except OSError:
                pass
            time.sleep(0.005)
        wall = time.perf_counter() - t0
        if proc.returncode:
            raise RuntimeError(f"llama-embedding exited {proc.returncode}")
        return wall, peak
    finally:
        path.unlink(missing_ok=True)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cores", default="1,2")
    ap.add_argument("--ctx", type=int, default=0, help="0 = llama-embedding defaults; N = -c/-b/-ub N (smaller buffers)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    profile = load_profile(os.environ.get("AFFINITY_PROFILE", "gx10_sim"))
    binary = profile.llama_bin_dir / "llama-embedding"
    model_path = Path.home() / "gemma4_affinity_sim" / "work" / ec.MODEL_DIR_NAME / a.model
    manifest = json.loads((root / "logs" / "embed" / a.model / "manifest.json").read_text(encoding="utf-8"))
    pooling = manifest["pooling"]
    records, _ = ec.collect_session_inputs(root, profile.name, 512, 0)
    pool = [(r["session_id"], i, text) for r in records for i, text in enumerate(r["chunks"]) if len(text) > 400]
    picks = [pool[int(i)] for i in np.linspace(0, len(pool) - 1, a.n)]
    usable = sorted(os.sched_getaffinity(0))
    result = {"model": a.model, "ctx_arg": a.ctx, "pooling": pooling, "n": a.n, "chunk_chars_median": float(np.median([len(t) for _, _, t in picks])),
              "gpu_mib": 0, "runs": {}}
    for cores in [int(x) for x in a.cores.split(",")]:
        os.sched_setaffinity(0, set(usable[:cores]))
        load, peaks = [], []
        for _ in range(5):
            wall, hwm = timed_embed(binary, model_path, ec.embedding_input(a.model, "a", "passage"), cores, a.ctx)
            load.append(wall)
            peaks.append(hwm)
        cold = []
        for _, _, text in picks:
            wall, hwm = timed_embed(binary, model_path, ec.embedding_input(a.model, text, "passage"), cores, a.ctx)
            cold.append(wall)
            peaks.append(hwm)
        peak_mib = max(peaks)
        warm = [max(c - float(np.median(load)), 0.0) for c in cold]
        row = {"affinity_cores": sorted(os.sched_getaffinity(0)), "threads": cores,
               "load_only_s_median": float(np.median(load)),
               "cold_s_p50": percentile(cold, 50), "cold_s_p95": percentile(cold, 95), "cold_s_max": float(max(cold)),
               "warm_est_s_p50": percentile(warm, 50), "warm_est_s_p95": percentile(warm, 95),
               "peak_rss_mib_children": peak_mib, "rss_note": "max VmHWM of the llama-embedding process itself, polled from /proc"}
        result["runs"][str(cores)] = row
        print(f"{a.model} cores={cores}: cold p50={row['cold_s_p50']:.2f}s p95={row['cold_s_p95']:.2f}s  load={row['load_only_s_median']:.2f}s  "
              f"warm_est p50={row['warm_est_s_p50']:.2f}s  peak RSS={peak_mib:.0f} MiB", flush=True)
    result["h4c"] = {"two_core_cold_p95_le_1s": result["runs"].get("2", {}).get("cold_s_p95", 9e9) <= 1.0,
                     "two_core_warm_p95_le_1s": result["runs"].get("2", {}).get("warm_est_s_p95", 9e9) <= 1.0,
                     "rss_le_300_mib": result["runs"][max(result["runs"])]["peak_rss_mib_children"] <= 300.0}
    print(json.dumps(result["h4c"]))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
