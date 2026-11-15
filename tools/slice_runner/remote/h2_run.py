"""gx10-side launcher for the H2 router-LoRA study (through gx10_run.py `h2`): fixed train / held-out split, fixed paths.

  h2 train --tag T [--lam 1.0 --mu 0.2 --lr 1e-3 --epochs 3 --accum 2 --rank 16 --max-windows N]
  h2 eval  --tag T
Runs scripts/h2_router_lora.py / h2_comp.py with the python given by $TRAIN_PY (torch / transformers / peft; the offload engine package is found through $OFFLOAD_ENGINE_DIR).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

UNSLOTH_PY = os.environ.get("TRAIN_PY", sys.executable)   # python with torch / transformers / peft
TRAIN = "coder_pilot,coder_01,coder_02,planner_pilot,planner_01,planner_02,writer_01"
HELD_OUT = "coder_03,planner_03,writer_02"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "comp-train", "comp-eval"])
    ap.add_argument("--tag", required=True)
    for k, d in (("lam", "1.0"), ("mu", "0.2"), ("lr", "1e-3"), ("epochs", "3"), ("accum", "2"), ("rank", "16"), ("alpha", "16"),
                 ("max-windows", "0"), ("hinge", "0.0"), ("nu", "0.0"), ("targets", "logs/h2/targets.json"), ("window", "768"), ("router", "logs/h2/adapter_c1"), ("lmw", "0.5"), ("sessions", ""), ("out", "")):
        ap.add_argument("--" + k, default=d)
    a = ap.parse_args()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    if a.mode == "comp-train":
        argv = [str(UNSLOTH_PY), "scripts/h2_comp.py", "train", "--router", a.router, "--out", f"logs/h2/adapter_{a.tag}",
                "--rank", a.rank, "--alpha", a.alpha, "--lr", a.lr, "--epochs", a.epochs, "--accum", a.accum, "--window", a.window,
                "--max-windows", a.max_windows, "--lm-weight", a.lmw]
        return subprocess.run(argv, env=env).returncode
    if a.mode == "comp-eval":
        argv = [str(UNSLOTH_PY), "scripts/h2_comp.py", "eval", "--router", a.router, "--peft", f"logs/h2/adapter_{a.tag}",
                "--sessions", (a.sessions or HELD_OUT), "--out", (a.out or f"logs/h2/eval_{a.tag}.json"), "--window", a.window]
        return subprocess.run(argv, env=env).returncode
    if a.mode == "train":
        argv = [str(UNSLOTH_PY), "scripts/h2_router_lora.py", "train", "--train", TRAIN, "--targets", a.targets,
                "--out", f"logs/h2/adapter_{a.tag}", "--rank", a.rank, "--alpha", a.alpha, "--lr", a.lr, "--epochs", a.epochs,
                "--accum", a.accum, "--lam", a.lam, "--mu", a.mu, "--hinge", a.hinge, "--nu", a.nu, "--window", a.window, "--max-windows", a.max_windows]
    else:
        argv = [str(UNSLOTH_PY), "scripts/h2_router_lora.py", "eval", "--adapter", f"logs/h2/adapter_{a.tag}", "--sessions", HELD_OUT,
                "--out", f"logs/h2/eval_{a.tag}.json", "--window", a.window]
    return subprocess.run(argv, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
