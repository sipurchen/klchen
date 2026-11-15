"""gx10-side helper (run through gx10_run.py `lorapb`): convert a PEFT LoRA adapter into a GGUF LoRA for llama.cpp.

  lorapb convert NAME      NAME in {h2_c1, h2_b2, h2_f2} (adapters produced by scripts/h2_*.py under logs/h2/adapter_<tag>)
                           or `external` (a PEFT adapter directory given through $EXTERNAL_ADAPTER_DIR)

Paths come from the environment: LLAMA_CPP_DIR (llama.cpp source tree), TRAIN_PY (python with torch/transformers),
BASE_HF_DIR (HF checkpoint of the base model), ADAPTER_ROOT, OUT_DIR. Nothing outside those locations is written.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
LLAMA = Path(os.environ.get("LLAMA_CPP_DIR", HOME / "llama.cpp"))
TRAIN_PY = os.environ.get("TRAIN_PY", sys.executable)
BASE_HF = Path(os.environ.get("BASE_HF_DIR", HOME / "models" / "gemma-4-26B-A4B-it"))
ADAPTER_ROOT = Path(os.environ.get("ADAPTER_ROOT", HOME / "gemma4_affinity_sim" / "repo_run" / "logs" / "h2"))
OUT = Path(os.environ.get("OUT_DIR", HOME / "gemma4_affinity_sim" / "repo_run" / "logs" / "lora"))
ADAPTERS = {"h2_c1": ADAPTER_ROOT / "adapter_c1", "h2_b2": ADAPTER_ROOT / "adapter_b2", "h2_f2": ADAPTER_ROOT / "adapter_f2"}
if os.environ.get("EXTERNAL_ADAPTER_DIR"):
    ADAPTERS["external"] = Path(os.environ["EXTERNAL_ADAPTER_DIR"])


def cmd_convert(name: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}.gguf"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(LLAMA / "gguf-py"))
    argv = [TRAIN_PY, str(LLAMA / "convert_lora_to_gguf.py"), "--base", str(BASE_HF), "--outfile", str(out), "--outtype", "f16",
            str(ADAPTERS[name])]
    r = subprocess.run(argv, env=env, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-3000:])
    print("exit", r.returncode, "size", out.stat().st_size if out.exists() else None)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("name", choices=sorted(ADAPTERS) or ["h2_c1"])
    a = ap.parse_args()
    return cmd_convert(a.name)


if __name__ == "__main__":
    sys.exit(main())
