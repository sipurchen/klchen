"""Emit the exact byte layout of the llama.cpp / ggml structs the Tier 2 ctypes harness touches.

The harness never transcribes a struct by hand: it treats every struct as an opaque block of the right size and writes the
few fields it needs at offsets measured here with the build's own headers (offsetof/sizeof from a tiny C program compiled
against that build's llama.h and ggml.h). A different llama.cpp build only needs this probe re-run, not new Python.

Run on the machine that owns the llama.cpp build (through gx10_run.py `layout`); needs gcc and the llama.cpp source tree
(include/llama.h, ggml/include). Output: logs/tier2/layout_<commit>.json (relative to the working directory).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

LLAMA_ROOT = Path.home() / "llama.cpp"

C_SRC = r"""
#include <stdio.h>
#include <stddef.h>
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#define OFF(S, f) printf("\"%s\": %zu, ", #f, offsetof(S, f))
int main(void) {
    printf("{\n");
    printf("\"llama_model_params\": {\"size\": %zu, ", sizeof(struct llama_model_params));
    OFF(struct llama_model_params, tensor_buft_overrides); OFF(struct llama_model_params, n_gpu_layers);
    printf("\"_\": 0},\n");
    printf("\"llama_context_params\": {\"size\": %zu, ", sizeof(struct llama_context_params));
    OFF(struct llama_context_params, n_ctx); OFF(struct llama_context_params, n_batch); OFF(struct llama_context_params, n_ubatch);
    OFF(struct llama_context_params, n_seq_max); OFF(struct llama_context_params, n_threads);
    OFF(struct llama_context_params, n_threads_batch); OFF(struct llama_context_params, cb_eval);
    OFF(struct llama_context_params, cb_eval_user_data); OFF(struct llama_context_params, no_perf);
    printf("\"_\": 0},\n");
    printf("\"llama_batch\": {\"size\": %zu, ", sizeof(struct llama_batch));
    OFF(struct llama_batch, n_tokens); OFF(struct llama_batch, token); OFF(struct llama_batch, pos);
    OFF(struct llama_batch, n_seq_id); OFF(struct llama_batch, seq_id); OFF(struct llama_batch, logits);
    printf("\"_\": 0},\n");
    printf("\"llama_model_tensor_buft_override\": {\"size\": %zu, ", sizeof(struct llama_model_tensor_buft_override));
    OFF(struct llama_model_tensor_buft_override, pattern); OFF(struct llama_model_tensor_buft_override, buft);
    printf("\"_\": 0},\n");
    printf("\"ggml_tensor\": {\"size\": %zu, ", sizeof(struct ggml_tensor));
    OFF(struct ggml_tensor, type); OFF(struct ggml_tensor, ne); OFF(struct ggml_tensor, nb); OFF(struct ggml_tensor, name);
    printf("\"_\": 0},\n");
    printf("\"constants\": {\"GGML_TYPE_I32\": %d, \"GGML_BACKEND_DEVICE_TYPE_CPU\": %d, \"GGML_MAX_NAME\": %d}\n",
           (int) GGML_TYPE_I32, (int) GGML_BACKEND_DEVICE_TYPE_CPU, (int) GGML_MAX_NAME);
    printf("}\n");
    return 0;
}
"""


def main() -> int:
    root = LLAMA_ROOT
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="layout_probe_") as td:
        src, exe = Path(td) / "probe.c", Path(td) / "probe"
        src.write_text(C_SRC, encoding="utf-8")
        r = subprocess.run(["gcc", "-o", str(exe), str(src), f"-I{root / 'include'}", f"-I{root / 'ggml' / 'include'}"],
                           capture_output=True, text=True)
        if r.returncode:
            print("compile failed:", r.stderr[-2000:])
            return 2
        out = subprocess.run([str(exe)], capture_output=True, text=True)
    layout = json.loads(out.stdout)
    layout["llama_cpp_commit"] = commit
    dest = Path("logs") / "tier2" / f"layout_{commit}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(layout, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(layout))
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
