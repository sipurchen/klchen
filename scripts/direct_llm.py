"""
direct_llm.py — Flash-MoE style direct GGUF inference (no Ollama)

Key techniques:
  1. Flash-MoE expert offloading: attention layers on GPU, MoE expert tensors
     stay in CPU RAM (only active experts compute = sparse MoE).
  2. MLX-style mmap: GGUF is memory-mapped (use_mmap=True). Unused expert
     pages are never pulled into RAM from disk, eliminating swap pressure.
  3. mlock: active model pages are pinned in RAM (use_mlock=True where
     RAM allows), preventing eviction to swap.
  4. CPU affinity: pin process to 3 cores, leave 1 for OS to prevent hang.
  5. Q4_K_M: models are already 4-bit quantized. We extract the GGUF blobs
     that Ollama stores and feed them directly to llama-cpp-python.
"""

import os, sys, time, json
from pathlib import Path
from typing import Optional

try:
    import psutil
    _proc = psutil.Process()
    # Pin to 3 cores, leave core 0 for OS (i5-4460 has 4 cores)
    _proc.cpu_affinity([1, 2, 3])
    _proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # Windows BELOW_NORMAL
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

# ─── GGUF blob paths (Ollama stores as raw GGUF with sha256 names) ─────────
BLOB_DIR = Path("E:/LLMmodel/blobs")
MODELS = {
    "gemma4-e4b": {
        "blob": "sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a",
        "size_gb": 9.1,
        # Flash-MoE routing: 12 layers on GPU (attn), expert FFN stays in RAM
        "n_gpu_layers": 12,
        "n_ctx": 1024,        # Tiny context → tiny KV cache → fits in 2GB VRAM
        "n_batch": 256,
        "n_threads": 3,
        "use_mmap": True,     # Memory-map GGUF → only active expert pages in RAM
        "use_mlock": True,    # Lock active pages → no swap eviction
        "type_k": "q8_0",     # KV cache quantization (reduces VRAM for KV)
        "type_v": "q8_0",
        "vision": True,
        "audio": True,
        "timeout_s": 900,
        "max_tokens": 60,
    },
    "gemma3-1b": {
        "blob": "sha256-7cd4618c1faf8b7233c6c906dac1694b6a47684b37b8895d470ac688520b9c01",
        "size_gb": 0.76,
        "n_gpu_layers": -1,   # 760MB < 1137MB free VRAM: fully on GPU
        "n_ctx": 2048,
        "n_batch": 512,
        "n_threads": 3,
        "use_mmap": True,
        "use_mlock": True,
        "type_k": "f16",
        "type_v": "f16",
        "vision": True,       # gemma3:1b supports vision
        "audio": False,
        "timeout_s": 120,
        "max_tokens": 250,
    },
    "deepseek-r1-1.5b": {
        "blob": "sha256-aabd4debf0c8f08881923f2c25fc0fdeed24435271c2b3e92c4af36704040dbc",
        "size_gb": 1.04,
        "n_gpu_layers": 20,   # ~1GB close to VRAM limit, partial offload
        "n_ctx": 2048,
        "n_batch": 512,
        "n_threads": 3,
        "use_mmap": True,
        "use_mlock": True,
        "type_k": "f16",
        "type_v": "f16",
        "vision": False,
        "audio": False,
        "timeout_s": 120,
        "max_tokens": 250,
    },
}

# ─── Cache: keep loaded model in memory between calls ─────────────────────
_loaded: dict[str, Llama] = {}

def _log(msg: str):
    sys.stdout.buffer.write((msg + "\n").encode("utf-8"))
    sys.stdout.flush()

def load_model(model_id: str) -> Optional[object]:
    """Load (or return cached) a Llama model using Flash-MoE config."""
    if not HAS_LLAMA_CPP:
        raise RuntimeError("llama-cpp-python not installed. Run: scripts/install_direct_inference.ps1")

    if model_id in _loaded:
        return _loaded[model_id]

    cfg = MODELS[model_id]
    gguf_path = str(BLOB_DIR / cfg["blob"])

    if not Path(gguf_path).exists():
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")

    _log(f"[load] {model_id} ({cfg['size_gb']}GB) | "
         f"GPU layers={cfg['n_gpu_layers']} | ctx={cfg['n_ctx']} | mmap=T mlock=T")

    t0 = time.time()
    # Build kwargs — type_k/type_v only supported in llama-cpp-python >= 0.2.70
    kwargs = dict(
        model_path   = gguf_path,
        n_gpu_layers = cfg["n_gpu_layers"],
        n_ctx        = cfg["n_ctx"],
        n_batch      = cfg["n_batch"],
        n_threads    = cfg["n_threads"],
        use_mmap     = cfg["use_mmap"],
        use_mlock    = cfg["use_mlock"],
        verbose      = False,
    )
    # Conditionally add KV cache quant (requires newer llama-cpp-python)
    try:
        import llama_cpp
        if hasattr(llama_cpp.Llama, 'type_k'):
            kwargs["type_k"] = cfg["type_k"]
            kwargs["type_v"] = cfg["type_v"]
    except Exception:
        pass

    llm = Llama(**kwargs)
    _loaded[model_id] = llm
    _log(f"[load] done in {time.time()-t0:.1f}s")
    return llm


def infer(model_id: str, prompt: str, images_b64: list[str] | None = None) -> dict:
    """
    Run inference directly via llama-cpp-python.
    Returns: {status, response, elapsed_s, tok_s}
    """
    cfg = MODELS[model_id]
    t0 = time.time()
    try:
        llm = load_model(model_id)

        # Vision: embed image as base64 in prompt if supported
        if images_b64 and cfg.get("vision"):
            result = llm.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text",    "text": prompt},
                        {"type": "image_url","image_url": {"url": f"data:image/png;base64,{images_b64[0]}"}}
                    ]
                }],
                max_tokens = cfg["max_tokens"],
                temperature= 0.3,
            )
        else:
            result = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens = cfg["max_tokens"],
                temperature= 0.3,
            )

        elapsed = time.time() - t0
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        n_gen = usage.get("completion_tokens", 1)
        tok_s = round(n_gen / elapsed, 2) if elapsed > 0 else 0

        return {
            "status":    "ok",
            "response":  text.strip()[:400],
            "elapsed_s": round(elapsed, 1),
            "tok_s":     tok_s,
        }
    except Exception as e:
        return {
            "status":    "error",
            "response":  str(e)[:200],
            "elapsed_s": round(time.time()-t0, 1),
            "tok_s":     0,
        }


def unload_model(model_id: str):
    """Free GPU/CPU memory for a model."""
    if model_id in _loaded:
        del _loaded[model_id]
        import gc; gc.collect()
        _log(f"[unload] {model_id} freed")


def system_info() -> dict:
    info = {"llama_cpp": HAS_LLAMA_CPP, "psutil": HAS_PSUTIL}
    if HAS_PSUTIL:
        m = psutil.virtual_memory()
        s = psutil.swap_memory()
        info["ram_free_gb"] = round(m.available / 1024**3, 1)
        info["ram_used_pct"] = m.percent
        info["swap_used_gb"] = round(s.used / 1024**3, 1)
        info["cpu_pct"] = psutil.cpu_percent(interval=1)
    return info


if __name__ == "__main__":
    print(json.dumps(system_info(), indent=2))
    print("\nModels available:")
    for k, v in MODELS.items():
        p = BLOB_DIR / v["blob"]
        ok = Path(p).exists()
        print(f"  {k}: {'[OK]' if ok else '[MISSING]'}  {v['size_gb']}GB  GPU={v['n_gpu_layers']}")
