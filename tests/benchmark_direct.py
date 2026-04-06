"""
benchmark_direct.py — Sequential LLM benchmark using direct GGUF inference
(No Ollama. Uses llama-cpp-python with Flash-MoE + mmap + mlock.)

Execution order (fast → slow):
  1. Gemma3:1b       (~30s/test, fully GPU)
  2. DeepSeek-r1:1.5b (~40s/test, partial GPU)
  3. Gemma4:E4b       (~5-10min/test, Flash-MoE expert routing)

CPU throttling: sleep + psutil gate between every test.
No swap: models are mmapped with mlock (pages pinned in RAM).
"""

import sys, os, json, time, subprocess, importlib
from pathlib import Path
from datetime import datetime

# ── Add scripts/ to path for direct_llm import ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    import psutil
    HAS_PSUTIL = True
    _proc = psutil.Process()
    _proc.cpu_affinity([1, 2, 3])          # Reserve core 0 for OS
    _proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    HAS_PSUTIL = False

import direct_llm as llm_engine

ASSET_DIR = Path(__file__).parent / "assets"
TEST_TEXT = open(ASSET_DIR / "test_text.txt", encoding="utf-8").read()
IMG_B64   = open(ASSET_DIR / "image_b64.txt").read()
AUD_B64   = open(ASSET_DIR / "audio_b64.txt").read()

RESULTS = {}
TOTAL_STEPS = 0
DONE_STEPS  = 0

# ─── Display ────────────────────────────────────────────────────────────────
def pb(done, total, w=30):
    f = int(w * done / total)
    return f"[{'█'*f}{'░'*(w-f)}] {int(100*done/total)}%"

def print_u(s):
    sys.stdout.buffer.write(s.encode("utf-8"))
    sys.stdout.flush()

def step(label):
    global DONE_STEPS
    DONE_STEPS += 1
    print_u(f"\n{pb(DONE_STEPS, TOTAL_STEPS)}  {label}\n")

# ─── CPU Throttle ───────────────────────────────────────────────────────────
def wait_cpu(threshold=60, max_wait=90):
    if not HAS_PSUTIL:
        time.sleep(10)
        return
    waited = 0
    while waited < max_wait:
        cpu = psutil.cpu_percent(interval=2)
        if cpu < threshold:
            return
        print_u(f"  [cool] CPU {cpu:.0f}% — waiting...\n")
        time.sleep(5); waited += 7
    print_u("  [cool] timeout, proceeding\n")

def pause(s=15, label=""):
    print_u(f"  [pause {label}] {s}s cooling down...\n")
    time.sleep(s)
    wait_cpu(threshold=60)

def mem_report():
    if not HAS_PSUTIL:
        return ""
    m = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return f"RAM free:{m.available//1024**3}GB  swap:{sw.used//1024**3}GB  CPU:{psutil.cpu_percent(interval=1):.0f}%"

# ─── TTS ────────────────────────────────────────────────────────────────────
def run_tts(text="Local AI inference without Ollama."):
    try:
        wav = str(ASSET_DIR / "tts_direct_out.wav").replace("/", "\\")
        safe = text.replace("'", "")
        r = subprocess.run(
            ['powershell', '-Command',
             f"Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
             f"$s.SetOutputToWaveFile('{wav}'); $s.Speak('{safe}'); $s.Dispose()"],
            capture_output=True, timeout=20
        )
        ok = Path(wav).exists()
        return {"status": "ok" if ok else "error",
                "response": "SAPI WAV generated" if ok else "SAPI failed",
                "elapsed_s": 2.0, "tok_s": 0}
    except Exception as e:
        return {"status": "error", "response": str(e)[:80], "elapsed_s": 0, "tok_s": 0}

# ─── Per-model test suite ───────────────────────────────────────────────────
def run_model_tests(model_id: str, cfg: dict):
    label = cfg["label"]
    RESULTS[label] = {}
    print_u(f"\n{'='*60}\n  Model: {label}\n  {mem_report()}\n{'='*60}\n")

    # -- Summarization
    step(f"{label} | Summarization")
    RESULTS[label]["summarization"] = llm_engine.infer(
        model_id,
        f"Summarize in 2 sentences:\n\n{TEST_TEXT}",
    )
    r = RESULTS[label]["summarization"]
    print_u(f"  [{r['status']}] {r['elapsed_s']}s | {r['tok_s']} tok/s\n")
    print_u(f"  {r['response'][:120]}\n")
    pause(12, "post-summarize")

    # -- Synonym
    step(f"{label} | Synonym Enhancement")
    RESULTS[label]["synonym"] = llm_engine.infer(
        model_id,
        "Rewrite with sophisticated vocabulary: 'AI is changing the way we work and live.'"
    )
    r = RESULTS[label]["synonym"]
    print_u(f"  [{r['status']}] {r['elapsed_s']}s | {r['tok_s']} tok/s\n")
    print_u(f"  {r['response'][:120]}\n")
    pause(12, "post-synonym")

    # -- Chat
    step(f"{label} | General Chat")
    RESULTS[label]["chat"] = llm_engine.infer(
        model_id,
        "List top 3 benefits of running LLMs locally. Be brief."
    )
    r = RESULTS[label]["chat"]
    print_u(f"  [{r['status']}] {r['elapsed_s']}s | {r['tok_s']} tok/s\n")

    # -- Vision
    if cfg["vision"]:
        pause(12, "pre-vision")
        step(f"{label} | Vision")
        RESULTS[label]["vision"] = llm_engine.infer(
            model_id,
            "Describe this image in one sentence.",
            images_b64=[IMG_B64]
        )
        r = RESULTS[label]["vision"]
        print_u(f"  [{r['status']}] {r['elapsed_s']}s\n  {r['response'][:100]}\n")
    else:
        RESULTS[label]["vision"] = {"status": "not_supported",
            "response": "Model does not support vision.", "elapsed_s": 0, "tok_s": 0}

    # -- Audio / STT
    if cfg["audio"]:
        pause(12, "pre-audio")
        step(f"{label} | Audio STT")
        RESULTS[label]["audio_stt"] = llm_engine.infer(
            model_id,
            "This is base64-encoded audio. Describe what you hear.",
            images_b64=[AUD_B64]
        )
        r = RESULTS[label]["audio_stt"]
        print_u(f"  [{r['status']}] {r['elapsed_s']}s\n  {r['response'][:100]}\n")
    else:
        RESULTS[label]["audio_stt"] = {"status": "not_supported",
            "response": "Audio STT not supported via llama-cpp direct.", "elapsed_s": 0, "tok_s": 0}

    # -- Video (N/A for all models)
    RESULTS[label]["video"] = {"status": "not_supported",
        "response": "Video input not supported in any local model via GGUF.", "elapsed_s": 0, "tok_s": 0}

# ─── Model sequence configuration ───────────────────────────────────────────
MODEL_SEQUENCE = [
    ("gemma3-1b",        {"label": "Gemma3 1B",       "vision": True,  "audio": False}),
    ("deepseek-r1-1.5b", {"label": "DeepSeek-r1 1.5B","vision": False, "audio": False}),
    ("gemma4-e4b",       {"label": "Gemma4 E4B",      "vision": True,  "audio": True }),
]

def count_steps():
    n = 0
    for _, cfg in MODEL_SEQUENCE:
        n += 3  # summarize + synonym + chat
        if cfg["vision"]: n += 1
        if cfg["audio"]:  n += 1
    n += 1  # TTS
    return n

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    global TOTAL_STEPS
    TOTAL_STEPS = count_steps()

    print_u(f"\n{'='*60}\n")
    print_u(f"  Direct GGUF Benchmark  |  {datetime.now():%Y-%m-%d %H:%M}\n")
    print_u(f"  Mode: Flash-MoE mmap+mlock, NO Ollama\n")
    print_u(f"  {mem_report()}\n")
    print_u(f"{'='*60}\n")

    if not llm_engine.HAS_LLAMA_CPP:
        print_u("\n[ERROR] llama-cpp-python not installed!\n")
        print_u("  Run:  powershell -File scripts/install_direct_inference.ps1\n")
        sys.exit(1)

    first = True
    for model_id, cfg in MODEL_SEQUENCE:
        if not first:
            # Unload previous model to free RAM before loading next
            prev_id = MODEL_SEQUENCE[MODEL_SEQUENCE.index((model_id, cfg)) - 1][0]
            llm_engine.unload_model(prev_id)
            pause(25, "model switch + RAM free")
        first = False
        run_model_tests(model_id, cfg)

    # Shared TTS test
    llm_engine.unload_model("gemma4-e4b")
    pause(15, "pre-TTS")
    step("TTS | Windows SAPI (shared)")
    tts_res = run_tts()
    for label in RESULTS:
        RESULTS[label]["tts"] = tts_res
    print_u(f"  [{tts_res['status']}] {tts_res['response']}\n")

    print_u(f"\n{pb(TOTAL_STEPS, TOTAL_STEPS)}  ALL DONE!\n")
    print_u(f"  {mem_report()}\n")

    out = Path(__file__).parent / "benchmark_direct_results.json"
    json.dump(RESULTS, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print_u(f"\nResults -> {out}\n")
