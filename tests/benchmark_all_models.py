# ============================================================================
# LLM Benchmark — CPU-Safe Sequential Runner
# i5-4460 / GT 1030 / 34GB RAM
#
# CPU safety strategy:
#   • Python process: cores 0+1  (affinity set at startup)
#   • Ollama / llama-server: cores 2+3  (set via psutil after start)
#   • num_thread=2 in Modelfile  (llama.cpp worker thread cap)
#   • OLLAMA_MAX_LOADED_MODELS=1 (no dual-load RAM spikes)
#   • Results saved after EVERY test  (crash = partial results preserved)
#   • Streaming API + short timeouts  (quick failure, no hang)
#
# Gemma4 E4B uses llama-server (NOT Ollama) with:
#   • bartowski text-only GGUF (5GB, 720 tensors, no multimodal encoders)
#   • CPU-only -ngl 0 (GT 1030 VRAM too small for embedding table)
#   • Raw /completion endpoint (bypass thinking chat template)
# ============================================================================
import httpx, json, time, sys, os, subprocess, signal
from pathlib import Path
from datetime import datetime

try:
    import psutil
    _self = psutil.Process()
    _self.cpu_affinity([0, 1])          # Python on cores 0+1
    _self.nice(psutil.NORMAL_PRIORITY_CLASS)
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

OLLAMA        = "http://localhost:11434"
LLAMASERVER   = "http://localhost:8080"
LLAMASERVER_EXE = Path(__file__).parent.parent / "bin/llama-cpp/llama-server.exe"
def _llms_dir() -> Path:
    import os
    if os.environ.get("LLMS_DIR"):
        return Path(os.environ["LLMS_DIR"])
    s = Path(__file__).parent.parent.parent / "LLMmodel"
    return s if s.exists() else Path(__file__).parent.parent / "LLMmodel"

GEMMA4_GGUF   = _llms_dir() / "gemma4_textonly" / "google_gemma-4-E4B-it-Q4_K_M.gguf"
ASSET_DIR     = Path(__file__).parent / "assets"
OUT_FILE      = Path(__file__).parent / "benchmark_results.json"
RESULTS       = {}
_llamaserver_proc = None

# Load assets
TEST_TEXT = open(ASSET_DIR / "test_text.txt", encoding="utf-8").read()
IMG_B64   = open(ASSET_DIR / "image_b64.txt").read()
AUD_B64   = open(ASSET_DIR / "audio_b64.txt").read()

# ─── Helpers ────────────────────────────────────────────────────────────────
def pu(s):
    sys.stdout.buffer.write(s.encode("utf-8"))
    sys.stdout.flush()

def save():
    """Write results to disk immediately — survives crashes."""
    json.dump(RESULTS, open(OUT_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def mem():
    if not HAS_PSUTIL: return ""
    m = psutil.virtual_memory()
    return f"RAM:{m.available//1024**3}GB free  swap:{psutil.swap_memory().used//1024**3}GB  CPU:{psutil.cpu_percent(interval=1):.0f}%"

TOTAL = 0
DONE  = 0
def step(label):
    global DONE
    DONE += 1
    f = int(30 * DONE / max(TOTAL, 1))
    bar = "█"*f + "░"*(30-f)
    pu(f"\n[{bar}] {int(100*DONE/max(TOTAL,1))}%  {label}\n")

# ─── llama-server lifecycle ───────────────────────────────────────────────────
def start_llamaserver():
    global _llamaserver_proc
    if not LLAMASERVER_EXE.exists():
        pu(f"  [llama-server] binary not found: {LLAMASERVER_EXE}\n")
        return False
    if not GEMMA4_GGUF.exists():
        pu(f"  [llama-server] GGUF not found: {GEMMA4_GGUF}\n")
        return False
    pu("  [llama-server] starting (CPU-only, ngl=0, t=4)...\n")
    _llamaserver_proc = subprocess.Popen(
        [str(LLAMASERVER_EXE), "-m", str(GEMMA4_GGUF),
         "-ngl", "0", "-c", "512", "-t", "4", "-np", "1",
         "--host", "127.0.0.1", "--port", "8080"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    # Pin to cores 2+3
    if HAS_PSUTIL:
        try:
            proc = psutil.Process(_llamaserver_proc.pid)
            proc.cpu_affinity([2, 3])
        except Exception as e:
            pu(f"  [llama-server] affinity warn: {e}\n")
    # Wait for server ready
    for i in range(60):
        time.sleep(1)
        try:
            r = httpx.get(f"{LLAMASERVER}/health", timeout=2)
            if r.status_code == 200:
                pu(f"  [llama-server] ready after {i+1}s\n")
                return True
        except Exception:
            pass
    pu("  [llama-server] timed out waiting for ready\n")
    return False

def stop_llamaserver():
    global _llamaserver_proc
    if _llamaserver_proc:
        pu("  [llama-server] stopping...\n")
        _llamaserver_proc.terminate()
        try:
            _llamaserver_proc.wait(timeout=10)
        except Exception:
            _llamaserver_proc.kill()
        _llamaserver_proc = None

# ─── Ollama process management ───────────────────────────────────────────────
def pin_ollama_cores():
    """Set Ollama to cores 2+3 immediately after it starts."""
    if not HAS_PSUTIL: return
    for p in psutil.process_iter(['pid', 'name']):
        if 'ollama' in p.info['name'].lower():
            try:
                proc = psutil.Process(p.info['pid'])
                proc.cpu_affinity([2, 3])
                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                pu(f"  [affinity] Ollama PID {p.info['pid']} pinned to cores 2+3\n")
            except Exception as e:
                pu(f"  [affinity] warn: {e}\n")

def unload_model(model_id):
    try:
        httpx.post(f"{OLLAMA}/api/generate",
            json={"model": model_id, "prompt": "", "keep_alive": 0},
            timeout=8)
        pu(f"  [unload] {model_id}\n")
    except Exception:
        pass

def unload_all():
    try:
        r = httpx.get(f"{OLLAMA}/api/ps", timeout=5)
        for m in r.json().get("models", []):
            unload_model(m["name"])
    except Exception:
        pass

# ─── CPU gate ────────────────────────────────────────────────────────────────
def cool(seconds=15, cpu_limit=70, tag=""):
    pu(f"  [cool{' '+tag if tag else ''}] {seconds}s pause...\n")
    time.sleep(seconds)
    if not HAS_PSUTIL: return
    # Re-pin Ollama every time (it may have restarted)
    pin_ollama_cores()
    waited = 0
    while waited < 60:
        cpu = psutil.cpu_percent(interval=2)
        if cpu < cpu_limit: return
        pu(f"  [cool] CPU {cpu:.0f}% > {cpu_limit}%, waiting...\n")
        time.sleep(5); waited += 7

# ─── Inference — streaming to avoid hang ─────────────────────────────────────
def _strip_think(text: str) -> str:
    """Remove DeepSeek-r1 <think>...</think> blocks; return visible output."""
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

def infer(model, prompt, images=None, timeout=90, max_tokens=150):
    is_deepseek = "deepseek" in model.lower()
    # DeepSeek-r1 uses <think> tokens that consume budget; give it more room
    effective_max = max_tokens * 3 if is_deepseek else max_tokens
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = images
    t0 = time.time()
    text = ""
    try:
        with httpx.stream("POST", f"{OLLAMA}/api/chat",
            json={"model": model, "messages": [msg], "stream": True,
                  "keep_alive": "30m",
                  "options": {"num_predict": effective_max, "temperature": 0.3,
                               "num_thread": 2}},
            timeout=timeout) as resp:
            for line in resp.iter_lines():
                if not line: continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                delta = chunk.get("message", {}).get("content", "")
                text += delta
                if chunk.get("done"):
                    elapsed = time.time() - t0
                    ec = chunk.get("eval_count", 0)
                    ed = chunk.get("eval_duration", 1) / 1e9
                    tok_s = round(ec / max(ed, 0.001), 2)
                    visible = _strip_think(text) if is_deepseek else text.strip()
                    return {"status": "ok", "response": visible[:400],
                            "elapsed_s": round(elapsed, 1), "tok_s": tok_s}
        elapsed = time.time() - t0
        visible = _strip_think(text) if is_deepseek else text.strip()
        return {"status": "ok", "response": visible[:400] or "(empty)",
                "elapsed_s": round(elapsed, 1), "tok_s": 0}
    except Exception as e:
        return {"status": "error", "response": str(e)[:120],
                "elapsed_s": round(time.time()-t0, 1), "tok_s": 0}

# ─── llama-server inference (Gemma4) ────────────────────────────────────────
def infer_llamaserver(prompt, timeout=180, max_tokens=80):
    """Use raw /completion endpoint to bypass Gemma4's thinking chat template."""
    # Build Gemma4 instruct prompt manually (no <|think|> insertion)
    formatted = (
        f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    t0 = time.time()
    try:
        resp = httpx.post(f"{LLAMASERVER}/completion",
            json={"prompt": formatted, "n_predict": max_tokens,
                  "temperature": 0.3, "stop": ["<end_of_turn>", "<eos>"]},
            timeout=timeout)
        data = resp.json()
        text = data.get("content", "").strip()
        tok_gen  = data.get("tokens_predicted", 0)
        elapsed  = time.time() - t0
        tok_s    = round(tok_gen / max(elapsed, 0.001), 2)
        return {"status": "ok", "response": text[:400] or "(empty)",
                "elapsed_s": round(elapsed, 1), "tok_s": tok_s}
    except Exception as e:
        return {"status": "error", "response": str(e)[:120],
                "elapsed_s": round(time.time()-t0, 1), "tok_s": 0}

def run_gemma4():
    """Run Gemma4 E4B suite via llama-server."""
    label = "Gemma4 E4B"
    RESULTS[label] = {}
    pu(f"\n{'='*55}\n  {label}  backend=llama-server  ngl=0  t=4\n")
    pu(f"  {mem()}\n{'='*55}\n")

    def do(key, prompt):
        step(f"{label} | {key}")
        r = infer_llamaserver(prompt, timeout=300, max_tokens=80)
        RESULTS[label][key] = r
        save()
        icon = "ok" if r["status"] == "ok" else "ERR"
        pu(f"  [{icon}] {r['elapsed_s']}s | {r['tok_s']} tok/s\n")
        pu(f"  {r['response'][:100]}\n")

    do("summarization", f"Summarize in 2 sentences:\n\n{TEST_TEXT}")
    cool(12, tag="post-summ")

    do("synonym", "Rewrite with sophisticated vocabulary: 'AI is changing the way we work and live.'")
    cool(12, tag="post-syn")

    do("chat", "Top 3 benefits of local LLMs. One line each.")

    # Vision not supported (text-only GGUF)
    for key in ("vision", "audio_stt", "video"):
        RESULTS[label][key] = {"status": "not_supported",
            "response": "Not available in text-only GGUF.", "elapsed_s": 0, "tok_s": 0}
    save()

# ─── TTS ─────────────────────────────────────────────────────────────────────
def run_tts():
    try:
        wav = str(ASSET_DIR / "tts_out.wav").replace("/", "\\")
        cmd = ['powershell', '-Command',
               "Add-Type -AssemblyName System.Speech; "
               "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
               f"$s.SetOutputToWaveFile('{wav}'); "
               "$s.Speak('Local AI inference on your own hardware.'); $s.Dispose()"]
        subprocess.run(cmd, capture_output=True, timeout=20)
        ok = Path(wav).exists()
        return {"status": "ok" if ok else "error",
                "response": "WAV file created via Windows SAPI" if ok else "SAPI failed",
                "elapsed_s": 2.0, "tok_s": 0}
    except Exception as e:
        return {"status": "error", "response": str(e)[:80], "elapsed_s": 0, "tok_s": 0}

# ─── Per-model suite ──────────────────────────────────────────────────────────
def run_model(mid, label, timeout, max_tok, has_vision, has_audio):
    RESULTS[label] = {}
    pu(f"\n{'='*55}\n  {label}  timeout={timeout}s  max_tokens={max_tok}\n")
    pu(f"  {mem()}\n{'='*55}\n")

    def do(key, prompt, imgs=None):
        step(f"{label} | {key}")
        r = infer(mid, prompt, images=imgs, timeout=timeout, max_tokens=max_tok)
        RESULTS[label][key] = r
        save()   # ← persist immediately
        icon = "ok" if r["status"] == "ok" else "ERR"
        pu(f"  [{icon}] {r['elapsed_s']}s | {r['tok_s']} tok/s\n")
        pu(f"  {r['response'][:100]}\n")

    do("summarization", f"Summarize in 2 sentences:\n\n{TEST_TEXT}")
    cool(12, tag="post-summ")

    do("synonym", "Rewrite with sophisticated vocabulary: 'AI is changing the way we work and live.'")
    cool(12, tag="post-syn")

    do("chat", "Top 3 benefits of local LLMs. One line each.")

    if has_vision:
        cool(12, tag="pre-vision")
        do("vision", "What objects and scene are shown in this image? Describe briefly.", imgs=[IMG_B64])
    else:
        RESULTS[label]["vision"] = {"status": "not_supported",
            "response": "Vision not supported.", "elapsed_s": 0, "tok_s": 0}
        save()

    if has_audio:
        cool(12, tag="pre-audio")
        do("audio_stt", "What can you infer from this base64 audio data?", imgs=[AUD_B64])
    else:
        RESULTS[label]["audio_stt"] = {"status": "not_supported",
            "response": "Audio STT not supported via Ollama.", "elapsed_s": 0, "tok_s": 0}
        save()

    RESULTS[label]["video"] = {"status": "not_supported",
        "response": "Video not supported in any local GGUF model.", "elapsed_s": 0, "tok_s": 0}
    save()

# ─── Ollama model sequence ────────────────────────────────────────────────────
OLLAMA_SEQUENCE = [
    # (ollama_id, label, timeout, max_tokens, vision, audio)
    ("gemma3:1b",        "Gemma3 1B",        90,  150, True,  False),
    ("deepseek-r1:1.5b", "DeepSeek-r1 1.5B", 90,  150, False, False),
]

def count_steps():
    n = 1  # TTS
    n += 3  # Gemma4: summ + synonym + chat (vision/audio = not_supported)
    for *_, vision, audio in OLLAMA_SEQUENCE:
        n += 3 + (1 if vision else 0) + (1 if audio else 0)
    return n

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TOTAL = count_steps()

    pu(f"\n{'='*55}\n  LLM Benchmark  |  {datetime.now():%Y-%m-%d %H:%M}\n")
    pu(f"  Sequential · CPU-safe · incremental saves\n")
    pu(f"  {mem()}\n{'='*55}\n")

    pin_ollama_cores()
    unload_all()
    cool(5, tag="init")

    # Run Ollama models
    ids = [s[0] for s in OLLAMA_SEQUENCE]
    for i, (mid, label, timeout, max_tok, vis, aud) in enumerate(OLLAMA_SEQUENCE):
        if i > 0:
            unload_model(ids[i-1])
            cool(25, tag=f"switch→{label}")
        run_model(mid, label, timeout, max_tok, vis, aud)

    # Unload all Ollama models before starting llama-server
    unload_all()
    cool(20, tag="pre-gemma4")

    # Run Gemma4 via llama-server
    if start_llamaserver():
        run_gemma4()
        stop_llamaserver()
    else:
        pu("  [llama-server] failed to start, skipping Gemma4\n")
        RESULTS["Gemma4 E4B"] = {k: {"status": "error",
            "response": "llama-server failed to start",
            "elapsed_s": 0, "tok_s": 0}
            for k in ("summarization","synonym","chat","vision","audio_stt","video","tts")}
        save()

    cool(10, tag="pre-TTS")

    step("TTS | Windows SAPI")
    tts = run_tts()
    for lbl in RESULTS:
        RESULTS[lbl]["tts"] = tts
    save()
    pu(f"  [{tts['status']}] {tts['response']}\n")

    pu(f"\n[{'█'*30}] 100%  DONE\n")
    pu(f"  {mem()}\n")
    pu(f"\nResults → {OUT_FILE}\n")
    pu("Run: python tests/generate_report.py\n")
