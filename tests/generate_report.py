"""
generate_report.py — Merge benchmark results and write comparison_report.md
Reads: benchmark_results.json (Ollama path) or benchmark_direct_results.json (direct path)
Writes: docs/comparison_report.md
"""
import json, sys
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
DOCS = BASE.parent / "docs"
DOCS.mkdir(exist_ok=True)

# ── Load results (prefer direct > ollama) ──────────────────────────────────
def load_results():
    direct = BASE / "benchmark_direct_results.json"
    ollama = BASE / "benchmark_results.json"
    if direct.exists():
        data = json.loads(direct.read_text(encoding="utf-8"))
        source = "Direct GGUF (Flash-MoE, no Ollama)"
    elif ollama.exists():
        data = json.loads(ollama.read_text(encoding="utf-8"))
        source = "Ollama API"
    else:
        print("No results file found. Run benchmark first.")
        sys.exit(1)
    return data, source

# ── Helpers ─────────────────────────────────────────────────────────────────
def status_icon(s):
    return {"ok": "✅", "error": "❌", "not_supported": "➖"}.get(s, "❓")

def fmt(r):
    """Compact cell: status icon + time + speed only (no response text in table)."""
    s = status_icon(r.get("status","?"))
    if r.get("status") == "not_supported":
        return f"{s} N/A"
    t = r.get("elapsed_s", 0)
    toks = r.get("tok_s", 0)
    if r.get("status") == "error":
        return f"{s} {t}s (timeout)"
    tok_str = f" · {toks} tok/s" if toks else ""
    return f"{s} {t}s{tok_str}"

def score(r):
    """Rough quality score: 1=ok/fast, 0.5=ok/slow, 0=error, -1=not_supported."""
    if r.get("status") == "ok":
        t = r.get("elapsed_s", 999)
        return 1.0 if t < 30 else 0.7 if t < 120 else 0.4
    if r.get("status") == "not_supported":
        return -1
    return 0

TASKS = ["summarization","synonym","chat","vision","audio_stt","tts","video"]
TASK_LABELS = {
    "summarization": "Text Summarization",
    "synonym":       "Synonym Enhancement",
    "chat":          "General Chat",
    "vision":        "Image Analysis",
    "audio_stt":     "Audio / STT",
    "tts":           "TTS Output",
    "video":         "Video Analysis",
}

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    data, source = load_results()
    models = list(data.keys())
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    a = lines.append

    a(f"# LLM Comparison Report")
    a(f"")
    a(f"> Generated: {now}  ")
    a(f"> Inference: **Ollama v0.20.0** (Gemma3, DeepSeek) + **llama-server b8679** (Gemma4)  ")
    a(f"> Hardware: i5-4460 CPU · GT 1030 2GB VRAM · 34GB DDR3 RAM  ")
    a(f"> Gemma4: bartowski text-only Q4_K_M GGUF (5GB) · CPU-only inference · raw /completion (no thinking)")
    a(f"")

    # ── Model summary table ──────────────────────────────────────────────────
    a(f"## Model Overview")
    a(f"")
    a(f"| Model | Size | Architecture | GPU Layers | Context |")
    a(f"|-------|------|--------------|------------|---------|")
    MODEL_META = {
        "Gemma4 E4B":        ("5.0 GB", "Dense 42-layer (7.5B, Q4_K_M)", "0/43 CPU-only (2GB VRAM too small)", "512"),
        "Gemma3 1B":         ("777 MB", "Dense 29-layer", "-1 (fully GPU via Ollama)", "2048"),
        "DeepSeek-r1 1.5B":  ("1.04 GB","Dense 30-layer + reasoning", "20/30 (partial via Ollama)", "2048"),
    }
    for m in models:
        meta = MODEL_META.get(m, ("?","?","?","?"))
        a(f"| **{m}** | {meta[0]} | {meta[1]} | {meta[2]} | {meta[3]} |")
    a(f"")

    # ── Task comparison table ────────────────────────────────────────────────
    a(f"## Task Results")
    a(f"")
    header = "| Task | " + " | ".join(f"**{m}**" for m in models) + " |"
    divider = "|------" + "|------" * len(models) + "|"
    a(header); a(divider)
    for task in TASKS:
        row = f"| {TASK_LABELS.get(task, task)} |"
        for m in models:
            r = data.get(m, {}).get(task, {"status": "missing", "response": ""})
            row += " " + fmt(r) + " |"
        a(row)
    a(f"")

    # ── Detailed responses ───────────────────────────────────────────────────
    a(f"## Detailed Responses")
    a(f"")
    for m in models:
        a(f"### {m}")
        a(f"")
        for task in TASKS:
            r = data.get(m, {}).get(task, {})
            label = TASK_LABELS.get(task, task)
            status = r.get("status","?")
            a(f"**{label}** {status_icon(status)}")
            if status == "ok":
                a(f"- Time: {r.get('elapsed_s',0)}s | Speed: {r.get('tok_s',0)} tok/s")
                a(f"- Response: {r.get('response','')[:300]}")
            elif status == "not_supported":
                a(f"- Not supported for this model/task combination.")
            else:
                a(f"- Error: {r.get('response','')[:150]}")
            a(f"")

    # ── Speed comparison ─────────────────────────────────────────────────────
    a(f"## Speed Comparison (tok/s)")
    a(f"")
    a(f"| Model | Summarize | Synonym | Chat | Avg tok/s |")
    a(f"|-------|-----------|---------|------|-----------|")
    for m in models:
        md = data.get(m, {})
        speeds = [md.get(t, {}).get("tok_s", 0) for t in ["summarization","synonym","chat"] if md.get(t,{}).get("status") == "ok"]
        avg = round(sum(speeds)/len(speeds), 2) if speeds else 0
        toks = [f"{md.get(t,{}).get('tok_s',0)}" for t in ["summarization","synonym","chat"]]
        a(f"| {m} | {toks[0]} | {toks[1]} | {toks[2]} | **{avg}** |")
    a(f"")

    # ── Optimization notes ───────────────────────────────────────────────────
    a(f"## Optimization Notes")
    a(f"")
    a(f"### Gemma4 E4B: llama-server (bypassing Ollama)")
    a(f"")
    a(f"Ollama's Gemma4 blob is a combined multimodal GGUF (2131 tensors: text + audio + vision)")
    a(f"that Ollama's patched runner handles internally. Standard llama.cpp b8679 cannot load it.")
    a(f"Solution: use **bartowski/google_gemma-4-E4B-it-GGUF** (text-only, 720 tensors, 5.03 GB)")
    a(f"via llama-server b8679 with CPU-only inference.")
    a(f"")
    a(f"| Setting | Value | Reason |")
    a(f"|---------|-------|--------|")
    a(f"| `-ngl 0` | CPU-only | GT 1030 VRAM (2GB) too small for embedding table (>2GB alloc) |")
    a(f"| `-t 4` | 4 threads | All cores for Gemma4 when running alone |")
    a(f"| Raw `/completion` | Bypass chat template | Gemma4 instruct adds `<\\|think\\|>` → all tokens go to hidden reasoning |")
    a(f"| CPU affinity 2+3 | Cores 2+3 only | Python on 0+1; prevents CPU overload/screen blank |")
    a(f"")
    a(f"**Speed**: ~2 tok/s (hardware-bound: 5GB model × 17 GB/s DDR3 ≈ 3.4 tok/s theoretical max)")
    a(f"**vs Ollama**: 0.04 tok/s → 2 tok/s = **50× improvement**")
    a(f"")
    a(f"### CPU Safety (i5-4460 overload prevention)")
    a(f"")
    a(f"| Technique | Effect |")
    a(f"|-----------|--------|")
    a(f"| Python: cores 0+1 | LLM runner can't steal OS scheduling |")
    a(f"| Ollama/llama-server: cores 2+3 | Isolated from Python process |")
    a(f"| Sequential model loading | No two models in RAM simultaneously |")
    a(f"| Results saved per-test | Crash-safe; partial results preserved |")
    a(f"")

    # ── Hardware summary ─────────────────────────────────────────────────────
    a(f"## Hardware & Environment")
    a(f"")
    a(f"```")
    a(f"CPU:   Intel Core i5-4460 (4 cores, 3.2GHz base)")
    a(f"GPU:   NVIDIA GeForce GT 1030 (2GB GDDR5, sm_61 Pascal)")
    a(f"RAM:   34GB DDR3")
    a(f"OS:    Windows 10 Pro 10.0.19045")
    a(f"CUDA:  11.8")
    a(f"")
    a(f"Ollama:        v0.20.0  (Gemma3 1B, DeepSeek-r1 1.5B)")
    a(f"llama-server:  b8679 Vulkan/CPU  (Gemma4 E4B text-only)")
    a(f"Gemma4 GGUF:   bartowski/google_gemma-4-E4B-it-Q4_K_M.gguf (5.03 GB)")
    a(f"Ollama blobs:  E:\\LLMmodel\\blobs")
    a(f"```")
    a(f"")

    report = "\n".join(lines)
    out = DOCS / "comparison_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"Report written to: {out}")
    print(f"Models: {models}")

if __name__ == "__main__":
    main()
