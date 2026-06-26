import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

budget = 850  # standard: Chrome+Edge closed, LINE open

# (layers, layer_mb, emb_mb, kv_mb_per_layer@4K_ctx, total_size_gb, note)
models = {
    "Qwen2.5-Coder-1.5B Q8":        (28,  22.0,  32.0, 3.0, 1.6, "full VRAM fit possible"),
    "Qwen3-1.7B Q4_K_M":            (28,  28.0,  32.0, 3.5, 1.0, "thinking mode, tiny"),
    "SmolLM2-1.7B Q8":              (24,  26.0,  32.0, 3.0, 1.8, "edge focus, fast"),
    "DeepSeek-R1-Distill-1.5B Q8":  (28,  28.0,  32.0, 3.5, 1.6, "reasoning distill"),
    "Qwen2.5-Coder-3B Q4_K_M":      (36,  40.0,  48.0, 4.0, 1.9, "strong code, best 3B"),
    "Qwen3-4B Q4_K_M":              (36,  55.0,  64.0, 5.0, 2.6, "thinking, best ratio"),
    "Phi-4-mini 3.8B Q4_K_M":       (32,  55.0,  64.0, 6.0, 2.5, "MS code focus"),
    "Gemma3-4B Q4_K_M":             (34,  55.0,  64.0, 5.0, 2.6, "multimodal capable"),
    "DeepSeek-Coder-V2-Lite Q4":    (27,  48.0,  64.0, 5.0, 8.9, "16B MoE 2.4B active"),
    "Qwen2.5-Coder-7B Q2_K":        (28, 100.0, 128.0, 8.0, 4.2, "larger, but Q2 quality hit"),
}

print("GT 1030 850MB standard -- 2026 coding LLM evaluation (ctx=4K, Vulkan)")
print()
print(f"{'Model':<34} {'ngl':>4} {'GPU%':>5} {'VRAM':>6} {'Size':>7}  Note")
print("-" * 84)
for name, (layers, lmb, emb, kv, size, note) in models.items():
    kv_total = kv * layers
    avail = budget - emb - kv_total
    ngl = max(0, min(layers, int(avail / lmb)))
    vram = emb + kv_total + ngl * lmb
    pct = 100 * ngl // layers
    flag = " <<BEST" if pct >= 70 else (" <OK" if pct >= 40 else "")
    print(f"{name:<34} {ngl:>4} {pct:>4}% {vram:>6.0f} {size:>6.1f}GB  {note}{flag}")

print()
print("Legend: <<BEST = 70%+ GPU  <OK = 40%+ GPU  (blank) = mostly CPU+RAM")
print()
print("Spec-Experts note:")
print("  DeepSeek-Coder-V2-Lite: MoE 2.4B active -> expert RAM pool fits well")
print("  All models: KV offload to disk unlocks 16K+ context regardless of ngl")
