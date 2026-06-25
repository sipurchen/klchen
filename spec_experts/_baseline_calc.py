import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# New standard: Chrome+Edge closed, LINE open
# Measured free: ~489MB
# Freed by closing Chrome: ~350MB, Edge: ~120MB
# LINE stays: -60MB
# Conservative budget: 850MB
budget = 850

models = {
    "Mixtral-8x7B Q2_K":           (32, 85.0,  256.0, 8.0),
    "DeepSeek-R1-Distill-7B Q4":   (28, 155.0, 128.0, 8.0),
    "Gemma4 E4B Q4 (MoE active)":  (46, 42.0,   96.0, 6.0),
    "Qwen2.5-14B Q2_K":            (40, 95.0,  256.0, 10.0),
    "Qwen2.5-7B Q4_K_M":           (28, 145.0, 128.0, 8.0),
    "DeepSeek-R1-1.5B Q8":         (28, 28.0,   32.0, 4.0),
    "Phi-4-mini Q4_K_M (3.8B)":    (32, 55.0,   64.0, 6.0),
}

print(f"New standard baseline: Chrome+Edge closed, LINE open")
print(f"VRAM budget: {budget}MB  (conservative; actual may be ~960MB)")
print()
print(f"{'Model':<38} {'ngl':>4} {'GPU%':>5} {'VRAM MB':>8} {'CPU layers':>11}")
print("-" * 72)
for name, (layers, layer_mb, emb_mb, kv_mb) in models.items():
    kv_total = kv_mb * layers
    avail = budget - emb_mb - kv_total
    ngl = max(0, min(layers, int(avail / layer_mb)))
    vram = emb_mb + kv_total + ngl * layer_mb
    cpu = layers - ngl
    pct = 100 * ngl // layers if layers else 0
    print(f"{name:<38} {ngl:>4} {pct:>4}% {vram:>8.0f} {cpu:>11}")

print()
print("Strategy notes:")
print("  - ngl=0 models: pure CPU+RAM, Spec-Experts KV path still works")
print("  - Gemma4 E4B MoE: embedding+shared layers GPU, experts RAM/disk")
print("  - 26B+ target: use Spec-Experts expert-only VRAM slot (~200MB per active expert)")
