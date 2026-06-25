"""
GT 1030 (2GB VRAM) Layer Offload Calculator
Given a model, compute max --n-gpu-layers to stay under VRAM budget.
"""

from dataclasses import dataclass

VRAM_BUDGET_MB = 1800  # leave 200MB for Vulkan overhead
VULKAN_OVERHEAD_MB = 200


@dataclass
class ModelProfile:
    name: str
    total_layers: int
    layer_size_mb: float    # per transformer layer (attn + ffn)
    embedding_mb: float
    kv_per_layer_mb: float  # at ctx=2048, f16


KNOWN_MODELS = {
    "mixtral-8x7b-q2": ModelProfile(
        name="Mixtral-8x7B Q2_K",
        total_layers=32,
        layer_size_mb=85.0,   # Q2_K: ~85MB/layer for 7B active
        embedding_mb=256.0,
        kv_per_layer_mb=8.0,  # ctx=2048
    ),
    "deepseek-r1-7b-q4": ModelProfile(
        name="DeepSeek-R1-Distill-7B Q4_K_M",
        total_layers=28,
        layer_size_mb=155.0,
        embedding_mb=128.0,
        kv_per_layer_mb=8.0,
    ),
    "gemma4-e4b-q4": ModelProfile(
        name="Gemma4 E4B Q4_K_M (MoE active)",
        total_layers=46,
        layer_size_mb=42.0,   # MoE: only active experts loaded
        embedding_mb=96.0,
        kv_per_layer_mb=6.0,
    ),
    "qwen25-14b-q2": ModelProfile(
        name="Qwen2.5-14B Q2_K",
        total_layers=40,
        layer_size_mb=95.0,
        embedding_mb=256.0,
        kv_per_layer_mb=10.0,
    ),
}


def calc_max_gpu_layers(model: ModelProfile, ctx: int = 2048) -> dict:
    kv_total = model.kv_per_layer_mb * model.total_layers
    available = VRAM_BUDGET_MB - model.embedding_mb - kv_total
    max_layers = max(0, int(available / model.layer_size_mb))
    max_layers = min(max_layers, model.total_layers)

    vram_used = model.embedding_mb + kv_total + max_layers * model.layer_size_mb

    return {
        "model": model.name,
        "total_layers": model.total_layers,
        "max_gpu_layers": max_layers,
        "gpu_pct": f"{100 * max_layers / model.total_layers:.0f}%",
        "vram_used_mb": f"{vram_used:.0f}",
        "remaining_on_cpu": model.total_layers - max_layers,
        "feasible": max_layers > 0,
    }


if __name__ == "__main__":
    print(f"GT 1030 VRAM budget: {VRAM_BUDGET_MB}MB\n")
    print(f"{'Model':<35} {'ngl':>4} {'GPU%':>5} {'VRAM(MB)':>9} {'Feasible':>8}")
    print("-" * 68)
    for key, profile in KNOWN_MODELS.items():
        r = calc_max_gpu_layers(profile)
        print(
            f"{r['model']:<35} {r['max_gpu_layers']:>4} "
            f"{r['gpu_pct']:>5} {r['vram_used_mb']:>9} {str(r['feasible']):>8}"
        )
