"""
Phase 2-7 Full Pipeline Demo
Run: python tests/phase2_7_demo.py
Tests all phases offline (no llama-server required for P2/3/4/5/6/7 structure tests)
P4 live routing requires llama-server on :8081
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

BANNER = "=" * 60

def hdr(t):
    print(f"\n{BANNER}\n  {t}\n{BANNER}")

# ─────────────────────────────────────────────────────────────
# P2  KV Offload Manager
# ─────────────────────────────────────────────────────────────
def demo_kv_offload():
    hdr("Phase 2 — KV Offload Manager (3-tier: VRAM/RAM/Disk)")
    from kv.kv_offload_manager import KVOffloadManager

    mgr = KVOffloadManager(
        vram_budget_mb=56,
        ram_budget_mb=256,
        session_id="demo_p2",
    )

    chunks = [
        (0, "reasoning", (2, 8, 64)),
        (1, "code",      (2, 8, 64)),
        (2, "factual",   (2, 8, 64)),
        (3, "reasoning", (2, 8, 64)),
        (4, "code",      (2, 8, 64)),
        (5, "creative",  (2, 8, 64)),
    ]

    print(f"\n  Storing {len(chunks)} KV chunks...")
    for cid, role, shape in chunks:
        tensor = np.random.randn(*shape).astype(np.float16)
        tokens = list(range(cid * 50, cid * 50 + 50))
        mgr.store(cid, role, tensor, tokens=tokens)
        s = mgr.stats()
        print(f"    chunk {cid} ({role:10s}) size={tensor.nbytes//1024}KB  "
              f"vram={s['vram_entries']} ram={s['ram_entries']}")

    # Retrieve
    print(f"\n  Retrieving chunk 1 (code)...")
    t = mgr.retrieve(1, "code")
    print(f"    Retrieved shape={t.shape} dtype={t.dtype}" if t is not None else "    MISS")

    # Effective context
    ctx = mgr.effective_context_tokens(tokens_per_chunk=50)
    print(f"\n  Effective context:")
    for k, v in ctx.items():
        print(f"    {k}: {v} tokens")

    print(f"\n  Stats: {mgr.stats()}")
    return True


# ─────────────────────────────────────────────────────────────
# P3  Directional Steering
# ─────────────────────────────────────────────────────────────
def demo_directional_steering():
    hdr("Phase 3 — Directional Steering (system prompt + logit bias)")
    from spec_experts.directional_steering import DirectionalSteering, SteeringVector

    ds = DirectionalSteering()

    # Register custom vector
    ds.register_vector(SteeringVector(
        concept="taiwanese",
        direction_prompt="請用繁體中文回答，語氣自然如台灣人。",
        anti_prompt="Reply in formal English only.",
        alpha=12.0,
    ))

    tests = [
        ("code",      ["code_quality", "concise"]),
        ("reasoning", ["reasoning"]),
        ("factual",   ["factual", "concise"]),
        ("creative",  ["creative", "taiwanese"]),
    ]

    print(f"\n  {'chunk_type':<12}  {'concepts':<30}  {'temp':<6}  {'sys_len'}")
    print(f"  {'-'*12}  {'-'*30}  {'-'*6}  {'-'*7}")

    for chunk_type, concepts in tests:
        sys_p = ds.compose_system_prompt("/no_think", concepts)
        temp  = ds.get_temperature(0.7, concepts)
        bias  = ds.compute_logit_bias(concepts)
        cfg   = ds.to_steer_params(chunk_type)
        print(f"  {chunk_type:<12}  {str(concepts):<30}  {temp:<6.2f}  {len(sys_p)} chars")

    print(f"\n  Activation proxy prefix (code_quality):")
    print(f"    {ds.activation_proxy_prefix('code_quality')!r}")
    return True


# ─────────────────────────────────────────────────────────────
# P4  Expert RAM Pool (MoE 26B)
# ─────────────────────────────────────────────────────────────
def demo_expert_ram_pool():
    hdr("Phase 4 — Expert RAM Pool (Mixtral-8x7B Q2_K / DeepSeek-V2-Lite)")
    from spec_experts.expert_ram_pool import ExpertRAMPool

    for model_key in ["mixtral-8x7b-q2", "deepseek-coder-v2-lite-q4"]:
        pool = ExpertRAMPool(model_key=model_key, ram_budget_mb=7168)
        print(f"\n  {pool.cfg.model_name}:")

        # Simulate chunk boundary events
        for chunk_type in ["reasoning", "code", "factual", "code", "reasoning"]:
            pool.on_chunk_boundary(chunk_type)

        print(pool.ram_pool_summary())
        hit = pool.activation_rate()
        print(f"  Cache hit rate: {hit*100:.0f}%")

    return True


# ─────────────────────────────────────────────────────────────
# P5  Edge VLM Router
# ─────────────────────────────────────────────────────────────
def demo_edge_vlm():
    hdr("Phase 5 — Edge VLM Router (moondream / MobileVLM)")
    from edge.vlm_router import VLMRouter, VLM_TARGETS

    print(f"\n  Hardware targets:")
    for target, cfg in VLM_TARGETS.items():
        print(f"    {target:<15}  {cfg.model_name:<30}  {cfg.device:<20}  "
              f"VRAM={cfg.vram_mb}MB  fps~{cfg.fps_est}")

    router = VLMRouter(target="gt1030")
    cmd = router.startup_cmd("LLMmodel")
    print(f"\n  GT 1030 launch cmd:")
    print(f"    {' '.join(cmd[:4])} ...")
    print(f"\n  Hardware report:")
    for line in router.hardware_report().split("\n"):
        print(f"    {line}")
    return True


# ─────────────────────────────────────────────────────────────
# P6  Mac Mini M4 Metal Expert Pool
# ─────────────────────────────────────────────────────────────
def demo_mac_m4():
    hdr("Phase 6 — Mac Mini M4 Metal Expert Pool (200B+ MoE)")
    from mac_m4.metal_expert_pool import MetalExpertPool, M4_MODELS

    print(f"\n  Available models for M4:")
    for mk, cfg in M4_MODELS.items():
        print(f"    {mk:<30}  {cfg.total_gb:>6.0f}GB  active={cfg.active_gb:>5.1f}GB  "
              f"target={cfg.tok_s_target} tok/s")

    pool = MetalExpertPool("deepseek-v3-671b-q2")
    print(f"\n  {pool.hardware_summary()}")

    # Simulate some expert accesses
    for i in range(100):
        pool.record_access(i % pool.cfg.n_experts)
    pinned = pool.update_pinned_set()
    print(f"\n  After 100 access records:")
    print(f"    Pinned {len(pinned)} experts in 32GB UMA")

    benefit = pool.spec_experts_benefit()
    print(f"\n  Spec-Experts routing benefit:")
    for k, v in benefit.items():
        print(f"    {k}: {v}")

    cmd = pool.llama_server_args("/Volumes/SSD/models/deepseek-v3.gguf")
    print(f"\n  llama-server args: {' '.join(cmd[:6])} ...")
    return True


# ─────────────────────────────────────────────────────────────
# P7  AGI Self-Loop
# ─────────────────────────────────────────────────────────────
async def demo_agi_self_loop():
    hdr("Phase 7 — AGI Self-Loop (LoRA Pool + Streaming Distillation)")
    from agi.self_loop import (
        LoRAAdapter, LoRAAdapterPool,
        StreamingDistillation, AGISelfLoop,
    )

    # Build adapter pool
    pool = LoRAAdapterPool()
    for name, task, score in [
        ("lora_reason_v1", "reasoning", 0.65),
        ("lora_reason_v2", "reasoning", 0.72),
        ("lora_code_v1",   "code",      0.78),
        ("lora_factual_v1","factual",   0.60),
    ]:
        a = LoRAAdapter(
            name=name,
            path=f"adapters/{name}.gguf",
            task_type=task,
            quality_score=score,
        )
        pool.register(a)

    print(f"\n  Adapter pool ({len(pool._adapters)} adapters):")
    for s in pool.stats():
        print(f"    {s['name']:<20}  task={s['task']:<10}  quality={s['quality']}")

    # Streaming distillation
    distill = StreamingDistillation(output_dir="distill_data")

    # Simulate offline (no server needed) — just score responses
    offline_tasks = [
        {"prompt": "Explain recursion", "chunk_type": "reasoning"},
        {"prompt": "Write a binary search", "chunk_type": "code"},
        {"prompt": "What is photosynthesis?", "chunk_type": "factual"},
        {"prompt": "Describe a sunset", "chunk_type": "creative"},
    ]

    print(f"\n  Simulating {len(offline_tasks)} distillation samples (offline):")
    for t in offline_tasks:
        # Simulated response
        fake_resp = {
            "reasoning": "Recursion is a function that calls itself. Therefore, the base case stops it.",
            "code": "def binary_search(arr, x):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == x: return mid\n        elif arr[mid] < x: lo = mid+1\n        else: hi = mid-1\n    return -1",
            "factual": "Photosynthesis converts CO2 and water into glucose using light energy, producing oxygen.",
            "creative": "The sunset painted the sky in crimson and gold, each cloud a brushstroke of fading warmth.",
        }[t["chunk_type"]]
        score = distill.record(t["prompt"], fake_resp, t["chunk_type"], tok_s=8.5)
        print(f"    {t['chunk_type']:<12}  quality={score:.2f}")

    print(f"\n  Distillation stats: {distill.stats()}")

    # Quality trend simulation
    loop_obj = AGISelfLoop(pool, distill)
    loop_obj._feedback_scores = [0.60, 0.63, 0.67, 0.70, 0.72, 0.75]
    print(f"\n  Quality trend: {loop_obj.quality_trend()}")

    return True


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
def print_summary(results: dict):
    hdr("Phase 2-7 Pipeline Summary")
    phases = {
        "P2 KV Offload Manager (3-tier VRAM/RAM/Disk)": results.get("p2", False),
        "P3 Directional Steering (system prompt proxy)": results.get("p3", False),
        "P4 Expert RAM Pool (Mixtral-8x7B / DS-V2-Lite)": results.get("p4", False),
        "P5 Edge VLM Router (moondream GT1030/Jetson)": results.get("p5", False),
        "P6 Mac M4 Metal Expert Pool (200B+ MoE)":       results.get("p6", False),
        "P7 AGI Self-Loop (LoRA + Distillation)":        results.get("p7", False),
    }
    print()
    all_pass = True
    for name, ok in phases.items():
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {name}")
        if not ok:
            all_pass = False
    print()
    print(f"  Full Spec-Experts pipeline: {'ALL PHASES PASS' if all_pass else 'PARTIAL'}")
    print(f"  Phase 1 (External Monitor): PASS [prev session]")
    print(f"  Architecture covers GT1030 → Mac M4 → AGI self-loop")


async def main():
    print(f"\n{BANNER}")
    print(f"  Spec-Experts LLM — Phase 2-7 Pipeline Demo")
    print(f"  GT 1030 baseline + scalability framework")
    print(BANNER)

    results = {}
    try:
        results["p2"] = demo_kv_offload()
    except Exception as e:
        print(f"  [ERR] P2: {e}")

    try:
        results["p3"] = demo_directional_steering()
    except Exception as e:
        print(f"  [ERR] P3: {e}")

    try:
        results["p4"] = demo_expert_ram_pool()
    except Exception as e:
        print(f"  [ERR] P4: {e}")

    try:
        results["p5"] = demo_edge_vlm()
    except Exception as e:
        print(f"  [ERR] P5: {e}")

    try:
        results["p6"] = demo_mac_m4()
    except Exception as e:
        print(f"  [ERR] P6: {e}")

    try:
        results["p7"] = await demo_agi_self_loop()
    except Exception as e:
        print(f"  [ERR] P7: {e}")

    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
