"""
Phase 6 — Mac Mini M4 Metal Expert Pool
200B+ MoE inference on Apple Silicon M4 (32GB Unified Memory).

Key advantages over GT 1030:
- Unified Memory Architecture (UMA): GPU & CPU share same 32GB pool
- M4 Neural Engine: 38 TOPS for specific ops
- Metal Performance Shaders (MPS): optimized matrix ops
- NVMe SSD ~7 GB/s: expert paging at scale
- Full ngl=99: all layers on GPU (no CPU-GPU split needed)

Target models:
- DeepSeek-V3 671B MoE Q2 (18.5GB active / 37B per-token)
- Mistral-Large 123B Q4 (63GB — needs swap with M4 Max)
- Qwen-72B Q4 (40GB — tight but fits)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time


@dataclass
class M4ExpertConfig:
    model_name: str
    n_experts: int
    top_k: int
    total_gb: float        # total model size GB
    active_gb: float       # active per-token GB
    ssd_gb_s: float = 7.0  # M4 SSD bandwidth
    mem_gb: int = 32       # Unified Memory
    tok_s_target: float = 20.0


M4_MODELS = {
    "deepseek-v3-671b-q2": M4ExpertConfig(
        model_name="DeepSeek-V3 671B Q2_K",
        n_experts=256,
        top_k=2,
        total_gb=210.0,    # total on SSD
        active_gb=18.5,    # active 37B parameters
        tok_s_target=15.0,
    ),
    "deepseek-r1-671b-q4": M4ExpertConfig(
        model_name="DeepSeek-R1 671B Q4",
        n_experts=256,
        top_k=2,
        total_gb=380.0,
        active_gb=37.0,
        tok_s_target=10.0,
    ),
    "mixtral-8x22b-q4": M4ExpertConfig(
        model_name="Mixtral-8x22B Q4",
        n_experts=8,
        top_k=2,
        total_gb=141.0,
        active_gb=26.0,
        tok_s_target=25.0,
    ),
    "qwen2-72b-q4": M4ExpertConfig(
        model_name="Qwen2.5-72B Q4 (Dense)",
        n_experts=0,
        top_k=0,
        total_gb=40.0,
        active_gb=40.0,
        tok_s_target=18.0,
    ),
}


class MetalExpertPool:
    """
    Expert pool management for Mac Mini M4 (32GB UMA).

    Strategy:
    - Top-32 most accessed experts pinned in 32GB UMA
    - Remaining experts on NVMe SSD (7 GB/s streaming)
    - Spec-Experts chunk routing reduces expert churn
    - Flash Attention enabled (Metal backend, no CPU/GPU split)
    - KV cache: 16K+ context via SSD streaming (no WDDM constraint)
    """

    def __init__(self, model_key: str = "deepseek-v3-671b-q2"):
        cfg = M4_MODELS.get(model_key)
        if cfg is None:
            raise ValueError(f"Unknown model: {model_key}")
        self.cfg = cfg
        self.model_key = model_key

        # Expert access frequency tracking
        self._access_counts: dict[int, int] = {}
        self._last_accessed: dict[int, float] = {}
        self._pinned: set[int] = set()

        # Calculate how many experts fit in UMA
        expert_size_gb = cfg.total_gb / max(cfg.n_experts, 1)
        self.experts_in_uma = min(
            cfg.n_experts,
            int(cfg.mem_gb * 0.85 / expert_size_gb) if expert_size_gb > 0 else cfg.n_experts,
        )
        self._expert_size_gb = expert_size_gb

    def record_access(self, expert_id: int):
        self._access_counts[expert_id] = self._access_counts.get(expert_id, 0) + 1
        self._last_accessed[expert_id] = time.time()

    def update_pinned_set(self) -> set[int]:
        """Recompute which experts should be pinned (top-N by access count)."""
        if self.cfg.n_experts == 0:  # dense model
            self._pinned = set()
            return self._pinned
        sorted_experts = sorted(
            self._access_counts.keys(),
            key=lambda e: self._access_counts[e],
            reverse=True,
        )
        self._pinned = set(sorted_experts[:self.experts_in_uma])
        return self._pinned

    def ssd_load_time(self, expert_id: int) -> float:
        """Estimate seconds to load one expert from SSD."""
        if expert_id in self._pinned:
            return 0.0  # already in UMA
        return self._expert_size_gb / self.cfg.ssd_gb_s

    def theoretical_tok_s(self, experts_cached: int) -> float:
        """
        Estimate tok/s given N experts already in UMA.
        Active bandwidth = active_gb / tok_s → rearranged.
        M4 UMA bandwidth ~120 GB/s.
        """
        uma_bw = 120.0  # GB/s
        active_on_uma = min(experts_cached, self.cfg.top_k)
        ssd_fraction = max(0, self.cfg.top_k - active_on_uma) / self.cfg.top_k
        effective_bw = uma_bw * (1 - ssd_fraction) + self.cfg.ssd_gb_s * ssd_fraction
        return min(
            self.cfg.tok_s_target,
            effective_bw / self.cfg.active_gb if self.cfg.active_gb > 0 else 0,
        )

    def llama_server_args(
        self,
        model_path: str,
        ctx: int = 16384,
        threads: int = 8,
        port: int = 8080,
    ) -> list[str]:
        """Generate optimal llama-server command for M4."""
        return [
            "llama-server",
            "--model", model_path,
            "--n-gpu-layers", "99",       # full Metal offload
            "--ctx-size", str(ctx),
            "--threads", str(threads),
            "--flash-attn",               # Flash Attention — works on Metal
            "--port", str(port),
            "--mlock",                    # lock UMA; no WDDM eviction on macOS
            "--n-predict", "4096",
        ]

    def spec_experts_benefit(self) -> dict:
        """
        Quantify Spec-Experts routing benefit on M4 vs naive routing.
        Fewer expert switches = less SSD paging.
        """
        if self.cfg.n_experts == 0:
            return {"benefit": "N/A (dense model)"}
        avg_switch_cost_s = self._expert_size_gb / self.cfg.ssd_gb_s
        chunks_per_session = 20
        switches_naive = chunks_per_session * self.cfg.top_k
        switches_spec  = chunks_per_session * 1.2  # Spec-Experts: ~1.2 new experts per chunk
        saved_s = (switches_naive - switches_spec) * avg_switch_cost_s
        return {
            "model": self.cfg.model_name,
            "experts_in_uma": self.experts_in_uma,
            "expert_size_gb": round(self._expert_size_gb, 2),
            "switches_naive": switches_naive,
            "switches_spec_experts": switches_spec,
            "time_saved_per_session_s": round(saved_s, 2),
            "theoretical_tok_s_full_uma": round(self.theoretical_tok_s(self.cfg.n_experts), 1),
            "theoretical_tok_s_partial": round(self.theoretical_tok_s(self.experts_in_uma), 1),
        }

    def hardware_summary(self) -> str:
        cfg = self.cfg
        uma_fit = self.experts_in_uma
        return (
            f"Mac Mini M4 — {cfg.model_name}\n"
            f"  Total size: {cfg.total_gb:.0f} GB | Active/tok: {cfg.active_gb:.1f} GB\n"
            f"  Experts: {cfg.n_experts} total, top-{cfg.top_k} active, "
            f"{uma_fit} fit in 32GB UMA\n"
            f"  Target: {cfg.tok_s_target} tok/s | Flash Attention: YES\n"
            f"  SSD streaming: {cfg.ssd_gb_s} GB/s for non-pinned experts"
        )
