"""
Phase 7 — AGI Self-Loop
Streaming distillation + LoRA adapter pool + self-reinforcement loop.

Architecture:
  [Perception] → [Monitor] → [Chunk Classifier]
       → [Expert Router] → [Generation]
       → [Distillation Evaluator] → [LoRA Update]
       → [Adapter Pool] ← [Directional Steering]
              ↑__________[Feedback Loop]__________↑

Current implementation: simulation + scaffolding for future hardware.
LoRA fine-tuning requires gradient access; we use llama.cpp's
--lora-scaled flag to hot-swap pre-trained adapters as a proxy.
"""

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import httpx


# ── LoRA Adapter Pool ────────────────────────────────────────────────────────

@dataclass
class LoRAAdapter:
    name: str
    path: str              # .gguf lora adapter file
    task_type: str         # "reasoning" | "code" | "factual" | "creative"
    quality_score: float = 0.0
    usage_count: int = 0
    last_updated: float = field(default_factory=time.time)
    scale: float = 1.0     # llama-server --lora-scaled factor


class LoRAAdapterPool:
    """
    Hot-swap LoRA adapters via llama-server --lora-scaled.
    Adapters are pre-trained; pool selects best for each chunk type.
    """

    def __init__(self, adapters_dir: str = "adapters"):
        self._adapters: dict[str, LoRAAdapter] = {}
        self._dir = Path(adapters_dir)
        self._quality_history: dict[str, list[float]] = {}

    def register(self, adapter: LoRAAdapter):
        self._adapters[adapter.name] = adapter

    def best_for(self, task_type: str) -> Optional[LoRAAdapter]:
        """Select highest-quality adapter for the given task type."""
        candidates = [
            a for a in self._adapters.values()
            if a.task_type == task_type
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.quality_score)

    def update_quality(self, adapter_name: str, score: float):
        """Update quality score via exponential moving average."""
        a = self._adapters.get(adapter_name)
        if a is None:
            return
        alpha = 0.3
        a.quality_score = alpha * score + (1 - alpha) * a.quality_score
        self._quality_history.setdefault(adapter_name, []).append(score)

    def lora_args(self, adapter: LoRAAdapter) -> list[str]:
        """Generate --lora-scaled args for llama-server."""
        if not Path(adapter.path).exists():
            return []
        return ["--lora-scaled", adapter.path, str(adapter.scale)]

    def stats(self) -> list[dict]:
        return [
            {
                "name": a.name,
                "task": a.task_type,
                "quality": round(a.quality_score, 3),
                "usage": a.usage_count,
            }
            for a in sorted(self._adapters.values(), key=lambda x: -x.quality_score)
        ]


# ── Streaming Distillation ───────────────────────────────────────────────────

@dataclass
class DistillationSample:
    prompt: str
    response: str
    chunk_type: str
    quality_score: float
    tok_s: float
    timestamp: float = field(default_factory=time.time)


class StreamingDistillation:
    """
    Collect (prompt, response) pairs from the self-loop.
    Evaluate quality and select high-quality samples for potential LoRA training.
    In current form: stores samples and produces training-ready JSONL.
    """

    QUALITY_METRICS = {
        "code": lambda r: (
            ("def " in r or "class " in r or "import " in r) * 0.4 +
            (len(r) > 100) * 0.3 +
            (r.count("\n") > 3) * 0.3
        ),
        "reasoning": lambda r: (
            (len(r.split()) > 50) * 0.4 +
            ("therefore" in r.lower() or "because" in r.lower() or "thus" in r.lower()) * 0.3 +
            (r.count(".") > 3) * 0.3
        ),
        "factual": lambda r: (
            (len(r.split()) > 20) * 0.5 +
            (any(c.isupper() for c in r)) * 0.3 +
            (r.count(",") > 1) * 0.2
        ),
        "creative": lambda r: (
            (len(r.split()) > 30) * 0.4 +
            (len(set(r.lower().split())) / max(len(r.split()), 1) > 0.6) * 0.6
        ),
    }

    def __init__(self, output_dir: str = "distill_data"):
        self._samples: list[DistillationSample] = []
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def score(self, response: str, chunk_type: str) -> float:
        metric = self.QUALITY_METRICS.get(chunk_type, lambda r: 0.5)
        return float(metric(response))

    def record(self, prompt: str, response: str, chunk_type: str, tok_s: float):
        score = self.score(response, chunk_type)
        sample = DistillationSample(
            prompt=prompt,
            response=response,
            chunk_type=chunk_type,
            quality_score=score,
            tok_s=tok_s,
        )
        self._samples.append(sample)
        return score

    def flush_high_quality(self, threshold: float = 0.7) -> Path:
        """Write high-quality samples to JSONL for LoRA fine-tuning."""
        good = [s for s in self._samples if s.quality_score >= threshold]
        out = self._dir / f"distill_{int(time.time())}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for s in good:
                rec = {
                    "prompt": s.prompt,
                    "completion": s.response,
                    "chunk_type": s.chunk_type,
                    "quality": s.quality_score,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return out

    def stats(self) -> dict:
        if not self._samples:
            return {"total": 0}
        scores = [s.quality_score for s in self._samples]
        return {
            "total": len(self._samples),
            "mean_quality": round(sum(scores) / len(scores), 3),
            "high_quality": sum(1 for s in scores if s >= 0.7),
            "chunk_types": {
                ct: sum(1 for s in self._samples if s.chunk_type == ct)
                for ct in set(s.chunk_type for s in self._samples)
            },
        }


# ── Self-Loop Controller ─────────────────────────────────────────────────────

class AGISelfLoop:
    """
    Autonomous self-reinforcement loop.
    Each iteration:
      1. Generate response for task (with current adapters + steering)
      2. Score quality via distillation evaluator
      3. Update adapter quality scores
      4. After N iterations, export high-quality samples
      5. (Future) trigger LoRA training job on high-quality JSONL

    Current hardware (GT 1030): runs simulation with quality tracking.
    Target hardware (Mac Mini M4): enables actual LoRA gradient updates.
    """

    def __init__(
        self,
        adapter_pool: LoRAAdapterPool,
        distillation: StreamingDistillation,
        server_url: str = "http://127.0.0.1:8081",
    ):
        self.adapters = adapter_pool
        self.distill = distillation
        self.server_url = server_url
        self._iteration = 0
        self._feedback_scores: list[float] = []

    async def _generate(self, prompt: str, chunk_type: str, max_tokens: int = 512) -> tuple[str, float]:
        """Call llama-server and measure tok/s."""
        t0 = time.time()
        payload = {
            "messages": [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                )
                data = r.json()
            choices = data.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            elapsed = time.time() - t0
            tokens = len(content.split())
            tok_s = tokens / elapsed if elapsed > 0 else 0
            return content, tok_s
        except Exception as e:
            return f"[ERROR: {e}]", 0.0

    async def run_iteration(self, task: dict) -> dict:
        """
        Single self-loop iteration.
        task = {prompt, chunk_type, adapter_name (optional)}
        """
        self._iteration += 1
        prompt = task["prompt"]
        chunk_type = task.get("chunk_type", "reasoning")

        # Select adapter
        adapter = self.adapters.best_for(chunk_type)

        # Generate
        response, tok_s = await self._generate(prompt, chunk_type)

        # Score
        quality = self.distill.record(prompt, response, chunk_type, tok_s)

        # Update adapter quality
        if adapter:
            self.adapters.update_quality(adapter.name, quality)
            adapter.usage_count += 1

        self._feedback_scores.append(quality)

        return {
            "iteration": self._iteration,
            "chunk_type": chunk_type,
            "adapter": adapter.name if adapter else None,
            "quality": round(quality, 3),
            "tok_s": round(tok_s, 1),
            "response_preview": response[:100],
        }

    async def run_loop(
        self,
        tasks: list[dict],
        export_threshold: float = 0.7,
        on_iteration: Optional[Callable] = None,
    ) -> dict:
        """Run full self-loop over task list."""
        results = []
        for task in tasks:
            r = await self.run_iteration(task)
            results.append(r)
            if on_iteration:
                on_iteration(r)

        # Export high-quality samples
        out_path = self.distill.flush_high_quality(export_threshold)

        scores = self._feedback_scores
        return {
            "iterations": len(results),
            "mean_quality": round(sum(scores) / len(scores), 3) if scores else 0,
            "quality_trend": "improving" if len(scores) > 2 and scores[-1] > scores[0] else "stable",
            "exported_samples": str(out_path),
            "distill_stats": self.distill.stats(),
            "adapter_stats": self.adapters.stats(),
        }

    def quality_trend(self) -> str:
        """Report quality improvement across iterations."""
        if len(self._feedback_scores) < 3:
            return "insufficient data"
        # Simple linear regression slope
        n = len(self._feedback_scores)
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(self._feedback_scores) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, self._feedback_scores))
        den = sum((x - x_mean) ** 2 for x in xs)
        slope = num / den if den else 0
        return f"slope={slope:+.4f}/iter ({'improving' if slope > 0 else 'degrading'})"
