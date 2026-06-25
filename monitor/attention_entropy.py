"""
P1.1 Attention Entropy Monitor
Hook to llama-server logprob SSE stream, detect semantic jump points.
"""

import asyncio
import json
import math
import httpx
from dataclasses import dataclass, field
from typing import AsyncIterator

ENTROPY_DROP_THRESHOLD = 0.35  # tunable: lower = more sensitive
WINDOW_SIZE = 8  # tokens to smooth over


@dataclass
class ChunkBoundaryCandidate:
    token_index: int
    entropy: float
    boundary_type: str = "entropy_drop"


@dataclass
class EntropyMonitor:
    server_url: str = "http://127.0.0.1:8080"
    threshold: float = ENTROPY_DROP_THRESHOLD
    _entropy_history: list = field(default_factory=list)
    _token_index: int = 0

    def _compute_entropy(self, logprobs: list[float]) -> float:
        """Shannon entropy from top-k logprobs."""
        probs = [math.exp(lp) for lp in logprobs]
        total = sum(probs)
        if total == 0:
            return 0.0
        probs = [p / total for p in probs]
        return -sum(p * math.log(p + 1e-12) for p in probs)

    def _detect_jump(self, entropy: float) -> bool:
        if len(self._entropy_history) < WINDOW_SIZE:
            return False
        avg = sum(self._entropy_history[-WINDOW_SIZE:]) / WINDOW_SIZE
        return (avg - entropy) > self.threshold

    async def stream_boundaries(
        self, prompt: str, max_tokens: int = 512
    ) -> AsyncIterator[ChunkBoundaryCandidate]:
        """
        POST to llama-server /completion with logprobs enabled,
        yield ChunkBoundaryCandidate on entropy drops.
        Requires llama-server built with --log-disable false and n_probs > 0.
        """
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "n_probs": 10,  # top-10 logprobs per token
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.server_url}/completion",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    token_probs = data.get("completion_probabilities", [])
                    if not token_probs:
                        self._token_index += 1
                        continue

                    logprobs = [p["logprob"] for p in token_probs[0].get("probs", [])]
                    if not logprobs:
                        self._token_index += 1
                        continue

                    entropy = self._compute_entropy(logprobs)

                    if self._detect_jump(entropy):
                        yield ChunkBoundaryCandidate(
                            token_index=self._token_index,
                            entropy=entropy,
                        )

                    self._entropy_history.append(entropy)
                    self._token_index += 1


async def _demo():
    monitor = EntropyMonitor()
    prompt = "<think>Let me reason step by step...</think>\nThe answer is"
    print("Streaming boundaries:")
    async for b in monitor.stream_boundaries(prompt, max_tokens=200):
        print(f"  jump at token {b.token_index}, entropy={b.entropy:.3f}")


if __name__ == "__main__":
    asyncio.run(_demo())
