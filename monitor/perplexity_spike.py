"""
P1.2 Perplexity Spike Monitor
Sliding window z-score spike detector on token surprise scores.
"""

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class SpikeEvent:
    token_index: int
    perplexity: float
    z_score: float
    boundary_type: str = "perplexity_spike"


class PerplexitySpikeMonitor:
    def __init__(self, window: int = 20, z_threshold: float = 2.0):
        self.window = window
        self.z_threshold = z_threshold
        self._ppl_window: deque[float] = deque(maxlen=window)
        self._token_index = 0

    def _ppl_from_logprob(self, logprob: float) -> float:
        return math.exp(-logprob)

    def feed_token(self, logprob: float) -> SpikeEvent | None:
        ppl = self._ppl_from_logprob(logprob)
        event = None

        if len(self._ppl_window) >= self.window:
            values = list(self._ppl_window)
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 1e-6
            z = (ppl - mean) / std
            if z > self.z_threshold:
                event = SpikeEvent(
                    token_index=self._token_index,
                    perplexity=ppl,
                    z_score=z,
                )

        self._ppl_window.append(ppl)
        self._token_index += 1
        return event
