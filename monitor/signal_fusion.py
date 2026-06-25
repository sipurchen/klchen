"""
P1.4 Signal Fusion
Merge entropy / perplexity / role_tag boundaries into unified ChunkBoundary.
Priority: role_tag (10) > entropy_drop (5) > perplexity_spike (3)
Hysteresis: suppress duplicate boundaries within MERGE_WINDOW tokens.
"""

from dataclasses import dataclass, field
from typing import Any

MERGE_WINDOW = 12  # tokens: collapse candidates within this range


@dataclass
class ChunkBoundary:
    token_index: int
    boundary_type: str
    priority: int
    signals: list[str] = field(default_factory=list)


class SignalFusion:
    def __init__(self, merge_window: int = MERGE_WINDOW):
        self.merge_window = merge_window
        self._pending: list[Any] = []  # raw candidates
        self._last_emitted: int = -999

    def add_candidate(self, candidate: Any) -> ChunkBoundary | None:
        """
        Accept candidate from any monitor.
        Returns ChunkBoundary if ready to emit, else None.
        Candidate must have: token_index, boundary_type, priority (or default).
        """
        priority = getattr(candidate, "priority", 3)
        token_index = candidate.token_index

        # deterministic role_tag: emit immediately, reset hysteresis
        if getattr(candidate, "boundary_type", "") == "role_tag":
            self._last_emitted = token_index
            self._pending.clear()
            return ChunkBoundary(
                token_index=token_index,
                boundary_type="role_tag",
                priority=10,
                signals=["role_tag"],
            )

        # suppress if too close to last boundary
        if abs(token_index - self._last_emitted) < self.merge_window:
            return None

        # accumulate probabilistic signals
        self._pending.append(candidate)

        # emit when 2+ signals cluster within merge_window
        cluster = [
            c for c in self._pending
            if abs(c.token_index - token_index) <= self.merge_window
        ]
        if len(cluster) >= 2:
            best = max(cluster, key=lambda c: getattr(c, "priority", 3))
            self._pending = [c for c in self._pending if c not in cluster]
            self._last_emitted = best.token_index
            return ChunkBoundary(
                token_index=best.token_index,
                boundary_type=best.boundary_type,
                priority=getattr(best, "priority", 3),
                signals=[getattr(c, "boundary_type", "unknown") for c in cluster],
            )

        return None
