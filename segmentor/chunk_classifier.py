"""
P2.3 Chunk Classifier
Classify chunk type from role_tag + entropy profile.
Output: chunk_type ∈ {reasoning, factual, creative, code, unknown}
Determines: Distill Path (resource-rich) vs KV Prefix Path (edge).
"""

from dataclasses import dataclass
from enum import Enum


class ChunkType(str, Enum):
    REASONING = "reasoning"
    FACTUAL = "factual"
    CREATIVE = "creative"
    CODE = "code"
    UNKNOWN = "unknown"


class RoutePath(str, Enum):
    DISTILL = "distill"   # resource-rich: LoRA fine-tune
    KV_PREFIX = "kv"      # edge: KV cache reuse


ROLE_TAG_MAP = {
    "<think>": ChunkType.REASONING,
    "</think>": ChunkType.REASONING,
    "```": ChunkType.CODE,
    "# ": ChunkType.FACTUAL,
    "## ": ChunkType.FACTUAL,
}

# entropy profiles (mean entropy per chunk type, empirical estimates)
ENTROPY_PROFILE = {
    ChunkType.REASONING: (1.2, 2.5),   # (min, max) expected range
    ChunkType.CODE: (0.3, 1.5),
    ChunkType.FACTUAL: (0.8, 2.0),
    ChunkType.CREATIVE: (2.0, 4.0),
}


@dataclass
class ClassifiedChunk:
    chunk_id: int
    chunk_type: ChunkType
    route: RoutePath
    mean_entropy: float
    role_tag: str = ""


def classify_chunk(
    chunk_id: int,
    role_tag: str,
    entropy_values: list[float],
    vram_available_mb: float = 500.0,  # current free VRAM
) -> ClassifiedChunk:
    # role_tag takes priority
    chunk_type = ROLE_TAG_MAP.get(role_tag, ChunkType.UNKNOWN)

    mean_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 1.0

    # fallback: infer from entropy if no role tag
    if chunk_type == ChunkType.UNKNOWN:
        if mean_entropy < 1.0:
            chunk_type = ChunkType.CODE
        elif mean_entropy < 1.8:
            chunk_type = ChunkType.FACTUAL
        elif mean_entropy < 2.5:
            chunk_type = ChunkType.REASONING
        else:
            chunk_type = ChunkType.CREATIVE

    # route decision: distill if enough VRAM, else KV prefix
    route = RoutePath.DISTILL if vram_available_mb > 800 else RoutePath.KV_PREFIX

    return ClassifiedChunk(
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        route=route,
        mean_entropy=mean_entropy,
        role_tag=role_tag,
    )
