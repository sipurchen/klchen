"""
Phase 2 — KV Offload Manager
三層 KV 快取：VRAM(simulated) → RAM → Disk
LRU eviction + prompt-replay workaround for llama.cpp HTTP API
"""

import os
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

from kv.kv_serializer import save_kv_chunk, load_kv_chunk, KV_BASE_DIR

# ── Constants ────────────────────────────────────────────────────────────────
VRAM_BUDGET_MB = 56        # KV-only VRAM budget on GT 1030 (ctx=2048 fp16)
RAM_BUDGET_MB  = 512       # RAM KV budget (34 GB available; keep conservative)
DTYPE          = np.float16

# priority weights: reasoning chunks re-accessed less; factual/code more
CHUNK_PRIORITY = {"reasoning": 0.5, "code": 1.5, "factual": 1.2, "creative": 0.8}


@dataclass
class KVEntry:
    session_id: str
    chunk_id: int
    role: str
    shape: tuple          # (n_layers, n_heads, head_dim)
    size_bytes: int
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    tier: str = "disk"   # "vram" | "ram" | "disk"
    data: Optional[np.ndarray] = None  # None when on disk


class KVOffloadManager:
    """
    Manages KV cache across VRAM → RAM → Disk tiers.
    GT 1030 llama.cpp doesn't expose KV injection; we simulate via
    prompt replay: store the original token sequence and re-prefill.
    """

    def __init__(
        self,
        vram_budget_mb: float = VRAM_BUDGET_MB,
        ram_budget_mb: float = RAM_BUDGET_MB,
        session_id: str = "default",
    ):
        self.session_id = session_id
        self.vram_budget = int(vram_budget_mb * 1024 * 1024)
        self.ram_budget  = int(ram_budget_mb  * 1024 * 1024)

        # OrderedDict acts as LRU (most-recently-used at end)
        self._vram: OrderedDict[tuple, KVEntry] = OrderedDict()
        self._ram:  OrderedDict[tuple, KVEntry] = OrderedDict()

        self._vram_used = 0
        self._ram_used  = 0

        self._lock = threading.Lock()

        # prompt store: chunk_id → token list (for replay)
        self._prompt_store: dict[int, list[int]] = {}

    # ── Key = (session_id, chunk_id, role) ───────────────────────────────────

    def _key(self, chunk_id: int, role: str) -> tuple:
        return (self.session_id, chunk_id, role)

    # ── Eviction ─────────────────────────────────────────────────────────────

    def _evict_cost(self, entry: KVEntry) -> float:
        age = time.time() - entry.last_access
        pri = CHUNK_PRIORITY.get(entry.role, 1.0)
        return age * entry.size_bytes / pri

    def _evict_from_vram(self, needed: int):
        """Move LRU entries from VRAM → RAM until needed bytes free."""
        while self._vram_used + needed > self.vram_budget and self._vram:
            # pop the LRU (first item) with lowest evict cost first
            worst_key = min(self._vram, key=lambda k: self._evict_cost(self._vram[k]))
            entry = self._vram.pop(worst_key)
            self._vram_used -= entry.size_bytes
            entry.tier = "ram"
            self._ensure_ram_space(entry.size_bytes)
            self._ram[worst_key] = entry
            self._ram.move_to_end(worst_key)
            self._ram_used += entry.size_bytes

    def _evict_from_ram(self, needed: int):
        """Move LRU entries from RAM → Disk until needed bytes free."""
        while self._ram_used + needed > self.ram_budget and self._ram:
            worst_key = min(self._ram, key=lambda k: self._evict_cost(self._ram[k]))
            entry = self._ram.pop(worst_key)
            self._ram_used -= entry.size_bytes
            # flush to disk
            if entry.data is not None:
                save_kv_chunk(
                    entry.session_id, entry.chunk_id, entry.role,
                    entry.data, layer_end=entry.shape[0],
                )
                entry.data = None
            entry.tier = "disk"

    def _ensure_vram_space(self, needed: int):
        self._evict_from_vram(needed)

    def _ensure_ram_space(self, needed: int):
        self._evict_from_ram(needed)

    # ── Public API ────────────────────────────────────────────────────────────

    def store(
        self,
        chunk_id: int,
        role: str,
        kv_tensor: np.ndarray,
        tokens: Optional[list] = None,
    ):
        """Store KV tensor. Attempts VRAM first, falls back to RAM, then disk."""
        key = self._key(chunk_id, role)
        size = kv_tensor.nbytes

        if tokens:
            self._prompt_store[chunk_id] = tokens

        entry = KVEntry(
            session_id=self.session_id,
            chunk_id=chunk_id,
            role=role,
            shape=kv_tensor.shape,
            size_bytes=size,
            data=kv_tensor.copy(),
        )

        with self._lock:
            # try VRAM
            if size <= self.vram_budget:
                self._ensure_vram_space(size)
                entry.tier = "vram"
                self._vram[key] = entry
                self._vram.move_to_end(key)
                self._vram_used += size
                return

            # try RAM
            if size <= self.ram_budget:
                self._ensure_ram_space(size)
                entry.tier = "ram"
                self._ram[key] = entry
                self._ram.move_to_end(key)
                self._ram_used += size
                return

            # disk only
            save_kv_chunk(
                entry.session_id, chunk_id, role,
                kv_tensor, layer_end=kv_tensor.shape[0],
            )
            entry.data = None
            entry.tier = "disk"

    def retrieve(self, chunk_id: int, role: str) -> Optional[np.ndarray]:
        """Retrieve KV tensor; promotes from disk→RAM if needed."""
        key = self._key(chunk_id, role)

        with self._lock:
            # check VRAM
            if key in self._vram:
                entry = self._vram[key]
                entry.last_access = time.time()
                entry.access_count += 1
                self._vram.move_to_end(key)
                return entry.data

            # check RAM
            if key in self._ram:
                entry = self._ram[key]
                entry.last_access = time.time()
                entry.access_count += 1
                self._ram.move_to_end(key)
                # promote to VRAM if fits
                if entry.size_bytes <= self.vram_budget:
                    self._ensure_vram_space(entry.size_bytes)
                    self._ram.pop(key)
                    self._ram_used -= entry.size_bytes
                    entry.tier = "vram"
                    self._vram[key] = entry
                    self._vram_used += entry.size_bytes
                return entry.data

            # load from disk
            data = load_kv_chunk(self.session_id, chunk_id, role)
            if data is None:
                return None

            entry = KVEntry(
                session_id=self.session_id,
                chunk_id=chunk_id,
                role=role,
                shape=data.shape,
                size_bytes=data.nbytes,
                data=data,
                tier="ram",
            )
            self._ensure_ram_space(data.nbytes)
            self._ram[key] = entry
            self._ram_used += data.nbytes
            return data

    def get_replay_tokens(self, chunk_id: int) -> Optional[list]:
        """Return stored token ids for prompt replay (llama.cpp KV workaround)."""
        return self._prompt_store.get(chunk_id)

    def build_replay_prompt(self, up_to_chunk_id: int, tokenizer_encode=None) -> str:
        """
        Build a concatenated prompt from stored chunks for KV replay.
        llama.cpp HTTP API lacks KV injection; re-prefilling is the workaround.
        """
        parts = []
        for cid in sorted(self._prompt_store.keys()):
            if cid <= up_to_chunk_id:
                parts.extend(self._prompt_store[cid])
        if tokenizer_encode is not None:
            return tokenizer_encode(parts)
        return parts

    def stats(self) -> dict:
        with self._lock:
            return {
                "vram_entries": len(self._vram),
                "vram_used_mb": self._vram_used / 1e6,
                "vram_budget_mb": self.vram_budget / 1e6,
                "ram_entries": len(self._ram),
                "ram_used_mb": self._ram_used / 1e6,
                "ram_budget_mb": self.ram_budget / 1e6,
                "disk_chunks": len(list(KV_BASE_DIR.glob(
                    f"{self.session_id}/*.kvbin"
                ))) if (KV_BASE_DIR / self.session_id).exists() else 0,
                "prompt_store_chunks": len(self._prompt_store),
            }

    def effective_context_tokens(
        self,
        tokens_per_chunk: int = 512,
        bytes_per_token: int = 64,  # fp16, 2 layers, 2 heads, 128 dim = 1024B; approx
    ) -> dict:
        """Estimate total effective context across all tiers."""
        disk = self.stats()["disk_chunks"]
        ram  = self.stats()["ram_entries"]
        vram = self.stats()["vram_entries"]
        return {
            "vram_ctx": vram * tokens_per_chunk,
            "ram_ctx":  ram  * tokens_per_chunk,
            "disk_ctx": disk * tokens_per_chunk,
            "total_ctx": (vram + ram + disk) * tokens_per_chunk,
        }
