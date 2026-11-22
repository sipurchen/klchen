"""Shared llama-embedding wrapper for the embedding studies.

Fixes three pitfalls of calling the binary one text at a time:
  * llama-embedding splits its prompt file at the embedding separator (default newline) and returns one vector per piece, so a
    multi-line text silently yields several vectors; here a private separator is used and the vector count is checked;
  * the binary aborts on a prompt longer than the model window (BERT-style models: 512 tokens); texts are truncated to an
    estimated token budget and a failed batch is retried in halves, then with a shorter text;
  * pooling is whatever the GGUF declares (bge: CLS, MiniLM/e5: mean, Qwen3-Embedding: last) instead of a forced value.
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SEP = "<#sep#>"
POOLING_NAMES = {0: "none", 1: "mean", 2: "cls", 3: "last", 4: "rank"}


def estimated_tokens(text: str) -> float:
    """cautious token estimate (word-piece models: ~2.2 ASCII chars per token, ~1.3 tokens per CJK character)"""
    cjk = sum(1 for c in text if ord(c) >= 0x2E80)
    return (len(text) - cjk) / 2.2 + cjk * 1.3


def truncate(text: str, max_tokens: int) -> str:
    text = text.replace(SEP, " ").strip() or " "
    if estimated_tokens(text) <= max_tokens:
        return text
    lo, hi = 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimated_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def pooling_type(model_path: Path) -> str | None:
    """pooling type declared in the GGUF header (None when the gguf package is unavailable)"""
    try:
        from gguf import GGUFReader
    except ImportError:
        return None
    for key, field in GGUFReader(str(model_path)).fields.items():
        if key.endswith(".pooling_type"):
            return POOLING_NAMES.get(int(field.parts[field.data[0]][0]), "unknown")
    return None


def _run(binary: Path, model_path: Path, texts: list[str], threads: int, timeout: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(SEP.join(texts))
        path = Path(handle.name)
    try:
        r = subprocess.run([str(binary), "-m", str(model_path), "-f", str(path), "-t", str(threads), "-ngl", "0",
                            "--embd-separator", SEP, "--embd-output-format", "json", "--no-warmup"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=timeout)
        if r.returncode:
            raise RuntimeError(f"llama-embedding exited {r.returncode}: {r.stderr[-400:]}")
        start = r.stdout.index("{")
        data = json.loads(r.stdout[start:])["data"]
        if len(data) != len(texts):
            raise RuntimeError(f"expected {len(texts)} vectors, got {len(data)}")
        rows = sorted(data, key=lambda x: x.get("index", 0))
        return np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    finally:
        path.unlink(missing_ok=True)


def embed_texts(binary: Path, model_path: Path, texts: list[str], *, threads: int = 2, max_tokens: int = 480,
                batch: int = 24, timeout: int = 1800) -> np.ndarray:
    """L2-normalised vectors [len(texts), d], in input order"""
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch):
        piece = [truncate(t, max_tokens) for t in texts[i:i + batch]]
        budget = max_tokens
        while True:
            try:
                out.append(_run(binary, model_path, piece, threads, timeout))
                break
            except RuntimeError:
                if len(piece) > 1:                       # retry the batch as two halves
                    half = len(piece) // 2
                    out.append(embed_texts(binary, model_path, piece[:half], threads=threads, max_tokens=budget,
                                           batch=half, timeout=timeout))
                    out.append(embed_texts(binary, model_path, piece[half:], threads=threads, max_tokens=budget,
                                           batch=len(piece) - half, timeout=timeout))
                    break
                budget = int(budget * 0.7)
                if budget < 16:
                    raise
                piece = [truncate(piece[0], budget)]
    vectors = np.vstack(out)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("zero-norm embedding")
    return (vectors / norms).astype(np.float32)
