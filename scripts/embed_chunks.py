"""Extract causal and non-causal span embeddings / 抽取因果與非因果 span 向量。

Runs only through the restricted runner.  Span text is reconstructed from the
same tokenizer token pieces used by ``segment_chunk_types.py``; this preserves
the token-index contract recorded in each ``*.spans.json`` file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from affinity_pipeline.embedding_io import embed_texts, pooling_type
from affinity_pipeline.config import REPO_ROOT, load_profile
from segment_chunk_types import USER_FOOTER, USER_HEADER, build_char_offsets, run_tokenize


MODEL_NAMES = (
    "bge-small-en-v1.5-q8_0.gguf",
    "all-MiniLM-L6-v2.Q8_0.gguf",
    "multilingual-e5-small-Q8_0.gguf",
    "Qwen3-Embedding-0.6B-Q8_0.gguf",
)
TFIDF_MODEL_NAME = "tfidf_char"
EXTRACTOR_NAMES = MODEL_NAMES + (TFIDF_MODEL_NAME,)
MODEL_DIR_NAME = "embed_models"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
E5_PREFIXES = {"multilingual-e5-small-Q8_0.gguf"}
QWEN_PREFIXES = {"Qwen3-Embedding-0.6B-Q8_0.gguf"}
QWEN_QUERY_INSTRUCTION = "Instruct: Retrieve relevant passages that answer the query\nQuery: "


def token_text(tokens: list[tuple[int, str]], start: int, end: int) -> str:
    """Join inclusive-start/exclusive-end tokenizer pieces / 連接 tokenizer token 區間。"""
    if not 0 <= start <= end <= len(tokens):
        raise ValueError(f"invalid token range [{start}:{end}] for {len(tokens)} tokens")
    return "".join(piece for _, piece in tokens[start:end])


def first_user_prompt(body: str, turns: list[dict]) -> str:
    """Return first user content excluding turn markers / 回傳第一個 user 內容且排除標記。"""
    if not turns:
        raise ValueError("session has no turns")
    turn = turns[0]
    text = body[turn["user_start"]:turn["user_end"]]
    if not text.startswith(USER_HEADER) or not text.endswith(USER_FOOTER):
        raise ValueError("first user turn does not have expected markers")
    return text[len(USER_HEADER):-len(USER_FOOTER)]


def context_text(prompt: str, tokens: list[tuple[int, str]], spans: list[dict], index: int,
                 previous_token_limit: int = 512) -> str:
    """Build causal context from first prompt and prior span / 用首 prompt 與前一 span 建因果脈絡。"""
    if index == 0:
        return prompt
    previous = spans[index - 1]
    start = max(previous["start"], previous["end"] - previous_token_limit)
    prior = token_text(tokens, start, previous["end"])
    return f"{prompt}\n{prior}"


def embedding_input(model_name: str, text: str, role: str) -> str:
    """Apply model-documented passage/query prefixes / 套用模型文件的 passage/query 前綴。"""
    if role not in {"passage", "query"}:
        raise ValueError(f"unknown embedding role: {role}")
    if model_name in E5_PREFIXES:
        return f"{role}: {text}"
    if model_name.startswith("bge-") and role == "query":
        return f"{BGE_QUERY_INSTRUCTION}{text}"
    if model_name in QWEN_PREFIXES and role == "query":
        return f"{QWEN_QUERY_INSTRUCTION}{text}"
    return text


def parse_embedding(stdout: str) -> np.ndarray:
    """Parse llama-embedding JSON output / 解析 llama-embedding JSON 輸出。"""
    def visit(value: object) -> np.ndarray | None:
        """Find OpenAI-style nested embedding row / 尋找 OpenAI 格式的巢狀 embedding 列。"""
        if isinstance(value, dict):
            candidate = value.get("embedding")
            if isinstance(candidate, list):
                vector = np.asarray(candidate, dtype=np.float32)
                if vector.ndim == 1 and vector.size:
                    return vector
            # llama-embedding JSON may return embeddings as a one-row matrix / llama-embedding JSON 可能以單列矩陣回傳 embeddings。
            candidates = value.get("embeddings")
            if isinstance(candidates, list) and candidates:
                vector = np.asarray(candidates[0], dtype=np.float32)
                if vector.ndim == 1 and vector.size:
                    return vector
            for child in value.values():
                found = visit(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            # Some llama-embedding versions emit a direct numeric array / 部分 llama-embedding 版本直接輸出數值陣列。
            if value and all(isinstance(item, (int, float)) for item in value):
                vector = np.asarray(value, dtype=np.float32)
                if vector.ndim == 1 and vector.size:
                    return vector
            for child in value:
                found = visit(child)
                if found is not None:
                    return found
        return None

    decoder = json.JSONDecoder()
    for offset, char in enumerate(stdout):
        if char not in "[{":
            continue
        try:
            data, _ = decoder.raw_decode(stdout[offset:])
        except json.JSONDecodeError:
            continue
        vector = visit(data)
        if vector is not None:
            return vector
    raise ValueError("llama-embedding JSON did not contain one non-empty embedding")


def normalise(vector: np.ndarray) -> np.ndarray:
    """Return L2-normalised float32 vector / 回傳 L2 正規化 float32 向量。"""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("embedding vector has zero L2 norm")
    return (vector / norm).astype(np.float32, copy=False)


def tfidf_feature_counts(text: str, *, dimensions: int = 4096) -> np.ndarray:
    """Hash character 3--5 grams into a fixed feature space / 雜湊字元 3--5 gram 至固定特徵空間。"""
    if dimensions < 1:
        raise ValueError("TF-IDF dimensions must be positive")
    canonical = " ".join(text.lower().split())
    # Preserve one normalisable feature for empty control spans / 為空 control span 保留可正規化特徵。
    if not canonical:
        canonical = "<empty-span>"
    indices: list[int] = []
    for width in (3, 4, 5):
        for start in range(max(0, len(canonical) - width + 1)):
            gram = canonical[start:start + width].encode("utf-8")
            indices.append(zlib.crc32(gram) % dimensions)
    if not indices:
        return np.zeros(dimensions, dtype=np.float32)
    return np.bincount(indices, minlength=dimensions).astype(np.float32, copy=False)


def tfidf_matrices(texts: list[str], *, dimensions: int = 4096) -> np.ndarray:
    """Build L2-normalised hashed character TF-IDF vectors / 建立 L2 正規化雜湊字元 TF-IDF 向量。"""
    counts = np.stack([tfidf_feature_counts(text, dimensions=dimensions) for text in texts])
    document_frequency = (counts > 0).sum(axis=0, dtype=np.int64)
    inverse_frequency = np.log((1.0 + len(texts)) / (1.0 + document_frequency)) + 1.0
    return np.stack([normalise(vector * inverse_frequency) for vector in counts]).astype(np.float32, copy=False)


def embed_one(binary: Path, model_path: Path, text: str, *, pooling: str | None = None, threads: int = 2) -> np.ndarray:
    """One text through the shared wrapper (newline-safe, truncated, GGUF pooling); `pooling` is kept for old callers and ignored"""
    return embed_texts(binary, model_path, [text], threads=threads)[0]


def embed_many(binary: Path, model_path: Path, texts: list[str], *, threads: int = 2) -> np.ndarray:
    return embed_texts(binary, model_path, texts, threads=threads)


def sha256_file(path: Path) -> str:
    """Return model SHA-256 / 回傳模型 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_session(root: Path, session_id: str, profile_name: str) -> tuple[list[tuple[int, str]], list[dict], str]:
    """Load and verify body-token reconstruction / 載入並驗證 body-token 重建。"""
    session_dir = root / "data" / "routing_corpus_v2" / "sessions"
    body_path = session_dir / f"{session_id}.body.txt"
    spans_path = session_dir / f"{session_id}.spans.json"
    turns_path = session_dir / f"{session_id}.turns.jsonl"
    body = body_path.read_text(encoding="utf-8")
    spans_doc = json.loads(spans_path.read_text(encoding="utf-8"))
    turns = [json.loads(line) for line in turns_path.read_text(encoding="utf-8").splitlines() if line]
    profile = load_profile(profile_name)
    tokens = run_tokenize(profile.model_path, body_path, profile=profile)
    build_char_offsets(tokens, body)
    if spans_doc["n_tokens"] != len(tokens):
        raise ValueError(f"{session_id}: spans n_tokens does not match tokenizer output")
    return tokens, spans_doc["spans"], first_user_prompt(body, turns)


def session_ids(root: Path) -> list[str]:
    """List corpus session identifiers / 列出語料 session 識別碼。"""
    directory = root / "data" / "routing_corpus_v2" / "sessions"
    return sorted(path.name.removesuffix(".spans.json") for path in directory.glob("*.spans.json"))


def collect_session_inputs(root: Path, profile_name: str, previous_token_limit: int,
                           sample_limit: int) -> tuple[list[dict], list[dict]]:
    """Collect aligned passage and causal inputs / 收集對齊的本文與因果輸入。"""
    records: list[dict] = []
    samples: list[dict] = []
    for session_id in session_ids(root):
        tokens, spans, prompt = load_session(root, session_id, profile_name)
        raw_chunks = [token_text(tokens, span["start"], span["end"]) for span in spans]
        raw_contexts = [context_text(prompt, tokens, spans, index, previous_token_limit)
                        for index in range(len(spans))]
        records.append({"session_id": session_id, "spans": spans, "chunks": raw_chunks, "contexts": raw_contexts})
        for index, span in enumerate(spans):
            if len(samples) >= sample_limit:
                break
            samples.append({"session": session_id, "span": index, "token_range": [span["start"], span["end"]],
                            "text_first_80": raw_chunks[index][:80]})
    return records, samples


def write_session_matrices(model_output: Path, records: list[dict], chunk_matrices: list[np.ndarray],
                           context_matrices: list[np.ndarray]) -> dict[str, dict[str, object]]:
    """Write per-session matrices and return their manifest records / 寫入各 session 矩陣並回傳 manifest 紀錄。"""
    outputs: dict[str, dict[str, object]] = {}
    for record, chunk_matrix, context_matrix in zip(records, chunk_matrices, context_matrices, strict=True):
        session_id = record["session_id"]
        if chunk_matrix.shape != context_matrix.shape:
            raise ValueError(f"{session_id}: chunk and ctx shapes differ")
        if chunk_matrix.shape[0] != len(record["spans"]):
            raise ValueError(f"{session_id}: vector row count does not match spans")
        session_output = model_output / session_id
        session_output.mkdir(parents=True, exist_ok=True)
        np.save(session_output / "chunk.npy", chunk_matrix)
        np.save(session_output / "ctx.npy", context_matrix)
        outputs[session_id] = {"n_spans": int(len(record["spans"])),
                               "chunk": f"{session_id}/chunk.npy", "ctx": f"{session_id}/ctx.npy"}
    return outputs


def extract_tfidf(root: Path, *, profile_name: str, output_root: Path,
                  previous_token_limit: int, sample_limit: int) -> dict:
    """Extract character TF-IDF baseline vectors / 抽取字元 TF-IDF 基準向量。"""
    records, samples = collect_session_inputs(root, profile_name, previous_token_limit, sample_limit)
    all_chunks = [text for record in records for text in record["chunks"]]
    all_contexts = [text for record in records for text in record["contexts"]]
    all_vectors = tfidf_matrices(all_chunks + all_contexts)
    boundary = len(all_chunks)
    chunk_vectors, context_vectors = all_vectors[:boundary], all_vectors[boundary:]
    chunk_matrices: list[np.ndarray] = []
    context_matrices: list[np.ndarray] = []
    offset = 0
    for record in records:
        size = len(record["spans"])
        chunk_matrices.append(chunk_vectors[offset:offset + size])
        context_matrices.append(context_vectors[offset:offset + size])
        offset += size
    model_output = output_root / TFIDF_MODEL_NAME
    model_output.mkdir(parents=True, exist_ok=True)
    outputs = write_session_matrices(model_output, records, chunk_matrices, context_matrices)
    manifest = {"schema": "embed_chunks/v1", "model_file": TFIDF_MODEL_NAME, "model_sha256": None,
                "quantization": "not_applicable", "dimensions": int(all_vectors.shape[1]),
                "pooling": "char_tfidf_hash_3_5", "normalization": "l2", "feature_hash": "crc32",
                "context_previous_span_token_limit": previous_token_limit, "chunk_role": "passage",
                "ctx_role": "query", "sessions": outputs, "samples": samples}
    (model_output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def extract_model(root: Path, model_name: str, *, profile_name: str, output_root: Path,
                  previous_token_limit: int = 512, sample_limit: int = 20) -> dict:
    """Extract all corpus vectors for one model / 對單一模型抽取所有語料向量。"""
    profile = load_profile(profile_name)
    binary = profile.llama_bin_dir / "llama-embedding"
    model_path = Path.home() / "gemma4_affinity_sim" / "work" / MODEL_DIR_NAME / model_name
    if not binary.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"missing llama-embedding binary or approved model {model_name}")
    model_output = output_root / model_name
    model_output.mkdir(parents=True, exist_ok=True)
    records, samples = collect_session_inputs(root, profile_name, previous_token_limit, sample_limit)
    chunk_matrices: list[np.ndarray] = []
    context_matrices: list[np.ndarray] = []
    dimensions: int | None = None
    pooling = pooling_type(model_path) or "gguf_default"
    for record in records:
        session_id = record["session_id"]
        chunks = [embedding_input(model_name, text, "passage") for text in record["chunks"]]
        contexts = [embedding_input(model_name, text, "query") for text in record["contexts"]]
        chunk_vectors = list(embed_many(binary, model_path, chunks))
        context_vectors = list(embed_many(binary, model_path, contexts))
        chunk_matrix = np.stack(chunk_vectors).astype(np.float32, copy=False)
        context_matrix = np.stack(context_vectors).astype(np.float32, copy=False)
        if chunk_matrix.shape != context_matrix.shape:
            raise ValueError(f"{session_id}: chunk and ctx shapes differ")
        if dimensions is None:
            dimensions = int(chunk_matrix.shape[1])
        elif dimensions != chunk_matrix.shape[1]:
            raise ValueError(f"{session_id}: embedding dimension changed")
        chunk_matrices.append(chunk_matrix)
        context_matrices.append(context_matrix)
    outputs = write_session_matrices(model_output, records, chunk_matrices, context_matrices)
    manifest = {
        "schema": "embed_chunks/v1",
        "model_file": model_name,
        "model_sha256": sha256_file(model_path),
        "quantization": "Q8_0",
        "dimensions": dimensions,
        "pooling": pooling,
        "normalization": "l2",
        "context_previous_span_token_limit": previous_token_limit,
        "chunk_role": "passage",
        "ctx_role": "query",
        "sessions": outputs,
        "samples": samples,
    }
    (model_output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    """Run command-line extraction or binary capability probe / 執行命令列抽取或 binary 探測。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=EXTRACTOR_NAMES)
    parser.add_argument("--profile", default="gx10_sim")
    parser.add_argument("--out", default="logs/embed")
    parser.add_argument("--previous-token-limit", type=int, default=512)
    parser.add_argument("--probe-help", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    binary = profile.llama_bin_dir / "llama-embedding"
    if args.probe_help:
        result = subprocess.run([str(binary), "--help"], capture_output=True, text=True, encoding="utf-8",
                                errors="replace", check=False, timeout=60)
        print(result.stdout)
        print(result.stderr)
        return result.returncode
    if args.model is None:
        parser.error("--model is required unless --probe-help is used")
    if args.previous_token_limit < 1:
        parser.error("--previous-token-limit must be positive")
    if args.model == TFIDF_MODEL_NAME:
        manifest = extract_tfidf(REPO_ROOT, profile_name=args.profile, output_root=Path(args.out),
                                 previous_token_limit=args.previous_token_limit, sample_limit=20)
    else:
        manifest = extract_model(REPO_ROOT, args.model, profile_name=args.profile, output_root=Path(args.out),
                                 previous_token_limit=args.previous_token_limit)
    print(json.dumps({"model": args.model, "dimensions": manifest["dimensions"],
                      "samples": len(manifest["samples"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
