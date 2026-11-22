"""Evaluate frozen Markdown retrieval corpus / 評測凍結的 Markdown 檢索語料。

The corpus builder normalises source text to LF before hashing and splitting.
Lexical baselines use only NumPy: BM25 plus fixed-space char 3--5 gram TF-IDF.
Embedding mode invokes llama-embedding on the constrained runner and writes
per-model vectors and a machine-readable summary under ``logs/rag_eval``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from affinity_pipeline.config import REPO_ROOT, load_profile
from affinity_pipeline.embedding_io import embed_texts, pooling_type


MAX_CHUNK_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 32
TFIDF_DIMENSIONS = 4096
BOOTSTRAP_SAMPLES = 10_000
MODEL_NAMES = (
    "bge-small-en-v1.5-q8_0.gguf",
    "all-MiniLM-L6-v2.Q8_0.gguf",
    "multilingual-e5-small-Q8_0.gguf",
    "Qwen3-Embedding-0.6B-Q8_0.gguf",
)
E5_MODELS = {"multilingual-e5-small-Q8_0.gguf"}
QWEN_MODELS = {"Qwen3-Embedding-0.6B-Q8_0.gguf"}
QWEN_QUERY_INSTRUCTION = "Instruct: Retrieve relevant passages that answer the query\nQuery: "
EXCLUDED_DOC_FILES = {
    "docs/embedding_research_plan_2026-11-19.md",
    "docs/rag_architecture_2026-11-19.md",
}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", re.UNICODE)


def lf_normalise(text: str) -> str:
    """Normalise CRLF and CR to LF / 將 CRLF 與 CR 正規化為 LF。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256_text(text: str) -> str:
    """Return UTF-8 SHA-256 / 回傳 UTF-8 SHA-256。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corpus_paths(root: Path) -> list[Path]:
    """List frozen corpus sources and exclude research planning outputs / 列出凍結來源並排除研究規劃產物。"""
    docs = root / "docs"
    return [path for path in sorted(docs.rglob("*.md"))
            if path.relative_to(root).as_posix() not in EXCLUDED_DOC_FILES
            and "embedding_plan" not in path.relative_to(root).parts]


def corpus_manifest(root: Path) -> dict:
    """Build an LF-normalised source manifest / 建立 LF 正規化來源清單。"""
    files = []
    for path in corpus_paths(root):
        text = lf_normalise(path.read_text(encoding="utf-8"))
        files.append({"file": path.relative_to(root).as_posix(), "sha256_lf": sha256_text(text),
                      "bytes_utf8_lf": len(text.encode("utf-8"))})
    return {"schema": "rag_eval_corpus_manifest/v1", "normalization": "LF", "excluded": sorted(EXCLUDED_DOC_FILES) + ["docs/embedding_plan/**"], "files": files}


def tokenise(text: str) -> list[str]:
    """Return deterministic mixed-language chunk tokens / 回傳中英混合切分的穩定 token。"""
    return TOKEN_RE.findall(text)


def _split_token_window(tokens: list[str], maximum: int, overlap: int) -> list[list[str]]:
    """Split token stream into overlapping windows / 將 token 串分成帶重疊的窗口。"""
    if maximum < 1 or not 0 <= overlap < maximum:
        raise ValueError("invalid chunk maximum or overlap")
    if not tokens:
        return []
    return [tokens[start:start + maximum] for start in range(0, len(tokens), maximum - overlap)]


def build_heading_chunks(root: Path, manifest: dict, maximum: int = MAX_CHUNK_TOKENS,
                         overlap: int = CHUNK_OVERLAP_TOKENS) -> list[dict]:
    """Split each Markdown heading section to bounded chunks / 以標題節切成上限 chunk。"""
    chunks: list[dict] = []
    for file_info in manifest["files"]:
        relative = file_info["file"]
        text = lf_normalise((root / relative).read_text(encoding="utf-8"))
        headings = list(HEADING_RE.finditer(text))
        sections = [("(document preamble)", text[:headings[0].start()] if headings else text)]
        sections.extend((match.group(2), text[match.end():headings[index + 1].start()] if index + 1 < len(headings) else text[match.end():])
                        for index, match in enumerate(headings))
        for heading, section in sections:
            matches = list(TOKEN_RE.finditer(section))
            tokens = [match.group(0) for match in matches]
            for part_index, window in enumerate(_split_token_window(tokens, maximum, overlap)):
                start = part_index * (maximum - overlap)
                end = start + len(window)
                chunk_id = f"{relative}::{heading}::{part_index:03d}"
                chunks.append({"chunk_id": chunk_id, "file": relative, "heading": heading,
                               "text": section[matches[start].start():matches[end - 1].end()], "token_count": len(window),
                               "lang": "mixed", "source_sha256": file_info["sha256_lf"]})
    return chunks


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write UTF-8 JSONL with one object per line / 寫入每行一物件的 UTF-8 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def lexical_terms(text: str) -> list[str]:
    """Emit English words and Chinese character n-grams / 產生英文詞與中文字元 n-gram。"""
    terms = [value.lower() for value in WORD_RE.findall(text)]
    han_runs = re.findall(r"[\u3400-\u9fff]+", text)
    for run in han_runs:
        terms.extend(run[index:index + width] for width in (2, 3) for index in range(len(run) - width + 1))
    return terms


def char_tfidf(texts: list[str], dimensions: int = TFIDF_DIMENSIONS) -> np.ndarray:
    """Build L2-normalised hashed char 3--5 gram TF-IDF / 建立 L2 正規化雜湊字元 3--5 gram TF-IDF。"""
    counts = np.zeros((len(texts), dimensions), dtype=np.float32)
    for row, text in enumerate(texts):
        canonical = " ".join(text.lower().split())
        for width in (3, 4, 5):
            for start in range(max(0, len(canonical) - width + 1)):
                counts[row, zlib.crc32(canonical[start:start + width].encode("utf-8")) % dimensions] += 1.0
    document_frequency = (counts > 0).sum(axis=0, dtype=np.int64)
    weights = np.log((1.0 + len(texts)) / (1.0 + document_frequency)) + 1.0
    vectors = counts * weights
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0).astype(np.float32)


def bm25_scores(query: str, documents: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """Score documents with a NumPy BM25 baseline / 以 NumPy 計算 BM25 基準分數。"""
    document_terms = [lexical_terms(text) for text in documents]
    lengths = np.asarray([len(terms) for terms in document_terms], dtype=np.float64)
    average = float(lengths.mean()) if len(lengths) else 0.0
    query_terms = lexical_terms(query)
    scores = np.zeros(len(documents), dtype=np.float64)
    for term in dict.fromkeys(query_terms):
        frequencies = np.asarray([terms.count(term) for terms in document_terms], dtype=np.float64)
        document_frequency = int((frequencies > 0).sum())
        if document_frequency == 0:
            continue
        inverse_frequency = math.log(1.0 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5))
        denominator = frequencies + k1 * (1.0 - b + b * lengths / max(average, 1.0))
        scores += inverse_frequency * frequencies * (k1 + 1.0) / denominator
    return scores


def ranks_from_scores(scores: np.ndarray) -> list[int]:
    """Return descending score order with stable index tie-break / 回傳穩定索引破同分的降序排名。"""
    return np.lexsort((np.arange(len(scores)), -scores)).tolist()


def gold_chunk_indices(question: dict, chunks: list[dict]) -> set[int]:
    """Map all question evidence headings to chunk rows / 將題目證據標題映射至 chunk 列。"""
    matches: set[int] = set()
    for evidence in question["evidence"]:
        found = [index for index, chunk in enumerate(chunks)
                 if chunk["file"] == evidence["file"] and chunk["heading"] == evidence["heading"]]
        if not found:
            raise ValueError(f"{question['id']}: evidence heading has no chunk")
        matches.update(found)
    return matches


def question_metric(order: list[int], question: dict, chunks: list[dict]) -> dict[str, float]:
    """Score full evidence-set retrieval hit and reciprocal rank / 評分完整證據集合的命中與倒數排名。"""
    evidence_sets: list[set[int]] = []
    for evidence in question["evidence"]:
        rows = {index for index, chunk in enumerate(chunks)
                if chunk["file"] == evidence["file"] and chunk["heading"] == evidence["heading"]}
        evidence_sets.append(rows)
    cumulative: set[int] = set()
    first_complete: int | None = None
    for rank, index in enumerate(order, start=1):
        cumulative.add(index)
        if all(cumulative & evidence for evidence in evidence_sets):
            first_complete = rank
            break
    return {"hit@1": float(first_complete is not None and first_complete <= 1),
            "hit@3": float(first_complete is not None and first_complete <= 3),
            "hit@5": float(first_complete is not None and first_complete <= 5),
            "mrr": 0.0 if first_complete is None else 1.0 / first_complete}


def paired_bootstrap(differences: np.ndarray, samples: int = BOOTSTRAP_SAMPLES, seed: int = 20261119) -> dict:
    """Bootstrap paired question differences / 對題目配對差異進行 bootstrap。"""
    if differences.size == 0:
        raise ValueError("bootstrap needs at least one paired difference")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(samples, differences.size))
    means = differences[indices].mean(axis=1)
    return {"mean": float(differences.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "samples": samples, "seed": seed}


def evaluate_rankings(rankings: dict[str, list[list[int]]], questions: list[dict], chunks: list[dict]) -> dict:
    """Evaluate methods overall and by question language / 依全部與題目語言評測方法。"""
    outputs: dict[str, dict] = {}
    groups = {"all": list(range(len(questions))), "zh": [index for index, row in enumerate(questions) if row["lang"] == "zh"],
              "en": [index for index, row in enumerate(questions) if row["lang"] == "en"]}
    per_method: dict[str, list[dict[str, float]]] = {}
    for method, method_rankings in rankings.items():
        per_method[method] = [question_metric(order, question, chunks) for order, question in zip(method_rankings, questions, strict=True)]
        outputs[method] = {group: {metric: float(np.mean([per_method[method][index][metric] for index in indices]))
                                           for metric in ("hit@1", "hit@3", "hit@5", "mrr")}
                           for group, indices in groups.items()}
    comparisons: dict[str, dict] = {}
    for method in rankings:
        if method in {"bm25", "tfidf_char"}:
            continue
        comparisons[method] = {}
        for baseline in ("bm25", "tfidf_char"):
            comparisons[method][baseline] = {group: {metric: paired_bootstrap(np.asarray(
                [per_method[method][index][metric] - per_method[baseline][index][metric] for index in indices], dtype=np.float64))
                for metric in ("hit@1", "hit@3", "hit@5", "mrr")} for group, indices in groups.items()}
    return {"metrics": outputs, "paired_differences": comparisons, "question_count": {name: len(rows) for name, rows in groups.items()}}


def embedding_input(model: str, text: str, role: str) -> str:
    """Apply model-specific query/passage formatting / 套用模型專用的查詢與段落格式。"""
    if model in E5_MODELS:
        return f"{role}: {text}"
    if model.startswith("bge-") and role == "query":
        return "Represent this sentence for searching relevant passages: " + text
    if model in QWEN_MODELS and role == "query":
        return QWEN_QUERY_INSTRUCTION + text
    return text


def parse_embedding(stdout: str) -> np.ndarray:
    """Extract one OpenAI-style vector / 擷取一個 OpenAI 樣式向量。"""
    decoder = json.JSONDecoder()
    for position, char in enumerate(stdout):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stdout[position:])
        except json.JSONDecodeError:
            continue
        stack = [value]
        while stack:
            candidate = stack.pop()
            if isinstance(candidate, dict):
                vector = candidate.get("embedding")
                if isinstance(vector, list) and vector:
                    return normalise(np.asarray(vector, dtype=np.float32))
                stack.extend(candidate.values())
            elif isinstance(candidate, list):
                stack.extend(candidate)
    raise ValueError("llama-embedding JSON did not contain a vector")


def normalise(vector: np.ndarray) -> np.ndarray:
    """L2-normalise one vector / L2 正規化單一向量。"""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("zero-norm embedding")
    return (vector / norm).astype(np.float32, copy=False)


def embed_one(binary: Path, model_path: Path, text: str, pooling: str | None = None) -> np.ndarray:
    """One text through the shared wrapper (newline-safe, truncated, GGUF pooling)"""
    return embed_texts(binary, model_path, [text])[0]


def sha256_file(path: Path) -> str:
    """Hash a model file / 雜湊模型檔。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_questions(path: Path) -> list[dict]:
    """Load validated JSONL questions / 載入已驗證的 JSONL 題目。"""
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(questions) != 40 or {question["lang"] for question in questions} != {"zh", "en"}:
        raise ValueError("questions.jsonl must contain the expected bilingual evaluation set")
    return questions


def build_local(root: Path, output: Path) -> tuple[list[dict], list[dict], dict]:
    """Freeze sources, build chunks, and evaluate lexical baselines / 凍結來源、切分並評測詞彙基準。"""
    manifest = corpus_manifest(root)
    chunks = build_heading_chunks(root, manifest)
    questions = load_questions(root / "data" / "rag_eval" / "questions.jsonl")
    write_jsonl(output / "chunks.jsonl", chunks)
    (output / "corpus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    passages = [chunk["text"] for chunk in chunks]
    rankings = {
        "bm25": [ranks_from_scores(bm25_scores(question["question"], passages)) for question in questions],
    }
    matrix = char_tfidf(passages + [question["question"] for question in questions])
    passage_matrix, query_matrix = matrix[:len(chunks)], matrix[len(chunks):]
    rankings["tfidf_char"] = [ranks_from_scores(passage_matrix @ query) for query in query_matrix]
    summary = evaluate_rankings(rankings, questions, chunks)
    summary["corpus"] = {"chunk_count": len(chunks), "max_chunk_tokens": MAX_CHUNK_TOKENS, "overlap_tokens": CHUNK_OVERLAP_TOKENS}
    summary["methods"] = ["bm25", "tfidf_char"]
    (output / "lexical_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return chunks, questions, summary


def evaluate_embedding(root: Path, model: str, output: Path) -> dict:
    """Embed frozen chunks and questions then compare against lexical methods / 嵌入凍結 chunk 與題目並比較詞彙法。"""
    chunks, questions, lexical = build_local(root, output)
    profile = load_profile("gx10_sim")
    binary = profile.llama_bin_dir / "llama-embedding"
    model_path = Path.home() / "gemma4_affinity_sim" / "work" / "embed_models" / model
    if not binary.is_file() or not model_path.is_file():
        raise FileNotFoundError("llama-embedding binary or approved model is unavailable")
    pooling = pooling_type(model_path) or "gguf_default"
    passages = [embedding_input(model, chunk["text"], "passage") for chunk in chunks]
    queries = [embedding_input(model, question["question"], "query") for question in questions]
    passage_vectors = embed_texts(binary, model_path, passages)
    query_vectors = embed_texts(binary, model_path, queries)
    model_dir = output / model
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / "embeddings.npy", passage_vectors)
    np.save(model_dir / "questions.npy", query_vectors)
    tfidf_matrix = char_tfidf([chunk["text"] for chunk in chunks] + [question["question"] for question in questions])
    lexical_rankings = {
        "bm25": [ranks_from_scores(bm25_scores(question["question"], [chunk["text"] for chunk in chunks])) for question in questions],
        "tfidf_char": [ranks_from_scores(tfidf_matrix[:len(chunks)] @ query) for query in tfidf_matrix[len(chunks):]],
    }
    rankings = lexical_rankings | {model: [ranks_from_scores(passage_vectors @ vector) for vector in query_vectors]}
    result = evaluate_rankings(rankings, questions, chunks)
    result["corpus"] = lexical["corpus"]
    result["embedding"] = {"model_file": model, "model_sha256": sha256_file(model_path), "dimensions": int(passage_vectors.shape[1]),
                           "pooling": pooling, "normalization": "l2", "prefix_query": embedding_input(model, "", "query"),
                           "prefix_passage": embedding_input(model, "", "passage")}
    best_lexical = max(result["metrics"]["bm25"]["all"]["hit@3"], result["metrics"]["tfidf_char"]["all"]["hit@3"])
    hit3 = result["metrics"][model]["all"]["hit@3"]
    result["h6_retrieval"] = {"hit_at_3": hit3, "best_lexical_hit_at_3": best_lexical, "difference": hit3 - best_lexical,
                              "criterion": "hit@3 >= 0.85 and difference >= 0.05", "met": hit3 >= 0.85 and hit3 - best_lexical >= 0.05}
    (model_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": model, "h6_retrieval": result["h6_retrieval"], "chunk_count": len(chunks)}, ensure_ascii=False))
    return result


def probe_embedding(model: str) -> dict:
    """Emit one model vector for runner verification / 輸出一個模型向量以驗證 runner。"""
    profile = load_profile("gx10_sim")
    binary = profile.llama_bin_dir / "llama-embedding"
    model_path = Path.home() / "gemma4_affinity_sim" / "work" / "embed_models" / model
    if not binary.is_file() or not model_path.is_file():
        raise FileNotFoundError("llama-embedding binary or approved model is unavailable")
    pooling = "last" if model in QWEN_MODELS else "mean"
    text = embedding_input(model, "retrieval probe", "query")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(text)
        input_path = Path(handle.name)
    try:
        diagnostics = {}
        for input_name, input_args in (("prompt", ["-p", text]), ("file", ["-f", str(input_path)])):
            command = [str(binary), "-m", str(model_path), *input_args, "-t", "2", "-ngl", "0", "--pooling", pooling,
                       "--embd-output-format", "json", "--no-warmup"]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=900)
            diagnostics[input_name] = {"returncode": result.returncode, "stdout_length": len(result.stdout),
                                       "stderr_length": len(result.stderr), "stdout_prefix": result.stdout[:120],
                                       "stderr_prefix": result.stderr[:120]}
            if result.returncode == 0:
                try:
                    vector = parse_embedding(result.stdout)
                except ValueError:
                    continue
                return {"model": model, "dimensions": int(vector.size), "l2_norm": float(np.linalg.norm(vector)),
                        "input_mode": input_name, "diagnostics": diagnostics}
        return {"model": model, "vector_found": False, "diagnostics": diagnostics}
    finally:
        input_path.unlink(missing_ok=True)


def main() -> int:
    """Run local lexical or constrained embedding evaluation / 執行本機詞彙或受限嵌入評測。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("lexical", "embed", "probe"), required=True)
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--out", default="data/rag_eval")
    args = parser.parse_args()
    if args.mode in {"embed", "probe"} and args.model is None:
        parser.error("--model is required for embed or probe mode")
    if args.mode == "lexical" and args.model is not None:
        parser.error("--model is only valid for embed or probe mode")
    if args.mode == "lexical":
        _, _, result = build_local(REPO_ROOT, Path(args.out))
        print(json.dumps({"methods": result["methods"], "corpus": result["corpus"]}, ensure_ascii=False))
    elif args.mode == "embed":
        evaluate_embedding(REPO_ROOT, args.model, Path(args.out))
    else:
        print(json.dumps(probe_embedding(args.model), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
