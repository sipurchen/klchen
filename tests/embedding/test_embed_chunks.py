"""Unit tests for span text and embedding-output helpers / span 文字與向量輸出輔助函式測試。"""
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from embed_chunks import (  # noqa: E402
    QWEN_QUERY_INSTRUCTION,
    context_text,
    embedding_input,
    normalise,
    parse_embedding,
    tfidf_feature_counts,
    tfidf_matrices,
    token_text,
)


def test_token_text_uses_span_token_range():
    """Token interval joins exactly its text / token 區間精確連接其文字。"""
    tokens = [(1, "one"), (2, " two"), (3, " three")]
    assert token_text(tokens, 1, 3) == " two three"


def test_context_first_span_is_only_user_prompt():
    """First span has user prompt only / 第一個 span 僅含 user prompt。"""
    assert context_text("prompt", [(1, "a")], [{"start": 0, "end": 1}], 0) == "prompt"


def test_context_uses_previous_span_tail_limit():
    """Later context keeps prompt and final prior tokens / 後續脈絡保留 prompt 與前段末 token。"""
    tokens = [(i, str(i)) for i in range(5)]
    spans = [{"start": 0, "end": 4}, {"start": 4, "end": 5}]
    assert context_text("q", tokens, spans, 1, previous_token_limit=2) == "q\n23"


def test_e5_and_qwen_prefix_roles_are_distinct():
    """E5 and Qwen apply query/passages prefixes / E5 與 Qwen 套用 query/passage 前綴。"""
    assert embedding_input("multilingual-e5-small-Q8_0.gguf", "text", "passage") == "passage: text"
    assert embedding_input("multilingual-e5-small-Q8_0.gguf", "text", "query") == "query: text"
    assert embedding_input("Qwen3-Embedding-0.6B-Q8_0.gguf", "text", "query") == QWEN_QUERY_INSTRUCTION + "text"
    assert embedding_input("Qwen3-Embedding-0.6B-Q8_0.gguf", "text", "passage") == "text"


def test_json_output_parses_and_normalises_vector():
    """OpenAI JSON vector parses then has unit L2 norm / OpenAI JSON 向量解析後為單位 L2 範數。"""
    stdout = 'log\n{"object":"list","data":[{"index":0,"embedding":[3,4]}]}\n'
    vector = normalise(parse_embedding(stdout))
    assert vector.dtype == np.float32
    assert np.allclose(vector, np.asarray([0.6, 0.8], dtype=np.float32))


def test_json_output_parses_plural_embedding_matrix():
    """llama-embedding plural JSON matrix yields its first row / llama-embedding 複數 JSON 矩陣取第一列。"""
    stdout = '{"embeddings":[[3,4]]}'
    assert np.array_equal(parse_embedding(stdout), np.asarray([3, 4], dtype=np.float32))


def test_json_output_parses_direct_embedding_array():
    """A direct JSON vector parses as one embedding / 直接 JSON 向量解析為一個 embedding。"""
    assert np.array_equal(parse_embedding("[3,4]"), np.asarray([3, 4], dtype=np.float32))


def test_tfidf_matrices_share_space_and_are_normalised():
    """TF-IDF rows share fixed dimensions and unit norms / TF-IDF 列共享固定維度且範數為單位值。"""
    vectors = tfidf_matrices(["alpha beta", "alpha gamma"], dimensions=32)
    assert vectors.shape == (2, 32)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), np.ones(2))


def test_tfidf_empty_span_has_deterministic_nonzero_features():
    """Empty spans receive a fixed L2-normalisable feature / 空 span 有固定可 L2 正規化特徵。"""
    counts = tfidf_feature_counts("", dimensions=32)
    vectors = tfidf_matrices(["", "alpha"], dimensions=32)
    assert np.count_nonzero(counts) > 0
    assert np.allclose(np.linalg.norm(vectors, axis=1), np.ones(2))
