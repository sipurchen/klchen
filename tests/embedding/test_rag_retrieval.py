"""Tests for heading chunking and retrieval metrics / 標題切分與檢索指標測試。"""
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_rag_retrieval import (  # noqa: E402
    build_heading_chunks,
    char_tfidf,
    lf_normalise,
    paired_bootstrap,
    question_metric,
)


def test_lf_normalise_removes_windows_newlines():
    """Line endings become LF / 換行正規化為 LF。"""
    assert lf_normalise("a\r\nb\rc\n") == "a\nb\nc\n"


def test_heading_chunks_are_bounded_and_overlap(tmp_path):
    """Heading chunks obey the token cap and overlap / 標題 chunk 符合 token 上限與重疊。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "sample.md"
    source.write_text("# Evidence\n" + " ".join(f"t{index}" for index in range(9)), encoding="utf-8")
    manifest = {"files": [{"file": "docs/sample.md", "sha256_lf": "a"}]}
    chunks = build_heading_chunks(tmp_path, manifest, maximum=4, overlap=1)
    assert [chunk["token_count"] for chunk in chunks] == [4, 4, 3]
    assert chunks[0]["heading"] == "Evidence"
    assert chunks[0]["text"].split()[-1] == chunks[1]["text"].split()[0]


def test_question_metric_requires_all_two_passages():
    """Two-passage questions hit only after both headings arrive / 兩段題需兩個標題皆出現才命中。"""
    chunks = [{"file": "docs/a.md", "heading": "A"}, {"file": "docs/b.md", "heading": "B"}, {"file": "docs/c.md", "heading": "C"}]
    question = {"evidence": [{"file": "docs/a.md", "heading": "A"}, {"file": "docs/b.md", "heading": "B"}]}
    metrics = question_metric([0, 2, 1], question, chunks)
    assert metrics == {"hit@1": 0.0, "hit@3": 1.0, "hit@5": 1.0, "mrr": 1 / 3}


def test_char_tfidf_is_normalised_and_shared_space():
    """Character TF-IDF has shared unit rows / 字元 TF-IDF 具共享單位列。"""
    vectors = char_tfidf(["中文檢索", "English retrieval"], dimensions=64)
    assert vectors.shape == (2, 64)
    assert np.allclose(np.linalg.norm(vectors, axis=1), np.ones(2))


def test_paired_bootstrap_is_seeded():
    """Paired bootstrap is deterministic for its seed / 配對 bootstrap 對固定種子可重現。"""
    values = np.asarray([0.0, 1.0], dtype=np.float64)
    assert paired_bootstrap(values, samples=50, seed=7) == paired_bootstrap(values, samples=50, seed=7)
