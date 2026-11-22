"""E08 helpers: question selection and the fixed scoring rule (no server, no model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import rag_e2e as r  # noqa: E402


def test_scoring_rule_text_and_numbers():
    assert r.fact_present("router.proj", "The LoRA is attached to `router.proj` in each layer.")
    assert r.fact_present("ΔH = 0.41", "the entropy dropped by 0.41 at token 45")      # all numbers present
    assert not r.fact_present("W = 128: 0.986", "the hit rate was 0.986")                # 128 missing
    s = r.score_answer(["a b", "tok 45", "5.1 tok/s"], "A  B and 45 but speed unknown")
    assert s["present"] == [True, True, False] and abs(s["fraction"] - 2 / 3) < 1e-9 and not s["all"]


def test_selection_is_12_per_language_with_four_two_passage():
    qs = []
    for lang in ("zh", "en"):
        for i in range(20):
            qs.append({"id": f"{lang}{i:02d}", "lang": lang, "needs_two_passages": i % 2 == 0 and i < 10, "key_facts": ["x"]})
    chosen = r.select_questions(qs)
    assert len(chosen) == 24
    for lang in ("zh", "en"):
        rows = [q for q in chosen if q["lang"] == lang]
        assert len(rows) == 12 and sum(q["needs_two_passages"] for q in rows) == 4


def test_retrieve_respects_top_k(tmp_path):
    """top_k limits the emb and bm25 retrievals; gold keeps its own rule (one chunk per evidence, at most TOP_K)"""
    import json
    import numpy as np
    chunks = [{"file": "f.md", "heading": f"h{i}", "text": f"alpha beta {i}" if i else "gamma delta"} for i in range(5)]
    rag = tmp_path / "logs" / "rag_eval"
    (rag / "m.gguf").mkdir(parents=True)
    (tmp_path / "logs" / "rag_eval" / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    np.save(rag / "m.gguf" / "embeddings.npy", np.eye(5, dtype=np.float32))
    np.save(rag / "m.gguf" / "questions.npy", np.array([[0.1, 0.9, 0.5, 0.0, 0.0]], dtype=np.float32))
    q = {"id": "Q", "question": "alpha beta 3", "evidence": [{"file": "f.md", "heading": "h3", "quote": "alpha beta 3"}], "key_facts": ["x"]}
    for k in (1, 2, 3):
        out = r.retrieve(tmp_path, "m.gguf", [q], [q], top_k=k)
        assert len(out["emb"][0]) == k and len(out["bm25"][0]) == k
        assert [c["heading"] for c in out["gold"][0]] == ["h3"]
    assert r.retrieve(tmp_path, "m.gguf", [q], [q])["emb"][0][0]["heading"] == "h1"      # default still TOP_K = 3
    assert len(r.retrieve(tmp_path, "m.gguf", [q], [q])["emb"][0]) == 3
