#!/usr/bin/env python3
"""E10: why does multilingual-e5-small reach hit@3 = 0.10 on the English questions of the retrieval set?

Measurements (all on the same 498 chunks and 40 questions as E07B):
  1. reproduction with the E07B metric (`eval_rag_retrieval.question_metric`: the full evidence set must be in the top-k)
  2. input variants: base (`query: `/`passage: `), no_prefix, swapped, both_query, trunc256
  3. an answer-oriented metric that is not tied to one file/heading: a chunk counts when it holds >= 50% of the key facts
  4. language of the retrieved chunks for the English questions, against the language mix of the corpus, for e5 and for the other
     three embedding models (their saved vectors in logs/rag_eval/<model>/)
  5. sentence-vector sanity: similar / dissimilar English, Chinese and cross-language pairs, and question-to-gold-quote similarity
     against random chunks
  python scripts/e5_diagnose.py --out logs/e5_diagnose
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import eval_rag_retrieval as er  # noqa: E402
import rag_e2e as e2e  # noqa: E402
from affinity_pipeline.config import load_profile  # noqa: E402
from affinity_pipeline.embedding_io import embed_texts, truncate  # noqa: E402

MODEL = "multilingual-e5-small-Q8_0.gguf"
OTHERS = ["bge-small-en-v1.5-q8_0.gguf", "all-MiniLM-L6-v2.Q8_0.gguf", "Qwen3-Embedding-0.6B-Q8_0.gguf"]
VARIANTS = ["base", "no_prefix", "swapped", "both_query", "trunc256"]
CJK_ZH, CJK_MIXED_LOW = 0.5, 0.1


def cjk_ratio(text: str) -> float:
    return sum(1 for c in text if 0x2E80 <= ord(c) <= 0x9FFF) / len(text) if text else 0.0


def chunk_language(text: str) -> str:
    r = cjk_ratio(text)
    return "zh" if r >= CJK_ZH else ("en" if r < CJK_MIXED_LOW else "mixed")


def formatted(text: str, role: str, variant: str) -> str:
    budget = 256 if variant == "trunc256" else 480
    text = truncate(text, budget)
    if variant == "no_prefix":
        return text
    if variant == "swapped":
        return f"{'passage' if role == 'query' else 'query'}: {text}"
    if variant == "both_query":
        return f"query: {text}"
    return f"{role}: {text}"          # base and trunc256


def e5_paths():
    profile = load_profile("gx10_sim")
    return profile.llama_bin_dir / "llama-embedding", Path.home() / "gemma4_affinity_sim" / "work" / "embed_models" / MODEL


def retrieval_metrics(passage_vectors, query_vectors, questions, chunks, support):
    """hit@3 with the E07B metric (all 40 and the answerable 37), by language, plus the answer-oriented hit@3"""
    rows = []
    for qi, q in enumerate(questions):
        order = np.argsort(-(passage_vectors @ query_vectors[qi]), kind="stable").tolist()
        m = er.question_metric(order, q, chunks)
        answerable = all(any(c["file"] == e["file"] and c["heading"] == e["heading"] for c in chunks) for e in q["evidence"])
        rows.append({"id": q["id"], "lang": q["lang"], "answerable": answerable, "hit3": m["hit@3"], "mrr": m["mrr"],
                     "answer_hit3": float(bool(support[qi][order[:3]].any())), "top3": order[:3]})

    def agg(sel, key):
        v = [r[key] for r in rows if sel(r)]
        return round(float(np.mean(v)), 3) if v else None
    out = {}
    for key in ("hit3", "mrr", "answer_hit3"):
        out[key] = {"all40": agg(lambda r: True, key), "zh": agg(lambda r: r["lang"] == "zh", key), "en": agg(lambda r: r["lang"] == "en", key),
                    "answerable37": agg(lambda r: r["answerable"], key)}
    return out, rows


def language_of_top3(rows, chunks, questions):
    langs = [chunk_language(c["text"]) for c in chunks]
    share_corpus = {k: round(langs.count(k) / len(langs), 3) for k in ("zh", "en", "mixed")}
    en_rows = [r for r in rows if r["lang"] == "en"]
    top = [langs[i] for r in en_rows for i in r["top3"]]
    share_top = {k: round(top.count(k) / len(top), 3) for k in ("zh", "en", "mixed")}
    return {"corpus_share": share_corpus, "top3_share_for_english_questions": share_top}


def sentence_checks(binary, model_path, chunks, questions):
    pairs = [("embedding", "vector", "high", "synonym, English"), ("expert routing pool", "expert pool routing", "high", "paraphrase, English"),
             ("model fine-tuning", "data augmentation", "low", "unrelated, English"),
             ("embeddings are vectors", "embeddings represent text in vector space", "high", "similar meaning, English"),
             ("嵌入向量", "向量表示", "high", "synonym, Chinese"), ("專家路由", "路由專家", "high", "paraphrase, Chinese"),
             ("expert routing", "專家路由", "high", "translation pair"), ("英文文本", "中文文本", "low", "different language names")]
    texts = [f"query: {a}" for a, _, _, _ in pairs] + [f"query: {b}" for _, b, _, _ in pairs]
    v = embed_texts(binary, model_path, texts)
    n = len(pairs)
    res = [{"a": a, "b": b, "expected": e, "note": d, "cosine": round(float(v[i] @ v[n + i]), 4)} for i, (a, b, e, d) in enumerate(pairs)]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="logs/e5_diagnose")
    a = ap.parse_args()
    root = Path(a.root)
    chunks = [json.loads(line) for line in (root / "logs" / "rag_eval" / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    questions = er.load_questions(root / "data" / "rag_eval" / "questions.jsonl")
    binary, model_path = e5_paths()
    support = np.zeros((len(questions), len(chunks)), dtype=bool)
    for qi, q in enumerate(questions):
        for ci, c in enumerate(chunks):
            support[qi, ci] = e2e.score_answer(q["key_facts"], c["text"])["fraction"] >= 0.5
    report: dict = {"model": MODEL, "chunks": len(chunks), "questions": len(questions),
                    "cjk_thresholds": {"zh": CJK_ZH, "mixed_lower": CJK_MIXED_LOW},
                    "questions_with_a_supporting_chunk": int(support.any(axis=1).sum()), "variants": {}}
    base_rows = None
    for variant in VARIANTS:
        print("variant", variant, flush=True)
        pv = embed_texts(binary, model_path, [formatted(c["text"], "passage", variant) for c in chunks])
        qv = embed_texts(binary, model_path, [formatted(q["question"], "query", variant) for q in questions])
        metrics, rows = retrieval_metrics(pv, qv, questions, chunks, support)
        report["variants"][variant] = metrics
        print("  hit@3", metrics["hit3"], "answer-oriented", metrics["answer_hit3"], flush=True)
        if variant == "base":
            base_rows, base_pv, base_qv = rows, pv, qv
            report["language_of_top3_e5_base"] = language_of_top3(rows, chunks, questions)
            gold = []
            for qi, q in enumerate(questions):
                idx = [ci for ci, c in enumerate(chunks) if any(c["file"] == e["file"] and c["heading"] == e["heading"] for e in q["evidence"])]
                if idx:
                    gold.append((q["lang"], float(np.mean(pv[idx] @ qv[qi]))))
            rnd = (pv @ qv.T)
            report["cosine_question_to_gold_chunks"] = {
                lang: round(float(np.mean([g for l, g in gold if l == lang])), 4) for lang in ("zh", "en")}
            report["cosine_question_to_all_chunks_mean"] = {
                lang: round(float(np.mean([rnd[:, qi].mean() for qi, q in enumerate(questions) if q["lang"] == lang])), 4) for lang in ("zh", "en")}
    other = {}
    for m in OTHERS:
        base = root / "logs" / "rag_eval" / m
        P, Q = np.load(base / "embeddings.npy"), np.load(base / "questions.npy")
        _, rows = retrieval_metrics(P, Q, questions, chunks, support)
        other[m] = language_of_top3(rows, chunks, questions)["top3_share_for_english_questions"]
    report["language_of_top3_other_models_english_questions"] = other
    report["sentence_checks"] = sentence_checks(binary, model_path, chunks, questions)
    samples = []
    for r in base_rows:
        if r["lang"] == "en" and len(samples) < 6:
            samples.append({"id": r["id"], "gold": [(e["file"], e["heading"]) for e in next(q for q in questions if q["id"] == r["id"])["evidence"]],
                            "top3": [(chunks[i]["file"], chunks[i]["heading"]) for i in r["top3"]]})
    report["english_question_samples_base"] = samples
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("language_of_top3_e5_base", "language_of_top3_other_models_english_questions")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
