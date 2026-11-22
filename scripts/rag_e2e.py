#!/usr/bin/env python3
"""E08: end-to-end retrieval-augmented answering with Gemma4 26B-A4B under the restricted gx10 profile.

24 of the 40 questions (12 zh + 12 en; per language the first 4 'needs_two_passages' questions by id and the first 8 others)
are answered by llama-server with temperature 0 under up to four conditions: none (no context), bm25 (top-3 BM25 chunks),
emb (top-3 chunks of the chosen embedding retriever) and gold (the evidence chunks of the question, up to 3: an upper bound
for perfect retrieval). Scoring rule (fixed before any run): a key fact is present when its normalized text is a substring of the normalized
answer, or, for facts that contain numbers, when every number of the fact appears among the numbers of the answer. The score of
a question is the fraction of its key facts present; the criterion (plan section 3, H6) is the mean score with retrieval minus
the mean score without, >= +0.20, with at most 1024 added prompt tokens per question; it is reported for every retrieval condition.
  python scripts/rag_e2e.py --retriever bge-small-en-v1.5-q8_0.gguf --out logs/rag_e2e
"""
import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from affinity_pipeline import gen_session as gs  # noqa: E402
from affinity_pipeline.config import load_profile  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_rag_retrieval as er  # noqa: E402

TURN_OPEN, TURN_CLOSE, MODEL_OPEN = "<|turn>user\n", "<turn|>\n", "<|turn>model\n"
INSTRUCTION = "Answer in the language of the question, in at most three sentences."
RAG_INSTRUCTION = ("Answer the question using only the documents below. If they do not contain the answer, say so. "
                   "Answer in the language of the question, in at most three sentences.")
TOP_K = 3
SEED = 1


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s*`_\\]+", "", text)


def numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", unicodedata.normalize("NFKC", text))


def fact_present(fact: str, answer: str) -> bool:
    if normalize(fact) in normalize(answer):
        return True
    nums = numbers(fact)
    return bool(nums) and all(n in set(numbers(answer)) for n in nums)


def score_answer(key_facts: list[str], answer: str) -> dict:
    present = [fact_present(f, answer) for f in key_facts]
    return {"fraction": sum(present) / len(present), "all": all(present), "present": present}


def select_questions(questions: list[dict], per_lang: int = 12, two: int = 4) -> list[dict]:
    chosen: list[dict] = []
    for lang in ("zh", "en"):
        rows = sorted((q for q in questions if q["lang"] == lang), key=lambda q: q["id"])
        multi = [q for q in rows if q.get("needs_two_passages")][:two]
        single = [q for q in rows if not q.get("needs_two_passages")][:per_lang - len(multi)]
        chosen += sorted(multi + single, key=lambda q: q["id"])
    return chosen


def build_prompt(question: str, passages: list[dict] | None) -> str:
    if not passages:
        return f"{TURN_OPEN}{INSTRUCTION}\n\nQuestion: {question}{TURN_CLOSE}{MODEL_OPEN}"
    docs = "\n\n".join(f"[{i + 1}] {p['file']} :: {p['heading']}\n{p['text']}" for i, p in enumerate(passages))
    return f"{TURN_OPEN}{RAG_INSTRUCTION}\n\n{docs}\n\nQuestion: {question}{TURN_CLOSE}{MODEL_OPEN}"


def gold_chunks(question: dict, chunks: list[dict]) -> list[dict]:
    """per evidence: the chunk of its heading that contains the start of the quote (else the first chunk of the heading); max TOP_K"""
    picked: list[dict] = []
    for ev in question["evidence"]:
        rows = [c for c in chunks if c["file"] == ev["file"] and c["heading"] == ev["heading"]]
        if not rows:
            raise ValueError(f"{question['id']}: evidence heading has no chunk")
        head = normalize(ev["quote"])[:40]
        best = next((c for c in rows if head in normalize(c["text"])), rows[0])
        if best not in picked:
            picked.append(best)
    return picked[:TOP_K]


def retrieve(root: Path, retriever: str, questions: list[dict], all_questions: list[dict], top_k: int = TOP_K) -> dict[str, list[list[dict]]]:
    chunks = [json.loads(line) for line in (root / "logs" / "rag_eval" / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]   # the exact chunk set that was embedded
    base = root / "logs" / "rag_eval" / retriever
    passages, queries = np.load(base / "embeddings.npy"), np.load(base / "questions.npy")
    index = {q["id"]: i for i, q in enumerate(all_questions)}
    texts = [c["text"] for c in chunks]
    out: dict[str, list[list[dict]]] = {"emb": [], "bm25": [], "gold": []}
    for q in questions:
        order = np.argsort(-(passages @ queries[index[q["id"]]]), kind="stable")[:top_k]
        out["emb"].append([chunks[int(i)] for i in order])
        order = np.argsort(-er.bm25_scores(q["question"], texts), kind="stable")[:top_k]
        out["bm25"].append([chunks[int(i)] for i in order])
        out["gold"].append(gold_chunks(q, chunks))
    return out


def complete(base_url: str, prompt: str, max_tokens: int, timeout: float = 1800) -> dict:
    body = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.0, "seed": SEED, "cache_prompt": False}
    req = urllib.request.Request(f"{base_url}/completion", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--retriever", required=True)
    ap.add_argument("--out", default="logs/rag_e2e")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--ngl", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--port", type=int, default=8261)
    ap.add_argument("--conditions", default="none,bm25,emb,gold")
    ap.add_argument("--top-k", type=int, default=TOP_K, help="number of top chunks to retrieve for bm25 and emb (gold always uses TOP_K)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N selected questions")
    a = ap.parse_args()
    root = Path(a.root)
    all_questions = [json.loads(line) for line in (root / "data" / "rag_eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    chunk_rows = [json.loads(line) for line in (root / "logs" / "rag_eval" / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    answerable = [q for q in all_questions
                  if all(any(c["file"] == ev["file"] and c["heading"] == ev["heading"] for c in chunk_rows) for ev in q["evidence"])]
    print(f"answerable questions (evidence inside the embedded corpus): {len(answerable)} of {len(all_questions)}", flush=True)
    chosen = select_questions(answerable)
    if a.limit:
        chosen = chosen[:a.limit]
    retrieved = retrieve(root, a.retriever, chosen, all_questions, top_k=a.top_k)
    profile = load_profile("gx10_sim")
    out_dir = root / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = gs.build_server_argv(profile, a.port, ctx=a.ctx, ngl=a.ngl, extra_args=["-ub", "64"])
    server = gs.start_server(profile, argv, out_dir / "server.log", ready_timeout_s=900, check_vram=True)
    conditions = [c for c in a.conditions.split(",") if c]
    rows = []
    try:
        base_url = f"http://127.0.0.1:{a.port}"
        for qi, q in enumerate(chosen):
            row = {"id": q["id"], "lang": q["lang"], "needs_two_passages": bool(q.get("needs_two_passages")), "key_facts": q["key_facts"]}
            for name in conditions:
                passages = None if name == "none" else retrieved[name][qi]
                t0 = time.time()
                r = complete(base_url, build_prompt(q["question"], passages), a.max_tokens)
                t = r.get("timings", {})
                row[name] = {"answer": r.get("content", ""), "wall_s": time.time() - t0, "prompt_tokens": r.get("tokens_evaluated"),
                             "prompt_ms": t.get("prompt_ms"), "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                             "score": score_answer(q["key_facts"], r.get("content", "")),
                             "retrieved": None if passages is None else [{"file": p["file"], "heading": p["heading"]} for p in passages]}
            rows.append(row)
            print(q["id"], q["lang"], " ".join(f"{c}={row[c]['score']['fraction']:.2f}" for c in conditions),
                  f"wall={sum(row[c]['wall_s'] for c in conditions):.0f}s", flush=True)
            (out_dir / "answers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        server.stop()
    mean = {c: float(np.mean([r[c]["score"]["fraction"] for r in rows])) for c in conditions}
    summary = {"retriever": a.retriever, "n": len(rows), "conditions": conditions, "mean_score": mean,
               "all_facts": {c: float(np.mean([r[c]["score"]["all"] for r in rows])) for c in conditions},
               "by_lang": {lang: {c: float(np.mean([r[c]["score"]["fraction"] for r in rows if r["lang"] == lang])) for c in conditions}
                           for lang in ("zh", "en")},
               "vs_none": {}}
    for c in conditions:
        if c == "none" or "none" not in conditions:
            continue
        diffs = np.asarray([r[c]["score"]["fraction"] - r["none"]["score"]["fraction"] for r in rows])
        boot = diffs[np.random.default_rng(0).integers(0, len(diffs), size=(10000, len(diffs)))].mean(axis=1)
        added = [(r[c]["prompt_tokens"] or 0) - (r["none"]["prompt_tokens"] or 0) for r in rows]
        summary["vs_none"][c] = {"difference": float(diffs.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                 "added_prompt_tokens_median": float(np.median(added)), "added_prompt_tokens_max": int(max(added)),
                                 "prompt_ms_median": float(np.median([r[c]["prompt_ms"] or 0 for r in rows])),
                                 "met": bool(diffs.mean() >= 0.20 and max(added) <= 1024)}
    summary["none_prompt_ms_median"] = float(np.median([r["none"]["prompt_ms"] or 0 for r in rows])) if "none" in conditions else None
    summary["criterion"] = "difference >= +0.20 and added tokens <= 1024"
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
