#!/usr/bin/env python3
"""E03 (direction A): embedding-retrieved expert pools versus hand-typed pools, leave-one-session-out.

For a held-out session every span gets a per-layer pool of C experts chosen from the routing profiles of the spans of the
OTHER sessions:
  type      : the E01 definition (top-C of the span's content type, all-type pool for control/user_prompt)
  knn       : cosine k-nearest training spans in an embedding space; profile = sum of neighbour routing profiles weighted by
              clipped cosine similarity, each training profile divided by its token count; top-C per layer
  softtype  : weights over the four content-type centroids (clipped cosine, normalized) mix the four type profiles
and two ways of using a pool:
  static : the pool is fixed for the span (a hit = the token's expert is in the pool)
  spec   : the pool is preloaded at the span boundary (counted as loads) and the cache keeps serving misses with LRU inside C
Pre-registered primary (plan section 3.1): representation bge-small-en-v1.5-q8_0, source ctx (causal), knn k=8.
Everything else is secondary. Self-check: the type baselines and reactive are recomputed with scripts/embed_pool_eval.py and must
match exactly.
  python scripts/a_embedding_pools.py --tags base_all,base_new --caps 32,48,57,64 --out logs/a_embedding_pools.json
"""
import argparse
import json
import sys
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import embed_pool_eval as e01  # noqa: E402
from affinity_pipeline import tier2_stats as st  # noqa: E402

N_EXPERTS = 128
CONTENT = st.CONTENT_TYPES
PRIMARY = ("bge-small-en-v1.5-q8_0.gguf", "ctx")
REPS = ["bge-small-en-v1.5-q8_0.gguf", "all-MiniLM-L6-v2.Q8_0.gguf", "multilingual-e5-small-Q8_0.gguf",
        "Qwen3-Embedding-0.6B-Q8_0.gguf", "tfidf_char"]
K_NEIGHBORS = 8


def load_all(root: Path, tags: list[str]) -> dict[str, tuple[np.ndarray, list[dict]]]:
    out: dict = {}
    for tag in tags:
        for name, value in e01.load_sessions(root, tag).items():
            if name in out:
                raise ValueError(f"session {name} appears in more than one tag")
            out[name] = value
    return out


def span_counts(topk: np.ndarray, spans: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """per-span expert counts [S, L, E] and token counts [S]"""
    layers = topk.shape[1]
    counts = np.zeros((len(spans), layers, N_EXPERTS), dtype=np.float64)
    ntok = np.zeros(len(spans), dtype=np.float64)
    layer_index = np.arange(layers)[None, :, None]
    for j, sp in enumerate(spans):
        a, b = int(sp["start"]), int(sp["end"])
        ntok[j] = b - a
        if b > a:
            idx = np.broadcast_to(layer_index, topk[a:b].shape)
            np.add.at(counts[j], (idx.ravel(), topk[a:b].astype(np.int64).ravel()), 1)
    return counts, ntok


def load_embeddings(root: Path, rep: str, source: str, names: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for name in names:
        f = root / "logs" / "embed" / rep / name / f"{source}.npy"
        out[name] = np.load(f).astype(np.float64)
    return out


def top_pools(profile: np.ndarray, cap: int) -> np.ndarray:
    """[L, E] -> [L, cap] deterministic top-C expert ids"""
    return np.argsort(-profile, axis=1, kind="stable")[:, :cap]


class Corpus:
    """everything needed to build pools for one held-out session"""

    def __init__(self, sessions, counts, ntok, types):
        self.sessions, self.counts, self.ntok, self.types = sessions, counts, ntok, types
        self.names = sorted(sessions)

    def train_names(self, held_out: str) -> list[str]:
        return [n for n in self.names if n != held_out]


def type_profiles(corpus: Corpus, held_out: str, kinds=CONTENT) -> dict[str, np.ndarray]:
    """per span type: summed counts / summed tokens over the OTHER sessions, [L, E]"""
    prof = {}
    for t in kinds:
        c, n = 0.0, 0.0
        for name in corpus.train_names(held_out):
            mask = np.array([x == t for x in corpus.types[name]])
            if mask.any():
                c = c + corpus.counts[name][mask].sum(axis=0)
                n += corpus.ntok[name][mask].sum()
        if n > 0:
            prof[t] = c / n
    return prof


def profiles_type(corpus: Corpus, held_out: str) -> np.ndarray:
    """E01 type pools as profiles [S, L, E]: every span type has its own pool (E01 counts all span types)"""
    kinds = sorted({t for n in corpus.names for t in corpus.types[n]})
    tp = type_profiles(corpus, held_out, kinds)
    glob = sum(corpus.counts[n].sum(axis=0) for n in corpus.train_names(held_out)) / sum(
        corpus.ntok[n].sum() for n in corpus.train_names(held_out))
    return np.stack([tp[t] if t in tp else glob for t in corpus.types[held_out]])


def profiles_knn(corpus: Corpus, emb: dict[str, np.ndarray], held_out: str) -> np.ndarray:
    train = corpus.train_names(held_out)
    tr_emb = np.vstack([emb[n] for n in train])
    tr_prof = np.concatenate([corpus.counts[n] / np.maximum(corpus.ntok[n], 1)[:, None, None] for n in train])
    out = []
    for q in emb[held_out]:
        sims = tr_emb @ q
        order = np.argsort(-sims, kind="stable")[:K_NEIGHBORS]
        w = np.clip(sims[order], 0.0, None)
        if w.sum() == 0.0:
            w = np.ones(len(order))
        out.append(np.tensordot(w, tr_prof[order], axes=(0, 0)))
    return np.stack(out)


def profiles_softtype(corpus: Corpus, emb: dict[str, np.ndarray], held_out: str) -> tuple[np.ndarray, np.ndarray]:
    """returns ([S, L, E] profiles, [S] max normalized type weight)"""
    tp = type_profiles(corpus, held_out)
    kinds = [t for t in CONTENT if t in tp]
    cents = []
    for t in kinds:
        rows = [emb[n][np.array([x == t for x in corpus.types[n]])] for n in corpus.train_names(held_out)]
        v = np.vstack([r for r in rows if len(r)]).mean(axis=0)
        cents.append(v / (np.linalg.norm(v) or 1.0))
    cents = np.stack(cents)
    profs, wmax = [], []
    for q in emb[held_out]:
        w = np.clip(cents @ q, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(kinds)) / len(kinds)
        profs.append(sum(wi * tp[t] for wi, t in zip(w, kinds)))
        wmax.append(float(w.max()))
    return np.stack(profs), np.asarray(wmax)


def replay_pools(topk: np.ndarray, spans: list[dict], pools: np.ndarray, cap: int) -> dict[str, np.ndarray]:
    """pools [S, L, C]. Returns per-span arrays: static_hits, static_loads, spec_hits, spec_loads (preloads included), slots"""
    n_spans, layers = len(spans), topk.shape[1]
    sh, sl, ph, pl, slots = (np.zeros(n_spans) for _ in range(5))
    width = topk.shape[2]
    for j, sp in enumerate(spans):
        a, b = int(sp["start"]), int(sp["end"])
        if b <= a:
            continue
        seg = topk[a:b]
        slots[j] = seg.size
        mask = np.zeros((layers, N_EXPERTS), dtype=bool)
        mask[np.arange(layers)[:, None], pools[j]] = True
        hits = int(mask[np.arange(layers)[None, :, None], seg].sum())
        sh[j] = hits
        sl[j] = seg.size - hits
        rows = seg.tolist()
        loads = 0
        for layer in range(layers):
            cache = OrderedDict((int(e), True) for e in sorted(pools[j, layer].tolist()))
            loads += cap                                   # preload at the span boundary
            for token in rows:
                loads += e01.lru_step(cache, token[layer], cap)
        pl[j] = loads
        ph[j] = seg.size - (loads - cap * layers)         # hits = served from cache without a miss load
    return {"static_hits": sh, "static_loads": sl, "spec_hits": ph, "spec_loads": pl, "slots": slots}


def reactive_session(topk: np.ndarray, cap: int) -> tuple[int, int]:
    layers = topk.shape[1]
    loads = 0
    for layer in range(layers):
        cache: OrderedDict = OrderedDict()
        for token in topk[:, layer, :].tolist():
            loads += e01.lru_step(cache, token, cap)
    return topk.size - loads, loads


def mixed_flags(types: list[str], wmax: np.ndarray) -> np.ndarray:
    """content-type span that follows a content span of a different type, or has a soft-type max weight < 0.6"""
    flags = np.zeros(len(types), dtype=bool)
    prev = None
    for j, t in enumerate(types):
        if t in CONTENT:
            flags[j] = (prev is not None and prev != t) or wmax[j] < 0.6
            prev = t
    return flags


def type_change_flags(types: list[str]) -> np.ndarray:
    """content-type span whose previous content-type span has a different type"""
    flags = np.zeros(len(types), dtype=bool)
    prev = None
    for j, t in enumerate(types):
        if t in CONTENT:
            flags[j] = prev is not None and prev != t
            prev = t
    return flags


def work(args):
    held_out, corpus, embs, caps, reps = args
    topk, spans = corpus.sessions[held_out]
    ntok_all = int(topk.shape[0])
    result: dict = {"tokens": ntok_all, "caps": {}, "mixed": None, "content": None}
    prof_sets: dict[str, np.ndarray] = {}
    wmax = None
    for rep in reps:
        for source in ("ctx", "chunk"):
            prof_sets[f"knn|{rep}|{source}"] = profiles_knn(corpus, embs[(rep, source)], held_out)
    for source in ("ctx", "chunk"):
        pr, w = profiles_softtype(corpus, embs[(PRIMARY[0], source)], held_out)
        prof_sets[f"softtype|{PRIMARY[0]}|{source}"] = pr
        if source == "ctx":
            wmax = w
    types = corpus.types[held_out]
    per_session_counts = {n: st.type_counts(corpus.sessions[n][0], corpus.sessions[n][1]) for n in corpus.names}
    all_counts = {n: sum(c.values()) for n, c in per_session_counts.items()}
    pooled_other: dict = {}
    for n, c in per_session_counts.items():
        if n != held_out:
            for t, v in c.items():
                pooled_other[t] = pooled_other.get(t, 0) + v
    global_other = sum(v for n, v in all_counts.items() if n != held_out)
    flags = mixed_flags(types, wmax)
    result["mixed"] = flags.tolist()
    result["type_change"] = type_change_flags(types).tolist()
    result["content"] = [t in CONTENT for t in types]
    for cap in caps:
        row = {"reactive": dict(zip(("hits", "loads"), reactive_session(topk, cap))), "sets": {}}
        typed, glob = e01._pool_sets(pooled_other, global_other, cap)       # exactly the E01 type pools
        type_pools = np.stack([np.stack([np.array(sorted(typed.get(t, glob)[l] if isinstance(typed.get(t, glob), list) else glob[l]))
                                         for l in range(topk.shape[1])]) for t in types])
        r = replay_pools(topk, spans, type_pools, cap)
        row["sets"]["type"] = {k: v.tolist() for k, v in r.items()}
        for key, prof in prof_sets.items():
            pools = np.stack([top_pools(p, cap) for p in prof])
            r = replay_pools(topk, spans, pools, cap)
            row["sets"][key] = {k: v.tolist() for k, v in r.items()}
        result["caps"][str(cap)] = row
    return held_out, result


def bootstrap(values: list[float], samples: int = 10000, seed: int = 0) -> dict:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(samples, len(v)))].mean(axis=1)
    return {"mean": float(v.mean()), "ci95_low": float(np.percentile(means, 2.5)), "ci95_high": float(np.percentile(means, 97.5))}


def summarize(raw: dict, caps: list[int], names: list[str], reps: list[str]) -> dict:
    out: dict = {"caps": {}}
    for cap in caps:
        c = str(cap)
        rows: dict = {}
        keys = list(raw[names[0]]["caps"][c]["sets"].keys())
        per_session: dict = {k: {"static": [], "spec": [], "static_loads": [], "spec_loads": []} for k in keys}
        react = {"hit": [], "loads": []}
        slots_s, tok_s = [], []
        for n in names:
            r = raw[n]["caps"][c]
            sl = {k: np.asarray(v["slots"]) for k, v in r["sets"].items()}
            any_key = keys[0]
            slots = sl[any_key].sum()
            tok = raw[n]["tokens"]
            slots_s.append(slots)
            tok_s.append(tok)
            react["hit"].append(r["reactive"]["hits"] / slots)
            react["loads"].append(r["reactive"]["loads"] / tok)
            for k in keys:
                s = r["sets"][k]
                per_session[k]["static"].append(float(np.sum(s["static_hits"]) / slots))
                per_session[k]["spec"].append(float(np.sum(s["spec_hits"]) / slots))
                per_session[k]["static_loads"].append(float(np.sum(s["static_loads"]) / tok))
                per_session[k]["spec_loads"].append(float(np.sum(s["spec_loads"]) / tok))
        rows["reactive"] = {"hit_rate": float(np.mean(react["hit"])), "loads_per_token": float(np.mean(react["loads"]))}
        for k in keys:
            ps = per_session[k]
            rows[k] = {"static_hit": float(np.mean(ps["static"])), "spec_hit": float(np.mean(ps["spec"])),
                       "static_loads_per_token": float(np.mean(ps["static_loads"])),
                       "spec_loads_per_token": float(np.mean(ps["spec_loads"]))}
        comps = {}
        base = per_session["type"]
        for k in keys:
            if k == "type":
                continue
            ps = per_session[k]
            comps[k] = {
                "static_hit_minus_type": bootstrap([a - b for a, b in zip(ps["static"], base["static"])]),
                "spec_hit_minus_type": bootstrap([a - b for a, b in zip(ps["spec"], base["spec"])]),
                "static_loads_minus_type": bootstrap([a - b for a, b in zip(ps["static_loads"], base["static_loads"])]),
                "spec_loads_minus_type": bootstrap([a - b for a, b in zip(ps["spec_loads"], base["spec_loads"])]),
                "spec_loads_minus_reactive": bootstrap([a - b for a, b in zip(ps["spec_loads"], react["loads"])]),
            }
        # mixed-chunk subset (content spans flagged mixed) for primary and baseline, pooled over sessions
        sub = {}
        for label, key in (("type", "type"), ("primary", f"knn|{PRIMARY[0]}|{PRIMARY[1]}")):
            hs = ps_ = sp_ = ss = 0.0
            for n in names:
                fl = np.asarray(raw[n]["mixed"], dtype=bool)
                ct = np.asarray(raw[n]["content"], dtype=bool)
                s = raw[n]["caps"][c]["sets"][key]
                tc = np.asarray(raw[n]["type_change"], dtype=bool)
                for sel_name, sel in (("mixed", fl), ("content_unmixed", ct & ~fl),
                                      ("type_change", tc), ("content_same_type", ct & ~tc)):
                    sub.setdefault(label, {}).setdefault(sel_name, [0.0, 0.0, 0.0])
                    sub[label][sel_name][0] += float(np.asarray(s["static_hits"])[sel].sum())
                    sub[label][sel_name][1] += float(np.asarray(s["spec_hits"])[sel].sum())
                    sub[label][sel_name][2] += float(np.asarray(s["slots"])[sel].sum())
        sub_out = {lab: {sel: {"static_hit": v[0] / v[2] if v[2] else None, "spec_hit": v[1] / v[2] if v[2] else None, "slots": v[2]}
                         for sel, v in d.items()} for lab, d in sub.items()}
        out["caps"][c] = {"policies": rows, "paired_vs_type": comps, "subsets": sub_out,
                          "primary_key": f"knn|{PRIMARY[0]}|{PRIMARY[1]}"}
    return out


def self_check(sessions, caps, summary_rows, names) -> dict:
    """recompute the E01 baselines and compare with the type/reactive numbers produced here (token-weighted pooled rates)"""
    e = e01.evaluate(sessions, caps, ["reactive", "static", "spec"])
    res = {}
    for cap in caps:
        p = e["caps"][str(cap)]["policies"]
        res[str(cap)] = {"e01": {k: (p[k]["hit_rate"], p[k]["loads_per_token"]) for k in p}}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tags", default="base_all,base_new")
    ap.add_argument("--caps", default="32,48,57,64")
    ap.add_argument("--reps", default=",".join(REPS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    caps = [int(x) for x in a.caps.split(",")]
    reps = [x for x in a.reps.split(",") if x]
    if PRIMARY[0] not in reps:
        reps.insert(0, PRIMARY[0])
    sessions = load_all(root, a.tags.split(","))
    names = sorted(sessions)
    counts, ntok, types = {}, {}, {}
    for n, (topk, spans) in sessions.items():
        counts[n], ntok[n] = span_counts(topk, spans)
        types[n] = [sp.get("type") for sp in spans]
    corpus = Corpus(sessions, counts, ntok, types)
    embs = {(rep, src): load_embeddings(root, rep, src, names) for rep in reps for src in ("ctx", "chunk")}
    jobs = [(n, corpus, embs, caps, reps) for n in names]
    with Pool(a.workers) as pool:
        raw = dict(pool.map(work, jobs, chunksize=1))
    summary = summarize(raw, caps, names, reps)
    check = self_check(sessions, caps, None, names)
    # token-weighted pooled type baselines from this script for the E01 comparison
    for cap in caps:
        c = str(cap)
        tot = {"static_h": 0.0, "spec_h": 0.0, "slots": 0.0, "static_l": 0.0, "spec_l": 0.0, "tok": 0.0, "react_h": 0.0, "react_l": 0.0}
        for n in names:
            r = raw[n]["caps"][c]
            s = r["sets"]["type"]
            tot["static_h"] += sum(s["static_hits"]); tot["spec_h"] += sum(s["spec_hits"]); tot["slots"] += sum(s["slots"])
            tot["static_l"] += sum(s["static_loads"]); tot["spec_l"] += sum(s["spec_loads"]); tot["tok"] += raw[n]["tokens"]
            tot["react_h"] += r["reactive"]["hits"]; tot["react_l"] += r["reactive"]["loads"]
        mine = {"reactive": (tot["react_h"] / tot["slots"], tot["react_l"] / tot["tok"]),
                "static": (tot["static_h"] / tot["slots"], tot["static_l"] / tot["tok"]),
                "spec": (tot["spec_h"] / tot["slots"], tot["spec_l"] / tot["tok"])}
        check[c]["mine"] = mine
        check[c]["max_abs_diff"] = max(abs(mine[k][i] - check[c]["e01"][k][i]) for k in mine for i in (0, 1))
        print(f"C={cap}: self-check max|diff| vs E01 = {check[c]['max_abs_diff']:.2e}", flush=True)
    out = {"schema": "a_embedding_pools/v1", "sessions": names, "caps": caps, "reps": reps, "k_neighbors": K_NEIGHBORS,
           "primary": {"rep": PRIMARY[0], "source": PRIMARY[1]}, "summary": summary, "self_check": check}
    for cap in caps:
        c = str(cap)
        rows = summary["caps"][c]["policies"]
        pk = summary["caps"][c]["primary_key"]
        cmp_ = summary["caps"][c]["paired_vs_type"][pk]
        print(f"C={cap}: type static={rows['type']['static_hit']:.3f} spec={rows['type']['spec_hit']:.3f}  reactive={rows['reactive']['hit_rate']:.3f}  "
              f"primary static={rows[pk]['static_hit']:.3f} spec={rows[pk]['spec_hit']:.3f}  "
              f"d_static={cmp_['static_hit_minus_type']['mean']:+.4f} [{cmp_['static_hit_minus_type']['ci95_low']:+.4f},{cmp_['static_hit_minus_type']['ci95_high']:+.4f}]  "
              f"d_spec={cmp_['spec_hit_minus_type']['mean']:+.4f} [{cmp_['spec_hit_minus_type']['ci95_low']:+.4f},{cmp_['spec_hit_minus_type']['ci95_high']:+.4f}]", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
