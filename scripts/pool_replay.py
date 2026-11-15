#!/usr/bin/env python3
"""Expert RAM pool replay on REAL per-token routing (Tier 2 sequences), replacing the Mixtral estimates of thesis 6.1-6.4.

For every session, layer and pool capacity C (experts kept in the fast tier per layer) three policies are replayed over the
actual expert ids the Gemma4 26B-A4B router chose:
  reactive : token-level LRU cache of C experts (load an expert only when a token needs it and it is not cached)
  static   : the type's C most-used experts, fixed per chunk type, counted on the OTHER sessions (leave-one-session-out,
             so the session being replayed never helps pick its own pool); unseen/untyped tokens use the all-type pool
  spec     : Spec-Experts: at every chunk boundary pre-load the type's top-C experts, then serve misses with LRU inside C
A "hit" is an expert of the token already in the pool; a "load" is one expert-layer fetched from the slow tier.
  python scripts/pool_replay.py --tag base_all --caps 32,48,57,64 [--out logs/replay_base_all.json]
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from affinity_pipeline import tier2_stats as st  # noqa: E402

SESSIONS_DIR = Path("data") / "routing_corpus_v2" / "sessions"
EXPERT_MIB = 4.14           # MiB per expert-layer (gate_up Q4_K 2.127 + down Q8_0 2.009), from the model's tensor sizes
DDR3_GBPS = 17.0            # nominal DDR3-1600 dual channel bandwidth of the GT 1030 machine


def load(root: Path, tag: str):
    sess = {}
    for f in sorted((root / "logs" / "tier2" / tag).glob("*.npz")):
        if f.stem in ("counts",):
            continue
        name = f.stem
        spans = json.loads((root / SESSIONS_DIR / f"{name}.spans.json").read_text(encoding="utf-8"))["spans"]
        sess[name] = (np.load(f)["topk"], spans)
    return sess


def lru_step(cache: OrderedDict, experts, cap: int) -> int:
    """serve one token's experts from an LRU cache of capacity cap; returns the number of loads (misses)"""
    miss = 0
    for e in experts:
        if e in cache:
            cache.move_to_end(e)
        else:
            miss += 1
            cache[e] = True
            if len(cache) > cap:
                cache.popitem(last=False)
    return miss


def replay_session(topk, spans, pooled_other: dict, glob_other: np.ndarray, cap: int):
    n, n_layers, k = topk.shape
    types = st.type_array(spans, n)
    sid = np.full(n, -1)
    for i, s in enumerate(spans):
        sid[s["start"]:s["end"]] = i
    total = n * n_layers * k
    r_miss = s_hit = sp_miss = sp_pre = 0
    for l in range(n_layers):
        pools_l = {t: set(np.argsort(-c[l])[:cap].tolist()) for t, c in pooled_other.items()}
        g = set(np.argsort(-glob_other[l])[:cap].tolist())
        rows = topk[:, l, :].tolist()
        lru_r, lru_s, cur_span = OrderedDict(), OrderedDict(), -2
        for i in range(n):
            ex = rows[i]
            t = types[i]
            pool = pools_l.get(t, g) if isinstance(t, str) else g
            r_miss += lru_step(lru_r, ex, cap)
            s_hit += sum(1 for e in ex if e in pool)
            if sid[i] != cur_span:                      # chunk boundary: pre-load the type pool, then serve misses with LRU
                cur_span = sid[i]
                lru_s = OrderedDict((e, True) for e in sorted(pool))
                sp_pre += len(pool)
            sp_miss += lru_step(lru_s, ex, cap)
    res = {"reactive": [total - r_miss, r_miss], "static": [s_hit, total - s_hit], "spec": [total - sp_miss, sp_miss + sp_pre]}
    return res, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tag", default="base_all")
    ap.add_argument("--caps", default="32,48,57,64")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    sess = load(root, a.tag)
    per = {n: st.type_counts(t, sp) for n, (t, sp) in sess.items()}
    allc = {n: sum(c.values()) for n, c in per.items()}
    out = {"tag": a.tag, "sessions": list(sess), "expert_mib": EXPERT_MIB, "ddr3_gbps": DDR3_GBPS, "caps": {}}
    for cap in [int(x) for x in a.caps.split(",")]:
        agg = {p: [0, 0] for p in ("reactive", "static", "spec")}
        tot_tokens = 0
        tot_slots = 0
        for name, (topk, spans) in sess.items():
            pooled = {}
            for other, cs in per.items():
                if other == name:
                    continue
                for t, c in cs.items():
                    pooled[t] = pooled.get(t, 0) + c
            glob = sum(c for o, c in allc.items() if o != name)
            r, total = replay_session(topk, spans, pooled, glob, cap)
            for p in agg:
                agg[p][0] += r[p][0]
                agg[p][1] += r[p][1]
            tot_slots += total
            tot_tokens += topk.shape[0]
        row = {}
        for p, (h, ld) in agg.items():
            loads_per_token = ld / tot_tokens                 # expert-layer loads per generated token (all 30 layers)
            ms = loads_per_token * EXPERT_MIB / 1024 / DDR3_GBPS * 1000
            row[p] = {"hit_rate": h / tot_slots, "loads_per_token": loads_per_token, "load_ms_per_token_ddr3": ms}
        out["caps"][str(cap)] = row
        print(f"C={cap:3d}: " + "  ".join(f"{p} hit={v['hit_rate']:.3f} loads/token={v['loads_per_token']:.1f} ({v['load_ms_per_token_ddr3']:.1f} ms)"
                                           for p, v in row.items()), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
