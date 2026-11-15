#!/usr/bin/env python3
"""Target expert sets for the H2 router-LoRA: per content type and layer, the m most used experts on the TRAINING sessions,
with m = ceil(ratio * |S(alpha=0.9)|). Reads the per-token Tier 2 sequences of a run tag (logs/tier2/<tag>/<session>.npz).

  python scripts/h2_targets.py --tag base_all --train coder_pilot,coder_01,coder_02,planner_pilot,planner_01,planner_02,writer_01
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from affinity_pipeline import tier2_stats as st  # noqa: E402

SESSIONS_DIR = Path("data") / "routing_corpus_v2" / "sessions"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base_all")
    ap.add_argument("--train", required=True)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--out", default="logs/h2/targets.json")
    a = ap.parse_args()
    counts = {}
    for name in a.train.split(","):
        topk = np.load(f"logs/tier2/{a.tag}/{name}.npz")["topk"]
        spans = json.loads((SESSIONS_DIR / f"{name}.spans.json").read_text(encoding="utf-8"))["spans"]
        for t, c in st.type_counts(topk, spans).items():
            counts[t] = counts.get(t, 0) + c
    out = {"ratio": a.ratio, "train_sessions": a.train.split(","), "targets": {}, "base_sizes": {}, "target_sizes": {}}
    for t in st.CONTENT_TYPES:
        c = counts[t]
        sizes, sets = [], []
        for l in range(c.shape[0]):
            p = c[l] / c[l].sum()
            s90 = len(st.affinity_set(p, 0.9))
            m = max(1, math.ceil(a.ratio * s90))
            order = np.argsort(-c[l])[:m]
            sets.append(sorted(int(e) for e in order))
            sizes.append(s90)
        out["targets"][t] = sets
        out["base_sizes"][t] = sizes
        out["target_sizes"][t] = [len(x) for x in sets]
        print(f"{t:10s} median base |S0.9|={np.median(sizes):.1f} target m={np.median([len(x) for x in sets]):.1f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
