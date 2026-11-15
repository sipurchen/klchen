"""Cross-check Tier 2 (per-token ids) against Tier 1 (llama-imatrix cumulative counts) for the same session.

For the whole-session imatrix job the per-expert counts of layer l are the number of tokens routed to each expert; summing the
one-hot of the Tier 2 per-token ids over all tokens must give the same matrix (test plan T6.3, there stated as Pearson r >= 0.90).
Usage (through gx10_run.py `xcheck`):  xcheck TAG SESSION [SESSION ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
from aggregate_expert_affinity import read_counts  # noqa: E402
from affinity_pipeline.tier2_stats import onehot  # noqa: E402


def main() -> int:
    tag, sessions = sys.argv[1], sys.argv[2:]
    jobs = json.loads(Path("data/routing_corpus_v2/jobs.json").read_text(encoding="utf-8"))
    for s in sessions:
        whole = [j for j in jobs if j["session"] == s and j.get("is_whole_session")]
        if len(whole) != 1:
            print(s, "no unique whole-session job", len(whole))
            continue
        t1 = np.asarray(read_counts(Path(whole[0]["outfile"])), dtype=np.float64)
        topk = np.load(f"logs/tier2/{tag}/{s}.npz")["topk"]
        t2 = onehot(topk).sum(axis=0).astype(np.float64)
        r = np.corrcoef(t1.ravel(), t2.ravel())[0, 1]
        print(f"{s}: tier1 total={t1.sum():.0f} tier2 total={t2.sum():.0f} tokens(T2)={topk.shape[0]} job ctx={whole[0]['ctx']} "
              f"max|diff|={np.abs(t1 - t2).max():.0f} exact_equal_cells={(t1 == t2).mean():.4f} pearson_r={r:.6f}")
        per_layer_r = [np.corrcoef(t1[l], t2[l])[0, 1] for l in range(t1.shape[0])]
        print(f"   per-layer r min={min(per_layer_r):.4f} median={np.median(per_layer_r):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
