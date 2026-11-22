"""E03 helper tests on synthetic routing traces (no model files, no gx10)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import a_embedding_pools as a  # noqa: E402


def make_session(rng, n_tokens=40, layers=3, kinds=("reasoning", "code")):
    topk = rng.integers(0, 128, size=(n_tokens, layers, 8)).astype(np.uint8)
    half = n_tokens // 2
    spans = [{"start": 0, "end": half, "type": kinds[0]}, {"start": half, "end": n_tokens, "type": kinds[1]}]
    return topk, spans


def test_span_counts_and_loso_profiles():
    rng = np.random.default_rng(0)
    sessions = {n: make_session(rng) for n in ("s1", "s2", "s3")}
    counts, ntok, types = {}, {}, {}
    for n, (topk, spans) in sessions.items():
        counts[n], ntok[n] = a.span_counts(topk, spans)
        types[n] = [sp["type"] for sp in spans]
        assert counts[n].sum() == topk.size                      # every selection counted once
    corpus = a.Corpus(sessions, counts, ntok, types)
    emb = {n: rng.normal(size=(2, 5)) for n in sessions}
    prof = a.profiles_knn(corpus, emb, "s1")
    assert prof.shape == (2, 3, 128)
    # the held-out session's own counts must not enter the profile: zero its counts and nothing changes
    counts2 = dict(counts)
    counts2["s1"] = counts["s1"] * 0
    corpus2 = a.Corpus(sessions, counts2, ntok, types)
    assert np.allclose(prof, a.profiles_knn(corpus2, emb, "s1"))


def test_replay_pools_static_matches_naive_and_full_pool_hits_everything():
    rng = np.random.default_rng(1)
    topk, spans = make_session(rng)
    pools = np.stack([np.tile(np.arange(10), (3, 1)) for _ in spans])      # pool = experts 0..9 in every layer
    r = a.replay_pools(topk, spans, pools, 10)
    naive = int((topk < 10).sum())
    assert r["static_hits"].sum() == naive
    assert r["slots"].sum() == topk.size
    full = np.stack([np.tile(np.arange(128), (3, 1)) for _ in spans])
    r2 = a.replay_pools(topk, spans, full, 128)
    assert r2["static_hits"].sum() == topk.size and r2["spec_hits"].sum() == topk.size


def test_softtype_weights_and_mixed_flags():
    types = ["control", "reasoning", "code", "code", "factual"]
    wmax = np.array([1.0, 0.9, 0.9, 0.5, 0.9])
    flags = a.mixed_flags(types, wmax)
    assert flags.tolist() == [False, False, True, True, True]      # type change, low weight, type change
