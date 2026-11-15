"""Statistics over per-token expert sequences (Tier 2): locality, per-type affinity sets, base-vs-LoRA comparison.

All functions are pure numpy and take `topk`: int array [n_tokens, n_layers, k] of the expert ids chosen for each token in each
MoE layer, and `spans`: the [{start, end, type}, ...] list of a session's `*.spans.json` (token index ranges).
"""
from math import comb

import numpy as np

N_EXPERTS = 128


def random_overlap_probability(n_experts: int = N_EXPERTS, k: int = 8) -> float:
    """P(two independent uniform k-subsets of n_experts share at least one expert): the no-structure baseline for the
    locality statistic Pr[E_i ∩ E_{i+1} ≠ ∅]."""
    return 1.0 - comb(n_experts - k, k) / comb(n_experts, k)


def onehot(topk: np.ndarray, n_experts: int = N_EXPERTS) -> np.ndarray:
    """bool [n_tokens, n_layers, n_experts]"""
    n, n_layers, k = topk.shape
    m = np.zeros((n, n_layers, n_experts), dtype=bool)
    ii = np.arange(n)[:, None, None]
    ll = np.arange(n_layers)[None, :, None]
    m[ii, ll, topk.astype(np.int64)] = True
    return m


def type_array(spans: list, n_tokens: int) -> np.ndarray:
    """object array [n_tokens] with the span type of every token ('' for tokens outside every span)."""
    t = np.full(n_tokens, "", dtype=object)
    for s in spans:
        t[s["start"]:s["end"]] = s["type"]
    return t


def pair_intersections(topk: np.ndarray, n_experts: int = N_EXPERTS) -> np.ndarray:
    """|E_i ∩ E_{i+1}| for every consecutive token pair and layer: int [n_tokens - 1, n_layers]"""
    m = onehot(topk, n_experts)
    return (m[:-1] & m[1:]).sum(axis=2)


def shuffled_overlap(m: np.ndarray, idx: np.ndarray, rng) -> np.ndarray:
    """Control: P(overlap) between tokens of the same group paired in a RANDOM order. It keeps each expert's marginal usage
    (so popular experts still collide) but destroys temporal adjacency; the gap to the real value is the genuine locality.
    Returns bool-mean per layer."""
    perm = rng.permutation(idx)
    return ((m[perm[:-1]] & m[perm[1:]]).sum(axis=2) > 0).mean(axis=0)


def locality(topk: np.ndarray, spans: list, n_experts: int = N_EXPERTS, seed: int = 0) -> dict:
    """Pr[E_i ∩ E_{i+1} ≠ ∅] and mean |E_i ∩ E_{i+1}| per layer, for consecutive-token pairs grouped by
    'all', each span type (both tokens in the same span type) and 'boundary' (the two tokens belong to different spans)."""
    inter = pair_intersections(topk, n_experts)
    types = type_array(spans, topk.shape[0])
    a, b = types[:-1], types[1:]
    sid = np.full(topk.shape[0], -1)
    for i, s in enumerate(spans):
        sid[s["start"]:s["end"]] = i
    same = (sid[:-1] == sid[1:]) & (sid[:-1] >= 0)
    groups = {"all": np.ones(len(a), dtype=bool), "boundary": (sid[:-1] != sid[1:]) & (sid[:-1] >= 0) & (sid[1:] >= 0)}
    for t in sorted({s["type"] for s in spans}):
        groups[t] = same & (a == t)
    out = {}
    m_all = onehot(topk, n_experts)
    rng = np.random.default_rng(seed)
    for name, mask in groups.items():
        n = int(mask.sum())
        if n == 0:
            continue
        sel = inter[mask]
        if name == "all":
            idx = np.arange(topk.shape[0])
        elif name == "boundary":
            idx = None
        else:
            idx = np.where(types == name)[0]
        shuf = shuffled_overlap(m_all, idx, rng) if idx is not None and len(idx) > 2 else np.full(topk.shape[1], np.nan)
        out[name] = {"pairs": n,
                     "p_overlap_shuffled_per_layer": shuf.tolist(),
                     "p_overlap_per_layer": (sel > 0).mean(axis=0).tolist(),
                     "mean_intersection_per_layer": sel.mean(axis=0).tolist(),
                     "p_overlap_mean": float((sel > 0).mean()),
                     "p_overlap_median_layer": float(np.median((sel > 0).mean(axis=0)))}
    return out


def type_counts(topk: np.ndarray, spans: list, n_experts: int = N_EXPERTS) -> dict:
    """{type: float64 [n_layers, n_experts]} number of times each expert was chosen, per span type."""
    types = type_array(spans, topk.shape[0])
    m = onehot(topk, n_experts)
    return {t: m[types == t].sum(axis=0).astype(np.float64) for t in sorted({s["type"] for s in spans})}


def affinity_set(p: np.ndarray, alpha: float) -> set:
    """Smallest set of experts whose probability mass reaches alpha (same rule as scripts/aggregate_expert_affinity.py)."""
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    k = int(np.searchsorted(cum, alpha) + 1)
    return set(order[:k].tolist())


def set_sizes(counts: np.ndarray, alpha: float) -> list:
    """per-layer affinity-set size for a counts matrix [n_layers, n_experts]"""
    sizes = []
    for l in range(counts.shape[0]):
        tot = counts[l].sum()
        sizes.append(len(affinity_set(counts[l] / tot, alpha)) if tot > 0 else 0)
    return sizes


def jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


CONTENT_TYPES = ("reasoning", "code", "factual", "creative")


def window_hit(topk: np.ndarray, window: int, n_experts: int = N_EXPERTS) -> tuple:
    """Dynamic (per-step) selection: for token i the 'cache' is the union of experts used by tokens i-window..i-1.
    Returns (hit_rate [n_layers], mean_union_size [n_layers]) where hit_rate is the fraction of a token's k experts found in
    that union, averaged over tokens i >= window."""
    m = onehot(topk, n_experts)
    n, n_layers, _ = m.shape
    hits, sizes = np.zeros(n_layers), np.zeros(n_layers)
    for l in range(n_layers):
        c = np.cumsum(m[:, l, :].astype(np.int32), axis=0)
        win = c[window - 1:n - 1] - np.vstack([np.zeros((1, n_experts), np.int32), c[:n - window - 1]])
        union = win > 0
        cur = m[window:, l, :]
        hits[l] = (cur & union).sum() / cur.sum()
        sizes[l] = union.sum(axis=1).mean()
    return hits, sizes


def type_conditioning(counts: dict, alpha: float = 0.9) -> dict:
    """Effect of knowing the chunk type: per-type affinity-set size versus the set of the type-agnostic mixture
    (counts pooled over the content types), as a fraction. This is what chunk-boundary conditioning buys."""
    types = [t for t in CONTENT_TYPES if t in counts]
    pooled = sum(counts[t] for t in types)
    g = float(np.median(set_sizes(pooled, alpha)))
    per = {t: float(np.median(set_sizes(counts[t], alpha))) for t in types}
    return {"alpha": alpha, "global_median_size": g, "per_type_median_size": per,
            "per_type_reduction": {t: 1 - v / g for t, v in per.items()},
            "mean_reduction": float(np.mean([1 - v / g for v in per.values()]))}


def compare_runs(base: dict, other: dict, alpha: float = 0.9) -> dict:
    """base / other: {type: counts [L, E]}. Per type: median per-layer set size in each run, relative change and the
    median per-layer Jaccard between the two sets. Used for H2 (LoRA attached vs not)."""
    out = {}
    for t in sorted(set(base) & set(other)):
        sb, so = set_sizes(base[t], alpha), set_sizes(other[t], alpha)
        jac = []
        for l in range(base[t].shape[0]):
            pb, po = base[t][l] / base[t][l].sum(), other[t][l] / other[t][l].sum()
            jac.append(jaccard(affinity_set(pb, alpha), affinity_set(po, alpha)))
        mb, mo = float(np.median(sb)), float(np.median(so))
        out[t] = {"median_size_base": mb, "median_size_other": mo,
                  "relative_change": (mo - mb) / mb if mb else float("nan"), "median_layer_jaccard": float(np.median(jac))}
    return out
