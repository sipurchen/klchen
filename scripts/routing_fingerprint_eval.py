#!/usr/bin/env python3
"""Evaluate routing-fingerprint hypotheses from Tier 2 expert traces.

C1 uses each span's first ``prefix_tokens`` routing selections as a normalized
per-layer expert histogram.  A leave-one-session-out cosine kNN combines other
spans' later-token profiles to build a C-sized expert pool for every layer.
C2 learns adjacent-layer expert transition counts from other sessions and
compares their top-8 prediction with a target-layer popularity baseline.

Example / 範例:
  python scripts/routing_fingerprint_eval.py --tags base_all,base_new \
      --out logs/routing_fingerprint_eval.json
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from embed_pool_eval import lru_step, st


N_EXPERTS = 128
DEFAULT_CAP = 57
DEFAULT_PREFIX_TOKENS = 32
DEFAULT_NEIGHBORS = 8
DEFAULT_BOOTSTRAP_SAMPLES = 10_000


def load_sessions(root: Path, tags: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Load Tier 2 arrays from all requested tags / 載入指定 tag 的 Tier 2 陣列。"""
    sessions: dict[str, np.ndarray] = {}
    for tag in tags:
        tag_dir = root / "logs" / "tier2" / tag
        for trace in sorted(tag_dir.glob("*.npz")):
            if trace.stem == "counts":
                continue
            session_id = f"{tag}/{trace.stem}"
            if session_id in sessions:
                raise ValueError(f"duplicate session id: {session_id}")
            sessions[session_id] = np.load(trace)["topk"].astype(np.uint8, copy=False)
    if not sessions:
        raise ValueError(f"no Tier 2 traces for tags: {','.join(tags)}")
    return sessions


def load_spans(root: Path, session_id: str) -> list[dict]:
    """Read matching spans by trace stem / 依軌跡檔名讀取對應 span。"""
    stem = session_id.rsplit("/", 1)[1]
    source = root / "data" / "routing_corpus_v2" / "sessions" / f"{stem}.spans.json"
    return json.loads(source.read_text(encoding="utf-8"))["spans"]


def onehot_counts(topk: np.ndarray) -> np.ndarray:
    """Count selected experts per layer / 逐層計數已選專家。"""
    tokens, layers, width = topk.shape
    counts = np.zeros((layers, N_EXPERTS), dtype=np.float64)
    layer_index = np.broadcast_to(np.arange(layers)[None, :, None], (tokens, layers, width))
    np.add.at(counts, (layer_index.ravel(), topk.astype(np.int64, copy=False).ravel()), 1)
    return counts


def normalized_histogram(topk: np.ndarray) -> np.ndarray:
    """Return unit-length routing histogram / 回傳單位長度路由直方圖。"""
    flat = onehot_counts(topk).ravel()
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm else flat


def span_examples(topk: np.ndarray, spans: list[dict], prefix_tokens: int) -> list[dict]:
    """Split spans into causal prefix and later routing target / 將 span 拆為因果前綴與後段目標。"""
    examples: list[dict] = []
    token_count = topk.shape[0]
    for span_index, span in enumerate(spans):
        start = max(0, int(span["start"]))
        end = min(token_count, int(span["end"]))
        prefix_end = start + prefix_tokens
        if end <= prefix_end:
            continue
        examples.append({
            "span_index": span_index,
            "feature": normalized_histogram(topk[start:prefix_end]),
            "target_counts": onehot_counts(topk[prefix_end:end]),
            "target_topk": topk[prefix_end:end],
        })
    return examples


def top_pool(counts: np.ndarray, cap: int) -> np.ndarray:
    """Select a deterministic top-C pool per layer / 每層選擇可重現的 top-C 池。"""
    if not 1 <= cap <= N_EXPERTS:
        raise ValueError(f"cap must be 1..{N_EXPERTS}")
    return np.argsort(-counts, axis=1, kind="stable")[:, :cap]


def hit_rate(target_topk: np.ndarray, pool: np.ndarray) -> tuple[int, int]:
    """Return selected-expert hits and slots / 回傳已選專家的命中數與槽位數。"""
    selected = pool[np.arange(pool.shape[0])[:, None], :]
    hits = sum(np.isin(target_topk[:, layer, :], selected[layer]).sum() for layer in range(pool.shape[0]))
    return int(hits), int(target_topk.size)


def weighted_profile(query: np.ndarray, training: list[dict], neighbors: int) -> np.ndarray:
    """Combine nearest later-token profiles / 合併最近鄰的後段 token 剖面。"""
    if not training:
        raise ValueError("training examples are required")
    similarities = np.asarray([float(np.dot(query, row["feature"])) for row in training])
    order = np.argsort(-similarities, kind="stable")[:min(neighbors, len(training))]
    weights = np.clip(similarities[order], 0.0, None)
    if float(weights.sum()) == 0.0:
        weights = np.ones(len(order), dtype=np.float64)
    profile = np.zeros_like(training[0]["target_counts"])
    for weight, index in zip(weights, order, strict=True):
        profile += weight * training[int(index)]["target_counts"]
    return profile


def c1_session_scores(
    sessions: dict[str, np.ndarray],
    spans_by_session: dict[str, list[dict]],
    cap: int,
    prefix_tokens: int,
    neighbors: int,
) -> dict[str, dict]:
    """Evaluate C1 with held-out-session kNN / 以留出 session kNN 評測 C1。"""
    examples = {name: span_examples(topk, spans_by_session[name], prefix_tokens) for name, topk in sessions.items()}
    scores: dict[str, dict] = {}
    for held_out, rows in examples.items():
        training = [row for name, source_rows in examples.items() if name != held_out for row in source_rows]
        if not rows or not training:
            raise ValueError(f"C1 requires prefix and training spans for {held_out}")
        static_counts = sum((onehot_counts(topk) for name, topk in sessions.items() if name != held_out))
        static_pool = top_pool(static_counts, cap)
        knn_hits = knn_slots = static_hits = static_slots = 0
        for row in rows:
            knn_pool = top_pool(weighted_profile(row["feature"], training, neighbors), cap)
            hits, slots = hit_rate(row["target_topk"], knn_pool)
            knn_hits += hits
            knn_slots += slots
            hits, slots = hit_rate(row["target_topk"], static_pool)
            static_hits += hits
            static_slots += slots
        scores[held_out] = {
            "eligible_spans": len(rows),
            "knn_hit_rate": knn_hits / knn_slots,
            "static_all_hit_rate": static_hits / static_slots,
            "delta": knn_hits / knn_slots - static_hits / static_slots,
        }
    return scores


PREFIX_POLICIES = (
    "reactive",
    "spec_type",
    "static_type",
    "fingerprint_hybrid",
    "fingerprint_static_tail",
)


def _type_pools(
    sessions: dict[str, np.ndarray],
    spans_by_session: dict[str, list[dict]],
    held_out: str,
    cap: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build LOSO type and fallback pools / 建立留一 session 外的類型與備援池。"""
    layer_count = next(iter(sessions.values())).shape[1]
    typed_counts: dict[str, np.ndarray] = {}
    global_counts = np.zeros((layer_count, N_EXPERTS), dtype=np.float64)
    for name, topk in sessions.items():
        if name == held_out:
            continue
        global_counts += onehot_counts(topk)
        for content_type, counts in st.type_counts(topk, spans_by_session[name]).items():
            typed_counts[content_type] = typed_counts.get(
                content_type, np.zeros_like(global_counts)
            ) + counts
    return {kind: top_pool(counts, cap) for kind, counts in typed_counts.items()}, top_pool(global_counts, cap)


def _serve_lru(cache: OrderedDict[int, bool], experts: np.ndarray, cap: int) -> tuple[int, int]:
    """Serve one token with E01 LRU accounting / 使用 E01 LRU 計帳服務單一 token。"""
    loads = lru_step(cache, experts.tolist(), cap)
    return int(experts.size - loads), loads


def _serve_static(experts: np.ndarray, pool: np.ndarray, layer: int) -> tuple[int, int]:
    """Serve one token with a fixed layer pool / 以固定逐層池服務單一 token。"""
    hits = int(np.isin(experts, pool[layer]).sum())
    return hits, int(experts.size - hits)


def prefix_cost_session_scores(
    sessions: dict[str, np.ndarray],
    spans_by_session: dict[str, list[dict]],
    cap: int,
    prefix_tokens: int,
    neighbors: int,
) -> dict[str, dict]:
    """Replay all tokens of qualifying spans with LOSO profiles / 以留一剖面重播合格 span 的全部 token。

    Only spans with a full prefix and a non-empty tail are included. Reactive
    caches persist across qualifying spans in each held-out session. The
    fingerprint-hybrid charges a C-expert preload at every tail switch, as E01
    does for its ``spec`` policy.
    """
    examples = {
        name: span_examples(topk, spans_by_session[name], prefix_tokens)
        for name, topk in sessions.items()
    }
    scores: dict[str, dict] = {}
    for held_out, rows in examples.items():
        training = [row for name, source_rows in examples.items() if name != held_out for row in source_rows]
        if not rows or not training:
            raise ValueError(f"prefix-cost replay requires prefix and training spans for {held_out}")
        type_pools, fallback_pool = _type_pools(sessions, spans_by_session, held_out, cap)
        topk = sessions[held_out]
        layer_count = topk.shape[1]
        reactive = [OrderedDict() for _ in range(layer_count)]
        hybrid = [OrderedDict() for _ in range(layer_count)]
        static_tail = [OrderedDict() for _ in range(layer_count)]
        spec = [OrderedDict() for _ in range(layer_count)]
        totals = {name: {"hits": 0, "loads": 0, "preloads": 0} for name in PREFIX_POLICIES}
        token_count = 0
        for row in rows:
            span = spans_by_session[held_out][row["span_index"]]
            content_type = str(span["type"])
            type_pool = type_pools.get(content_type, fallback_pool)
            fingerprint_pool = top_pool(weighted_profile(row["feature"], training, neighbors), cap)
            start = int(span["start"])
            prefix_end = start + prefix_tokens
            end = min(topk.shape[0], int(span["end"]))
            for layer in range(layer_count):
                spec[layer] = OrderedDict((int(expert), True) for expert in sorted(type_pool[layer]))
                totals["spec_type"]["preloads"] += cap
            for token in range(start, end):
                in_prefix = token < prefix_end
                token_count += 1
                for layer in range(layer_count):
                    experts = topk[token, layer, :]
                    hits, loads = _serve_lru(reactive[layer], experts, cap)
                    totals["reactive"]["hits"] += hits
                    totals["reactive"]["loads"] += loads
                    hits, loads = _serve_lru(spec[layer], experts, cap)
                    totals["spec_type"]["hits"] += hits
                    totals["spec_type"]["loads"] += loads
                    hits, loads = _serve_static(experts, type_pool, layer)
                    totals["static_type"]["hits"] += hits
                    totals["static_type"]["loads"] += loads
                    if in_prefix:
                        hits, loads = _serve_lru(hybrid[layer], experts, cap)
                        totals["fingerprint_hybrid"]["hits"] += hits
                        totals["fingerprint_hybrid"]["loads"] += loads
                        hits, loads = _serve_lru(static_tail[layer], experts, cap)
                        totals["fingerprint_static_tail"]["hits"] += hits
                        totals["fingerprint_static_tail"]["loads"] += loads
                    elif token == prefix_end:
                        hybrid[layer] = OrderedDict((int(expert), True) for expert in sorted(fingerprint_pool[layer]))
                        totals["fingerprint_hybrid"]["preloads"] += cap
                        hits, loads = _serve_lru(hybrid[layer], experts, cap)
                        totals["fingerprint_hybrid"]["hits"] += hits
                        totals["fingerprint_hybrid"]["loads"] += loads
                        hits, loads = _serve_static(experts, fingerprint_pool, layer)
                        totals["fingerprint_static_tail"]["hits"] += hits
                        totals["fingerprint_static_tail"]["loads"] += loads
                    else:
                        hits, loads = _serve_lru(hybrid[layer], experts, cap)
                        totals["fingerprint_hybrid"]["hits"] += hits
                        totals["fingerprint_hybrid"]["loads"] += loads
                        hits, loads = _serve_static(experts, fingerprint_pool, layer)
                        totals["fingerprint_static_tail"]["hits"] += hits
                        totals["fingerprint_static_tail"]["loads"] += loads
        slots = token_count * layer_count * topk.shape[2]
        scores[held_out] = {
            "eligible_spans": len(rows),
            "tokens": token_count,
            "policies": {
                name: {
                    "hit_rate": values["hits"] / slots,
                    "loads_per_token": (values["loads"] + values["preloads"]) / token_count,
                    "loads": values["loads"],
                    "preloads": values["preloads"],
                }
                for name, values in totals.items()
            },
        }
    return scores


def prefix_cost_scores(
    sessions: dict[str, np.ndarray],
    spans_by_session: dict[str, list[dict]],
    caps: tuple[int, ...],
    prefix_tokens: int,
    neighbors: int,
    bootstrap_samples: int,
) -> dict:
    """Summarize prefix-inclusive policy replay / 彙整前綴含入策略重播。"""
    result: dict[str, dict] = {"caps": {}}
    for cap in caps:
        per_session = prefix_cost_session_scores(sessions, spans_by_session, cap, prefix_tokens, neighbors)
        policies = {}
        for policy in PREFIX_POLICIES:
            policies[policy] = {
                metric: float(np.mean([row["policies"][policy][metric] for row in per_session.values()]))
                for metric in ("hit_rate", "loads_per_token")
            }
        comparisons = {}
        for baseline in ("spec_type", "reactive"):
            comparisons[f"fingerprint_hybrid_minus_{baseline}"] = {
                metric: paired_bootstrap(
                    [
                        row["policies"]["fingerprint_hybrid"][metric] - row["policies"][baseline][metric]
                        for row in per_session.values()
                    ],
                    bootstrap_samples,
                )
                for metric in ("hit_rate", "loads_per_token")
            }
        result["caps"][str(cap)] = {
            "policies": policies,
            "per_session": per_session,
            "paired_bootstrap": comparisons,
        }
    return result


def transition_counts(topk: np.ndarray) -> np.ndarray:
    """Count adjacent-layer expert co-selections / 計數相鄰層專家共同選擇。"""
    _, layers, width = topk.shape
    transitions = np.zeros((layers - 1, N_EXPERTS, N_EXPERTS), dtype=np.float64)
    for layer in range(layers - 1):
        source = topk[:, layer, :].astype(np.int64, copy=False)
        target = topk[:, layer + 1, :].astype(np.int64, copy=False)
        for source_slot in range(width):
            for target_slot in range(width):
                np.add.at(transitions[layer], (source[:, source_slot], target[:, target_slot]), 1)
    return transitions


def c2_session_scores(sessions: dict[str, np.ndarray]) -> dict[str, dict]:
    """Evaluate C2 against target-layer popularity / 對照目標層熱門 top-8 評測 C2。"""
    transitions_by_session = {name: transition_counts(topk) for name, topk in sessions.items()}
    counts_by_session = {name: onehot_counts(topk) for name, topk in sessions.items()}
    scores: dict[str, dict] = {}
    for held_out, topk in sessions.items():
        learned = sum(value for name, value in transitions_by_session.items() if name != held_out)
        popular = sum(value for name, value in counts_by_session.items() if name != held_out)
        learned_recall = popular_recall = comparisons = 0.0
        tokens, layers, width = topk.shape
        for layer in range(layers - 1):
            popular_top8 = top_pool(popular[layer + 1:layer + 2], width)[0]
            for token in range(tokens):
                source = topk[token, layer, :].astype(np.int64, copy=False)
                predicted = np.argsort(-learned[layer, source].sum(axis=0), kind="stable")[:width]
                actual = topk[token, layer + 1, :]
                learned_recall += np.isin(actual, predicted).sum() / width
                popular_recall += np.isin(actual, popular_top8).sum() / width
                comparisons += 1
        scores[held_out] = {
            "learned_recall_at_8": learned_recall / comparisons,
            "popular_recall_at_8": popular_recall / comparisons,
            "delta": learned_recall / comparisons - popular_recall / comparisons,
            "token_layer_pairs": int(comparisons),
        }
    return scores


def paired_bootstrap(values: list[float], samples: int = DEFAULT_BOOTSTRAP_SAMPLES, seed: int = 0) -> dict:
    """Bootstrap the mean paired difference / 對配對差值平均數進行 bootstrap。"""
    if not values:
        raise ValueError("paired values are required")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "samples": samples,
        "seed": seed,
    }


def evaluate(
    sessions: dict[str, np.ndarray],
    spans_by_session: dict[str, list[dict]],
    cap: int = DEFAULT_CAP,
    prefix_tokens: int = DEFAULT_PREFIX_TOKENS,
    neighbors: int = DEFAULT_NEIGHBORS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    prefix_cost_caps: tuple[int, ...] = (32, 48, 57),
) -> dict:
    """Run C1 and C2 evaluation / 執行 C1 與 C2 評測。"""
    shapes = {topk.shape[1:] for topk in sessions.values()}
    if len(shapes) != 1 or next(iter(shapes))[1] != 8:
        raise ValueError("all traces require the same layer count and top-8 width")
    c1_scores = c1_session_scores(sessions, spans_by_session, cap, prefix_tokens, neighbors)
    c2_scores = c2_session_scores(sessions)
    return {
        "schema": "routing_fingerprint_eval/v1",
        "session_count": len(sessions),
        "sessions": sorted(sessions),
        "c1": {
            "cap": cap,
            "prefix_tokens": prefix_tokens,
            "neighbors": neighbors,
            "per_session": c1_scores,
            "paired_bootstrap": paired_bootstrap([row["delta"] for row in c1_scores.values()], bootstrap_samples),
        },
        "c2": {
            "per_session": c2_scores,
            "paired_bootstrap": paired_bootstrap([row["delta"] for row in c2_scores.values()], bootstrap_samples),
        },
        "prefix_cost_inclusive": {
            "prefix_tokens": prefix_tokens,
            "policies": list(PREFIX_POLICIES),
            **prefix_cost_scores(
                sessions, spans_by_session, prefix_cost_caps, prefix_tokens, neighbors, bootstrap_samples
            ),
        },
    }


def main() -> int:
    """Run command-line evaluation / 執行命令列評測。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--tags", default="base_all,base_new")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--prefix-tokens", type=int, default=DEFAULT_PREFIX_TOKENS)
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--prefix-cost-caps", default="32,48,57")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tags = tuple(value for value in args.tags.split(",") if value)
    root = Path(args.root)
    sessions = load_sessions(root, tags)
    spans_by_session = {name: load_spans(root, name) for name in sessions}
    prefix_cost_caps = tuple(int(value) for value in args.prefix_cost_caps.split(",") if value)
    result = evaluate(
        sessions, spans_by_session, args.cap, args.prefix_tokens, args.neighbors,
        args.bootstrap_samples, prefix_cost_caps,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    c1 = result["c1"]["paired_bootstrap"]
    c2 = result["c2"]["paired_bootstrap"]
    print(f"sessions={result['session_count']} C1 delta={c1['mean']:.3f} CI95=[{c1['ci95_low']:.3f},{c1['ci95_high']:.3f}]")
    print(f"C2 delta={c2['mean']:.3f} CI95=[{c2['ci95_low']:.3f},{c2['ci95_high']:.3f}]")
    for cap, row in result["prefix_cost_inclusive"]["caps"].items():
        summary = "  ".join(
            f"{name} hit={values['hit_rate']:.3f} loads/token={values['loads_per_token']:.1f}"
            for name, values in row["policies"].items()
        )
        print(f"prefix C={cap}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
