#!/usr/bin/env python3
"""Replay registered expert-pool policies against Tier 2 routing traces.

The default policies reproduce S8: ``reactive`` (token LRU), ``static``
(type-conditioned, leave-one-session-out pools), and ``spec`` (a static pool
preloaded at each chunk boundary).  Later slices can call :func:`register_policy`
without changing the LOSO accounting.

Example / 範例:
  python scripts/embed_pool_eval.py --tag base_all --caps 32,48,57,64 \
      --out logs/embed_pool_eval_base_all.json
"""
import argparse
import json
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from affinity_pipeline import tier2_stats as st  # noqa: E402


SESSIONS_DIR = Path("data") / "routing_corpus_v2" / "sessions"
EXPERT_MIB = 4.14
DDR3_GBPS = 17.0


class PoolPolicy:
    """Per-layer pool policy interface / 每層專家池策略介面。"""

    def begin_span(self, pool: set[int]) -> None:
        """Apply a chunk boundary / 套用區塊邊界。"""

    def serve(self, experts: Iterable[int], pool: set[int]) -> tuple[int, int]:
        """Return ``(hits, loads)`` for one token / 回傳單 token 的命中與載入。"""
        raise NotImplementedError


def lru_step(cache: OrderedDict[int, bool], experts: Iterable[int], cap: int) -> int:
    """Serve one token with an LRU pool / 以 LRU 池服務一個 token。"""
    misses = 0
    for expert in experts:
        if expert in cache:
            cache.move_to_end(expert)
            continue
        misses += 1
        cache[expert] = True
        if len(cache) > cap:
            cache.popitem(last=False)
    return misses


class ReactivePolicy(PoolPolicy):
    """Token-level LRU baseline / token 級 LRU 基準。"""

    def __init__(self, cap: int):
        self.cap = cap
        self.cache: OrderedDict[int, bool] = OrderedDict()

    def begin_span(self, pool: set[int]) -> None:
        """Reactive LRU has no boundary preload / Reactive LRU 不在邊界預載。"""

    def serve(self, experts: Iterable[int], pool: set[int]) -> tuple[int, int]:
        values = tuple(experts)
        loads = lru_step(self.cache, values, self.cap)
        return len(values) - loads, loads


class StaticPolicy(PoolPolicy):
    """Fixed type-conditioned pool / 固定內容類型條件池。"""

    def begin_span(self, pool: set[int]) -> None:
        """Static pools have no state / 靜態池沒有狀態。"""

    def serve(self, experts: Iterable[int], pool: set[int]) -> tuple[int, int]:
        values = tuple(experts)
        hits = sum(expert in pool for expert in values)
        return hits, len(values) - hits


class SpecPolicy(PoolPolicy):
    """Preload at each chunk then use LRU / 每個區塊預載後用 LRU。"""

    def __init__(self, cap: int):
        self.cap = cap
        self.cache: OrderedDict[int, bool] = OrderedDict()
        self.preloads = 0

    def begin_span(self, pool: set[int]) -> None:
        self.cache = OrderedDict((expert, True) for expert in sorted(pool))
        self.preloads += len(pool)

    def serve(self, experts: Iterable[int], pool: set[int]) -> tuple[int, int]:
        values = tuple(experts)
        loads = lru_step(self.cache, values, self.cap)
        return len(values) - loads, loads


PolicyFactory = Callable[[int], PoolPolicy]
POLICIES: dict[str, PolicyFactory] = {}


def register_policy(name: str, factory: PolicyFactory) -> None:
    """Register an E03-compatible policy factory / 註冊可供 E03 使用的策略工廠。"""
    if not name or "," in name:
        raise ValueError("policy name must be a non-empty comma-free string")
    POLICIES[name] = factory


register_policy("reactive", ReactivePolicy)
register_policy("static", lambda cap: StaticPolicy())
register_policy("spec", SpecPolicy)


def load_sessions(root: Path, tag: str) -> dict[str, tuple[np.ndarray, list[dict]]]:
    """Load local or fetched Tier 2 traces / 載入本機或已擷取的 Tier 2 軌跡。"""
    sessions: dict[str, tuple[np.ndarray, list[dict]]] = {}
    for trace in sorted((root / "logs" / "tier2" / tag).glob("*.npz")):
        if trace.stem == "counts":
            continue
        spans_path = root / SESSIONS_DIR / f"{trace.stem}.spans.json"
        spans = json.loads(spans_path.read_text(encoding="utf-8"))["spans"]
        sessions[trace.stem] = (np.load(trace)["topk"], spans)
    if not sessions:
        raise ValueError(f"no Tier 2 traces for tag {tag!r}")
    return sessions


def _pool_sets(counts: dict[str, np.ndarray], global_counts: np.ndarray, cap: int) -> tuple[dict[str, list[set[int]]], list[set[int]]]:
    """Build top-C pools per type and layer / 為每類型與層建立 top-C 池。"""
    typed = {
        kind: [set(np.argsort(-layer_counts)[:cap].tolist()) for layer_counts in kind_counts]
        for kind, kind_counts in counts.items()
    }
    global_pool = [set(np.argsort(-layer_counts)[:cap].tolist()) for layer_counts in global_counts]
    return typed, global_pool


def replay_session(
    topk: np.ndarray,
    spans: list[dict],
    pooled_other: dict[str, np.ndarray],
    global_other: np.ndarray,
    cap: int,
    policy_names: Iterable[str],
) -> tuple[dict[str, dict[str, int]], int]:
    """Replay one held-out session using pools counted only from other sessions.

    對一個留出 session 重播；傳入的 pooled_other 與 global_other 必須只含其他
    session 的計數，讓呼叫端可明確驗證 LOSO 邊界。
    """
    names = tuple(policy_names)
    unknown = [name for name in names if name not in POLICIES]
    if unknown:
        raise ValueError(f"unregistered policies: {unknown}")
    token_count, layer_count, top_k = topk.shape
    types = st.type_array(spans, token_count)
    span_ids = np.full(token_count, -1, dtype=np.int32)
    for span_id, span in enumerate(spans):
        span_ids[span["start"]:span["end"]] = span_id
    typed_pools, global_pools = _pool_sets(pooled_other, global_other, cap)
    totals = {name: {"hits": 0, "loads": 0, "preloads": 0} for name in names}
    for layer in range(layer_count):
        policies = {name: POLICIES[name](cap) for name in names}
        current_span = -2
        for token in range(token_count):
            content_type = types[token]
            pool = typed_pools.get(content_type, global_pools)[layer]
            if span_ids[token] != current_span:
                current_span = int(span_ids[token])
                for policy in policies.values():
                    policy.begin_span(pool)
            experts = topk[token, layer, :].tolist()
            for name, policy in policies.items():
                hits, loads = policy.serve(experts, pool)
                totals[name]["hits"] += hits
                totals[name]["loads"] += loads
        for name, policy in policies.items():
            totals[name]["preloads"] += getattr(policy, "preloads", 0)
    return totals, token_count * layer_count * top_k


def evaluate(sessions: dict[str, tuple[np.ndarray, list[dict]]], caps: Iterable[int], policy_names: Iterable[str]) -> dict:
    """Evaluate all capacities with session-level LOSO / 以 session 為單位執行 LOSO。"""
    names = tuple(policy_names)
    per_session = {name: st.type_counts(topk, spans) for name, (topk, spans) in sessions.items()}
    all_counts = {name: sum(counts.values()) for name, counts in per_session.items()}
    result: dict = {"caps": {}}
    for cap in caps:
        aggregate = {name: {"hits": 0, "loads": 0, "preloads": 0} for name in names}
        session_rows: dict[str, dict] = {}
        total_tokens = 0
        total_slots = 0
        for held_out, (topk, spans) in sessions.items():
            pooled_other: dict[str, np.ndarray] = {}
            for session_name, counts in per_session.items():
                if session_name == held_out:
                    continue
                for content_type, values in counts.items():
                    pooled_other[content_type] = pooled_other.get(content_type, 0) + values
            global_other = sum(values for session_name, values in all_counts.items() if session_name != held_out)
            raw, slots = replay_session(topk, spans, pooled_other, global_other, cap, names)
            token_count = int(topk.shape[0])
            total_tokens += token_count
            total_slots += slots
            session_rows[held_out] = {}
            for name, values in raw.items():
                aggregate[name]["hits"] += values["hits"]
                aggregate[name]["loads"] += values["loads"] + values["preloads"]
                session_rows[held_out][name] = {
                    "hit_rate": values["hits"] / slots,
                    "loads_per_token": (values["loads"] + values["preloads"]) / token_count,
                }
        policies = {}
        for name, values in aggregate.items():
            loads_per_token = values["loads"] / total_tokens
            policies[name] = {
                "hit_rate": values["hits"] / total_slots,
                "loads_per_token": loads_per_token,
                "load_ms_per_token_ddr3": loads_per_token * EXPERT_MIB / 1024 / DDR3_GBPS * 1000,
            }
        result["caps"][str(cap)] = {"policies": policies, "per_session": session_rows}
    return result


def main() -> int:
    """Run the command-line evaluator / 執行命令列評測器。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--tag", default="base_all")
    parser.add_argument("--caps", default="32,48,57,64")
    parser.add_argument("--policies", default="reactive,static,spec")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    caps = tuple(int(value) for value in args.caps.split(","))
    policy_names = tuple(value for value in args.policies.split(",") if value)
    sessions = load_sessions(Path(args.root), args.tag)
    result = evaluate(sessions, caps, policy_names)
    output = {
        "schema": "embed_pool_eval/v1",
        "tag": args.tag,
        "session_count": len(sessions),
        "sessions": sorted(sessions),
        "expert_mib": EXPERT_MIB,
        "ddr3_gbps": DDR3_GBPS,
        "policy_names": list(policy_names),
        **result,
    }
    for cap in caps:
        values = output["caps"][str(cap)]["policies"]
        summary = "  ".join(
            f"{name} hit={row['hit_rate']:.3f} loads/token={row['loads_per_token']:.1f}"
            for name, row in values.items()
        )
        print(f"C={cap:3d}: {summary}", flush=True)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
