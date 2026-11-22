"""Unit tests for E05 routing fingerprint analysis / E05 路由指紋分析單元測試。"""
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from routing_fingerprint_eval import (  # noqa: E402
    c1_session_scores,
    c2_session_scores,
    paired_bootstrap,
    prefix_cost_scores,
    prefix_cost_session_scores,
    span_examples,
)


def _trace(values: list[list[list[int]]]) -> np.ndarray:
    """Create a uint8 Tier 2 trace / 建立 uint8 Tier 2 軌跡。"""
    return np.asarray(values, dtype=np.uint8)


def test_c1_loso_keeps_held_out_later_profile_out_of_knn_training():
    """C1 training excludes the held-out session / C1 訓練排除留出 session。"""
    sessions = {
        "train": _trace([[[0]], [[0]], [[0]], [[0]]]),
        "held": _trace([[[1]], [[1]], [[1]], [[1]]]),
    }
    spans = {name: [{"start": 0, "end": 4}] for name in sessions}
    result = c1_session_scores(sessions, spans, cap=1, prefix_tokens=2, neighbors=1)
    assert result["held"]["knn_hit_rate"] == 0.0
    assert result["held"]["static_all_hit_rate"] == 0.0


def test_span_examples_use_first_prefix_tokens_and_later_target():
    """C1 feature and target retain the causal split / C1 特徵與目標保留因果切分。"""
    trace = _trace([[[0]], [[0]], [[1]], [[1]]])
    rows = span_examples(trace, [{"start": 0, "end": 4}], prefix_tokens=2)
    assert len(rows) == 1
    assert rows[0]["target_counts"][0, 1] == 2
    assert rows[0]["target_counts"][0, 0] == 0


def test_c2_transition_and_popularity_scores_are_session_level():
    """C2 returns one recall pair per held-out session / C2 每個留出 session 回傳一組 recall。"""
    sessions = {
        "one": _trace([[[0], [1]], [[0], [1]]]),
        "two": _trace([[[0], [1]], [[0], [1]]]),
    }
    result = c2_session_scores(sessions)
    assert set(result) == {"one", "two"}
    assert result["one"]["learned_recall_at_8"] == 1.0
    assert result["one"]["popular_recall_at_8"] == 1.0


def test_paired_bootstrap_is_seeded_and_returns_mean_interval():
    """Bootstrap is deterministic for its documented seed / bootstrap 對文件所列 seed 可重現。"""
    first = paired_bootstrap([0.0, 0.2], samples=100, seed=7)
    second = paired_bootstrap([0.0, 0.2], samples=100, seed=7)
    assert first == second
    assert first["mean"] == 0.1
    assert first["ci95_low"] <= first["mean"] <= first["ci95_high"]


def test_prefix_cost_replay_counts_all_qualifying_span_tokens_and_hybrid_preload():
    """Hybrid charges one per-layer preload and scores prefix plus tail / Hybrid 計入逐層預載與前綴後段。"""
    sessions = {
        "train": _trace([[[0]], [[0]], [[0]], [[0]]]),
        "held": _trace([[[0]], [[0]], [[1]], [[1]]]),
    }
    spans = {name: [{"start": 0, "end": 4, "type": "reasoning"}] for name in sessions}
    result = prefix_cost_session_scores(sessions, spans, cap=1, prefix_tokens=2, neighbors=1)
    held = result["held"]
    assert held["tokens"] == 4
    assert held["policies"]["fingerprint_hybrid"]["preloads"] == 1
    assert held["policies"]["fingerprint_hybrid"]["hit_rate"] == 0.5
    assert held["policies"]["fingerprint_hybrid"]["loads_per_token"] == 0.75
    assert held["policies"]["fingerprint_static_tail"]["hit_rate"] == 0.25


def test_prefix_cost_summary_bootstraps_hybrid_against_spec_and_reactive():
    """Summary exposes both required session-paired baselines / 彙整輸出兩項逐 session 配對基準。"""
    sessions = {
        "one": _trace([[[0]], [[0]], [[0]], [[0]]]),
        "two": _trace([[[1]], [[1]], [[1]], [[1]]]),
    }
    spans = {name: [{"start": 0, "end": 4, "type": "reasoning"}] for name in sessions}
    result = prefix_cost_scores(sessions, spans, (1,), prefix_tokens=2, neighbors=1, bootstrap_samples=20)
    comparisons = result["caps"]["1"]["paired_bootstrap"]
    assert set(comparisons) == {
        "fingerprint_hybrid_minus_spec_type",
        "fingerprint_hybrid_minus_reactive",
    }
    for row in comparisons.values():
        assert row["hit_rate"]["samples"] == 20
        assert row["loads_per_token"]["samples"] == 20


def test_prefix_cost_type_pool_excludes_held_out_session():
    """Type baseline counts only other sessions / 類型基準只計其他 session。"""
    sessions = {
        "train": _trace([[[0]]] * 17),
        "held": _trace([[[0]], [[0]]] + [[[1]]] * 15),
    }
    spans = {name: [{"start": 0, "end": 17, "type": "reasoning"}] for name in sessions}
    result = prefix_cost_session_scores(sessions, spans, cap=1, prefix_tokens=2, neighbors=1)
    assert result["held"]["policies"]["static_type"]["hit_rate"] == 2 / 17
