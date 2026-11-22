"""Unit tests for E01's policy registry and LOSO replay / E01 策略註冊與 LOSO 重播單元測試。"""
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from embed_pool_eval import evaluate, register_policy  # noqa: E402


def _topk(rows: list[list[int]]) -> np.ndarray:
    """Create one-layer traces / 建立單層軌跡。"""
    return np.asarray(rows, dtype=np.uint8)[:, None, :]


def test_loso_static_pool_excludes_held_out_session():
    """Held-out-only expert must miss / 僅在留出 session 出現的專家必須未命中。"""
    sessions = {
        "train": (_topk([[0], [0], [0]]), [{"start": 0, "end": 3, "type": "code"}]),
        "held": (_topk([[1], [1], [1]]), [{"start": 0, "end": 3, "type": "code"}]),
    }
    result = evaluate(sessions, caps=(1,), policy_names=("static",))
    held = result["caps"]["1"]["per_session"]["held"]["static"]
    assert held["hit_rate"] == 0.0


def test_spec_policy_counts_chunk_preload_and_reactive_is_registered():
    """Spec counts its preload; default reactive remains usable / Spec 計預載且 reactive 可用。"""
    sessions = {
        "one": (_topk([[0], [0]]), [{"start": 0, "end": 2, "type": "code"}]),
        "two": (_topk([[0], [0]]), [{"start": 0, "end": 2, "type": "code"}]),
    }
    result = evaluate(sessions, caps=(1,), policy_names=("reactive", "spec"))
    policies = result["caps"]["1"]["policies"]
    assert policies["reactive"]["hit_rate"] == 0.5
    assert policies["spec"]["hit_rate"] == 1.0
    assert policies["spec"]["loads_per_token"] == 0.5


def test_registered_policy_factory_is_used():
    """External slices can register a policy / 後續切片可註冊策略。"""
    class AlwaysHit:
        def begin_span(self, pool):
            pass

        def serve(self, experts, pool):
            return len(tuple(experts)), 0

    register_policy("test_always_hit", lambda cap: AlwaysHit())
    sessions = {
        "one": (_topk([[3]]), [{"start": 0, "end": 1, "type": "code"}]),
        "two": (_topk([[4]]), [{"start": 0, "end": 1, "type": "code"}]),
    }
    result = evaluate(sessions, caps=(1,), policy_names=("test_always_hit",))
    assert result["caps"]["1"]["policies"]["test_always_hit"]["hit_rate"] == 1.0
