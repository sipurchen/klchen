"""Tier 2 (per-token expert ids): statistics, struct layout file, and the eval-callback logic without libllama."""
import ctypes
import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from affinity_pipeline import tier2_stats as st  # noqa: E402
from affinity_pipeline.tier2_ctypes import Layout, Tier2Runner, blob_type, parse_topk, peek, poke  # noqa: E402

LAYOUT_FILE = ROOT / "config" / "tier2_layouts" / "d327bf1.json"


def make_topk(rows, n_layers=2):
    """rows: list of lists of expert ids (same for every layer) -> uint8 [n, n_layers, k]"""
    a = np.array(rows, dtype=np.uint8)
    return np.repeat(a[:, None, :], n_layers, axis=1)


def test_random_overlap_probability_matches_hypergeometric():
    assert st.random_overlap_probability(128, 8) == pytest.approx(1 - comb(120, 8) / comb(128, 8))
    assert 0.40 < st.random_overlap_probability() < 0.42
    assert st.random_overlap_probability(8, 8) == 1.0


def test_parse_topk_reads_strided_view():
    ne0, ne1, nb1 = 3, 4, 6 * 4          # rows of 6 int32 slots, only the first 3 are the top-k view
    full = np.arange(ne1 * 6, dtype=np.int32).reshape(ne1, 6)
    buf = full.tobytes()[: (ne1 - 1) * nb1 + ne0 * 4]
    out = parse_topk(buf, ne0, ne1, nb1)
    assert out.shape == (4, 3) and (out == full[:, :3]).all()


def test_parse_topk_rejects_unaligned_stride():
    with pytest.raises(ValueError):
        parse_topk(b"\0" * 64, 2, 2, 10)


def test_pair_intersections_and_locality_groups():
    # tokens 0-2 share expert 1; token 3 is disjoint; spans: a=[0,2) b=[2,4)
    topk = make_topk([[0, 1], [1, 2], [1, 3], [4, 5]])
    inter = st.pair_intersections(topk, 8)
    assert inter[:, 0].tolist() == [1, 1, 0]
    spans = [{"start": 0, "end": 2, "type": "a"}, {"start": 2, "end": 4, "type": "b"}]
    loc = st.locality(topk, spans, n_experts=8)
    assert loc["all"]["pairs"] == 3 and loc["all"]["p_overlap_mean"] == pytest.approx(2 / 3)
    assert loc["a"]["pairs"] == 1 and loc["a"]["p_overlap_mean"] == 1.0
    assert loc["boundary"]["pairs"] == 1 and loc["boundary"]["p_overlap_mean"] == 1.0   # token 1 -> token 2 share expert 1
    assert loc["b"]["pairs"] == 1 and loc["b"]["p_overlap_mean"] == 0.0
    assert len(loc["all"]["p_overlap_shuffled_per_layer"]) == 2


def test_shuffled_control_is_lower_when_order_matters():
    # runs of identical tokens: adjacent overlap is 1.0, a random pairing mostly mixes the two runs
    topk = make_topk([[0, 1]] * 50 + [[2, 3]] * 50)
    spans = [{"start": 0, "end": 100, "type": "t"}]
    loc = st.locality(topk, spans, n_experts=8)["t"]
    assert loc["p_overlap_mean"] > 0.98
    assert np.mean(loc["p_overlap_shuffled_per_layer"]) < 0.7


def test_type_counts_sum_to_tokens_times_k():
    topk = make_topk([[0, 1], [0, 2], [3, 4]])
    spans = [{"start": 0, "end": 2, "type": "x"}, {"start": 2, "end": 3, "type": "y"}]
    c = st.type_counts(topk, spans, 8)
    assert c["x"].sum() == 2 * 2 * 2 and c["y"].sum() == 1 * 2 * 2     # tokens * k * layers
    assert c["x"][0, 0] == 2 and c["x"][0, 1] == 1


def test_affinity_set_matches_aggregate_script():
    from aggregate_expert_affinity import affinity_set as agg_set
    rng = np.random.default_rng(1)
    for alpha in (0.8, 0.9):
        p = rng.dirichlet(np.ones(128) * 0.3)
        assert st.affinity_set(p, alpha) == agg_set(p, alpha)


def test_set_sizes_and_compare_identical_runs():
    rng = np.random.default_rng(2)
    base = {"code": rng.random((3, 16)) * 10 + 1, "reasoning": rng.random((3, 16)) * 10 + 1}
    cmp_ = st.compare_runs(base, {k: v.copy() for k, v in base.items()}, 0.9)
    for v in cmp_.values():
        assert v["relative_change"] == 0 and v["median_layer_jaccard"] == 1.0
    # concentrating one run on 2 experts shrinks the set
    conc = {k: np.where(np.arange(16) < 2, 1000.0, 0.0)[None, :].repeat(3, axis=0) for k in base}
    assert st.compare_runs(base, conc, 0.9)["code"]["relative_change"] < -0.5


def test_window_hit_constant_stream_and_union_size():
    topk = make_topk([[0, 1]] * 40)
    hit, size = st.window_hit(topk, 8, n_experts=8)
    assert hit.min() == 1.0 and size.max() == 2.0
    rnd = np.random.default_rng(3).integers(0, 8, size=(400, 2, 2)).astype(np.uint8)   # may contain duplicate ids, fine
    hit2, size2 = st.window_hit(rnd, 4, n_experts=8)
    assert 0 < hit2.mean() <= 1 and size2.mean() <= 8


def test_type_conditioning_reduction():
    broad = np.ones((2, 32))
    narrow = np.zeros((2, 32))
    narrow[:, :4] = 1
    counts = {"reasoning": broad, "code": narrow, "factual": broad, "creative": broad}
    tc = st.type_conditioning(counts, 0.9)
    assert tc["per_type_reduction"]["code"] > 0.5 and tc["per_type_reduction"]["reasoning"] <= 0.05


# ---------------------------------------------------------------- layout file + callback without libllama
def test_layout_file_has_every_field_the_harness_uses():
    L = Layout.load(LAYOUT_FILE)
    for s, f in (("llama_model_params", "n_gpu_layers"), ("llama_model_params", "tensor_buft_overrides"),
                 ("llama_context_params", "cb_eval"), ("llama_context_params", "cb_eval_user_data"),
                 ("llama_context_params", "n_ubatch"), ("llama_batch", "logits"), ("llama_batch", "seq_id"),
                 ("llama_model_tensor_buft_override", "buft"), ("ggml_tensor", "name"), ("ggml_tensor", "nb")):
        assert 0 <= L.off(s, f) < L.size(s), (s, f)
    assert L.const("GGML_TYPE_I32") == 26 and L.const("GGML_MAX_NAME") == 64


def test_poke_peek_roundtrip_in_blob():
    B = blob_type(32)
    b = B()
    poke(b, 8, ctypes.c_int32, 1234)
    poke(b, 16, ctypes.c_void_p, 0xDEADBEEF)
    assert peek(b, 8, ctypes.c_int32) == 1234 and peek(b, 16, ctypes.c_void_p) == 0xDEADBEEF


def _fake_runner():
    r = object.__new__(Tier2Runner)
    r.layout = Layout.load(LAYOUT_FILE)
    r._rows, r._err, r._cb_calls = {}, None, 0
    return r


def _tensor(runner, name: bytes, rows, nb1_slots=6, ttype=26):
    L = runner.layout
    blob = ctypes.create_string_buffer(L.size("ggml_tensor"))
    base = ctypes.addressof(blob)
    ctypes.memmove(base + L.off("ggml_tensor", "name"), name + b"\0", len(name) + 1)
    ctypes.c_int32.from_address(base + L.off("ggml_tensor", "type")).value = ttype
    ne0, ne1 = len(rows[0]), len(rows)
    ctypes.c_int64.from_address(base + L.off("ggml_tensor", "ne")).value = ne0
    ctypes.c_int64.from_address(base + L.off("ggml_tensor", "ne") + 8).value = ne1
    ctypes.c_size_t.from_address(base + L.off("ggml_tensor", "nb") + 8).value = nb1_slots * 4
    full = np.zeros((ne1, nb1_slots), dtype=np.int32)
    full[:, :ne0] = rows
    data = full.tobytes()[: (ne1 - 1) * nb1_slots * 4 + ne0 * 4]
    return blob, base, data


def test_eval_callback_filters_and_collects(monkeypatch):
    r = _fake_runner()
    blob, addr, data = _tensor(r, b"ffn_moe_topk-7", [[1, 2], [3, 4], [5, 6]])
    other, oaddr, _ = _tensor(r, b"ffn_moe_gate-7", [[0, 0]])
    assert r._on_tensor(addr, True, None) is True            # we want this tensor
    assert r._on_tensor(oaddr, True, None) is False          # and not this one
    assert r._on_tensor(oaddr, False, None) is True          # data call for a foreign tensor: just continue

    def fake_get(t, buf, offset, size):
        ctypes.memmove(buf, data, size)
    monkeypatch.setattr(r, "_sym", lambda name: fake_get)
    assert r._on_tensor(addr, False, None) is True
    assert r._err is None and 7 in r._rows
    assert r._rows[7][0].tolist() == [[1, 2], [3, 4], [5, 6]]


def test_eval_callback_aborts_on_wrong_dtype(monkeypatch):
    r = _fake_runner()
    blob, addr, data = _tensor(r, b"ffn_moe_topk-0", [[1, 2]], ttype=0)    # F32, not I32
    monkeypatch.setattr(r, "_sym", lambda name: (lambda *a: None))
    assert r._on_tensor(addr, False, None) is False
    assert "expected I32" in r._err


# ---------------------------------------------------------------- H2 router-LoRA helpers (pure parts, no torch)
def test_h2_targets_sizes_follow_ratio(tmp_path, monkeypatch):
    import math
    from affinity_pipeline.tier2_stats import affinity_set
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(128) * 0.2)
    s90 = len(affinity_set(p, 0.9))
    m = max(1, math.ceil(0.6 * s90))
    assert 1 <= m < s90                       # the target set is strictly tighter than the alpha=0.9 set it is derived from
    top = set(np.argsort(-p)[:m].tolist())
    assert top <= affinity_set(p, 0.9)        # and made of the most-used experts, so it is a subset of that set


# ---------------------------------------------------------------- pool replay (thesis 6.3)
def test_lru_step_counts_misses_and_respects_capacity():
    from collections import OrderedDict
    from pool_replay import lru_step
    cache = OrderedDict()
    assert lru_step(cache, [1, 2, 3], 3) == 3          # cold: three loads
    assert lru_step(cache, [1, 2, 3], 3) == 0          # all cached
    assert lru_step(cache, [4], 3) == 1 and len(cache) == 3 and 1 not in cache   # 1 was the least recently used


def test_replay_session_constant_stream_all_policies_hit():
    from pool_replay import replay_session
    topk = make_topk([[0, 1]] * 20, n_layers=2)
    spans = [{"start": 0, "end": 20, "type": "code"}]
    other = {"code": np.where(np.arange(8) < 2, 10.0, 0.0)[None, :].repeat(2, axis=0)}   # other sessions also used experts 0 and 1
    glob = other["code"]
    res, total = replay_session(topk, spans, other, glob, cap=2)
    assert total == 20 * 2 * 2
    assert res["static"][0] == total                       # every expert is in the type pool
    assert res["reactive"][1] == 2 * 2                      # only the cold start loads (2 experts x 2 layers)
    assert res["spec"][0] == total                         # the pre-loaded pool already holds both experts
    assert res["spec"][1] == 2 * 2                          # and the boundary pre-load is the only load
