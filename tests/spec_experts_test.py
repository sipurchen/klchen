"""
Spec-Experts Integration Test
Tests: env check → model load → chunk detection → infer → KV serialize → cleanup
Run AFTER run_spec_experts.ps1 has started the controller.
"""

import asyncio
import json
import sys
import time
import httpx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitor.role_tag_parser import RoleTagParser
from monitor.perplexity_spike import PerplexitySpikeMonitor
from monitor.signal_fusion import SignalFusion
from segmentor.chunk_classifier import classify_chunk
from kv.kv_serializer import save_kv_chunk, load_kv_chunk

import numpy as np

CONTROLLER = "http://127.0.0.1:8090"
RESULTS_FILE = Path(__file__).parent / "results_text.json"

results = []


def log(tag: str, passed: bool, detail: str = "", duration_ms: int = 0):
    mark = "PASS" if passed else "FAIL"
    line = f"[{mark}] {tag}"
    if duration_ms:
        line += f" ({duration_ms}ms)"
    if detail:
        line += f" -- {detail}"
    print(line)
    results.append({"test": tag, "passed": passed, "detail": detail, "ms": duration_ms})


# ── Test 1: VRAM check ────────────────────────────────────────────────────────
def test_vram():
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"], text=True
        ).strip().split(",")
        free, total = int(out[0]), int(out[1])
        ok = free >= 700  # warn below 700MB (standard is 850)
        log("VRAM check", ok, f"free={free}MB total={total}MB (target>=850MB)")
    except Exception as e:
        log("VRAM check", False, str(e))


# ── Test 2: Controller health ─────────────────────────────────────────────────
async def test_controller_health():
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{CONTROLLER}/health")
        ok = r.status_code == 200
        ms = int((time.time() - t0) * 1000)
        log("Controller health", ok, r.text[:80], ms)
        return ok
    except Exception as e:
        log("Controller health", False, str(e))
        return False


# ── Test 3: Chunk boundary detection ─────────────────────────────────────────
def test_chunk_detection():
    cot_sample = (
        "<think>\nLet me analyze this step by step.\n"
        "First, I check the imports.\n</think>\n"
        "```python\ndef solve(n):\n    return n * 2\n```\n"
        "The answer is 42."
    )
    parser = RoleTagParser()
    bounds = parser.feed_text(cot_sample)
    ok = len(bounds) >= 3  # expect <think>, </think>, ```
    log("Chunk boundary detection", ok,
        f"found {len(bounds)} boundaries: {[b.tag for b in bounds]}")

    # classify chunks
    for i, b in enumerate(bounds):
        c = classify_chunk(chunk_id=i, role_tag=b.tag, entropy_values=[1.5, 1.2, 1.8])
        print(f"       chunk {i}: tag={b.tag!r} type={c.chunk_type} route={c.route}")


# ── Test 4: KV serializer ─────────────────────────────────────────────────────
def test_kv_serializer():
    kv = np.random.rand(2, 32, 64).astype(np.float16)
    path = save_kv_chunk("test_session", chunk_id=0, role="reasoning", kv_tensor=kv)
    loaded = load_kv_chunk("test_session", chunk_id=0, role="reasoning")
    ok = loaded is not None and np.allclose(kv, loaded, atol=1e-3)
    log("KV serializer round-trip", ok,
        f"saved {path} shape={kv.shape} dtype={kv.dtype}")
    # cleanup
    path.unlink(missing_ok=True)
    path.parent.rmdir() if path.parent.exists() and not any(path.parent.iterdir()) else None


# ── Test 5: Infer — coding chunk ──────────────────────────────────────────────
async def test_infer_coding():
    payload = {
        "prompt": "Write a Python function that returns the nth Fibonacci number using memoization.",
        "chunk_type": "code",
        "session_id": "test_coding",
    }
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{CONTROLLER}/infer", json=payload)
        ms = int((time.time() - t0) * 1000)
        if r.status_code == 200:
            data = r.json()
            resp = data.get("response", "")
            ok = "def" in resp and len(resp) > 50
            log("Infer coding chunk", ok,
                f"model={data.get('model')} len={len(resp)} tok/s~={512000//max(ms,1)}", ms)
            if ok:
                print(f"       preview: {resp[:120].strip()}...")
        else:
            log("Infer coding chunk", False, f"HTTP {r.status_code}: {r.text[:100]}")
    except Exception as e:
        log("Infer coding chunk", False, str(e))


# ── Test 6: Infer — reasoning chunk ──────────────────────────────────────────
async def test_infer_reasoning():
    payload = {
        "prompt": "If a train travels 60km/h for 2.5 hours, how far does it go? Show your reasoning.",
        "chunk_type": "reasoning",
        "session_id": "test_reasoning",
    }
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{CONTROLLER}/infer", json=payload)
        ms = int((time.time() - t0) * 1000)
        if r.status_code == 200:
            data = r.json()
            resp = data.get("response", "")
            ok = "150" in resp  # 60 * 2.5 = 150
            log("Infer reasoning chunk", ok,
                f"model={data.get('model')} correct={'yes' if ok else 'no'}", ms)
        else:
            log("Infer reasoning chunk", False, f"HTTP {r.status_code}")
    except Exception as e:
        log("Infer reasoning chunk", False, str(e))


# ── Test 7: Analyze chunks endpoint ───────────────────────────────────────────
async def test_analyze_chunks():
    payload = {
        "text": "<think>Reasoning here.</think>\n```python\nprint('hello')\n```",
        "session_id": "test_chunks",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{CONTROLLER}/analyze_chunks", json=payload)
        ok = r.status_code == 200 and r.json().get("total", 0) >= 2
        log("Analyze chunks endpoint", ok, f"chunks={r.json().get('total')}")
    except Exception as e:
        log("Analyze chunks endpoint", False, str(e))


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print("\n===== Spec-Experts Integration Test =====")
    print(f"Controller: {CONTROLLER}")
    print()

    test_vram()
    test_chunk_detection()
    test_kv_serializer()

    ctrl_ok = await test_controller_health()
    if ctrl_ok:
        await test_analyze_chunks()
        await test_infer_coding()
        await test_infer_reasoning()
    else:
        log("Infer tests", False, "skipped — controller not available")
        log("Analyze chunks", False, "skipped — controller not available")

    # ── Summary ──
    passed = sum(1 for r in results if r["passed"])
    total  = len(results)
    print(f"\n{'='*42}")
    print(f"Results: {passed}/{total} passed")

    # save results
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {RESULTS_FILE}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
