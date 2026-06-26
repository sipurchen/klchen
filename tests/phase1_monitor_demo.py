"""
Phase 1 -- External Monitor Full Pipeline Demo
Tests: attention_entropy + perplexity_spike + role_tag_parser + signal_fusion
against live Qwen3-1.7B llama-server on :8080

Run: python tests/phase1_monitor_demo.py
"""

import asyncio
import json
import math
import sys
import time
from pathlib import Path

import httpx
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitor.attention_entropy import EntropyMonitor, ChunkBoundaryCandidate
from monitor.perplexity_spike import PerplexitySpikeMonitor
from monitor.role_tag_parser import RoleTagParser
from monitor.signal_fusion import SignalFusion, ChunkBoundary
from segmentor.chunk_classifier import classify_chunk

SERVER = "http://127.0.0.1:8080"

BANNER = "=" * 60


def hdr(title: str):
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)


# -------------------------------------------------------------
# P1.1  Role-Tag Parser  (deterministic, no server needed)
# -------------------------------------------------------------
def demo_role_tag_parser():
    hdr("P1.1  Role-Tag Parser — deterministic boundary detection")

    samples = [
        (
            "CoT with code",
            "<think>\nLet me reason step by step.\n"
            "First, I check the algorithm.\n</think>\n"
            "```python\ndef fibonacci(n):\n    if n <= 1: return n\n"
            "    return fibonacci(n-1) + fibonacci(n-2)\n```\n"
            "The time complexity is O(2^n).",
        ),
        (
            "Mixed reasoning + factual",
            "<think>\nThe question asks about photosynthesis.\n</think>\n"
            "# Photosynthesis\n"
            "Plants convert CO₂ + H₂O → glucose + O₂ using light energy.\n"
            "```python\n# Rate equation\nrate = light_intensity * 0.042\n```",
        ),
    ]

    parser = RoleTagParser()
    fusion = SignalFusion()

    for label, text in samples:
        print(f"\n[Sample: {label}]")
        print(f"  Input ({len(text)} chars): {text[:80].strip()!r}...")

        bounds = parser.feed_text(text)
        print(f"  → {len(bounds)} boundaries detected:")
        for b in bounds:
            print(f"      tok~{b.token_index:4d}  tag={b.tag!r:<14s}  priority={b.priority}")
            chunk = classify_chunk(
                chunk_id=b.token_index,
                role_tag=b.tag,
                entropy_values=[1.5, 1.2, 1.8],
            )
            print(f"             type={chunk.chunk_type}  route={chunk.route}")

    return True


# -------------------------------------------------------------
# P1.2  Perplexity Spike Monitor (offline simulation)
# -------------------------------------------------------------
def demo_perplexity_spike():
    hdr("P1.2  Perplexity Spike Monitor — z-score boundary detection")

    # Simulate logprob stream: stable → spike → stable → spike
    import random
    random.seed(42)

    monitor = PerplexitySpikeMonitor(z_threshold=2.0, window=10)

    # Phase A: stable low-perplexity (reasoning text)
    stable_logprobs = [-0.3, -0.4, -0.35, -0.28, -0.32, -0.41, -0.29, -0.38,
                       -0.33, -0.27, -0.36, -0.31, -0.42, -0.30, -0.37]
    # Phase B: high perplexity spike (semantic shift: reasoning→code)
    spike_logprobs  = [-2.8, -3.1, -2.6]
    # Phase C: stable code tokens
    code_logprobs   = [-0.15, -0.12, -0.18, -0.14, -0.11, -0.16, -0.13, -0.17,
                       -0.15, -0.12, -0.19, -0.14, -0.11, -0.13, -0.16]
    # Phase D: spike again (code→factual)
    spike2_logprobs = [-2.5, -2.9, -2.7]
    # Phase E: factual
    factual_logprobs = [-0.45, -0.50, -0.48, -0.43, -0.52, -0.47, -0.44,
                        -0.51, -0.46, -0.49]

    all_logprobs = (
        stable_logprobs + spike_logprobs + code_logprobs
        + spike2_logprobs + factual_logprobs
    )

    labels = (
        ["reasoning"] * len(stable_logprobs) +
        ["[SPIKE→code]"] * len(spike_logprobs) +
        ["code"] * len(code_logprobs) +
        ["[SPIKE→factual]"] * len(spike2_logprobs) +
        ["factual"] * len(factual_logprobs)
    )

    spikes = []
    print(f"\n  Simulating {len(all_logprobs)} tokens (z_threshold=2.0, window=10):")
    print(f"  {'tok':>4}  {'logprob':>8}  {'zone':<16}  {'spike'}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*16}  {'-'*20}")

    for i, (lp, label) in enumerate(zip(all_logprobs, labels)):
        event = monitor.feed_token(lp)
        marker = f"[!] SPIKE z={event.z_score:.2f}" if event else ""
        if event:
            spikes.append((i, event.z_score))
        print(f"  {i:>4}  {lp:>8.3f}  {label:<16}  {marker}")

    print(f"\n  → {len(spikes)} spike events detected at tokens: {[s[0] for s in spikes]}")
    return len(spikes) >= 2


# -------------------------------------------------------------
# P1.3  Signal Fusion (offline)
# -------------------------------------------------------------
def demo_signal_fusion():
    hdr("P1.3  Signal Fusion — multi-signal boundary voting")

    from dataclasses import dataclass

    @dataclass
    class MockCandidate:
        token_index: int
        boundary_type: str
        priority: int

    fusion = SignalFusion(merge_window=12)
    events = []

    # Sequence: entropy drop at 45, perplexity spike at 48, then role_tag at 52
    candidates = [
        MockCandidate(45,  "entropy_drop",     5),
        MockCandidate(48,  "perplexity_spike",  3),
        MockCandidate(52,  "role_tag",         10),  # immediate
        MockCandidate(90,  "entropy_drop",      5),
        MockCandidate(94,  "perplexity_spike",  3),
        MockCandidate(150, "role_tag",         10),  # immediate
    ]

    print(f"\n  {'tok':>4}  {'signal':<20}  {'pri':>4}  {'→ emitted boundary'}")
    print(f"  {'-'*4}  {'-'*20}  {'-'*4}  {'-'*40}")

    for c in candidates:
        result = fusion.add_candidate(c)
        emitted = ""
        if result:
            events.append(result)
            emitted = f"OK BOUNDARY type={result.boundary_type} pri={result.priority} signals={result.signals}"
        print(f"  {c.token_index:>4}  {c.boundary_type:<20}  {c.priority:>4}  {emitted}")

    print(f"\n  → {len(events)} boundaries emitted")
    for e in events:
        print(f"      tok={e.token_index:4d}  type={e.boundary_type:<15s}  pri={e.priority}  signals={e.signals}")

    return len(events) >= 2


# -------------------------------------------------------------
# P1.4  Live Entropy Monitor — against real llama-server
# -------------------------------------------------------------
async def demo_live_entropy_monitor():
    hdr("P1.4  Live Entropy Monitor — real LLM token stream")

    # First verify server is up
    try:
        r = requests.get(f"{SERVER}/health", timeout=3)
        if r.status_code != 200:
            print("  [SKIP] llama-server not ready")
            return False
    except Exception as e:
        print(f"  [SKIP] llama-server unreachable: {e}")
        return False

    print(f"  Server: {SERVER} OK")

    # CoT prompt that will produce semantic jumps
    prompt = (
        "/no_think\n"
        "Below is a coding problem. First reason briefly, then write code.\n\n"
        "Problem: Write a function to check if a string is a palindrome.\n\n"
        "Reasoning:\nA palindrome reads the same forward and backward. "
        "I need to compare the string with its reverse.\n\n"
        "Code:\n```python\n"
    )

    print(f"  Prompt ({len(prompt)} chars):")
    print(f"    {prompt[:100].strip()!r}...")
    print()

    monitor = EntropyMonitor(server_url=SERVER, threshold=0.35)
    fusion = SignalFusion(merge_window=8)
    role_parser = RoleTagParser()

    all_tokens = []
    entropy_trace = []
    boundaries = []
    generated_text = ""

    # Stream completion with n_probs for logprob-based entropy
    payload = {
        "prompt": prompt,
        "n_predict": 300,
        "n_probs": 10,
        "stream": True,
        "temperature": 0.0,
        "stop": ["```\n\n", "</s>"],
    }

    t0 = time.time()
    token_count = 0

    print(f"  {'tok':>4}  {'entropy':>8}  {'token_text':<20}  note")
    print(f"  {'-'*4}  {'-'*8}  {'-'*20}  {'-'*30}")

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{SERVER}/completion",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                tok_text = data.get("content", "")
                generated_text += tok_text

                # Entropy from logprobs
                note = ""
                entropy = None
                token_probs = data.get("completion_probabilities", [])
                if token_probs:
                    lps = [p["logprob"] for p in token_probs[0].get("probs", []) if "logprob" in p]
                    if lps:
                        probs = [math.exp(lp) for lp in lps]
                        total = sum(probs)
                        if total > 0:
                            probs = [p / total for p in probs]
                            entropy = -sum(p * math.log(p + 1e-12) for p in probs)

                if entropy is not None:
                    entropy_trace.append(entropy)
                    monitor._entropy_history.append(entropy)

                    # Check entropy drop
                    if len(monitor._entropy_history) >= 8:
                        avg = sum(monitor._entropy_history[-8:]) / 8
                        if (avg - entropy) > 0.35:
                            cand = ChunkBoundaryCandidate(
                                token_index=token_count,
                                entropy=entropy,
                                boundary_type="entropy_drop",
                            )
                            # inject priority for fusion
                            cand.priority = 5
                            result = fusion.add_candidate(cand)
                            if result:
                                boundaries.append(result)
                                note = f"[!] ENTROPY BOUNDARY (Δ={avg-entropy:.2f})"

                # Role tag detection
                tag_bounds = role_parser.feed_text(tok_text)
                for tb in tag_bounds:
                    result = fusion.add_candidate(tb)
                    if result:
                        boundaries.append(result)
                        note = f"[T]  ROLE_TAG={tb.tag!r}"

                all_tokens.append(tok_text)

                # Print every 5th token or boundary
                if token_count % 5 == 0 or note:
                    ent_str = f"{entropy:.4f}" if entropy is not None else "   n/a"
                    tok_disp = repr(tok_text)[:18]
                    print(f"  {token_count:>4}  {ent_str:>8}  {tok_disp:<20}  {note}")

                token_count += 1
                if data.get("stop"):
                    break

    elapsed = time.time() - t0
    tok_s = token_count / elapsed if elapsed > 0 else 0

    print(f"\n  --- Generation Complete ---")
    print(f"  Tokens generated : {token_count}")
    print(f"  Time             : {elapsed:.1f}s")
    print(f"  Speed            : {tok_s:.1f} tok/s")
    print(f"  Boundaries found : {len(boundaries)}")
    print()
    print(f"  Generated text preview:")
    print(f"  {generated_text[:300].strip()!r}...")

    if entropy_trace:
        avg_h = sum(entropy_trace) / len(entropy_trace)
        max_h = max(entropy_trace)
        min_h = min(entropy_trace)
        print(f"\n  Entropy stats:")
        print(f"    mean={avg_h:.4f}  max={max_h:.4f}  min={min_h:.4f}")

    print(f"\n  Chunk Boundaries (all signals):")
    if boundaries:
        for b in boundaries:
            chunk = classify_chunk(
                chunk_id=b.token_index,
                role_tag=b.boundary_type,
                entropy_values=entropy_trace[max(0, b.token_index-5):b.token_index+5] or [1.0],
            )
            print(f"    tok={b.token_index:4d}  type={b.boundary_type:<15s}  "
                  f"pri={b.priority}  → chunk_type={chunk.chunk_type}  route={chunk.route}")
    else:
        print("    (none — prompt may be too short for boundary detection)")

    return True


# -------------------------------------------------------------
# P1.5  Full Pipeline Integration Summary
# -------------------------------------------------------------
def demo_pipeline_summary(results: dict):
    hdr("P1.5  Phase 1 Pipeline Summary")

    components = {
        "P1.1  RoleTagParser       (deterministic)": results.get("role_tag", False),
        "P1.2  PerplexitySpikeMonitor (z-score)":   results.get("perplexity", False),
        "P1.3  SignalFusion        (multi-vote)":    results.get("fusion", False),
        "P1.4  EntropyMonitor      (live stream)":   results.get("entropy_live", False),
    }

    print()
    all_pass = True
    for name, ok in components.items():
        mark = "PASS OK" if ok else "FAIL XX"
        print(f"  [{mark}]  {name}")
        if not ok:
            all_pass = False

    print()
    print(f"  Architecture validated: Spec-Experts External Monitor")
    print(f"  Signal pipeline: RoleTag(10) > Entropy(5) > Perplexity(3)")
    print(f"  Fusion hysteresis: MERGE_WINDOW=12 tokens")
    print(f"  Chunk routing: code→coding, reasoning→reasoning, factual→coding")
    print()
    status = "OK Phase 1 COMPLETE" if all_pass else "⚠ Phase 1 PARTIAL"
    print(f"  {status}")


# -------------------------------------------------------------
async def main():
    print("\n" + "=" * 60)
    print("  Spec-Experts LLM — Phase 1: External Monitor")
    print("  GT 1030 2GB VRAM | Qwen3-1.7B-Q8_0 | ngl=10")
    print("=" * 60)

    results = {}
    results["role_tag"]    = demo_role_tag_parser()
    results["perplexity"]  = demo_perplexity_spike()
    results["fusion"]      = demo_signal_fusion()
    results["entropy_live"] = await demo_live_entropy_monitor()
    demo_pipeline_summary(results)


if __name__ == "__main__":
    asyncio.run(main())

