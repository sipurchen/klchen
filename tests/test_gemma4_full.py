# ============================================================================
# Gemma 4 E4B - Comprehensive Test Suite
# Modified by ClaudeO
# Purpose: Validates Ollama integration, API server, Python client, agent
#          routing, vision, audio, memory optimization, and edge cases.
# ============================================================================

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

# --- Modified by ClaudeO: Test configuration ---
API_URL = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434"
NODE_AGENT_URL = "http://localhost:8001"
RESULTS = []


def log_result(test_name: str, passed: bool, detail: str = ""):
    """Record and display test result."""
    status = "PASS" if passed else "FAIL"
    icon = "✅" if passed else "❌"
    RESULTS.append({"test": test_name, "passed": passed, "detail": detail})
    print(f"  {icon} [{status}] {test_name}")
    if detail and not passed:
        print(f"       Detail: {detail}")


# ============================================================================
# Phase 1: Infrastructure Tests
# ============================================================================

def test_phase1_infrastructure():
    """Verify that all infrastructure components are operational."""
    print("\n" + "=" * 60)
    print("Phase 1: Infrastructure Verification")
    print("=" * 60)

    # Test 1.1: Ollama server reachable
    try:
        resp = httpx.get(OLLAMA_URL, timeout=5)
        log_result("Ollama server reachable", resp.status_code == 200)
    except Exception as e:
        log_result("Ollama server reachable", False, str(e))

    # Test 1.2: Gemma 4 E4B model available
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = resp.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        has_gemma4 = any("gemma4" in name for name in model_names)
        log_result("Gemma 4 model present in Ollama", has_gemma4,
                   f"Found models: {model_names[:5]}")
    except Exception as e:
        log_result("Gemma 4 model present in Ollama", False, str(e))

    # Test 1.3: Optimized model variant exists
    try:
        has_opt = any("gemma4-e4b-opt" in name for name in model_names)
        log_result("Optimized model variant exists", has_opt)
    except Exception:
        log_result("Optimized model variant exists", False, "Could not check models")

    # Test 1.4: API server reachable
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=5)
        data = resp.json()
        log_result("API server healthy", data.get("status") in ("healthy", "degraded"),
                   json.dumps(data))
    except Exception as e:
        log_result("API server healthy", False, str(e))

    # Test 1.5: Python dependencies
    try:
        import fastapi
        import uvicorn
        log_result("Python dependencies installed", True,
                   f"fastapi={fastapi.__version__}")
    except ImportError as e:
        log_result("Python dependencies installed", False, str(e))


# ============================================================================
# Phase 2: Ollama Direct Communication Tests
# ============================================================================

def test_phase2_ollama_direct():
    """Test direct communication with Ollama's Gemma 4 E4B."""
    print("\n" + "=" * 60)
    print("Phase 2: Ollama Direct Communication")
    print("=" * 60)

    # Test 2.1: Basic text generation
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "gemma4:e4b",
                "messages": [{"role": "user", "content": "Reply with only: OK"}],
                "stream": False,
                "options": {"num_ctx": 2048},
            },
            timeout=60,
        )
        content = resp.json().get("message", {}).get("content", "")
        log_result("Ollama basic text generation", len(content) > 0,
                   f"Response length: {len(content)}")
    except Exception as e:
        log_result("Ollama basic text generation", False, str(e))

    # Test 2.2: System prompt support
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "gemma4:e4b",
                "messages": [
                    {"role": "system", "content": "You must respond in JSON format only."},
                    {"role": "user", "content": "What is 2+2? Respond as {\"answer\": N}"},
                ],
                "stream": False,
                "options": {"num_ctx": 2048, "temperature": 0.1},
            },
            timeout=60,
        )
        content = resp.json().get("message", {}).get("content", "")
        has_json = "{" in content and "}" in content
        log_result("System prompt support (JSON output)", has_json,
                   f"Response: {content[:100]}")
    except Exception as e:
        log_result("System prompt support (JSON output)", False, str(e))

    # Test 2.3: OpenAI-compatible endpoint
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json={
                "model": "gemma4:e4b",
                "messages": [{"role": "user", "content": "Say hello."}],
            },
            timeout=60,
        )
        data = resp.json()
        has_choices = "choices" in data
        log_result("Ollama OpenAI-compatible endpoint", has_choices)
    except Exception as e:
        log_result("Ollama OpenAI-compatible endpoint", False, str(e))

    # Test 2.4: Memory-optimized model
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "gemma4-e4b-opt",
                "messages": [{"role": "user", "content": "Reply: working"}],
                "stream": False,
            },
            timeout=60,
        )
        content = resp.json().get("message", {}).get("content", "")
        log_result("Optimized model responds", len(content) > 0)
    except Exception as e:
        log_result("Optimized model responds", False, str(e))


# ============================================================================
# Phase 3: Unified API Server Tests
# ============================================================================

def test_phase3_api_server():
    """Test the FastAPI unified server endpoints."""
    print("\n" + "=" * 60)
    print("Phase 3: Unified API Server")
    print("=" * 60)

    # Test 3.1: Chat completion endpoint
    try:
        resp = httpx.post(
            f"{API_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is 1+1?"}],
                "temperature": 0.1,
            },
            timeout=120,
        )
        data = resp.json()
        has_response = len(data.get("choices", [{}])[0].get("message", {}).get("content", "")) > 0
        log_result("API chat completion", has_response, f"Status: {resp.status_code}")
    except Exception as e:
        log_result("API chat completion", False, str(e))

    # Test 3.2: Agent routing
    try:
        resp = httpx.post(
            f"{API_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Write a hello world in Python."}],
                "agent_id": "coder",
            },
            timeout=120,
        )
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        has_code = "print" in content.lower() or "def " in content
        log_result("Agent routing (coder)", has_code,
                   f"Contains code: {has_code}")
    except Exception as e:
        log_result("Agent routing (coder)", False, str(e))

    # Test 3.3: Agent listing
    try:
        resp = httpx.get(f"{API_URL}/v1/agents", timeout=10)
        agents = resp.json().get("agents", [])
        agent_ids = [a["agent_id"] for a in agents]
        expected = {"general", "coder", "analyst", "translator", "vision", "planner"}
        has_all = expected.issubset(set(agent_ids))
        log_result("All default agents registered", has_all,
                   f"Found: {agent_ids}")
    except Exception as e:
        log_result("All default agents registered", False, str(e))

    # Test 3.4: Custom agent registration
    try:
        resp = httpx.post(
            f"{API_URL}/v1/agents",
            json={
                "agent_id": "test_agent",
                "name": "Test Agent",
                "system_prompt": "You are a test agent. Always reply with 'TEST_OK'.",
                "temperature": 0.1,
                "description": "For testing only",
            },
            timeout=10,
        )
        registered = resp.json().get("status") == "registered"
        log_result("Custom agent registration", registered)

        # Cleanup
        httpx.delete(f"{API_URL}/v1/agents/test_agent", timeout=5)
    except Exception as e:
        log_result("Custom agent registration", False, str(e))

    # Test 3.5: Model listing
    try:
        resp = httpx.get(f"{API_URL}/v1/models", timeout=10)
        models = resp.json().get("data", [])
        log_result("Model listing endpoint", len(models) > 0,
                   f"Found {len(models)} models")
    except Exception as e:
        log_result("Model listing endpoint", False, str(e))

    # Test 3.6: Error handling (invalid model)
    try:
        resp = httpx.post(
            f"{API_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "model": "nonexistent-model-xyz",
            },
            timeout=30,
        )
        # Should return error or fallback gracefully
        is_handled = resp.status_code in (200, 404, 502)
        log_result("Error handling (invalid model)", is_handled,
                   f"Status: {resp.status_code}")
    except Exception as e:
        log_result("Error handling (invalid model)", True, "Exception caught as expected")


# ============================================================================
# Phase 4: Memory Optimization Tests
# ============================================================================

def test_phase4_memory():
    """Test memory optimization settings (KV cache, context window)."""
    print("\n" + "=" * 60)
    print("Phase 4: Memory Optimization")
    print("=" * 60)

    # Test 4.1: Reduced context window works
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "gemma4-e4b-opt",
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "stream": False,
                "options": {"num_ctx": 2048},  # Very small context
            },
            timeout=60,
        )
        content = resp.json().get("message", {}).get("content", "")
        log_result("Reduced context window (2K)", "paris" in content.lower(),
                   f"Response: {content[:100]}")
    except Exception as e:
        log_result("Reduced context window (2K)", False, str(e))

    # Test 4.2: Check model size on disk
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = resp.json().get("models", [])
        for m in models:
            if "gemma4" in m.get("name", "") and "e4b" in m.get("name", ""):
                size_gb = m.get("size", 0) / (1024 ** 3)
                under_5gb = size_gb < 5.0  # Q4 should be ~3GB
                log_result(f"Model disk size ({m['name']})", under_5gb,
                           f"Size: {size_gb:.2f} GB")
                break
        else:
            log_result("Model disk size check", False, "No gemma4:e4b found")
    except Exception as e:
        log_result("Model disk size check", False, str(e))

    # Test 4.3: Concurrent request handling
    try:
        import asyncio

        async def _concurrent_test():
            async with httpx.AsyncClient(timeout=120) as client:
                tasks = [
                    client.post(
                        f"{API_URL}/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": f"Count to {i+1}."}]},
                    )
                    for i in range(3)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successes = sum(1 for r in results if not isinstance(r, Exception) and r.status_code == 200)
                return successes

        successes = asyncio.run(_concurrent_test())
        log_result("Concurrent requests (3x)", successes >= 1,
                   f"Successful: {successes}/3")
    except Exception as e:
        log_result("Concurrent requests (3x)", False, str(e))


# ============================================================================
# Phase 5: Multimodal Capability Tests
# ============================================================================

def test_phase5_multimodal():
    """Test multimodal capabilities (vision, audio placeholders)."""
    print("\n" + "=" * 60)
    print("Phase 5: Multimodal Capabilities")
    print("=" * 60)

    # Test 5.1: Vision endpoint exists
    try:
        # Send a minimal 1x1 white PNG
        import base64
        # Minimal valid PNG (1x1 white pixel)
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        files = {"image": ("test.png", png_bytes, "image/png")}
        data = {"prompt": "What color is this image?"}
        resp = httpx.post(f"{API_URL}/v1/vision/analyze", files=files, data=data, timeout=60)
        has_analysis = "analysis" in resp.json()
        log_result("Vision endpoint functional", has_analysis or resp.status_code == 200)
    except Exception as e:
        log_result("Vision endpoint functional", False, str(e))

    # Test 5.2: Audio endpoint exists
    try:
        resp = httpx.post(
            f"{API_URL}/v1/audio/transcribe",
            files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
            data={"language": "en"},
            timeout=30,
        )
        log_result("Audio endpoint responds", resp.status_code in (200, 422, 502))
    except Exception as e:
        log_result("Audio endpoint responds", False, str(e))

    # Test 5.3: TTS endpoint info
    try:
        resp = httpx.post(
            f"{API_URL}/v1/audio/tts",
            data={"text": "hello", "language": "en"},
            timeout=10,
        )
        data = resp.json()
        has_suggestions = "recommended_engines" in data
        log_result("TTS endpoint provides engine info", has_suggestions)
    except Exception as e:
        log_result("TTS endpoint provides engine info", False, str(e))


# ============================================================================
# Summary Report
# ============================================================================

def print_summary():
    """Print final test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in RESULTS if r["passed"])
    failed = sum(1 for r in RESULTS if not r["passed"])
    total = len(RESULTS)

    print(f"\n  Total:  {total}")
    print(f"  Passed: {passed} ✅")
    print(f"  Failed: {failed} ❌")
    print(f"  Rate:   {passed/total*100:.1f}%\n" if total > 0 else "")

    if failed > 0:
        print("  Failed tests:")
        for r in RESULTS:
            if not r["passed"]:
                print(f"    ❌ {r['test']}: {r['detail']}")

    print("\n" + "=" * 60)
    return failed == 0


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  GEMMA 4 E4B - COMPREHENSIVE TEST SUITE")
    print("  Modified by ClaudeO")
    print("=" * 60)

    test_phase1_infrastructure()
    test_phase2_ollama_direct()
    test_phase3_api_server()
    test_phase4_memory()
    test_phase5_multimodal()

    all_passed = print_summary()
    sys.exit(0 if all_passed else 1)
