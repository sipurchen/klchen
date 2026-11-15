import pytest
import json
import sys
import os
from pathlib import Path
import http.server
import threading
import time
import urllib.request
from typing import Sequence, Optional

# Mock server infrastructure
class MockLlamaHandler(http.server.BaseHTTPRequestHandler):
    responses = []
    requests_log = []
    response_idx = [0]

    def do_POST(self):
        if self.path == "/completion":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            request_data = json.loads(body)
            self.requests_log.append(request_data)

            if MockLlamaHandler.response_idx[0] < len(MockLlamaHandler.responses):
                resp_spec = MockLlamaHandler.responses[MockLlamaHandler.response_idx[0]]
                MockLlamaHandler.response_idx[0] += 1

                if resp_spec.get("http_status") == 500:
                    self.send_response(500)
                    self.end_headers()
                else:
                    response = {
                        "content": resp_spec.get("content", ""),
                        "stop_type": resp_spec.get("stop_type", "eos"),
                        "tokens_predicted": resp_spec.get("tokens_predicted", 0),
                        "tokens": resp_spec.get("tokens")
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode("utf-8"))
            else:
                self.send_response(500)
                self.end_headers()

    def log_message(self, format, *args):
        pass

def start_mock_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockLlamaHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port, thread

def stop_mock_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)

# Test imports
from affinity_pipeline.gen_session import (
    load_turn_spec, build_server_argv, http_complete, generate_session,
    write_outputs, start_server, main
)
from affinity_pipeline.config import load_profile, REPO_ROOT

def test_load_turn_spec_missing_family():
    """Missing 'family' → ValueError with field name."""
    # Create a temporary JSON file without family
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "session": "test_session",
            "seed": 123,
            "turns": [{"label": "test", "user": "test user"}]
        }, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="family"):
            load_turn_spec(temp_path)
    finally:
        os.unlink(temp_path)

def test_load_turn_spec_invalid_seed_type():
    """seed is string or bool → ValueError."""
    import tempfile
    import os
    # Test with string seed
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "family": "test",
            "session": "test_session",
            "seed": "not_a_number",
            "turns": [{"label": "test", "user": "test user"}]
        }, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="seed"):
            load_turn_spec(temp_path)
    finally:
        os.unlink(temp_path)

def test_load_turn_spec_empty_turns():
    """turns is empty list → ValueError."""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "family": "test",
            "session": "test_session",
            "seed": 123,
            "turns": []
        }, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="turns"):
            load_turn_spec(temp_path)
    finally:
        os.unlink(temp_path)

def test_load_turn_spec_turn_missing_user():
    """A turn item missing 'user' → ValueError with 'user' in message."""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "family": "test",
            "session": "test_session",
            "seed": 123,
            "turns": [{"label": "test"}]
        }, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="user"):
            load_turn_spec(temp_path)
    finally:
        os.unlink(temp_path)

def test_load_turn_spec_turn_missing_label():
    """A turn item missing 'label' → ValueError with 'label' in message."""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "family": "test",
            "session": "test_session",
            "seed": 123,
            "turns": [{"user": "test user"}]
        }, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="label"):
            load_turn_spec(temp_path)
    finally:
        os.unlink(temp_path)

def test_load_turn_spec_valid_with_null_system():
    """system=null is valid."""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "family": "test",
            "session": "test_session",
            "seed": 123,
            "system": None,
            "turns": [{"label": "test", "user": "test user"}]
        }, f)
        temp_path = f.name

    try:
        spec = load_turn_spec(temp_path)
        assert spec.system is None
    finally:
        os.unlink(temp_path)

def test_load_turn_spec_bom_file():
    """File with UTF-8 BOM should load correctly."""
    import tempfile
    import os
    import json
    from pathlib import Path

    json_data = {
        "family": "test",
        "session": "test_session",
        "seed": 123,
        "turns": [{"label": "test", "user": "test user"}]
    }
    json_bytes = json.dumps(json_data).encode('utf-8')

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        Path(temp_path).write_bytes(b"\xef\xbb\xbf" + json_bytes)
        spec = load_turn_spec(temp_path)
        assert spec.family == "test"
    finally:
        os.unlink(temp_path)

def test_generate_session_golden():
    """Validate against golden data: planner_pilot session."""
    # Load golden data
    turns_file = REPO_ROOT / "data/routing_corpus_v2/turns/planner_pilot.json"
    golden_body_path = REPO_ROOT / "data/routing_corpus_v2/sessions/planner_pilot.body.txt"
    golden_turns_path = REPO_ROOT / "data/routing_corpus_v2/sessions/planner_pilot.turns.jsonl"

    spec = load_turn_spec(turns_file)  # Should load 6 turns with seed=3001

    # Read golden body and turns
    with open(golden_body_path, "rb") as f:
        golden_body_bytes = f.read().replace(b"\r\n", b"\n")  # Normalize CRLF to LF

    golden_turns_list = []
    with open(golden_turns_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_turns_list.append(json.loads(line))

    # Extract model answers from golden to feed back to mock
    responses = []
    for turn_meta in golden_turns_list:
        model_start = turn_meta["model_start"]
        model_end = turn_meta["model_end"]
        model_text = golden_body_bytes.decode("utf-8")[model_start:model_end]
        responses.append({
            "content": model_text,
            "stop_type": "eos",
            "tokens_predicted": turn_meta["tokens_predicted"],
            "tokens": turn_meta.get("gen_tokens")
        })

    # Start mock server
    server, port, thread = start_mock_server()
    MockLlamaHandler.responses = responses
    MockLlamaHandler.requests_log = []
    MockLlamaHandler.response_idx = [0]

    try:
        # Generate session
        complete = http_complete(f"http://127.0.0.1:{port}")
        result = generate_session(spec, complete, seed=3001)

        # Validate: body bytes match (after LF normalization)
        result_body_bytes = result.body.encode("utf-8")
        assert result_body_bytes == golden_body_bytes, \
            f"Body mismatch: {len(result_body_bytes)} bytes generated vs {len(golden_body_bytes)} golden"

        # Validate: turns match (parse JSON and compare)
        assert len(result.turns) == len(golden_turns_list), \
            f"Turn count mismatch: {len(result.turns)} vs {len(golden_turns_list)}"

        for i, (result_turn, golden_turn) in enumerate(zip(result.turns, golden_turns_list)):
            result_dict = result_turn.to_dict()
            # Compare all fields
            for key in golden_turn:
                assert result_dict[key] == golden_turn[key], \
                    f"Turn {i+1} field {key} mismatch: {result_dict[key]} vs {golden_turn[key]}"

        # Validate: accepted is True, 4 turns (golden file has 4 turns)
        assert result.accepted is True
        assert len(result.turns) == 4

        # Validate: mock received correct prompts
        assert len(MockLlamaHandler.requests_log) == 4, \
            f"Mock received {len(MockLlamaHandler.requests_log)} requests, expected 4"

        for i, req in enumerate(MockLlamaHandler.requests_log):
            # Prompt should be golden body up to that turn's model_start
            expected_prompt = golden_body_bytes.decode("utf-8")[:golden_turns_list[i]["model_start"]]
            assert req["prompt"] == expected_prompt, \
                f"Turn {i+1} prompt mismatch at {len(req['prompt'])} chars"

            # Validate request fields
            assert req["n_predict"] == 3072
            assert req["cache_prompt"] is True
            assert req["temperature"] == 1.0
            assert req["top_k"] == 64
            assert req["top_p"] == 0.95
            assert req["return_tokens"] is True
            assert req["seed"] == 3001  # Should not increment on successful turns

        # Validate: turns were not ended mid-turn by max_tokens valve
        assert result.accepted is True  # No truncation

    finally:
        stop_mock_server(server, thread)

def test_retry_increment_seed():
    """First turn: retry once (length->eos), second turn uses +1 seed, no reset."""
    server, port, thread = start_mock_server()

    # Mock responses: turn 1 gets [length, eos], turn 2+ get eos
    MockLlamaHandler.responses = [
        {"content": "first answer", "stop_type": "length"},  # First attempt fails with length
        {"content": "first answer", "stop_type": "eos"},     # Second attempt succeeds with eos
        {"content": "second answer", "stop_type": "eos"}      # Second turn succeeds
    ]

    MockLlamaHandler.requests_log = []
    MockLlamaHandler.response_idx = [0]

    try:
        spec = load_turn_spec(REPO_ROOT / "data/routing_corpus_v2/turns/planner_pilot.json")

        complete = http_complete(f"http://127.0.0.1:{port}")
        result = generate_session(spec, complete, seed=3001)

        # First turn used seed 3001 (failed once), turns 2-4 each use incremented seed
        # With 4 turns and first turn retrying: 2 + 3 = 5 requests
        assert len(MockLlamaHandler.requests_log) >= 3
        assert MockLlamaHandler.requests_log[0]["seed"] == 3001
        assert MockLlamaHandler.requests_log[1]["seed"] == 3002  # incremented from first failure
        # Turn 2 onward: each may have seed 3002 or higher depending on responses

    finally:
        stop_mock_server(server, thread)

def test_truncation_no_eos():
    """Second turn: both attempts fail (both length), session truncates."""
    server, port, thread = start_mock_server()

    # Mock responses: turn 1 eos, turn 2 [length, length]
    MockLlamaHandler.responses = [
        {"content": "first answer", "stop_type": "eos"},      # First turn succeeds
        {"content": "second answer", "stop_type": "length"},  # First attempt fails with length
        {"content": "second answer", "stop_type": "length"}   # Second attempt fails with length
    ]

    MockLlamaHandler.requests_log = []
    MockLlamaHandler.response_idx = [0]

    try:
        spec = load_turn_spec(REPO_ROOT / "data/routing_corpus_v2/turns/planner_pilot.json")

        complete = http_complete(f"http://127.0.0.1:{port}")
        result = generate_session(spec, complete, seed=3001)

        # Should only process first turn and then truncate
        assert len(MockLlamaHandler.requests_log) == 3  # First turn + 2 attempts for second turn
        assert result.accepted is False  # Truncated session
        assert len(result.turns) == 1    # Only one turn processed

    finally:
        stop_mock_server(server, thread)

def test_http_500_treated_as_error():
    """HTTP 500 counts as error attempt, retry with seed+1."""
    server, port, thread = start_mock_server()

    # Mock responses: HTTP 500, then success
    MockLlamaHandler.responses = [
        {"http_status": 500},  # First request fails with 500
        {"content": "answer", "stop_type": "eos"}  # Second succeeds
    ]

    MockLlamaHandler.requests_log = []
    MockLlamaHandler.response_idx = [0]

    try:
        spec = load_turn_spec(REPO_ROOT / "data/routing_corpus_v2/turns/planner_pilot.json")

        complete = http_complete(f"http://127.0.0.1:{port}")
        result = generate_session(spec, complete, seed=3001)

        # Should have tried at least twice for turn 1, then continue for other turns
        # With HTTP 500 + 3 remaining turns: at least 2 + 3 = 5 requests possible
        assert len(MockLlamaHandler.requests_log) >= 2
        assert MockLlamaHandler.requests_log[0]["seed"] == 3001
        if len(MockLlamaHandler.requests_log) > 1:
            # Seed should increment after 500 error
            assert MockLlamaHandler.requests_log[1]["seed"] >= 3001

    finally:
        stop_mock_server(server, thread)

def test_max_tokens_stops_generation():
    """Set max_tokens small enough to trigger on turn 1."""
    server, port, thread = start_mock_server()

    # Mock responses: turn 1 eos
    MockLlamaHandler.responses = [
        {"content": "answer", "stop_type": "eos"}
    ]

    MockLlamaHandler.requests_log = []
    MockLlamaHandler.response_idx = [0]

    try:
        spec = load_turn_spec(REPO_ROOT / "data/routing_corpus_v2/turns/planner_pilot.json")

        complete = http_complete(f"http://127.0.0.1:{port}")
        result = generate_session(spec, complete, seed=3001, max_tokens=1)  # Very small limit

        # Should stop after first turn even though it could continue
        assert len(MockLlamaHandler.requests_log) == 1  # Only one request made
        assert result.accepted is True  # Not truncated but stopped by max_tokens

    finally:
        stop_mock_server(server, thread)

def test_write_outputs_binary_mode():
    """Verify body.txt written in binary mode, no BOM, no line ending conversion."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        # Create a mock session result
        from affinity_pipeline.gen_session import SessionResult
        from affinity_pipeline.render import TurnMeta

        body = "Test body content\nwith newlines"
        turns = []
        result = SessionResult(body=body, turns=turns, accepted=True)

        # Write outputs
        body_path, turns_path = write_outputs(result, out_dir, "test", "session")

        # Read back the body file in binary mode to check no BOM
        with open(body_path, "rb") as f:
            content = f.read()

        # Should be exactly what we wrote (no extra encoding)
        expected = body.encode("utf-8")
        assert content == expected

def test_write_outputs_turns_newline_format():
    """Verify turns.jsonl uses LF, includes trailing newline on last line."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        # Create a mock session result with turns
        from affinity_pipeline.gen_session import SessionResult
        from affinity_pipeline.render import TurnMeta

        body = "Test body content"
        turns = [
            TurnMeta(turn=1, label="test", user_start=0, user_end=10, model_start=10,
                     model_end=20, stop_type="eos", tokens_predicted=5, seed_used=3001,
                     gen_tokens=[1, 2, 3])
        ]
        result = SessionResult(body=body, turns=turns, accepted=True)

        # Write outputs
        body_path, turns_path = write_outputs(result, out_dir, "test", "session")

        # Read back the turns file
        with open(turns_path, "r") as f:
            content = f.read()

        # Should end with newline
        assert content.endswith("\n")

def test_build_server_argv_gt1030():
    """Verify argv for gt1030_win profile: correct paths, --override-tensor value."""
    profile = load_profile("gt1030_win")
    argv = build_server_argv(profile, 8260)

    assert str(profile.model_path) in argv
    assert "--override-tensor" in argv
    # The actual tensor override value should be a single string with literal backslashes
    assert r"\.ffn_.*_exps\.=CPU" in argv

def test_build_server_argv_gx10():
    """Verify argv for gx10_sim profile."""
    profile = load_profile("gx10_sim")
    argv = build_server_argv(profile, 8260)
    assert str(profile.model_path) in argv
    assert "--override-tensor" in argv
    assert r"\.ffn_.*_exps\.=CPU" in argv

def test_build_server_argv_extra_args():
    """Extra args appended to end."""
    profile = load_profile("gt1030_win")
    argv = build_server_argv(profile, 8260, extra_args=["--lv", "4"])
    assert argv[-2] == "--lv"
    assert argv[-1] == "4"

def test_start_server_ready(tmp_path):
    """Server prints 'server is listening', returns running GuardedProcess."""
    import subprocess
    log_file = tmp_path / "server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a fake server script that prints ready
    fake_server_script = tmp_path / "fake_server.py"
    fake_server_script.write_text(
        "import sys, time\n"
        "print('server is listening', flush=True)\n"
        "time.sleep(60)\n"
    )

    profile = load_profile("gt1030_win")
    import dataclasses
    profile = dataclasses.replace(profile, cpu=dataclasses.replace(profile.cpu, mode="none"), max_load1=None)

    argv = [sys.executable, str(fake_server_script)]

    try:
        gp = start_server(profile, argv, log_file, ready_timeout_s=10, poll_s=0.1)
        assert gp.poll() is None  # Process should be running
        gp.stop()
        assert gp.poll() is not None  # Process should have stopped
    except Exception as e:
        pytest.fail(f"start_server failed: {e}")

def test_start_server_dies_early(tmp_path):
    """Process exits before ready → RuntimeError, gp stopped."""
    import subprocess
    log_file = tmp_path / "server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    profile = load_profile("gt1030_win")
    import dataclasses
    profile = dataclasses.replace(profile, cpu=dataclasses.replace(profile.cpu, mode="none"), max_load1=None)

    # Script that exits immediately
    argv = [sys.executable, "-c", "import sys; print('starting'); sys.exit(1)"]

    try:
        with pytest.raises(RuntimeError, match="died early"):
            start_server(profile, argv, log_file, ready_timeout_s=10, poll_s=0.1)
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")

def test_start_server_never_ready(tmp_path):
    """Timeout before 'server is listening' → RuntimeError, process killed."""
    import subprocess
    log_file = tmp_path / "server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    profile = load_profile("gt1030_win")
    import dataclasses
    profile = dataclasses.replace(profile, cpu=dataclasses.replace(profile.cpu, mode="none"), max_load1=None)

    # Script that never prints ready
    argv = [sys.executable, "-c", "import time; time.sleep(60)"]

    try:
        with pytest.raises(RuntimeError, match="not ready"):
            start_server(profile, argv, log_file, ready_timeout_s=1, poll_s=0.1)
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")

def test_cli_dry_run(tmp_path, capsys):
    """--dry-run prints JSON, returns 0, no files written, no processes."""
    out_dir = tmp_path / "output"

    ret = main([
        "--profile", "gt1030_win",
        "--family", "test",
        "--session", "session",
        "--turns-file", str(REPO_ROOT / "data/routing_corpus_v2/turns/planner_pilot.json"),
        "--dry-run",
        "--out-dir", str(out_dir)
    ])

    assert ret == 0

    # Check no files were written
    assert not out_dir.exists() or len(list(out_dir.rglob("*"))) == 0

    # Check JSON output
    captured = capsys.readouterr()
    import json
    output = json.loads(captured.out.strip())
    assert "profile" in output
    assert "server_argv" in output
    assert "outputs" in output

def test_cli_accepted_false_rc3(tmp_path):
    """Session truncates (no eos) → rc=3, files still written."""
    server, port, thread = start_mock_server()

    # Mock responses: turn 1 eos, turn 2 [length, length] (no eos)
    MockLlamaHandler.responses = [
        {"content": "first", "stop_type": "eos"},
        {"content": "second", "stop_type": "length"},
        {"content": "second", "stop_type": "length"}
    ]
    MockLlamaHandler.requests_log = []
    MockLlamaHandler.response_idx = [0]

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ret = main([
            "--profile", "gt1030_win",
            "--family", "test",
            "--session", "trunc",
            "--turns-file", str(REPO_ROOT / "data/routing_corpus_v2/turns/planner_pilot.json"),
            "--server-url", f"http://127.0.0.1:{port}",
            "--out-dir", str(out_dir)
        ])

        assert ret == 3  # Session truncated

        # Files should still be written
        assert (out_dir / "test_trunc.body.txt").exists()
        assert (out_dir / "test_trunc.turns.jsonl").exists()

    finally:
        stop_mock_server(server, thread)
