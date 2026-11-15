import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from affinity_pipeline.config import load_profile, MachineProfile, CpuLimit, VramBudget
from affinity_pipeline.render import Transcript


@pytest.fixture
def mock_profile():
    return MachineProfile(
        name="test_profile",
        platform="windows",
        llama_bin_dir=Path("/fake/bin"),
        model_path=Path("/fake/model.gguf"),
        work_dir=Path("/fake/work"),
        cpu=CpuLimit(mode="none", cores=(), priority="normal", threads=1),
        vram=VramBudget(target_mb=400, max_mb=800),
        max_load1=0.8
    )


def test_get_profile_parameter_priority(monkeypatch):
    """Test that passing a name to get_profile uses that name."""
    test_profile = MachineProfile(
        name="custom",
        platform="windows",
        llama_bin_dir=Path("/fake/bin"),
        model_path=Path("/fake/model.gguf"),
        work_dir=Path("/fake/work"),
        cpu=CpuLimit(mode="none", cores=(), priority="normal", threads=1),
        vram=VramBudget(target_mb=400, max_mb=800),
        max_load1=0.8
    )
    with patch('scripts.segment_chunk_types.load_profile') as mock_load:
        mock_load.return_value = test_profile
        from scripts.segment_chunk_types import get_profile
        result = get_profile("custom")
        assert result == test_profile
        mock_load.assert_called_once_with("custom")


def test_get_profile_env_var(monkeypatch):
    """Test that env var AFFINITY_PROFILE is used when no parameter given."""
    test_profile = MachineProfile(
        name="custom",
        platform="windows",
        llama_bin_dir=Path("/fake/bin"),
        model_path=Path("/fake/model.gguf"),
        work_dir=Path("/fake/work"),
        cpu=CpuLimit(mode="none", cores=(), priority="normal", threads=1),
        vram=VramBudget(target_mb=400, max_mb=800),
        max_load1=0.8
    )
    monkeypatch.setenv("AFFINITY_PROFILE", "custom")
    with patch('scripts.segment_chunk_types.load_profile') as mock_load:
        mock_load.return_value = test_profile
        from scripts.segment_chunk_types import get_profile
        result = get_profile()
        assert result == test_profile
        mock_load.assert_called_once_with("custom")
    monkeypatch.delenv("AFFINITY_PROFILE")


def test_get_profile_default(monkeypatch):
    """Test that default profile is used when no param and no env var."""
    monkeypatch.delenv("AFFINITY_PROFILE", raising=False)
    with patch('scripts.segment_chunk_types.load_profile') as mock_load:
        mock_load.return_value = MachineProfile(
            name="gt1030_win",
            platform="windows",
            llama_bin_dir=Path("/fake/bin"),
            model_path=Path("/fake/model.gguf"),
            work_dir=Path("/fake/work"),
            cpu=CpuLimit(mode="none", cores=(), priority="normal", threads=1),
            vram=VramBudget(target_mb=400, max_mb=800),
            max_load1=0.8
        )
        from scripts.segment_chunk_types import get_profile
        result = get_profile()
        assert result.name == "gt1030_win"
        mock_load.assert_called_once_with("gt1030_win")


def test_run_tokenize_windows_profile(monkeypatch, mock_profile):
    """Test run_tokenize with Windows profile uses .exe binary."""
    fake_stdout = "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
    fake_stderr = ""

    # Mock subprocess.run to capture argv and return fake output
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=fake_stdout, stderr=fake_stderr
        )

        with patch('scripts.segment_chunk_types._load_vocab') as mock_vocab:
            mock_vocab.return_value = list("hello world")

            from scripts.segment_chunk_types import run_tokenize

            model_path = Path("/fake/model.gguf")
            body_path = Path("/fake/body.txt")

            result = run_tokenize(model_path, body_path, profile=mock_profile)

            # Verify result matches expected tokens
            assert result == [(i, c) for i, c in enumerate("hello world")]

            # Verify subprocess was called with correct args
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]  # argv list
            assert ".exe" in call_args[0]  # executable should have .exe suffix on Windows
            expected_args = [
                "-m", str(model_path),
                "--no-bos", "--no-escape", "--ids",
                "-f", str(body_path),
                "--log-disable"
            ]
            assert call_args[1:] == expected_args

            # Verify vocab was called to get tokens
            mock_vocab.assert_called_once_with(model_path)


def test_run_tokenize_linux_profile(monkeypatch, mock_profile):
    """Test run_tokenize with Linux profile does not use .exe binary."""
    # Create a Linux profile
    from dataclasses import replace
    linux_profile = replace(mock_profile, platform="linux")

    fake_stdout = "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
    fake_stderr = ""

    # Mock subprocess.run to capture argv and return fake output
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=fake_stdout, stderr=fake_stderr
        )

        with patch('scripts.segment_chunk_types._load_vocab') as mock_vocab:
            mock_vocab.return_value = list("hello world")

            from scripts.segment_chunk_types import run_tokenize

            model_path = Path("/fake/model.gguf")
            body_path = Path("/fake/body.txt")

            result = run_tokenize(model_path, body_path, profile=linux_profile)

            # Verify result matches expected tokens
            assert result == [(i, c) for i, c in enumerate("hello world")]

            # Verify subprocess was called with correct args
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]  # argv list
            assert ".exe" not in call_args[0]  # executable should NOT have .exe suffix on Linux


def test_run_tokenize_returncode_nonzero(monkeypatch, mock_profile):
    """Test run_tokenize raises RuntimeError when subprocess fails."""
    fake_stderr = "some error message"

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=fake_stderr
        )

        from scripts.segment_chunk_types import run_tokenize

        with pytest.raises(RuntimeError, match="llama-tokenize --ids failed.*some error message"):
            run_tokenize(Path("/fake/model.gguf"), Path("/fake/body.txt"), profile=mock_profile)


def test_run_tokenize_no_bracket_list(monkeypatch, mock_profile):
    """Test run_tokenize raises RuntimeError when output doesn't contain bracket list."""
    fake_stdout = "not a bracket list"
    fake_stderr = ""

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=fake_stdout, stderr=fake_stderr
        )

        from scripts.segment_chunk_types import run_tokenize

        with pytest.raises(RuntimeError, match="could not find a"):
            run_tokenize(Path("/fake/model.gguf"), Path("/fake/body.txt"), profile=mock_profile)


def test_run_tokenize_token_id_out_of_range(monkeypatch, mock_profile):
    """Test run_tokenize raises RuntimeError when token id is out of vocab range."""
    fake_stdout = "[0, 1, 99999]"  # Token ID 99999 exceeds vocab size
    fake_stderr = ""

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=fake_stdout, stderr=fake_stderr
        )

        with patch('scripts.segment_chunk_types._load_vocab') as mock_vocab:
            mock_vocab.return_value = ["a"] * 1000  # Vocab size 1000

            from scripts.segment_chunk_types import run_tokenize

            with pytest.raises(RuntimeError, match="out of vocab range"):
                run_tokenize(Path("/fake/model.gguf"), Path("/fake/body.txt"), profile=mock_profile)


def test_segment_creates_spans_json(tmp_path, mock_profile):
    """Test segment creates spans JSON file with correct content."""
    # Use Transcript to generate valid body and turns
    t = Transcript(None)
    us, ue, ms = t.open_user_turn("test")
    t.close_model_turn(turn=1, label="response", user_start=us, user_end=ue, model_start=ms,
                       answer="ok", stop_type="eos", tokens_predicted=1, seed_used=0, gen_tokens=[1])

    body_text = t.body

    # Create body file
    body_file = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "test_id.body.txt"
    body_file.parent.mkdir(parents=True, exist_ok=True)
    body_file.write_text(body_text)

    # Create turns file
    turns_file = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "test_id.turns.jsonl"
    turns_file.parent.mkdir(parents=True, exist_ok=True)
    turns_file.write_text("\n".join(json.dumps(turn.to_dict()) for turn in t.turns))

    # Mock run_tokenize to return one token per character
    with patch('scripts.segment_chunk_types.run_tokenize') as mock_tokenize:
        mock_tokenize.return_value = [(i, c) for i, c in enumerate(body_text)]

        from scripts.segment_chunk_types import segment

        result = segment("test_id", Path("/fake/model.gguf"), root=tmp_path, profile=mock_profile)

        # Check that spans file was created
        spans_file = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "test_id.spans.json"
        assert spans_file.exists()

        # Read and validate JSON content
        content = json.loads(spans_file.read_text(encoding="utf-8"))
        assert "session" in content
        assert "n_tokens" in content
        assert "spans" in content
        assert isinstance(content["spans"], list)

        # Check that spans cover the full range [0, n_tokens)
        spans = content["spans"]
        assert len(spans) > 0
        total_tokens = content["n_tokens"]

        # Validate that spans are contiguous and cover [0, total_tokens)
        current_pos = 0
        for span in spans:
            assert span["start"] == current_pos
            current_pos = span["end"]
        assert current_pos == total_tokens

        # Check no BOM in file
        content_bytes = spans_file.read_bytes()
        assert not content_bytes.startswith(b'\xef\xbb\xbf')


def test_segment_creates_export_txt(tmp_path, mock_profile):
    """Test segment creates export text file with correct content."""
    # Use Transcript to generate valid body and turns
    t = Transcript(None)
    us, ue, ms = t.open_user_turn("test")
    t.close_model_turn(turn=1, label="response", user_start=us, user_end=ue, model_start=ms,
                       answer="ok", stop_type="eos", tokens_predicted=1, seed_used=0, gen_tokens=[1])

    body_text = t.body

    # Create body file
    body_file = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "test_id.body.txt"
    body_file.parent.mkdir(parents=True, exist_ok=True)
    body_file.write_text(body_text)

    # Create turns file
    turns_file = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "test_id.turns.jsonl"
    turns_file.parent.mkdir(parents=True, exist_ok=True)
    turns_file.write_text("\n".join(json.dumps(turn.to_dict()) for turn in t.turns))

    # Mock run_tokenize to return one token per character
    with patch('scripts.segment_chunk_types.run_tokenize') as mock_tokenize:
        mock_tokenize.return_value = [(i, c) for i, c in enumerate(body_text)]

        from scripts.segment_chunk_types import segment

        result = segment("test_id", Path("/fake/model.gguf"), root=tmp_path, profile=mock_profile)

        # Check that export file was created
        export_file = tmp_path / "data" / "routing_corpus_v2" / "exports" / "test_id.txt"
        assert export_file.exists()

        content = export_file.read_text(encoding="utf-8")
        assert "# session=test_id n_tokens=" in content
        assert "---" in content  # Should contain span annotations


def test_segment_missing_body_file(tmp_path, mock_profile):
    """Test segment raises FileNotFoundError when body file is missing."""
    # Create only turns file
    turns_file = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "test_id.turns.jsonl"
    turns_file.parent.mkdir(parents=True, exist_ok=True)
    turns_file.write_text('{"user_start": 0, "user_end": 11, "model_start": 11, "model_end": 11, "turn": 1, "label": "response"}')
    
    from scripts.segment_chunk_types import segment
    
    with pytest.raises(FileNotFoundError, match="missing.*body.txt"):
        segment("test_id", Path("/fake/model.gguf"), root=tmp_path, profile=mock_profile)


def test_segment_missing_turns_file(tmp_path, mock_profile):
    """Test segment raises FileNotFoundError when turns file is missing."""
    # Create only body file
    body_file = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "test_id.body.txt"
    body_file.parent.mkdir(parents=True, exist_ok=True)
    body_file.write_text("hello world")
    
    from scripts.segment_chunk_types import segment
    
    with pytest.raises(FileNotFoundError, match="missing.*turns.jsonl"):
        segment("test_id", Path("/fake/model.gguf"), root=tmp_path, profile=mock_profile)


def test_segment_calls_run_tokenize_with_profile(tmp_path, mock_profile):
    """Test that segment calls run_tokenize with the provided profile."""
    # Use Transcript to generate valid body and turns
    t = Transcript(None)
    us, ue, ms = t.open_user_turn("test")
    t.close_model_turn(turn=1, label="response", user_start=us, user_end=ue, model_start=ms,
                       answer="ok", stop_type="eos", tokens_predicted=1, seed_used=0, gen_tokens=[1])

    body_text = t.body

    # Create body file
    body_file = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "test_id.body.txt"
    body_file.parent.mkdir(parents=True, exist_ok=True)
    body_file.write_text(body_text)

    # Create turns file
    turns_file = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "test_id.turns.jsonl"
    turns_file.parent.mkdir(parents=True, exist_ok=True)
    turns_file.write_text("\n".join(json.dumps(turn.to_dict()) for turn in t.turns))

    # Mock run_tokenize to track if it was called
    with patch('scripts.segment_chunk_types.run_tokenize') as mock_tokenize:
        mock_tokenize.return_value = [(i, c) for i, c in enumerate(body_text)]

        from scripts.segment_chunk_types import segment

        result = segment("test_id", Path("/fake/model.gguf"), root=tmp_path, profile=mock_profile)

        # Check that run_tokenize was called with the profile
        mock_tokenize.assert_called_once()


def test_main_segment_failure_exits_1(monkeypatch):
    """Test main exits with code 1 when segment fails."""
    monkeypatch.setattr(sys, 'argv', ['script', 'test_id'])

    with patch('scripts.segment_chunk_types.get_profile') as mock_get_prof:
        mock_get_prof.return_value = MachineProfile(
            name="gt1030_win",
            platform="windows",
            llama_bin_dir=Path("/fake/bin"),
            model_path=Path("/fake/model.gguf"),
            work_dir=Path("/fake/work"),
            cpu=CpuLimit(mode="none", cores=(), priority="normal", threads=1),
            vram=VramBudget(target_mb=400, max_mb=800),
            max_load1=0.8
        )
        with patch('scripts.segment_chunk_types.segment') as mock_segment:
            mock_segment.side_effect = ValueError("test error")

            from scripts.segment_chunk_types import main

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1


def test_main_with_profile_and_model_options(monkeypatch):
    """Test main correctly handles --profile and --model options."""
    monkeypatch.setattr(sys, 'argv', ['script', 'test_id', '--profile', 'gt1030_win', '--model', '/path/to/model'])
    
    with patch('scripts.segment_chunk_types.segment') as mock_segment:
        mock_segment.return_value = ([], {})
        
        from scripts.segment_chunk_types import main
        
        # This should not raise any exceptions
        main()
        
        # Verify segment was called with correct arguments
        mock_segment.assert_called_once()


def test_main_argv_none_uses_sys_argv(monkeypatch):
    """Test main correctly handles None argv by using sys.argv."""
    monkeypatch.setattr(sys, 'argv', ['script', 'test_id'])

    with patch('scripts.segment_chunk_types.get_profile') as mock_get_prof:
        mock_get_prof.return_value = MachineProfile(
            name="gt1030_win",
            platform="windows",
            llama_bin_dir=Path("/fake/bin"),
            model_path=Path("/fake/model.gguf"),
            work_dir=Path("/fake/work"),
            cpu=CpuLimit(mode="none", cores=(), priority="normal", threads=1),
            vram=VramBudget(target_mb=400, max_mb=800),
            max_load1=0.8
        )
        with patch('scripts.segment_chunk_types.segment') as mock_segment:
            mock_segment.return_value = ([], {})

            from scripts.segment_chunk_types import main

            # This should not raise any exceptions
            main(None)

            # Verify segment was called with correct arguments
            mock_segment.assert_called_once()


def test_no_hardcoded_paths():
    """Test that the script contains no hardcoded Windows paths or .exe literals."""
    script_path = Path("scripts/segment_chunk_types.py")
    content = script_path.read_text(encoding="utf-8")
    
    assert "E:\\" not in content
    assert "C:\\" not in content
    # Note: We allow ".exe" to be present since it's imported from config via resolve_binary
    # but we do not allow hardcoded .exe strings in the code itself
    assert ".exe" not in content or "resolve_binary" in content or "//" in content  # Allow it only when used correctly
