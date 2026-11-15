import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.build_imatrix_jobs import resolve_job_path, build_windows


def test_resolve_job_path_absolute():
    """Test that absolute paths are returned as-is."""
    root = Path("/tmp/root")
    path = "/absolute/path/to/file"
    result = resolve_job_path(path, root)
    assert result == Path("/absolute/path/to/file")


def test_resolve_job_path_relative():
    """Test that relative paths are resolved against root."""
    root = Path("/tmp/root")
    path = "data/file.txt"
    result = resolve_job_path(path, root)
    assert result == Path("/tmp/root/data/file.txt")


def test_resolve_job_path_empty_string():
    """Test behavior with empty string path."""
    root = Path("/tmp/root")
    path = ""
    result = resolve_job_path(path, root)
    # Path("") is relative, so it becomes root / ""
    assert result == root / path


def test_build_jobs_infile_outfile_are_relative_posix(tmp_path):
    """Test that infile/outfile paths in jobs are relative POSIX strings."""
    from scripts.build_imatrix_jobs import build_jobs_for_session

    # Create synthetic session files
    body_path = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "test_id.body.txt"
    spans_path = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "test_id.spans.json"
    turns_path = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "test_id.turns.jsonl"

    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_text = "test body"
    body_path.write_text(body_text)
    turns_path.write_text('{"turn": 0, "user_start": 0, "user_end": 4, "model_start": 4, "model_end": 9, "label": "response"}')
    spans_path.write_text('{"session": "test_id", "n_tokens": 9, "spans": [{"start": 0, "end": 4, "type": "user_prompt"}, {"start": 4, "end": 9, "type": "control"}]}')

    with patch('scripts.build_imatrix_jobs.run_tokenize', return_value=[(i, c) for i, c in enumerate(body_text)]):
        result = build_jobs_for_session("test_id", Path("/fake/model.gguf"), root=tmp_path)

        for job in result["jobs"]:
            # Check paths are strings
            assert isinstance(job["infile"], str)
            assert isinstance(job["outfile"], str)

            # Check paths are relative (not absolute)
            assert not job["infile"].startswith("/")
            assert not job["outfile"].startswith("/")

            # Check paths use POSIX (forward slashes)
            assert "/" in job["infile"]
            assert "/" in job["outfile"]

            # No .exe suffix
            assert ".exe" not in job["infile"]
            assert ".exe" not in job["outfile"]


def test_build_jobs_ctx_equals_bk_plus_1(tmp_path):
    """Test that ctx == b_k + 1."""
    from scripts.build_imatrix_jobs import build_jobs_for_session

    body_path = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "t.body.txt"
    spans_path = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "t.spans.json"
    turns_path = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "t.turns.jsonl"

    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_text = "x" * 300
    body_path.write_text(body_text)
    turns_path.write_text('{"turn": 0, "user_start": 0, "user_end": 150, "model_start": 150, "model_end": 300, "label": "response"}')
    spans_path.write_text('{"session": "t", "n_tokens": 300, "spans": [{"start": 0, "end": 150, "type": "user_prompt"}, {"start": 150, "end": 300, "type": "control"}]}')

    with patch('scripts.build_imatrix_jobs.run_tokenize', return_value=[(i, c) for i, c in enumerate(body_text)]):
        result = build_jobs_for_session("t", Path("/fake/model.gguf"), root=tmp_path)

        for job in result["jobs"]:
            assert job["ctx"] == job["b_k"] + 1


def test_build_jobs_last_job_is_whole_session(tmp_path):
    """Test that is_whole_session is True only for the last job."""
    from scripts.build_imatrix_jobs import build_jobs_for_session

    body_path = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "t.body.txt"
    spans_path = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "t.spans.json"
    turns_path = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "t.turns.jsonl"

    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_text = "x" * 300
    body_path.write_text(body_text)
    turns_path.write_text('{"turn": 0, "user_start": 0, "user_end": 150, "model_start": 150, "model_end": 300, "label": "response"}')
    spans_path.write_text('{"session": "t", "n_tokens": 300, "spans": [{"start": 0, "end": 150, "type": "user_prompt"}, {"start": 150, "end": 300, "type": "control"}]}')

    with patch('scripts.build_imatrix_jobs.run_tokenize', return_value=[(i, c) for i, c in enumerate(body_text)]):
        result = build_jobs_for_session("t", Path("/fake/model.gguf"), root=tmp_path)

        assert len(result["jobs"]) > 0
        assert result["jobs"][-1]["is_whole_session"] is True

        for job in result["jobs"][:-1]:
            assert job["is_whole_session"] is False


def test_build_jobs_missing_body_or_spans(tmp_path):
    """Test that missing files raise FileNotFoundError with both paths."""
    from scripts.build_imatrix_jobs import build_jobs_for_session

    turns_path = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "t.turns.jsonl"
    turns_path.parent.mkdir(parents=True, exist_ok=True)
    turns_path.write_text('{}')

    with patch('scripts.build_imatrix_jobs.run_tokenize'):
        with pytest.raises(FileNotFoundError) as exc:
            build_jobs_for_session("t", Path("/fake/model.gguf"), root=tmp_path)

        err = str(exc.value)
        assert "body.txt" in err or "spans.json" in err


def test_build_windows_min_size_constraint():
    """Test that build_windows handles small span lists."""
    spans = [{"start": 0, "end": 10, "type": "user_prompt"}]
    windows = build_windows(spans)
    assert len(windows) >= 1


def test_no_hardcoded_paths():
    """Test that script has no hardcoded paths like E:\\ or .exe."""
    script_path = Path("scripts/build_imatrix_jobs.py")
    content = script_path.read_text(encoding="utf-8")

    # Check for Windows paths
    assert "E:\\" not in content
    assert "C:\\" not in content
    # .exe can appear in resolve_binary calls from config, but not in this script
    assert content.count(".exe") == 0
