import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripts.profile_imatrix import (
    build_imatrix_argv,
    job_ceiling_seconds,
    run_job,
    main
)
from affinity_pipeline.config import MachineProfile, CpuLimit, VramBudget


def make_profile(platform: str = "windows", max_mb: int = 1000) -> MachineProfile:
    """Construct a test profile without machine dependencies."""
    from affinity_pipeline.config import MachineProfile, CpuLimit, VramBudget
    return MachineProfile(
        name="test_profile",
        platform=platform,
        llama_bin_dir=Path("bin"),
        model_path=Path("model.gguf"),
        work_dir=Path("work"),
        cpu=CpuLimit(mode="none", cores=(), priority="normal", threads=2),
        vram=VramBudget(target_mb=max_mb // 2, max_mb=max_mb),
        max_load1=None
    )


@pytest.fixture
def fake_script():
    """Helper to generate small Python scripts for subprocess injection."""
    def gen(*, log_lines=None, write_file=None, sleep_s=0.0):
        lines = []
        if log_lines:
            for line in log_lines:
                lines.append(f'print({repr(line)})')
        if write_file:
            lines.append(f'from pathlib import Path')
            lines.append(f'Path({repr(str(write_file))}).write_bytes(b"x")')
        if sleep_s:
            lines.append(f'import time')
            lines.append(f'time.sleep({sleep_s})')
        return '\n'.join(lines)
    return gen


def test_build_imatrix_argv_windows():
    profile = make_profile("windows")
    infile = Path("input.txt")
    ctx = 512
    outfile = Path("output.gguf")
    
    argv = build_imatrix_argv(profile, infile, ctx, outfile)
    
    # Check executable ends with .exe on Windows
    assert argv[0].endswith(".exe")
    
    # Check exact order and content
    expected = [
        str(profile.llama_bin_dir / "llama-imatrix.exe"),
        "-m", str(profile.model_path),
        "-ngl", "9",
        "--override-tensor", r"\.ffn_.*_exps\.=CPU",
        "-c", str(ctx),
        "-b", "512",
        "-ub", "512",
        "-t", str(profile.cpu.threads),
        "-tb", str(profile.cpu.threads),
        "-fit", "off",
        "-f", str(infile),
        "--chunks", "1",
        "--parse-special",
        "--no-escape",
        "--no-ppl",
        "--output-format", "gguf",
        "-o", str(outfile)
    ]
    
    assert argv == expected


def test_build_imatrix_argv_linux():
    profile = make_profile("linux")
    infile = Path("input.txt")
    ctx = 512
    outfile = Path("output.gguf")
    
    argv = build_imatrix_argv(profile, infile, ctx, outfile)
    
    # Check executable does NOT end with .exe on Linux
    assert not argv[0].endswith(".exe")
    
    # Check exact order and content
    expected = [
        str(profile.llama_bin_dir / "llama-imatrix"),
        "-m", str(profile.model_path),
        "-ngl", "9",
        "--override-tensor", r"\.ffn_.*_exps\.=CPU",
        "-c", str(ctx),
        "-b", "512",
        "-ub", "512",
        "-t", str(profile.cpu.threads),
        "-tb", str(profile.cpu.threads),
        "-fit", "off",
        "-f", str(infile),
        "--chunks", "1",
        "--parse-special",
        "--no-escape",
        "--no-ppl",
        "--output-format", "gguf",
        "-o", str(outfile)
    ]
    
    assert argv == expected


def test_job_ceiling_seconds():
    assert job_ceiling_seconds(1) == 31  # ceil(30 + 0.12 * 1) = ceil(30.12) = 31
    assert job_ceiling_seconds(2057) == 277  # ceil(30 + 0.12 * 2057) = ceil(276.84) = 277
    assert job_ceiling_seconds(1000) == 150  # ceil(30 + 0.12 * 1000) = ceil(150.0) = 150
    assert job_ceiling_seconds(0) == 30  # ceil(30 + 0.12 * 0) = ceil(30) = 30


def test_run_job_returns_ok_when_log_has_verification_line(tmp_path, fake_script):
    root = tmp_path / "root"
    root.mkdir()
    (root / "logs" / "imatrix_v2").mkdir(parents=True)
    
    job = {
        "session": "test",
        "k": 1,
        "ctx": 512,
        "infile": "input.txt",
        "outfile": "output.gguf",
        "window_type": "code"
    }
    
    profile = make_profile()
    infile = root / job["infile"]
    outfile = root / job["outfile"]
    
    # Create a fake script that writes the verification line
    script_content = fake_script(
        log_lines=[
            "load_tensors: Vulkan0 compute buffer size = 500.0 MiB",
            "computing over 1 chunks, n_ctx=512"
        ],
        write_file=outfile,
        sleep_s=0.01
    )
    
    argv = [sys.executable, "-c", script_content]
    
    result = run_job(job, profile, root, argv=argv)
    
    assert result == "OK"


def test_run_job_returns_ok_unverified_when_missing_log_line(tmp_path, fake_script):
    root = tmp_path / "root"
    root.mkdir()
    (root / "logs" / "imatrix_v2").mkdir(parents=True)
    
    job = {
        "session": "test",
        "k": 1,
        "ctx": 512,
        "infile": "input.txt",
        "outfile": "output.gguf",
        "window_type": "code"
    }
    
    profile = make_profile()
    infile = root / job["infile"]
    outfile = root / job["outfile"]
    
    # Create a fake script that does NOT write the verification line
    script_content = fake_script(
        log_lines=[
            "load_tensors: Vulkan0 compute buffer size = 500.0 MiB"
        ],
        write_file=outfile,
        sleep_s=0.01
    )
    
    argv = [sys.executable, "-c", script_content]
    
    result = run_job(job, profile, root, argv=argv)
    
    assert result == "OK_UNVERIFIED"


def test_run_job_returns_no_output_when_outfile_missing(tmp_path, fake_script, capsys):
    root = tmp_path / "root"
    root.mkdir()
    (root / "logs" / "imatrix_v2").mkdir(parents=True)
    
    job = {
        "session": "test",
        "k": 1,
        "ctx": 512,
        "infile": "input.txt",
        "outfile": "output.gguf",
        "window_type": "code"
    }
    
    profile = make_profile()
    infile = root / job["infile"]
    outfile = root / job["outfile"]
    
    # Create a fake script that does NOT write the output file
    script_content = fake_script(
        log_lines=[
            "load_tensors: Vulkan0 compute buffer size = 500.0 MiB"
        ],
        sleep_s=0.01
    )
    
    argv = [sys.executable, "-c", script_content]
    
    result = run_job(job, profile, root, argv=argv)

    assert result == "NO_OUTPUT"

    # Capture stderr to verify log tail is printed (should contain the log content)
    captured = capsys.readouterr()
    # stderr should contain the log tail content (Vulkan0 buffer size line was printed)
    assert len(captured.err) > 0


def test_run_job_returns_timeout_when_ceiling_exceeded(tmp_path, fake_script):
    root = tmp_path / "root"
    root.mkdir()
    (root / "logs" / "imatrix_v2").mkdir(parents=True)
    
    job = {
        "session": "test",
        "k": 1,
        "ctx": 100,
        "infile": "input.txt",
        "outfile": "output.gguf",
        "window_type": "code"
    }
    
    profile = make_profile()
    
    # Monkeypatch job_ceiling_seconds to return a very short timeout
    with patch("scripts.profile_imatrix.job_ceiling_seconds", return_value=1):
        script_content = fake_script(
            log_lines=[
                "load_tensors: Vulkan0 compute buffer size = 500.0 MiB"
            ],
            sleep_s=10.0  # Sleep longer than timeout
        )
        
        argv = [sys.executable, "-c", script_content]
        
        result = run_job(job, profile, root, argv=argv)
        
        assert result == "TIMEOUT"


def test_run_job_returns_vram_abort_when_budget_exceeded(tmp_path, fake_script):
    root = tmp_path / "root"
    root.mkdir()
    (root / "logs" / "imatrix_v2").mkdir(parents=True)
    
    job = {
        "session": "test",
        "k": 1,
        "ctx": 256,
        "infile": "input.txt",
        "outfile": "output.gguf",
        "window_type": "code"
    }
    
    # Create a profile with small VRAM budget
    profile = make_profile(max_mb=1000)
    # This should cause VRAM_ABORT since 1100 > 1000
    
    script_content = fake_script(
        log_lines=[
            "load_tensors: Vulkan0 compute buffer size = 1100.0 MiB"
        ],
        sleep_s=10.0  # Sleep longer than timeout
    )
    
    argv = [sys.executable, "-c", script_content]
    
    result = run_job(job, profile, root, argv=argv)
    
    assert result == "VRAM_ABORT"


def test_main_dry_run_no_processes(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    
    jobs_file = root / "data" / "routing_corpus_v2" / "jobs.json"
    jobs_file.parent.mkdir(parents=True)
    
    jobs_data = [
        {
            "session": "test1",
            "k": 1,
            "ctx": 512,
            "infile": "input1.txt",
            "outfile": "output1.gguf",
            "window_type": "code"
        },
        {
            "session": "test2",
            "k": 2,
            "ctx": 1024,
            "infile": "input2.txt",
            "outfile": "output2.gguf",
            "window_type": "code"
        }
    ]
    
    with open(jobs_file, 'w') as f:
        json.dump(jobs_data, f)
    
    # Mock GuardedProcess to verify it's never instantiated
    with patch('scripts.profile_imatrix.GuardedProcess') as mock_gp:
        mock_gp.return_value.__enter__.return_value = MagicMock()
        
        exit_code = main([
            "--root", str(root),
            "--dry-run",
            "--profile", "gt1030_win"
        ])
        
        assert exit_code == 0
        
        # Verify GuardedProcess was never instantiated
        mock_gp.assert_not_called()


def test_main_jobs_file_not_found(tmp_path, capsys):
    root = tmp_path / "root"
    root.mkdir()
    
    # Don't create jobs.json
    
    exit_code = main([
        "--root", str(root),
        "--profile", "gt1030_win"
    ])
    
    assert exit_code == 1
    
    captured = capsys.readouterr()
    assert "Jobs file not found" in captured.err
    assert str(root / "data/routing_corpus_v2/jobs.json") in captured.err


def test_main_ad_hoc_file_missing_ctx(tmp_path, capsys):
    root = tmp_path / "root"
    root.mkdir()
    
    exit_code = main([
        "--root", str(root),
        "--file", "input.txt",
        "--out", "out.gguf"
        # Missing --ctx
    ])
    
    assert exit_code == 1
    
    captured = capsys.readouterr()
    assert "Ad-hoc mode needs" in captured.err or "--ctx" in captured.err


def test_main_results_json_format(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    
    jobs_file = root / "data" / "routing_corpus_v2" / "jobs.json"
    jobs_file.parent.mkdir(parents=True)
    
    jobs_data = [
        {
            "session": "test1",
            "k": 1,
            "ctx": 512,
            "infile": "input1.txt",
            "outfile": "output1.gguf",
            "window_type": "code"
        },
        {
            "session": "test2",
            "k": 2,
            "ctx": 1024,
            "infile": "input2.txt",
            "outfile": "output2.gguf",
            "window_type": "code"
        },
        {
            "session": "test3",
            "k": 3,
            "ctx": 2048,
            "infile": "input3.txt",
            "outfile": "output3.gguf",
            "window_type": "code"
        }
    ]
    
    with open(jobs_file, 'w') as f:
        json.dump(jobs_data, f)
    
    # Mock run_job to return specific statuses
    def mock_run_job(job, profile, root, *, argv=None, poll_s=2.0):
        if job["k"] == 1:
            return "OK"
        elif job["k"] == 2:
            return "TIMEOUT"
        else:
            return "NO_OUTPUT"
    
    with patch("scripts.profile_imatrix.run_job", side_effect=mock_run_job):
        exit_code = main([
            "--root", str(root),
            "--profile", "gt1030_win"
        ])
        
        assert exit_code == 0
    
    # Read job_results.json
    results_file = root / "logs" / "imatrix_v2" / "job_results.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    assert len(results) == 3
    
    for entry in results:
        assert "session" in entry
        assert "k" in entry
        assert "status" in entry
        assert "outfile" in entry


def test_static_check_no_hardcoded_paths():
    # Read profile_imatrix.py source
    script_path = Path("scripts/profile_imatrix.py")
    content = script_path.read_text()
    
    # Check for hardcoded paths or system-specific commands
    forbidden_patterns = [
        r"E:\\",
        r"C:\\",
        r"\.exe",
        r"taskset",
        r"ProcessorAffinity"
    ]
    
    for pattern in forbidden_patterns:
        assert not (pattern in content), f"Found forbidden pattern '{pattern}' in profile_imatrix.py"
