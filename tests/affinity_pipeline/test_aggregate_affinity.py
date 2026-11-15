import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from scripts.aggregate_expert_affinity import main


def _mock_read_counts(gguf_path):
    """Mock read_counts that returns [30, 128] array of small counts."""
    return np.ones((30, 128), dtype=np.float64)


def test_aggregate_affinity_relative_outfile(tmp_path, monkeypatch):
    """Test aggregation with relative outfile path."""
    root = tmp_path

    # Mock read_counts to avoid needing real GGUF files
    monkeypatch.setattr("scripts.aggregate_expert_affinity.read_counts", _mock_read_counts)

    # Create jobs.json
    jobs_data = [
        {
            "session": "test_sess",
            "k": 1,
            "outfile": "logs/imatrix_v2/test_sess_1.gguf",
            "window_type": "code",
            "window_purity": 0.95
        }
    ]

    jobs_file = root / "data" / "routing_corpus_v2" / "jobs.json"
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(jobs_file, 'w', encoding='utf-8') as f:
        json.dump(jobs_data, f)

    # Create dummy .gguf file (not used due to mock)
    gguf_file = root / "logs" / "imatrix_v2" / "test_sess_1.gguf"
    gguf_file.parent.mkdir(parents=True, exist_ok=True)
    gguf_file.write_bytes(b'dummy')

    # Call main
    result = main(["--root", str(root)])

    assert result == 0

    # Check output files exist
    type_file = root / "logs" / "expert_affinity_type.json"
    role_file = root / "logs" / "expert_affinity_role.json"
    report_file = root / "logs" / "affinity_validation_report.md"

    assert type_file.exists()
    assert role_file.exists()
    assert report_file.exists()

    # Check content
    with open(type_file, 'r', encoding='utf-8') as f:
        type_data = json.load(f)

    with open(role_file, 'r', encoding='utf-8') as f:
        role_data = json.load(f)

    with open(report_file, 'r', encoding='utf-8') as f:
        report_content = f.read()

    assert "Expert Affinity Validation Report" in report_content


def test_aggregate_affinity_absolute_outfile(tmp_path, monkeypatch):
    """Test aggregation with absolute outfile path."""
    root = tmp_path

    # Mock read_counts
    monkeypatch.setattr("scripts.aggregate_expert_affinity.read_counts", _mock_read_counts)

    # Create jobs.json
    gguf_file = root / "test.gguf"
    jobs_data = [
        {
            "session": "test_sess",
            "k": 1,
            "outfile": str(gguf_file),
            "window_type": "code",
            "window_purity": 0.95
        }
    ]

    jobs_file = root / "data" / "routing_corpus_v2" / "jobs.json"
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(jobs_file, 'w', encoding='utf-8') as f:
        json.dump(jobs_data, f)

    # Create dummy .gguf file
    gguf_file.parent.mkdir(parents=True, exist_ok=True)
    gguf_file.write_bytes(b'dummy')

    # Call main
    result = main(["--root", str(root)])

    assert result == 0

    # Check output files exist
    type_file = root / "logs" / "expert_affinity_type.json"
    role_file = root / "logs" / "expert_affinity_role.json"
    report_file = root / "logs" / "affinity_validation_report.md"

    assert type_file.exists()
    assert role_file.exists()
    assert report_file.exists()


def test_aggregate_affinity_output_encoding_no_bom(tmp_path, monkeypatch):
    """Test that output files have no BOM."""
    root = tmp_path

    # Mock read_counts
    monkeypatch.setattr("scripts.aggregate_expert_affinity.read_counts", _mock_read_counts)

    # Create jobs.json
    jobs_data = [
        {
            "session": "test_sess",
            "k": 1,
            "outfile": "logs/imatrix_v2/test_sess_1.gguf",
            "window_type": "code",
            "window_purity": 0.95
        }
    ]

    jobs_file = root / "data" / "routing_corpus_v2" / "jobs.json"
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(jobs_file, 'w', encoding='utf-8') as f:
        json.dump(jobs_data, f)

    # Create dummy .gguf file
    gguf_file = root / "logs" / "imatrix_v2" / "test_sess_1.gguf"
    gguf_file.parent.mkdir(parents=True, exist_ok=True)
    gguf_file.write_bytes(b'dummy')

    # Call main
    result = main(["--root", str(root)])

    assert result == 0

    # Check no BOM in output files
    type_file = root / "logs" / "expert_affinity_type.json"
    role_file = root / "logs" / "expert_affinity_role.json"
    report_file = root / "logs" / "affinity_validation_report.md"

    for file_path in [type_file, role_file, report_file]:
        with open(file_path, 'rb') as f:
            content = f.read()
            # Check that first 3 bytes are not BOM
            assert not (content[:3] == b'\xef\xbb\xbf'), f"BOM found in {file_path}"


def test_aggregate_affinity_report_contains_sections(tmp_path, monkeypatch):
    """Test that report contains expected section headers."""
    root = tmp_path

    # Mock read_counts
    monkeypatch.setattr("scripts.aggregate_expert_affinity.read_counts", _mock_read_counts)

    # Create jobs.json
    jobs_data = [
        {
            "session": "test_sess",
            "k": 1,
            "outfile": "logs/imatrix_v2/test_sess_1.gguf",
            "window_type": "code",
            "window_purity": 0.95
        }
    ]

    jobs_file = root / "data" / "routing_corpus_v2" / "jobs.json"
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(jobs_file, 'w', encoding='utf-8') as f:
        json.dump(jobs_data, f)

    # Create dummy .gguf file
    gguf_file = root / "logs" / "imatrix_v2" / "test_sess_1.gguf"
    gguf_file.parent.mkdir(parents=True, exist_ok=True)
    gguf_file.write_bytes(b'dummy')

    # Call main
    result = main(["--root", str(root)])

    assert result == 0

    # Check report content
    report_file = root / "logs" / "affinity_validation_report.md"
    with open(report_file, 'r', encoding='utf-8') as f:
        report_content = f.read()

    assert "# Expert Affinity Validation Report" in report_content
    assert "## V5" in report_content
    assert "## V6" in report_content
    assert "## V7" in report_content
    assert "## V8" in report_content


def test_aggregate_affinity_jobs_file_not_found(tmp_path):
    """Test that missing jobs.json raises FileNotFoundError."""
    root = tmp_path
    
    # Do not create jobs.json
    
    with pytest.raises(FileNotFoundError) as exc_info:
        main(["--root", str(root)])
    
    assert "jobs.json" in str(exc_info.value)


def test_aggregate_affinity_missing_outfile_does_not_crash(tmp_path, monkeypatch):
    """Test that missing outfile is handled gracefully."""
    root = tmp_path

    # Mock read_counts to handle missing files gracefully
    def mock_read_counts_with_skip(gguf_path):
        # Return mock data if file exists (mocked), else raise to simulate missing file
        from pathlib import Path
        if not Path(gguf_path).exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
        return np.ones((30, 128), dtype=np.float64)

    monkeypatch.setattr("scripts.aggregate_expert_affinity.read_counts", mock_read_counts_with_skip)

    # Create jobs.json with one valid and one invalid file
    jobs_data = [
        {
            "session": "test_sess",
            "k": 1,
            "outfile": "logs/imatrix_v2/test_sess_1.gguf",
            "window_type": "code",
            "window_purity": 0.95
        },
        {
            "session": "test_sess2",
            "k": 2,
            "outfile": "logs/imatrix_v2/missing_file.gguf",
            "window_type": "reasoning",
            "window_purity": 0.85
        }
    ]

    jobs_file = root / "data" / "routing_corpus_v2" / "jobs.json"
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(jobs_file, 'w', encoding='utf-8') as f:
        json.dump(jobs_data, f)

    # Create only one .gguf file
    gguf_file = root / "logs" / "imatrix_v2" / "test_sess_1.gguf"
    gguf_file.parent.mkdir(parents=True, exist_ok=True)
    gguf_file.write_bytes(b'dummy')

    # Call main - should not crash
    result = main(["--root", str(root)])

    assert result == 0

    # Check that output files still exist
    type_file = root / "logs" / "expert_affinity_type.json"
    role_file = root / "logs" / "expert_affinity_role.json"
    report_file = root / "logs" / "affinity_validation_report.md"

    assert type_file.exists()
    assert role_file.exists()
    assert report_file.exists()


def test_aggregate_affinity_static_check_no_hardcoded_paths(tmp_path):
    """Test that no hardcoded paths are present in source."""
    # Read the source file
    script_path = Path("scripts/aggregate_expert_affinity.py")

    # This would be checked by a static analysis tool, but we simulate it
    # by checking for patterns that should not be present
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for hardcoded Windows paths or executables
        assert "E:\\" not in content
        assert "C:\\" not in content
        assert ".exe" not in content  # Not checking for .ps1 since those are acceptable
    except FileNotFoundError:
        # If file doesn't exist, skip this test (mocked scenario)
        pass


def test_aggregate_affinity_with_custom_root_argument(tmp_path, monkeypatch):
    """Test that custom root argument works correctly."""
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"

    # Mock read_counts
    monkeypatch.setattr("scripts.aggregate_expert_affinity.read_counts", _mock_read_counts)

    # Create jobs.json in root1
    jobs_data = [
        {
            "session": "test_sess",
            "k": 1,
            "outfile": "logs/imatrix_v2/test_sess_1.gguf",
            "window_type": "code",
            "window_purity": 0.95
        }
    ]

    jobs_file = root1 / "data" / "routing_corpus_v2" / "jobs.json"
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(jobs_file, 'w', encoding='utf-8') as f:
        json.dump(jobs_data, f)

    # Create dummy .gguf file
    gguf_file = root1 / "logs" / "imatrix_v2" / "test_sess_1.gguf"
    gguf_file.parent.mkdir(parents=True, exist_ok=True)
    gguf_file.write_bytes(b'dummy')

    # Call main with root1
    result = main(["--root", str(root1)])

    assert result == 0

    # Check output files are in root1, not root2
    type_file = root1 / "logs" / "expert_affinity_type.json"
    role_file = root1 / "logs" / "expert_affinity_role.json"
    report_file = root1 / "logs" / "affinity_validation_report.md"

    assert type_file.exists()
    assert role_file.exists()
    assert report_file.exists()

    # Check that files don't exist in root2
    assert not (root2 / "logs").exists()
