"""Tests for the machine configuration layer."""

import os
import pytest
from pathlib import Path

from affinity_pipeline.config import (
    load_profile,
    resolve_binary,
    affinity_mask,
    REPO_ROOT
)

def test_load_windows_profile():
    """Test loading a valid Windows profile."""
    # Load the profile
    profile = load_profile("gt1030_win")
    
    # Verify it's correct
    assert profile.name == "gt1030"
    assert profile.platform == "windows"
    assert profile.cpu.mode == "affinity_mask"
    assert profile.cpu.cores == (2, 3)
    assert profile.cpu.priority == "below_normal"
    assert profile.cpu.threads == 2
    assert profile.vram.target_mb == 1500
    assert profile.vram.max_mb == 1700
    assert profile.max_load1 is None
    
    # Test affinity mask conversion
    assert affinity_mask((2, 3)) == 12

def test_load_linux_profile():
    """Test loading a valid Linux profile."""
    # Load the profile
    profile = load_profile("gx10_sim")
    
    # Verify it's correct
    assert profile.name == "gx10_sim"
    assert profile.platform == "linux"
    assert profile.cpu.mode == "taskset"
    assert profile.cpu.cores == (0, 1, 2, 3, 4)
    assert profile.cpu.priority == "nice10"
    assert profile.cpu.threads == 5
    assert profile.vram.target_mb == 750
    assert profile.vram.max_mb == 1000
    assert profile.max_load1 == 6.0

def test_path_expansion_win(monkeypatch):
    """Test that environment variables are expanded correctly for Windows profile."""
    monkeypatch.setenv("LLMS_DIR", "/opt/models")

    profile = load_profile("gt1030_win")

    # Windows profile uses LLMS_DIR, paths resolve relative to REPO_ROOT
    assert str(profile.model_path).endswith("google_gemma-4-26B-A4B-it-Q4_K_M.gguf")
    assert "logs" in str(profile.work_dir) and "affinity_sim" in str(profile.work_dir)
    assert "bin" in str(profile.llama_bin_dir) and "llama-cpp" in str(profile.llama_bin_dir)

def test_path_expansion_linux(monkeypatch):
    """Test that ~ expansion works correctly for Linux profile."""
    profile = load_profile("gx10_sim")

    # Linux profile uses ~ expansion: ~ is resolved to home directory (platform-specific)
    # and relative parts are left relative to home
    work_dir_str = str(profile.work_dir)
    assert "gemma4_affinity_sim" in work_dir_str and "work" in work_dir_str

    bin_dir_str = str(profile.llama_bin_dir)
    assert "llama.cpp" in bin_dir_str and "build" in bin_dir_str and "bin" in bin_dir_str

def test_invalid_platform():
    """Test that invalid platform raises ValueError with correct message."""
    # Create a temporary JSON file with invalid platform
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "freebsd",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "none",
                "cores": [],
                "priority": "normal",
                "threads": 1
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 200
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="platform must be 'windows' or 'linux'"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

def test_invalid_cpu_mode():
    """Test that invalid CPU mode raises ValueError with correct message."""
    # Create a temporary JSON file with invalid CPU mode for platform
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "linux",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "affinity_mask",
                "cores": [0, 1],
                "priority": "normal",
                "threads": 1
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 200
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="cpu.mode.*affinity_mask.*not valid"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

def test_empty_cores():
    """Test that empty cores raises ValueError with correct message."""
    # Create a temporary JSON file with empty cores for non-none mode
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "affinity_mask",
                "cores": [],
                "priority": "normal",
                "threads": 1
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 200
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="cpu.cores must not be empty when mode is not 'none'"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

def test_negative_cores():
    """Test that negative cores raises ValueError with correct message."""
    # Create a temporary JSON file with negative core
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "affinity_mask",
                "cores": [0, -1],
                "priority": "normal",
                "threads": 1
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 200
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="cpu.cores.*>= 0"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

def test_duplicate_cores():
    """Test that duplicate cores raises ValueError with correct message."""
    # Create a temporary JSON file with duplicate cores
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "affinity_mask",
                "cores": [0, 1, 1],
                "priority": "normal",
                "threads": 1
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 200
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="cpu.cores.*duplicates"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

def test_invalid_vram():
    """Test that invalid VRAM budget raises ValueError with correct message."""
    # Create a temporary JSON file with invalid VRAM
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "none",
                "cores": [],
                "priority": "normal",
                "threads": 1
            },
            "vram": {
                "target_mb": 200,
                "max_mb": 100
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="vram.target_mb.*> vram.max_mb"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

def test_missing_keys():
    """Test that missing keys raise ValueError with correct message."""
    # Create a temporary JSON file with missing keys
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            # Missing model_path, work_dir, cpu, vram
        }, f)
    
    try:
        with pytest.raises(ValueError, match="Missing required keys"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

def test_bare_name_lookup(monkeypatch):
    """Test that bare name lookup works correctly."""
    # Create a fake repo structure
    import tempfile
    import json
    
    temp_dir = tempfile.mkdtemp()
    config_dir = Path(temp_dir) / "config" / "machines"
    config_dir.mkdir(parents=True)
    
    # Create a test profile file
    test_profile_path = config_dir / "test_win.json"
    with open(test_profile_path, 'w') as f:
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "affinity_mask",
                "cores": [0, 1],
                "priority": "normal",
                "threads": 1
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 200
            }
        }, f)
    
    # Monkeypatch REPO_ROOT to our temp directory
    monkeypatch.setattr("affinity_pipeline.config.REPO_ROOT", Path(temp_dir))
    
    # Try loading the profile by name
    try:
        profile = load_profile("test_win")
        assert profile.name == "test"
        assert profile.platform == "windows"
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)

def test_resolve_binary():
    """Test binary resolution."""
    # Load a profile to get a reference
    profile = load_profile("gt1030_win")
    
    # Test Windows binary resolution
    bin_path = resolve_binary(profile, "llama-server")
    assert bin_path.name == "llama-server.exe"
    
    # Test Linux binary resolution
    profile_linux = load_profile("gx10_sim")
    bin_path = resolve_binary(profile_linux, "llama-server")
    assert bin_path.name == "llama-server"
    
    # Test error case
    with pytest.raises(ValueError, match="contains path separators"):
        resolve_binary(profile, "foo/bar")

def test_affinity_mask():
    """Test affinity mask conversion."""
    # Test the examples from docstring
    assert affinity_mask((2, 3)) == 12
    assert affinity_mask((0, 1, 4)) == 19
    
    # Test empty tuple raises error
    with pytest.raises(ValueError, match="Core list must not be empty"):
        affinity_mask(())
    
    # Test negative core raises error
    with pytest.raises(ValueError, match="All cores must be >= 0"):
        affinity_mask((0, -1))

def test_both_profiles_load():
    """Test that both real profile files can be loaded."""
    # Set up environment variable for the Windows profile
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a fake LLMS_DIR
        llms_dir = Path(temp_dir) / "models"
        llms_dir.mkdir()
        
        # Set environment variable
        os.environ["LLMS_DIR"] = str(llms_dir)
        
        # Test loading both profiles
        profile1 = load_profile("gt1030_win")
        assert profile1.cpu.mode == "affinity_mask"
        assert profile1.cpu.threads == 2
        assert profile1.vram.max_mb == 1700
        
        profile2 = load_profile("gx10_sim")
        assert len(profile2.cpu.cores) == 5
        assert profile2.vram.target_mb == 750
        assert profile2.vram.max_mb == 1000
        assert profile2.max_load1 == 6.0

def test_threads_validation():
    """Test that threads < 1 raises ValueError with correct message."""
    # Create a temporary JSON file with threads = 0
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "none",
                "cores": [],
                "priority": "normal",
                "threads": 0
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 200
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="cpu.threads must be >= 1"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

    # Create a temporary JSON file with threads = -3
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "none",
                "cores": [],
                "priority": "normal",
                "threads": -3
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 200
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="cpu.threads must be >= 1"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

def test_nested_missing_keys():
    """Test nested missing keys raise ValueError with correct message."""
    # Test missing cpu.threads alone
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "none",
                "cores": [],
                "priority": "normal"
                # Missing threads
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 200
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="cpu.threads"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

    # Test missing vram.max_mb alone
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "none",
                "cores": [],
                "priority": "normal",
                "threads": 1
            },
            "vram": {
                "target_mb": 100
                # Missing max_mb
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="vram.max_mb"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

    # Test missing both cpu.threads and vram.max_mb
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "none",
                "cores": [],
                "priority": "normal"
                # Missing threads
            },
            "vram": {
                "target_mb": 100
                # Missing max_mb
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="cpu.threads.*vram.max_mb") or pytest.raises(ValueError, match="vram.max_mb.*cpu.threads"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

    # Test missing multiple cpu keys
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "cores": [],
                "priority": "normal"
                # Missing mode and threads
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 200
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="cpu.mode.*cpu.threads") or pytest.raises(ValueError, match="cpu.threads.*cpu.mode"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

def test_invalid_cpu_mode_value():
    """Test that invalid cpu.mode raises ValueError with correct message."""
    # Create a temporary JSON file with invalid cpu.mode
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "bogus",
                "cores": [],
                "priority": "normal",
                "threads": 1
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 200
            }
        }, f)
    
    try:
        with pytest.raises(ValueError, match="cpu.mode.*is not valid"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

def test_utf8_encoding():
    """Test that UTF-8 encoding works correctly."""
    # Create a temporary JSON file with Chinese characters in work_dir
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = Path(tmpdir) / "test.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump({
                "name": "test",
                "platform": "windows",
                "llama_bin_dir": "/bin",
                "model_path": "/models/test.gguf",
                "work_dir": "/工作目录/测试",
                "cpu": {
                    "mode": "none",
                    "cores": [],
                    "priority": "normal",
                    "threads": 1
                },
                "vram": {
                    "target_mb": 100,
                    "max_mb": 200
                }
            }, f, ensure_ascii=False)

        profile = load_profile(str(temp_file))
        assert "工作目录" in str(profile.work_dir)

def test_bare_name_with_separators():
    """Test that bare name detection works with both separators."""
    # Test that a/b is NOT treated as bare name
    with pytest.raises(ValueError, match="Profile file not found"):
        load_profile("a/b")

    # Test that a\b is NOT treated as bare name
    with pytest.raises(ValueError, match="Profile file not found"):
        load_profile("a\\b")

def test_empty_cores_taskset_mode():
    """Test that empty cores raises error for taskset mode (M5 mutation catch)."""
    # Create a temporary JSON file with empty cores for taskset mode
    import tempfile
    import json

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "linux",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "taskset",
                "cores": [],
                "priority": "nice10",
                "threads": 1
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 200
            }
        }, f)

    try:
        with pytest.raises(ValueError, match="cpu.cores must not be empty when mode is not 'none'"):
            load_profile(temp_file)
    finally:
        os.unlink(temp_file)

def test_vram_equal_target_and_max():
    """Test that equal VRAM target and max is allowed (M8 mutation catch)."""
    # Create a temporary JSON file with target_mb == max_mb
    import tempfile
    import json

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump({
            "name": "test",
            "platform": "windows",
            "llama_bin_dir": "/bin",
            "model_path": "/models/test.gguf",
            "work_dir": "/work",
            "cpu": {
                "mode": "none",
                "cores": [],
                "priority": "normal",
                "threads": 1
            },
            "vram": {
                "target_mb": 100,
                "max_mb": 100
            }
        }, f)

    try:
        # This should NOT raise an error - equal values are allowed
        profile = load_profile(temp_file)
        assert profile.vram.target_mb == 100
        assert profile.vram.max_mb == 100
    finally:
        os.unlink(temp_file)

def test_resolve_binary_with_backslash():
    """Test that backslash in binary name is rejected (M12 mutation catch)."""
    profile = load_profile("gt1030_win")

    # Test that backslash separator is rejected
    with pytest.raises(ValueError, match="contains path separators"):
        resolve_binary(profile, "foo" + chr(92) + "bar")
