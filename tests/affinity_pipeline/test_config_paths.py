import os
import json
import pytest
from pathlib import Path
from affinity_pipeline import config

def test_env_var_expansion_dollar_bracket_form(tmp_path, monkeypatch):
    """Test ${VAR} expansion form"""
    monkeypatch.setenv("AFFINITY_T", str(tmp_path))
    
    profile_json = {
        "name": "test_profile",
        "platform": "linux",
        "llama_bin_dir": "bin",
        "model_path": "${AFFINITY_T}/m.gguf",
        "work_dir": "logs",
        "cpu": {"mode": "none", "cores": [], "priority": "normal", "threads": 1},
        "vram": {"target_mb": 100, "max_mb": 200}
    }
    
    profile_path = tmp_path / "profile.json"
    with open(profile_path, 'w') as f:
        json.dump(profile_json, f)
        
    profile = config.load_profile(str(profile_path))
    expected = tmp_path / "m.gguf"
    assert profile.model_path == expected

def test_env_var_expansion_dollar_bare_form(tmp_path, monkeypatch):
    """Test $VAR expansion form (both ${VAR} and $VAR should work)"""
    monkeypatch.setenv("AFFINITY_T", str(tmp_path))
    
    profile_json = {
        "name": "test_profile",
        "platform": "linux",
        "llama_bin_dir": "bin",
        "model_path": "$AFFINITY_T/m.gguf",
        "work_dir": "logs",
        "cpu": {"mode": "none", "cores": [], "priority": "normal", "threads": 1},
        "vram": {"target_mb": 100, "max_mb": 200}
    }
    
    profile_path = tmp_path / "profile.json"
    with open(profile_path, 'w') as f:
        json.dump(profile_json, f)
        
    profile = config.load_profile(str(profile_path))
    expected = tmp_path / "m.gguf"
    assert profile.model_path == expected

@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_env_var_percent_form_windows_only(tmp_path, monkeypatch):
    """Test %VAR% expansion (Windows only)"""
    monkeypatch.setenv("AFFINITY_T", str(tmp_path))
    
    profile_json = {
        "name": "test_profile",
        "platform": "linux",
        "llama_bin_dir": "bin",
        "model_path": "models/m.gguf",
        "work_dir": "%AFFINITY_T%/work",
        "cpu": {"mode": "none", "cores": [], "priority": "normal", "threads": 1},
        "vram": {"target_mb": 100, "max_mb": 200}
    }
    
    profile_path = tmp_path / "profile.json"
    with open(profile_path, 'w') as f:
        json.dump(profile_json, f)
        
    profile = config.load_profile(str(profile_path))
    expected = tmp_path / "work"
    assert profile.work_dir == expected

def test_relative_llama_bin_dir_resolved_to_repo_root(tmp_path, monkeypatch):
    """Test relative llama_bin_dir is resolved against REPO_ROOT"""
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    
    profile_json = {
        "name": "test_profile",
        "platform": "linux",
        "llama_bin_dir": "bin/llama-cpp",
        "model_path": "models/m.gguf",
        "work_dir": "logs",
        "cpu": {"mode": "none", "cores": [], "priority": "normal", "threads": 1},
        "vram": {"target_mb": 100, "max_mb": 200}
    }
    
    profile_path = tmp_path / "profile.json"
    with open(profile_path, 'w') as f:
        json.dump(profile_json, f)
        
    profile = config.load_profile(str(profile_path))
    expected = tmp_path / "bin" / "llama-cpp"
    assert profile.llama_bin_dir == expected

def test_relative_work_dir_resolved_to_repo_root(tmp_path, monkeypatch):
    """Test relative work_dir is resolved against REPO_ROOT"""
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    
    profile_json = {
        "name": "test_profile",
        "platform": "linux",
        "llama_bin_dir": "bin",
        "model_path": "models/m.gguf",
        "work_dir": "logs/affinity_sim",
        "cpu": {"mode": "none", "cores": [], "priority": "normal", "threads": 1},
        "vram": {"target_mb": 100, "max_mb": 200}
    }
    
    profile_path = tmp_path / "profile.json"
    with open(profile_path, 'w') as f:
        json.dump(profile_json, f)
        
    profile = config.load_profile(str(profile_path))
    expected = tmp_path / "logs" / "affinity_sim"
    assert profile.work_dir == expected

def test_bare_name_rejects_forward_slash(tmp_path, monkeypatch):
    """Test that "sub/a" is NOT treated as bare name"""
    # Create a valid file for the directory
    machine_dir = tmp_path / "config" / "machines"
    machine_dir.mkdir(parents=True)
    
    profile_json = {
        "name": "test_profile",
        "platform": "linux",
        "llama_bin_dir": "bin",
        "model_path": "models/m.gguf",
        "work_dir": "logs",
        "cpu": {"mode": "none", "cores": [], "priority": "normal", "threads": 1},
        "vram": {"target_mb": 100, "max_mb": 200}
    }
    
    valid_file = machine_dir / "sub" / "a.json"
    valid_file.parent.mkdir(parents=True, exist_ok=True)
    with open(valid_file, 'w') as f:
        json.dump(profile_json, f)
        
    # Try loading a path that contains forward slash
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError):
        config.load_profile("sub/a")

def test_bare_name_rejects_backslash(tmp_path, monkeypatch):
    """Test that "sub\a" (backslash form) is NOT treated as bare name"""
    # Create a valid file for the directory
    machine_dir = tmp_path / "config" / "machines"
    machine_dir.mkdir(parents=True)
    
    profile_json = {
        "name": "test_profile",
        "platform": "linux",
        "llama_bin_dir": "bin",
        "model_path": "models/m.gguf",
        "work_dir": "logs",
        "cpu": {"mode": "none", "cores": [], "priority": "normal", "threads": 1},
        "vram": {"target_mb": 100, "max_mb": 200}
    }
    
    valid_file = machine_dir / "sub" / "a.json"
    valid_file.parent.mkdir(parents=True, exist_ok=True)
    with open(valid_file, 'w') as f:
        json.dump(profile_json, f)
        
    # Try loading a path that contains backslash
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError):
        config.load_profile("sub" + chr(92) + "a")
