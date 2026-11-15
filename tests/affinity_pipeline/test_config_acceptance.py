"""Acceptor-added tests (S0 round-3 takeover). Each closes a behaviour that independent mutation testing showed was
unprotected after three slice rounds: `~` expansion, `.json`-suffix sources treated as paths, taskset rejected on
windows, and invalid-JSON error handling. Written by the acceptor, not by the Coder/Reviewer pair."""
import json
from pathlib import Path

import pytest

from affinity_pipeline import config
from affinity_pipeline.config import load_profile


def _profile(**overrides) -> dict:
    data = {
        "name": "acc", "platform": "windows", "llama_bin_dir": "bin", "model_path": "m.gguf", "work_dir": "w",
        "cpu": {"mode": "affinity_mask", "cores": [2, 3], "priority": "below_normal", "threads": 2},
        "vram": {"target_mb": 1, "max_mb": 2},
    }
    data.update(overrides)
    return data


def _write(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_tilde_expands_to_home(tmp_path, monkeypatch):
    # expanduser reads HOME on POSIX and USERPROFILE on Windows; set both so the test is platform-independent
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    prof = load_profile(_write(tmp_path / "p.json", _profile(model_path="~/models/m.gguf", llama_bin_dir="~/bin")))
    assert prof.model_path == tmp_path / "models" / "m.gguf"
    assert prof.llama_bin_dir == tmp_path / "bin"


def test_taskset_rejected_on_windows(tmp_path):
    data = _profile(cpu={"mode": "taskset", "cores": [0], "priority": "nice10", "threads": 1})
    path = _write(tmp_path / "w.json", data)
    with pytest.raises(ValueError) as exc:
        load_profile(path)
    assert "cpu.mode" in str(exc.value) and "w.json" in str(exc.value)


def test_source_ending_in_json_without_separator_is_a_path_not_a_bare_name(tmp_path, monkeypatch):
    # "p.json" has no path separator but DOES have a .json suffix -> contract rule 1: it is a file path
    # (resolved against the cwd), never a bare name looked up under REPO_ROOT/config/machines/.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write(cwd / "p.json", _profile(name="from-cwd"))
    fake_root = tmp_path / "root"
    (fake_root / "config" / "machines").mkdir(parents=True)
    monkeypatch.setattr(config, "REPO_ROOT", fake_root)
    monkeypatch.chdir(cwd)
    assert load_profile("p.json").name == "from-cwd"


def test_invalid_json_raises_value_error_naming_the_file(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_profile(bad)
    assert "broken.json" in str(exc.value)
    assert "Invalid JSON" in str(exc.value)
