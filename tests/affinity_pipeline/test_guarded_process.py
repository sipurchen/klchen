import os
import sys
import tempfile
import time
import signal
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from affinity_pipeline.config import CpuLimit, MachineProfile, VramBudget
from affinity_pipeline.guarded_process import (
    GuardedProcess,
    build_command,
    load_guard_ok,
)

if sys.platform == "win32":
    import ctypes


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def windows_profile():
    """Windows profile with affinity_mask mode (for testing SetProcessAffinityMask)."""
    return MachineProfile(
        name="test_windows",
        platform="windows",
        llama_bin_dir=Path("C:\\bin"),
        model_path=Path("C:\\model.gguf"),
        work_dir=Path("C:\\tmp"),
        cpu=CpuLimit(mode="affinity_mask", cores=(2, 3), priority="below_normal", threads=2),
        vram=VramBudget(target_mb=1000, max_mb=2000),
        max_load1=None,
    )


@pytest.fixture
def linux_profile():
    """Linux profile with taskset mode using actual available cores."""
    available_cores = os.sched_getaffinity(0)
    cores = tuple(sorted(available_cores))[:2]
    return MachineProfile(
        name="test_linux",
        platform="linux",
        llama_bin_dir=Path("/bin"),
        model_path=Path("/model.gguf"),
        work_dir=Path("/tmp"),
        cpu=CpuLimit(mode="taskset", cores=cores, priority="nice10", threads=2),
        vram=VramBudget(target_mb=1000, max_mb=2000),
        max_load1=None,
    )


@pytest.fixture
def no_limit_profile():
    """Cross-platform profile with mode="none" (no CPU limits)."""
    return MachineProfile(
        name="test_none",
        platform="windows" if sys.platform == "win32" else "linux",
        llama_bin_dir=Path("C:\\bin" if sys.platform == "win32" else "/bin"),
        model_path=Path("C:\\model.gguf" if sys.platform == "win32" else "/model.gguf"),
        work_dir=Path("C:\\tmp" if sys.platform == "win32" else "/tmp"),
        cpu=CpuLimit(mode="none", cores=(), priority="normal", threads=1),
        vram=VramBudget(target_mb=1000, max_mb=2000),
        max_load1=None,
    )


# ============================================================================
# Helper for process liveness check (T1 fix)
# ============================================================================

def _is_alive(pid: int) -> bool:
    """Check if process with given pid is still alive (not zombie).
    - Windows: GetExitCodeProcess returns STILL_ACTIVE (259)
    - Linux: /proc/<pid>/stat exists and state != 'Z'
    """
    if sys.platform == "win32":
        try:
            handle = ctypes.windll.kernel32.OpenProcess(0x400, False, pid)
            if handle == 0:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return exit_code.value == 259  # STILL_ACTIVE
                return False
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False
    else:
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                fields = f.read().split()
                state = fields[2]
                return state != 'Z'
        except (FileNotFoundError, IndexError, OSError):
            return False


# ============================================================================
# Test build_command
# ============================================================================

def test_build_command_windows_unchanged(windows_profile):
    """build_command on Windows returns argv unchanged."""
    argv = [sys.executable, "-c", "print('hi')"]
    result = build_command(argv, windows_profile)
    assert result == list(argv)


def test_build_command_none_mode(no_limit_profile):
    """build_command with mode='none' returns argv unchanged."""
    argv = [sys.executable, "-c", "print('hi')"]
    result = build_command(argv, no_limit_profile)
    assert result == list(argv)


def test_build_command_linux_taskset_nice10():
    """build_command on Linux with taskset and nice10."""
    profile = MachineProfile(
        name="test_linux",
        platform="linux",
        llama_bin_dir=Path("/bin"),
        model_path=Path("/model.gguf"),
        work_dir=Path("/tmp"),
        cpu=CpuLimit(mode="taskset", cores=(0, 1), priority="nice10", threads=2),
        vram=VramBudget(target_mb=1000, max_mb=2000),
        max_load1=None,
    )
    argv = [sys.executable, "-c", "print('test')"]
    result = build_command(argv, profile)
    assert result[0] == "taskset"
    assert result[1] == "-c"
    assert result[2] == "0,1"
    assert result[3:5] == ["nice", "-n"]
    assert result[5] == "10"
    assert result[6:] == argv


def test_build_command_linux_taskset_idle():
    """build_command on Linux with taskset and idle priority."""
    profile = MachineProfile(
        name="test_linux",
        platform="linux",
        llama_bin_dir=Path("/bin"),
        model_path=Path("/model.gguf"),
        work_dir=Path("/tmp"),
        cpu=CpuLimit(mode="taskset", cores=(0, 1), priority="idle", threads=2),
        vram=VramBudget(target_mb=1000, max_mb=2000),
        max_load1=None,
    )
    argv = [sys.executable, "-c", "print('test')"]
    result = build_command(argv, profile)
    assert result[0] == "taskset"
    assert result[5] == "19"
    assert result[6:] == argv


# ============================================================================
# Test load_guard_ok
# ============================================================================

def test_load_guard_ok_none(no_limit_profile):
    """load_guard_ok returns True when max_load1 is None."""
    ok, msg = load_guard_ok(no_limit_profile)
    assert ok is True
    assert "no load guard" in msg.lower()


def test_load_guard_ok_linux_below_threshold(monkeypatch):
    """load_guard_ok on Linux returns True when load is below threshold."""
    profile = MachineProfile(
        name="test_linux",
        platform="linux",
        llama_bin_dir=Path("/bin"),
        model_path=Path("/model.gguf"),
        work_dir=Path("/tmp"),
        cpu=CpuLimit(mode="taskset", cores=(0,), priority="normal", threads=1),
        vram=VramBudget(target_mb=1000, max_mb=2000),
        max_load1=2.0,
    )
    monkeypatch.setattr(os, "getloadavg", lambda: (1.0, 0.5, 0.5), raising=False)
    ok, msg = load_guard_ok(profile)
    assert ok is True


def test_load_guard_ok_linux_above_threshold(monkeypatch):
    """load_guard_ok on Linux returns False when load exceeds threshold."""
    profile = MachineProfile(
        name="test_linux",
        platform="linux",
        llama_bin_dir=Path("/bin"),
        model_path=Path("/model.gguf"),
        work_dir=Path("/tmp"),
        cpu=CpuLimit(mode="taskset", cores=(0,), priority="normal", threads=1),
        vram=VramBudget(target_mb=1000, max_mb=2000),
        max_load1=2.0,
    )
    monkeypatch.setattr(os, "getloadavg", lambda: (3.0, 2.5, 2.5), raising=False)
    ok, msg = load_guard_ok(profile)
    assert ok is False
    assert "3.0" in msg


@pytest.mark.skipif(sys.platform != "win32", reason="Windows platform only")
def test_load_guard_ok_windows(windows_profile):
    """load_guard_ok on Windows returns True (no load average available)."""
    profile_with_load = replace(windows_profile, max_load1=2.0)
    ok, msg = load_guard_ok(profile_with_load)
    assert ok is True
    assert "n/a" in msg.lower()


# ============================================================================
# Test real GuardedProcess (real subprocess)
# ============================================================================

@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
def test_guarded_process_affinity_windows(windows_profile):
    """GuardedProcess applies Windows affinity mask and priority correctly."""
    try:
        with GuardedProcess([sys.executable, "-c", "import time; time.sleep(5)"], windows_profile) as gp:
            pid = gp.pid
            assert pid is not None

            handle = ctypes.windll.kernel32.OpenProcess(0x400, False, pid)
            assert handle != 0
            try:
                affinity_mask = ctypes.c_ulong()
                if ctypes.windll.kernel32.GetProcessAffinityMask(handle, ctypes.byref(affinity_mask), ctypes.byref(ctypes.c_ulong())):
                    assert affinity_mask.value & 0xC == 0xC
                priority = ctypes.windll.kernel32.GetPriorityClass(handle)
                assert priority == 0x4000
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
    finally:
        pass


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-specific test")
def test_guarded_process_affinity_linux(linux_profile):
    """GuardedProcess applies Linux taskset correctly."""
    try:
        with GuardedProcess([sys.executable, "-c", "import time; time.sleep(5)"], linux_profile) as gp:
            pid = gp.pid
            assert pid is not None
            affinity = os.sched_getaffinity(pid)
            expected = set(linux_profile.cpu.cores)
            assert affinity == expected
    finally:
        pass


def test_guarded_process_context_manager_cleanup(no_limit_profile):
    """GuardedProcess cleanup on context manager exit."""
    gp = None
    try:
        with GuardedProcess([sys.executable, "-c", "import time; time.sleep(10)"], no_limit_profile) as gp:
            pid = gp.pid
            assert pid is not None
        assert gp.pid is None or not _is_alive(gp.pid)
    finally:
        if gp:
            gp.stop()


def test_guarded_process_exception_cleanup(no_limit_profile):
    """GuardedProcess cleans up even when exception occurs."""
    gp = None
    pid_before_exc = None
    try:
        with GuardedProcess([sys.executable, "-c", "import time; time.sleep(10)"], no_limit_profile) as gp:
            pid_before_exc = gp.pid
            raise ValueError("test exception")
    except ValueError:
        pass
    finally:
        if gp:
            assert gp.pid is None or not _is_alive(gp.pid)


def test_guarded_process_stop_idempotent(no_limit_profile):
    """GuardedProcess.stop() can be called multiple times."""
    gp = GuardedProcess([sys.executable, "-c", "import time; time.sleep(10)"], no_limit_profile)
    gp.start()
    pid = gp.pid
    assert pid is not None
    gp.stop()
    assert gp.pid is None
    gp.stop()
    assert gp.pid is None
    gp.stop()
    assert gp.pid is None


# ============================================================================
# T1: Test timeout flag (fixed from r2 feedback)
# ============================================================================

def test_guarded_process_p3_timeout_flag(no_limit_profile):
    """Test that timeout flag is set correctly and process is dead after timeout."""
    gp = GuardedProcess(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        no_limit_profile,
        timeout_s=0.5
    )
    gp.start()

    pid_at_start = gp.pid
    assert pid_at_start is not None
    assert _is_alive(pid_at_start) is True

    time.sleep(1.5)

    assert _is_alive(pid_at_start) is False
    assert gp.timed_out is True


# ============================================================================
# T2: Test invalid log path (fixed from r2 feedback)
# ============================================================================

def test_guarded_process_log_file_invalid_path(tmp_path, no_limit_profile):
    """Log file with non-existent parent directory should raise RuntimeError."""
    log_file = tmp_path / "no_such_dir" / "log.txt"

    with pytest.raises(RuntimeError):
        with GuardedProcess([sys.executable, "-c", "print('hi')"], no_limit_profile, log_path=log_file) as gp:
            pass


# ============================================================================
# T3: Test timed_out protection (new test from r2 feedback)
# ============================================================================

def test_guarded_process_timeout_not_triggered_on_quick_exit(no_limit_profile):
    """timed_out should be False if process exits before timeout fires."""
    gp = GuardedProcess(
        [sys.executable, "-c", "print('x')"],
        no_limit_profile,
        timeout_s=1
    )
    gp.start()

    time.sleep(2.5)

    assert gp.timed_out is False
    assert gp.poll() is not None

    gp.stop()


# ============================================================================
# T4: Test Windows idle priority (new test from r2 feedback)
# ============================================================================

@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific priority test")
def test_guarded_process_windows_priority_idle():
    """Windows idle priority (0x40) should be set and readable."""
    profile = MachineProfile(
        name="test_idle",
        platform="windows",
        llama_bin_dir=Path("C:\\bin"),
        model_path=Path("C:\\model.gguf"),
        work_dir=Path("C:\\tmp"),
        cpu=CpuLimit(mode="affinity_mask", cores=(0,), priority="idle", threads=1),
        vram=VramBudget(target_mb=1000, max_mb=2000),
        max_load1=None,
    )

    with GuardedProcess([sys.executable, "-c", "import time; time.sleep(5)"], profile) as gp:
        pid = gp.pid

        handle = ctypes.windll.kernel32.OpenProcess(0x400, False, pid)
        assert handle != 0
        try:
            priority_class = ctypes.windll.kernel32.GetPriorityClass(handle)
            assert priority_class == 0x40
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)


# ============================================================================
# Acceptor additions (S1 round-3 takeover): mutants G2/G5/G6/G14 survived the Coder/Reviewer suite
# ============================================================================

def test_load_guard_boundary_equal_is_ok(monkeypatch):
    """load == max_load1 must pass (guard is strictly '>'); runs on every OS (platform comes from the profile)."""
    profile = MachineProfile(
        name="t", platform="linux", llama_bin_dir=Path("x"), model_path=Path("m"), work_dir=Path("w"),
        cpu=CpuLimit(mode="none", cores=(), priority="normal", threads=1),
        vram=VramBudget(target_mb=1, max_mb=2), max_load1=2.0)
    monkeypatch.setattr(os, "getloadavg", lambda: (2.0, 0.0, 0.0), raising=False)
    ok, _ = load_guard_ok(profile)
    assert ok is True
    monkeypatch.setattr(os, "getloadavg", lambda: (2.01, 0.0, 0.0), raising=False)
    ok, msg = load_guard_ok(profile)
    assert ok is False and "2.01" in msg


def test_start_refused_when_load_guard_fails(monkeypatch, no_limit_profile):
    profile = replace(no_limit_profile, platform="linux", max_load1=1.0)
    monkeypatch.setattr(os, "getloadavg", lambda: (5.0, 0.0, 0.0), raising=False)
    gp = GuardedProcess([sys.executable, "-c", "print(1)"], profile)
    with pytest.raises(RuntimeError, match="5.00"):
        gp.start()
    assert gp.pid is None


def test_env_is_merged_over_os_environ(no_limit_profile, tmp_path, monkeypatch):
    """Child sees BOTH a variable inherited from os.environ and the overriding/added ones (G5: env replaced, not merged)."""
    monkeypatch.setenv("S1_INHERITED", "from_parent")
    monkeypatch.setenv("S1_OVERRIDDEN", "old")
    log = tmp_path / "env.log"
    code = ("import os; print(os.environ.get('S1_INHERITED'), os.environ.get('S1_OVERRIDDEN'), "
            "os.environ.get('S1_ADDED'), 'PATH' in os.environ or 'Path' in os.environ)")
    with GuardedProcess([sys.executable, "-c", code], no_limit_profile, log_path=log,
                        env={"S1_OVERRIDDEN": "new", "S1_ADDED": "x"}) as gp:
        gp.wait(timeout=30)
    assert log.read_text(encoding="utf-8").split() == ["from_parent", "new", "x", "True"]


def test_stop_kills_grandchild_process_tree(no_limit_profile, tmp_path):
    """stop() must kill the whole tree, not just the direct child (G2: taskkill without /T; Linux killpg)."""
    pidfile = tmp_path / "grandchild.pid"
    parent_code = "; ".join([
        "import subprocess, sys, time",
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
        f"open({str(pidfile)!r}, 'w').write(str(g.pid))",
        "time.sleep(60)",
    ])
    gp = GuardedProcess([sys.executable, "-c", parent_code], no_limit_profile)
    gp.start()
    grandchild = None
    try:
        deadline = time.time() + 20
        while time.time() < deadline and not (pidfile.exists() and pidfile.read_text().strip()):
            time.sleep(0.1)
        grandchild = int(pidfile.read_text().strip())
        assert _is_alive(grandchild) is True  # positive control: the grandchild really runs
        gp.stop()
        deadline = time.time() + 10
        while time.time() < deadline and _is_alive(grandchild):
            time.sleep(0.1)
        assert _is_alive(grandchild) is False
    finally:
        gp.stop()
        if grandchild is not None and _is_alive(grandchild):  # never leave a stray process behind
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(grandchild), "/F"], capture_output=True)
            else:
                os.kill(grandchild, signal.SIGKILL)


def test_stop_cancels_timeout_timer(no_limit_profile):
    """White-box: after stop() the timer must be cancelled immediately (G14). Behaviourally the callback is
    guarded by poll(), so only Timer.finished (set by cancel()) distinguishes 'cancelled' from 'still pending'."""
    gp = GuardedProcess([sys.executable, "-c", "import time; time.sleep(30)"], no_limit_profile, timeout_s=30)
    gp.start()
    timer = gp._timeout_timer
    assert timer is not None and not timer.finished.is_set()
    gp.stop()
    assert timer.finished.is_set()
    assert gp.timed_out is False
