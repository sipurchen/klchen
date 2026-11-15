import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

if sys.platform == "win32":
    import ctypes

from affinity_pipeline.config import MachineProfile, affinity_mask


def build_command(argv: Sequence[str], profile: MachineProfile) -> list[str]:
    """Build OS-specific command wrapper around argv using profile CPU limits.

    Rules:
    - If profile.platform == "linux" AND profile.cpu.mode == "taskset":
        Return ["taskset", "-c", "<comma-sep cores>", *prio, *argv]
        where prio depends on profile.cpu.priority:
          - "nice10" → ["nice", "-n", "10"]
          - "idle" → ["nice", "-n", "19"]
          - other ("normal", "below_normal") → []
    - If profile.platform == "windows" OR profile.cpu.mode == "none":
        Return list(argv) unchanged.
    - MUST NOT modify the input argv sequence.

    Args:
        argv: Command to wrap (e.g., [sys.executable, "-c", "print('hi')"])
        profile: MachineProfile with platform, cpu.mode, cpu.priority, cpu.cores

    Returns:
        Wrapped command as list[str].
    """
    if profile.platform == "windows" or profile.cpu.mode == "none":
        return list(argv)

    if profile.platform == "linux" and profile.cpu.mode == "taskset":
        cores_str = ",".join(str(c) for c in profile.cpu.cores)
        command = ["taskset", "-c", cores_str]

        # Add priority
        if profile.cpu.priority == "nice10":
            command.extend(["nice", "-n", "10"])
        elif profile.cpu.priority == "idle":
            command.extend(["nice", "-n", "19"])

        command.extend(argv)
        return command

    return list(argv)


def load_guard_ok(profile: MachineProfile) -> tuple[bool, str]:
    """Check if machine load allows starting new processes.

    Rules:
    - If profile.max_load1 is None → (True, "no load guard")
    - If profile.platform == "linux":
        Use os.getloadavg()[0] (1-minute average).
        If load > profile.max_load1 → (False, f"load {actual:.2f} > max {max}")
        Else → (True, "load ok")
    - If profile.platform == "windows":
        → (True, "n/a on windows")

    Args:
        profile: MachineProfile

    Returns:
        (ok: bool, message: str)
    """
    if profile.max_load1 is None:
        return True, "no load guard"

    if profile.platform == "linux":
        try:
            current_load = os.getloadavg()[0]
            if current_load > profile.max_load1:
                return False, f"load {current_load:.2f} > max {profile.max_load1}"
            else:
                return True, "load ok"
        except Exception:
            # If we can't get load average, assume it's okay
            return True, "load ok"
    else:
        return True, "n/a on windows"


class GuardedProcess:
    """Context manager for subprocess with CPU affinity, priority, timeout, and cleanup.

    Must be:
    - Usable as `with GuardedProcess(...) as proc: ...`
    - If __exit__ is called (normal or exception), stop() is invoked.
    - Exception in __exit__ must propagate (not swallowed).

    Attributes:
        pid: int | None (set after start())
        timed_out: bool (True iff timeout triggered while process was running)
    """

    def __init__(self, argv: Sequence[str], profile: MachineProfile, *,
                 cwd: str | Path | None = None,
                 env: dict[str, str] | None = None,
                 log_path: str | Path | None = None,
                 timeout_s: float | None = None):
        """Initialize (do not start process yet).

        Args:
            argv: Command + args (e.g., ["python", "-c", "..."])
            profile: MachineProfile with CPU/load settings
            cwd: Working directory for the process
            env: Environment variables (merged with os.environ if not None).
                 Semantics: actual env = {**os.environ, **env} if env is not None, else os.environ
            log_path: If not None, write stdout+stderr to this file (UTF-8, errors="replace")
            timeout_s: Kill process after this many seconds from start(). Timeout counting
                      starts when start() is called; if set, must trigger whether or not
                      wait() or __exit__ is called within the GuardedProcess lifetime.
        """
        self.argv = argv
        self.profile = profile
        self.cwd = cwd
        self.env = env
        self.log_path = log_path
        self.timeout_s = timeout_s

        self._proc = None
        self.pid = None
        self.timed_out = False
        self._timeout_timer = None
        self._log_file = None

    def start(self) -> "GuardedProcess":
        """Start the subprocess with affinity/priority applied.

        Rules:
        - Call load_guard_ok(profile); if not ok, raise RuntimeError with the message.
        - Call build_command() to wrap argv.
        - Launch subprocess.Popen.
        - On Windows: apply SetProcessAffinityMask and SetPriorityClass using ctypes.windll.
          Mask from affinity_mask(profile.cpu.cores).
          Priority: "below_normal"→0x4000, "idle"→0x40, "normal"→0x20.
          Check return values; on failure, call stop() and raise RuntimeError.
        - On Linux: use start_new_session=True so stop() can killpg.
        - On Windows: use subprocess.CREATE_NEW_PROCESS_GROUP.
        - If timeout_s is set, start timeout mechanism (Timer).
        - Store pid, set timed_out=False initially.
        - Return self.

        Raises:
            RuntimeError: On load guard fail or affinity/priority setup fail.
        """
        # Check load guard
        ok, message = load_guard_ok(self.profile)
        if not ok:
            raise RuntimeError(message)

        # Build command
        cmd = build_command(self.argv, self.profile)

        # Open log file if needed
        stdout_file = None
        stderr_file = None

        if self.log_path is not None:
            try:
                self._log_file = open(self.log_path, "w", encoding="utf-8", errors="replace")
                stdout_file = self._log_file
                stderr_file = self._log_file
            except Exception as e:
                raise RuntimeError(f"Failed to open log file {self.log_path}: {e}")

        # Start process
        try:
            if sys.platform == "win32":
                # Windows specific settings
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

                # Merge env properly: {**os.environ, **self.env}
                popen_env = {**os.environ}
                if self.env is not None:
                    popen_env.update(self.env)

                self._proc = subprocess.Popen(
                    cmd,
                    cwd=self.cwd,
                    env=popen_env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    creationflags=creationflags
                )
            else:
                # Linux specific settings
                # Merge env properly: {**os.environ, **self.env}
                popen_env = {**os.environ}
                if self.env is not None:
                    popen_env.update(self.env)

                self._proc = subprocess.Popen(
                    cmd,
                    cwd=self.cwd,
                    env=popen_env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True
                )

            self.pid = self._proc.pid

            # Apply affinity and priority on Windows (only if not "none" mode)
            if sys.platform == "win32" and self.profile.cpu.mode != "none":
                handle = ctypes.windll.kernel32.OpenProcess(0x1F0FFF, False, self.pid)
                if not handle:
                    raise RuntimeError("Failed to open process handle")

                mask = affinity_mask(self.profile.cpu.cores)
                success = ctypes.windll.kernel32.SetProcessAffinityMask(handle, mask)
                if not success:
                    self.stop()
                    ctypes.windll.kernel32.CloseHandle(handle)
                    raise RuntimeError("Failed to set process affinity mask")

                # Set priority
                priority_map = {
                    "below_normal": 0x4000,
                    "idle": 0x40,
                    "normal": 0x20
                }

                priority = priority_map.get(self.profile.cpu.priority, 0x20)
                success = ctypes.windll.kernel32.SetPriorityClass(handle, priority)
                if not success:
                    self.stop()
                    ctypes.windll.kernel32.CloseHandle(handle)
                    raise RuntimeError("Failed to set process priority")

                ctypes.windll.kernel32.CloseHandle(handle)

            # Start timeout mechanism
            if self.timeout_s is not None:
                def kill_process():
                    # Only set timed_out if the process is still running at timeout time
                    if self.poll() is None:
                        self.timed_out = True
                        self.stop()

                self._timeout_timer = threading.Timer(self.timeout_s, kill_process)
                self._timeout_timer.daemon = True
                self._timeout_timer.start()

        except Exception as e:
            if self._log_file:
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None
            raise RuntimeError(f"Failed to start process: {e}")

        return self

    def poll(self) -> int | None:
        """Check if process has terminated, return exit code or None if still running."""
        if self._proc is not None:
            return self._proc.poll()
        return None

    def wait(self, timeout: float | None = None) -> int:
        """Wait for process to finish, return exit code.

        Args:
            timeout: Max seconds to wait; if process still running after, raise subprocess.TimeoutExpired.

        Returns:
            Exit code (int).

        Critical: Cancel the timeout timer after process exits to prevent timed_out flag from
        being set incorrectly after a normal exit.
        """
        if self._proc is not None:
            try:
                rc = self._proc.wait(timeout=timeout)
                # CRITICAL: Cancel timer immediately after process exits to prevent
                # the kill_process callback from executing after normal exit
                if self._timeout_timer is not None:
                    self._timeout_timer.cancel()
                return rc
            except subprocess.TimeoutExpired:
                raise
        else:
            raise RuntimeError("Process not started")

    def stop(self, grace_s: float = 5.0) -> None:
        """Terminate the process tree and clean up (idempotent, no exception on error).

        Rules:
        - Check if process still running; if not, return.
        - On Linux: use os.killpg(os.getpgid(pid), signal.SIGTERM), wait grace_s, then SIGKILL if needed.
        - On Windows: use subprocess.run([taskkill /PID pid /T /F]) to kill the tree.
        - Close log file if open.
        - Set pid=None to indicate cleanup done.
        - This method is idempotent: calling it multiple times is safe.

        Args:
            grace_s: Seconds to wait after SIGTERM before SIGKILL (Linux only).
        """
        if self._proc is None or self.pid is None:
            return

        try:
            # Close log file
            if self._log_file:
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None

            # Cancel timeout timer
            if self._timeout_timer is not None:
                self._timeout_timer.cancel()

            if sys.platform == "win32":
                # Windows process tree termination
                subprocess.run(["taskkill", "/PID", str(self.pid), "/T", "/F"],
                               capture_output=True, check=False)
            else:
                # Linux process group termination
                try:
                    pgid = os.getpgid(self.pid)
                    os.killpg(pgid, signal.SIGTERM)

                    # Wait for grace period
                    start_time = time.time()
                    while self.poll() is None and (time.time() - start_time) < grace_s:
                        time.sleep(0.1)

                    # Force kill if still running
                    if self.poll() is None:
                        os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    # Already terminated
                    pass

        except Exception:
            # Ignore errors during cleanup
            pass

        finally:
            self.pid = None

    def __enter__(self) -> "GuardedProcess":
        """Context manager entry: call start()."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: call stop() and propagate any exception."""
        self.stop()
        # Do NOT return True (would swallow exceptions); let them propagate.
