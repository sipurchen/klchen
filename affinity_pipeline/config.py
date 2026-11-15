"""Machine configuration layer implementation."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class CpuLimit:
    mode: str            # "affinity_mask" (Windows) | "taskset" (Linux) | "none"
    cores: tuple[int, ...]
    priority: str        # "below_normal" | "idle" | "nice10" | "normal"
    threads: int

@dataclass(frozen=True)
class VramBudget:
    target_mb: int
    max_mb: int

@dataclass(frozen=True)
class MachineProfile:
    name: str
    platform: str        # "windows" | "linux"
    llama_bin_dir: Path
    model_path: Path
    work_dir: Path
    cpu: CpuLimit
    vram: VramBudget
    max_load1: float | None   # 1-minute load average threshold; None = no limit

def _resolve_path(path_str: str) -> Path:
    """Resolve a path string to an absolute Path, expanding ~ and environment variables."""
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    if not os.path.isabs(expanded):
        return REPO_ROOT / expanded
    return Path(expanded)

def _validate_platform_specific_cpu_mode(cpu: CpuLimit, platform: str, file_path: Path) -> None:
    """Validate CPU mode based on platform."""
    if cpu.mode == "affinity_mask" and platform != "windows":
        raise ValueError(f"{file_path}: cpu.mode '{cpu.mode}' not valid on platform '{platform}'")
    if cpu.mode == "taskset" and platform != "linux":
        raise ValueError(f"{file_path}: cpu.mode '{cpu.mode}' not valid on platform '{platform}'")
    if cpu.mode != "none" and len(cpu.cores) == 0:
        raise ValueError(f"{file_path}: cpu.cores must not be empty when mode is not 'none'")
    if any(core < 0 for core in cpu.cores):
        raise ValueError(f"{file_path}: cpu.cores: all cores must be >= 0")
    if len(cpu.cores) != len(set(cpu.cores)):
        raise ValueError(f"{file_path}: cpu.cores: must not contain duplicates")

def _validate_cpu_limit(cpu: CpuLimit, file_path: Path) -> None:
    """Validate CPU limit fields."""
    if cpu.threads < 1:
        raise ValueError(f"{file_path}: cpu.threads must be >= 1")
    if cpu.mode not in ("affinity_mask", "taskset", "none"):
        raise ValueError(f"{file_path}: cpu.mode '{cpu.mode}' is not valid")

def _validate_vram(vram: VramBudget, file_path: Path) -> None:
    """Validate VRAM budget."""
    if vram.target_mb > vram.max_mb:
        raise ValueError(f"{file_path}: vram.target_mb ({vram.target_mb}) > vram.max_mb ({vram.max_mb})")

def load_profile(source: str | os.PathLike) -> MachineProfile:
    """Load a machine profile from JSON.
    
    Args:
        source: Either a bare name (no path separators, no .json suffix) like "gt1030_win",
                which is resolved to REPO_ROOT / "config" / "machines" / f"{name}.json",
                or a full file path.
    
    Returns:
        A fully validated MachineProfile instance.
    
    Raises:
        ValueError: On validation failure (see Rules section below).
    """
    if isinstance(source, str) and '/' not in source and '\\' not in source and not source.endswith('.json'):
        # Bare name case
        file_path = REPO_ROOT / "config" / "machines" / f"{source}.json"
    else:
        file_path = Path(source)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Profile file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")

    # Extract required fields
    required_fields = ['name', 'platform', 'llama_bin_dir', 'model_path', 'work_dir', 'cpu', 'vram']
    missing_fields = [field for field in required_fields if field not in raw_data]
    if missing_fields:
        raise ValueError(f"Missing required keys in {file_path}: {missing_fields}")
    
    # Validate platform
    platform = raw_data['platform']
    if platform not in ('windows', 'linux'):
        raise ValueError(f"{file_path}: platform must be 'windows' or 'linux', got '{platform}'")
    
    # Resolve paths
    llama_bin_dir = _resolve_path(raw_data['llama_bin_dir'])
    model_path = _resolve_path(raw_data['model_path'])
    work_dir = _resolve_path(raw_data['work_dir'])
    
    # Parse CPU limit
    cpu_data = raw_data['cpu']
    required_cpu_fields = ['mode', 'cores', 'priority', 'threads']
    missing_cpu_fields = [field for field in required_cpu_fields if field not in cpu_data]

    # Parse VRAM budget
    vram_data = raw_data['vram']
    required_vram_fields = ['target_mb', 'max_mb']
    missing_vram_fields = [field for field in required_vram_fields if field not in vram_data]

    # Check all missing nested keys at once (V2 contract)
    all_missing = []
    all_missing.extend([f"cpu.{field}" for field in missing_cpu_fields])
    all_missing.extend([f"vram.{field}" for field in missing_vram_fields])
    if all_missing:
        raise ValueError(f"Missing required keys in {file_path}: {', '.join(all_missing)}")

    cpu_cores = tuple(cpu_data.get('cores', []))
    cpu = CpuLimit(
        mode=cpu_data['mode'],
        cores=cpu_cores,
        priority=cpu_data['priority'],
        threads=cpu_data['threads']
    )

    # Validate CPU
    _validate_cpu_limit(cpu, file_path)
    _validate_platform_specific_cpu_mode(cpu, platform, file_path)
    
    vram = VramBudget(
        target_mb=vram_data['target_mb'],
        max_mb=vram_data['max_mb']
    )

    # Validate VRAM
    _validate_vram(vram, file_path)
    
    # Parse max_load1
    max_load1 = raw_data.get('max_load1')
    
    return MachineProfile(
        name=raw_data['name'],
        platform=platform,
        llama_bin_dir=llama_bin_dir,
        model_path=model_path,
        work_dir=work_dir,
        cpu=cpu,
        vram=vram,
        max_load1=max_load1
    )

def resolve_binary(profile: MachineProfile, name: str) -> Path:
    """Resolve a binary name to its platform-specific path (no existence check).
    
    Args:
        name: Binary name, e.g., "llama-server"
    
    Returns:
        Windows: profile.llama_bin_dir / "llama-server.exe"
        Linux:   profile.llama_bin_dir / "llama-server"
    
    Raises:
        ValueError: If name contains path separators.
    """
    if '/' in name or '\\' in name:
        raise ValueError(f"Binary name '{name}' contains path separators")
    
    if profile.platform == "windows":
        return profile.llama_bin_dir / f"{name}.exe"
    else:
        return profile.llama_bin_dir / name

def affinity_mask(cores: tuple[int, ...]) -> int:
    """Convert core indices to Windows affinity bitmask.
    
    Examples:
        affinity_mask((2, 3)) == 12
        affinity_mask((0, 1, 4)) == 19
    
    Raises:
        ValueError: On empty tuple or negative core indices.
    """
    if len(cores) == 0:
        raise ValueError("Core list must not be empty")
    if any(core < 0 for core in cores):
        raise ValueError("All cores must be >= 0")
    
    mask = 0
    for core in cores:
        mask |= (1 << core)
    return mask
