from dataclasses import dataclass
from enum import Enum
import re

@dataclass(frozen=True)
class BufferReport:
    device_mb: dict[str, float]     # per-device buffer sum: {"Vulkan0": 1632.92}
    host_mb: float                  # sum of all _Host and CPU* device buffers (reference only)
    offloaded: tuple[int, int] | None   # from "offloaded N/M layers to GPU", else None
    complete: bool                  # True iff compute buffer line seen for (non-host) device

class BudgetVerdict(str, Enum):  # values are literal strings
    OK = "ok"
    WARN = "warn"
    ABORT = "abort"
    INCOMPLETE = "incomplete"

def parse_buffer_lines(text: str) -> BufferReport:
    """Parse llama.cpp buffer output log, extract device/host memory allocations.
    
    Rules:
    1. Sum only three buffer types: "model buffer size", "KV buffer size", "compute buffer size" (MiB, may have decimals).
       Output buffer size is NOT counted.
    2. Device name = token before "buffer size" on that line; may have leading whitespace/timestamp/tag prefix.
       Names ending in _Host or starting with CPU are *host* (CUDA_Host, CPU_Mapped, CPU, etc.); others are *devices*.
    3. Same device may have multiple KV lines (SWA + non-SWA) → sum all.
    4. If no buffer lines exist → device_mb={}, complete=False, offloaded=None.
    5. Parse "offloaded N/M layers to GPU" → (N, M) tuple; missing → None.
    
    Args:
        text: Raw log text.
    
    Returns:
        BufferReport with device sums, host sum, offloaded tuple, and complete flag.
    
    Raises:
        None (never). Even on empty input, return valid default.
    """
    # Initialize result components
    device_mb = {}
    host_mb = 0.0
    offloaded = None
    complete = False

    # Pattern to match buffer lines: handles both old format (no timestamp/I) and new format (with timestamp and I)
    # Matches: "load_tensors: device_name buffer_type buffer size = NNN.NN MiB"
    buffer_pattern = re.compile(r'(?:load_tensors|llama_context|llama_kv_cache|sched_reserve):\s+(\S+)\s+(model|KV|compute)\s+buffer\s+size\s*=\s*([\d.]+)\s+MiB')

    # Pattern to match offloaded line
    offloaded_pattern = re.compile(r'offloaded (\d+)/(\d+) layers to GPU')
    
    for line in text.splitlines():
        # Match buffer lines (use search, not match, to handle timestamps/prefixes)
        buffer_match = buffer_pattern.search(line)
        if buffer_match:
            device_name = buffer_match.group(1)
            buffer_type = buffer_match.group(2)
            size_mb = float(buffer_match.group(3))
            
            # Determine if it's a host or device buffer
            is_host = device_name.endswith("_Host") or device_name.startswith("CPU")

            if buffer_type in ("model", "KV", "compute"):
                if is_host:
                    host_mb += size_mb
                else:
                    # Add to device sum
                    if device_name not in device_mb:
                        device_mb[device_name] = 0.0
                    device_mb[device_name] += size_mb

                    # If compute buffer, mark as complete
                    if buffer_type == "compute":
                        complete = True
                        
        # Match offloaded line (use search, not match, to handle prefixes)
        offloaded_match = offloaded_pattern.search(line)
        if offloaded_match:
            offloaded = (int(offloaded_match.group(1)), int(offloaded_match.group(2)))
    
    return BufferReport(device_mb=device_mb, host_mb=host_mb, offloaded=offloaded, complete=complete)

def device_total_mb(report: BufferReport) -> float:
    """Sum all non-host device buffers in report.
    
    Returns:
        Sum of all values in report.device_mb. If empty, 0.0.
    """
    return sum(report.device_mb.values())

def check_budget(report: BufferReport, vram: "VramBudget") -> BudgetVerdict:
    """Verdict on VRAM usage vs budget.

    Rules:
    - complete=False → INCOMPLETE
    - device_total_mb > vram.max_mb → ABORT (strict >; == is OK)
    - device_total_mb > vram.target_mb AND device_total_mb < vram.max_mb → WARN (both endpoints excluded)
    - else → OK

    Args:
        report: BufferReport from parse_buffer_lines.
        vram: VramBudget with target_mb and max_mb fields.

    Returns:
        One of the four BudgetVerdict strings.
    """
    if not report.complete:
        return BudgetVerdict.INCOMPLETE

    total = device_total_mb(report)

    if total > vram.max_mb:
        return BudgetVerdict.ABORT
    elif total > vram.target_mb and total < vram.max_mb:
        return BudgetVerdict.WARN
    else:
        return BudgetVerdict.OK
