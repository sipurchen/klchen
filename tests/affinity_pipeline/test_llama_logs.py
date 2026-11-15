import pytest
from pathlib import Path
from affinity_pipeline.config import VramBudget
from affinity_pipeline.llama_logs import parse_buffer_lines, device_total_mb, check_budget, BufferReport, BudgetVerdict

def test_parse_b8679_vulkan():
    """Parse old Vulkan format log (no timestamps)."""
    text = """load_tensors:      Vulkan0 model buffer size =   915.42 MiB
llama_context: Vulkan_Host  output buffer size =     1.00 MiB
llama_kv_cache:    Vulkan0 KV buffer size =   128.00 MiB
llama_kv_cache:    Vulkan0 KV buffer size =    72.00 MiB
sched_reserve:    Vulkan0 compute buffer size =   517.50 MiB
load_tensors:      CPU_Mapped model buffer size = 16227.52 MiB
load_tensors: offloaded 9/31 layers to GPU"""
    
    report = parse_buffer_lines(text)
    assert report.device_mb["Vulkan0"] == pytest.approx(1632.92, abs=0.01)
    assert report.host_mb == pytest.approx(16227.52, abs=0.01)
    assert report.offloaded == (9, 31)
    assert report.complete is True

def test_parse_lv4_cuda():
    """Parse new CUDA format log (with timestamps)."""
    text = """0.06.103.480 I load_tensors:        CUDA0 model buffer size =   853.17 MiB
0.21.867.485 I llama_context:  CUDA_Host  output buffer size =     1.00 MiB
0.21.869.160 I llama_kv_cache:        CPU KV buffer size =    96.00 MiB
0.21.892.815 I llama_kv_cache:      CUDA0 KV buffer size =    64.00 MiB
0.21.893.261 I llama_kv_cache:      CUDA0 KV buffer size =    72.00 MiB
0.22.001.443 I sched_reserve:      CUDA0 compute buffer size =   347.63 MiB
0.06.103.491 I load_tensors: offloaded 9/31 layers to GPU"""
    
    report = parse_buffer_lines(text)
    assert report.device_mb["CUDA0"] == pytest.approx(1336.80, abs=0.01)
    assert report.host_mb == pytest.approx(96.00, abs=0.01)  # CPU KV buffer (CPU starts with "CPU" so is host); CUDA_Host output not counted
    assert report.offloaded == (9, 31)
    assert report.complete is True

def test_parse_empty():
    """Empty log returns default values."""
    report = parse_buffer_lines("")
    assert report.device_mb == {}
    assert report.host_mb == 0.0
    assert report.offloaded is None
    assert report.complete is False

def test_parse_no_compute_buffer():
    """Missing compute buffer line → complete=False."""
    text = """load_tensors:      Vulkan0 model buffer size =   915.42 MiB
llama_kv_cache:    Vulkan0 KV buffer size =   128.00 MiB"""
    
    report = parse_buffer_lines(text)
    assert report.device_mb["Vulkan0"] == pytest.approx(1043.42, abs=0.01)
    assert report.complete is False

def test_parse_multiple_devices():
    """Multiple devices on same log."""
    text = """load_tensors:      Vulkan0 model buffer size =   100.0 MiB
load_tensors:      Vulkan1 model buffer size =   200.0 MiB
sched_reserve:    Vulkan0 compute buffer size =    50.0 MiB
sched_reserve:    Vulkan1 compute buffer size =    75.0 MiB"""
    
    report = parse_buffer_lines(text)
    assert report.device_mb["Vulkan0"] == pytest.approx(150.0, abs=0.01)
    assert report.device_mb["Vulkan1"] == pytest.approx(275.0, abs=0.01)
    assert report.complete is True

def test_device_total_mb():
    """Sum of device buffers."""
    report = BufferReport(
        device_mb={"Vulkan0": 1000.0, "Vulkan1": 500.0},
        host_mb=2000.0,
        offloaded=(9, 31),
        complete=True
    )
    assert device_total_mb(report) == pytest.approx(1500.0, abs=0.01)

def test_device_total_mb_empty():
    """Empty device list."""
    report = BufferReport(device_mb={}, host_mb=100.0, offloaded=None, complete=False)
    assert device_total_mb(report) == 0.0

def test_check_budget_incomplete():
    """complete=False → INCOMPLETE."""
    report = BufferReport(device_mb={"GPU": 500.0}, host_mb=0.0, offloaded=None, complete=False)
    vram = VramBudget(target_mb=1000, max_mb=2000)
    assert check_budget(report, vram) == BudgetVerdict.INCOMPLETE

def test_check_budget_ok():
    """Within target → OK."""
    report = BufferReport(device_mb={"GPU": 500.0}, host_mb=0.0, offloaded=None, complete=True)
    vram = VramBudget(target_mb=1000, max_mb=2000)
    assert check_budget(report, vram) == BudgetVerdict.OK

def test_check_budget_warn():
    """Between target and max → WARN."""
    report = BufferReport(device_mb={"GPU": 1500.0}, host_mb=0.0, offloaded=None, complete=True)
    vram = VramBudget(target_mb=1000, max_mb=2000)
    assert check_budget(report, vram) == BudgetVerdict.WARN

def test_check_budget_abort():
    """Exceeds max → ABORT."""
    report = BufferReport(device_mb={"GPU": 2500.0}, host_mb=0.0, offloaded=None, complete=True)
    vram = VramBudget(target_mb=1000, max_mb=2000)
    assert check_budget(report, vram) == BudgetVerdict.ABORT

def test_check_budget_boundary_target():
    """Exactly at target → OK (not WARN)."""
    report = BufferReport(device_mb={"GPU": 1000.0}, host_mb=0.0, offloaded=None, complete=True)
    vram = VramBudget(target_mb=1000, max_mb=2000)
    assert check_budget(report, vram) == BudgetVerdict.OK

def test_check_budget_boundary_max():
    """Exactly at max → OK (not ABORT)."""
    report = BufferReport(device_mb={"GPU": 2000.0}, host_mb=0.0, offloaded=None, complete=True)
    vram = VramBudget(target_mb=1000, max_mb=2000)
    assert check_budget(report, vram) == BudgetVerdict.OK

def test_parse_offloaded_none():
    """No offloaded line → None."""
    text = """load_tensors:      Vulkan0 model buffer size =   915.42 MiB
sched_reserve:    Vulkan0 compute buffer size =   517.50 MiB"""
    
    report = parse_buffer_lines(text)
    assert report.offloaded is None
