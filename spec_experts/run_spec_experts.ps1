# run_spec_experts.ps1
# Start Spec-Experts controller session
# Standard env: Chrome+Edge closed, LINE open
# Auto-shuts down after 2 hours idle (or via /shutdown endpoint)

param(
    [switch]$SkipEnvCheck,
    [switch]$NoScheduler
)

chcp 65001 | Out-Null
$ErrorActionPreference = "Stop"
$ProjectDir = "\path\to\project"
$PythonExe  = "python"

function Write-Step { param($n, $msg) Write-Host "[$n] $msg" -ForegroundColor Cyan }
function Write-OK   { param($msg) Write-Host "    OK: $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "    WARN: $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "    FAIL: $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "===== Spec-Experts LLM Controller =====" -ForegroundColor Magenta
Write-Host "  GT 1030 850MB standard | auto-shutdown 2h idle"
Write-Host ""

# ── Step 1: env check ─────────────────────────────────────────────────────────
Write-Step 1 "Environment check"
if (-not $SkipEnvCheck) {
    $result = & $PythonExe "$ProjectDir\scripts\precheck_env.py" 2>&1
    Write-Host $result
    if ($LASTEXITCODE -eq 2) {
        Write-Fail "VRAM too low. Close Chrome and Edge first, then re-run."
        exit 1
    }
} else {
    Write-Warn "Env check skipped (-SkipEnvCheck)"
}
Write-OK "Environment acceptable"

# ── Step 2: verify models ─────────────────────────────────────────────────────
Write-Step 2 "Checking available models"
$models = @{
    "Qwen3-1.7B"         = "\path\to\LLMs\Qwen3-1.7B"
    "Qwen2.5-Coder-1.5B" = "\path\to\LLMs\Qwen2.5-Coder-1.5B"
    "DeepSeek-R1-1.5B"   = "\path\to\LLMs\DeepSeek-R1-1.5B"
    "Qwen2.5-Coder-3B"   = "\path\to\LLMs\Qwen2.5-Coder-3B"
    "DS-Coder-V2-Lite"   = "\path\to\LLMs\DeepSeek-Coder-V2-Lite"
}
$missingModels = @()
foreach ($name in $models.Keys) {
    $dir  = $models[$name]
    $gguf = Get-ChildItem $dir -Filter "*.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($gguf) { Write-OK "$name -> $($gguf.Name)" }
    else        { Write-Warn "$name not found at $dir (run download_spec_experts_models.ps1)"; $missingModels += $name }
}
if ($missingModels.Count -gt 0) {
    Write-Warn "$($missingModels.Count) model(s) missing. Controller will skip those chunk types."
}

# ── Step 3: start controller ──────────────────────────────────────────────────
Write-Step 3 "Starting controller on :8090"
$env:PYTHONPATH = $ProjectDir
$ControllerJob = Start-Job -Name "SpecExpertsController" -ScriptBlock {
    param($dir, $py)
    Set-Location $dir
    $env:PYTHONPATH = $dir
    & $py -m spec_experts.controller 2>&1
} -ArgumentList $ProjectDir, $PythonExe

Start-Sleep 3
$ctrl = Get-Job -Name "SpecExpertsController" -ErrorAction SilentlyContinue
if (-not $ctrl -or $ctrl.State -eq "Failed") {
    Write-Fail "Controller failed to start. Check Python path and dependencies."
    exit 1
}
Write-OK "Controller started (job id=$($ctrl.Id))"

# ── Step 4: wait for health ───────────────────────────────────────────────────
Write-Step 4 "Waiting for controller health"
$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        $r = Invoke-WebRequest "http://127.0.0.1:8090/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    Start-Sleep 1
}
if ($ready) { Write-OK "Controller healthy at http://127.0.0.1:8090" }
else        { Write-Warn "Controller not responding yet — may still be starting" }

# ── Step 5: register 2h scheduler (optional) ──────────────────────────────────
if (-not $NoScheduler) {
    Write-Step 5 "Registering 2-hour wake scheduler"
    $taskName = "SpecExpertsWake"
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Warn "Task $taskName already registered (skip)"
    } else {
        $action  = New-ScheduledTaskAction -Execute "powershell.exe" `
            -Argument "-NonInteractive -File `"$ProjectDir\spec_experts\run_spec_experts.ps1`" -SkipEnvCheck -NoScheduler"
        $trigger = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Hours 2) -Once -At (Get-Date)
        $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2) -StartWhenAvailable
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
        Write-OK "Task '$taskName' registered — wakes every 2 hours"
    }
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "===== Controller Running =====" -ForegroundColor Green
Write-Host "  Health:    http://127.0.0.1:8090/health"
Write-Host "  Status:    http://127.0.0.1:8090/status"
Write-Host "  Infer:     POST http://127.0.0.1:8090/infer"
Write-Host "  Shutdown:  POST http://127.0.0.1:8090/shutdown"
Write-Host "  Auto-off:  2 hours idle"
Write-Host ""
Write-Host "Run tests: python \path\to\project\tests\spec_experts_test.py" -ForegroundColor Yellow
Write-Host "Cleanup:   \path\to\project\spec_experts\cleanup.ps1" -ForegroundColor Yellow
