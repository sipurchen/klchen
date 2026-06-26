# cleanup.ps1
# Graceful shutdown of all Spec-Experts processes
# Releases VRAM, RAM, CPU after test/session complete

chcp 65001 | Out-Null

function Write-Step { param($n, $msg) Write-Host "[$n] $msg" -ForegroundColor Cyan }
function Write-OK   { param($msg) Write-Host "    OK: $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "    --: $msg" -ForegroundColor DarkGray }

Write-Host ""
Write-Host "===== Spec-Experts Cleanup =====" -ForegroundColor Magenta

# ── Step 1: graceful controller shutdown via API ───────────────────────────────
Write-Step 1 "Sending /shutdown to controller :8090"
try {
    Invoke-WebRequest "http://127.0.0.1:8090/shutdown" -Method POST -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop | Out-Null
    Write-OK "Controller shutdown signal sent"
    Start-Sleep 3
} catch {
    Write-Warn "Controller not responding (already stopped or not started)"
}

# ── Step 2: kill llama-server ─────────────────────────────────────────────────
Write-Step 2 "Killing llama-server.exe"
$llama = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($llama) {
    $llama | Stop-Process -Force
    Write-OK "Killed $($llama.Count) llama-server instance(s)"
} else {
    Write-Warn "No llama-server running"
}

# ── Step 3: kill Python API processes on spec-experts ports ───────────────────
Write-Step 3 "Killing Python on ports 8090, 8081"
foreach ($port in @(8090, 8081)) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        $pid_ = $conn.OwningProcess | Select-Object -First 1
        $proc = Get-Process -Id $pid_ -ErrorAction SilentlyContinue
        if ($proc) {
            $proc | Stop-Process -Force
            Write-OK "Killed PID $pid_ ($($proc.Name)) on :$port"
        }
    } else {
        Write-Warn "Nothing on :$port"
    }
}

# ── Step 4: stop PowerShell jobs ─────────────────────────────────────────────
Write-Step 4 "Stopping PowerShell jobs"
$jobs = Get-Job -Name "SpecExperts*" -ErrorAction SilentlyContinue
if ($jobs) {
    $jobs | Stop-Job
    $jobs | Remove-Job
    Write-OK "Stopped $($jobs.Count) job(s)"
} else {
    Write-Warn "No SpecExperts jobs found"
}

# ── Step 5: kill rpc-server if running ───────────────────────────────────────
Write-Step 5 "Checking rpc-server"
$rpc = Get-Process -Name "rpc-server" -ErrorAction SilentlyContinue
if ($rpc) { $rpc | Stop-Process -Force; Write-OK "Killed rpc-server" }
else       { Write-Warn "No rpc-server" }

# ── Step 6: VRAM status after cleanup ─────────────────────────────────────────
Write-Step 6 "VRAM after cleanup"
Start-Sleep 2
try {
    $vram = & nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>$null
    Write-OK "VRAM free/total: $vram MB"
} catch {
    Write-Warn "nvidia-smi not available"
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "===== Cleanup Complete =====" -ForegroundColor Green
Write-Host "System resources released."
