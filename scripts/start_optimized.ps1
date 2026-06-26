# Gemma 4 E4B - Optimized Startup Script (PowerShell)
# CPU-safe: Ollama locked to cores 2+3, num_thread=2 in Modelfile
# Prevents 100% CPU saturation on i5-4460 (4-core)

param(
    [string]$KvCacheType = "q8_0",
    [string]$KeepAlive   = "60m",
    [switch]$AggressiveQuant
)
if ($AggressiveQuant) { $KvCacheType = "q4_0" }

# Kill existing Ollama instances
Get-Process | Where-Object { $_.Name -like "*ollama*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 3

# Load local paths config (gitignored) if present
$_pathsCfg = Join-Path (Split-Path $PSScriptRoot -Parent) "config\paths.ps1"
if (Test-Path $_pathsCfg) { . $_pathsCfg }
$_LLMsDir = if ($env:LLMS_DIR) { $env:LLMS_DIR } else { Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) "LLMmodel" }

# Environment variables
$env:OLLAMA_MODELS           = $_LLMsDir
$env:OLLAMA_HOST             = "0.0.0.0:11434"
$env:OLLAMA_LOAD_TIMEOUT     = "20m"          # 9.6GB model needs >5min cold start
$env:OLLAMA_KEEP_ALIVE       = $KeepAlive
$env:OLLAMA_KV_CACHE_TYPE    = $KvCacheType
$env:OLLAMA_GPU_OVERHEAD     = "134217728"    # 128MB reserved VRAM
$env:OLLAMA_NUM_PARALLEL     = "1"            # one request at a time
$env:OLLAMA_MAX_LOADED_MODELS = "1"           # never dual-load (RAM spike)
$env:OLLAMA_FLASH_ATTENTION  = "true"
$env:CUDA_VISIBLE_DEVICES    = "0"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Gemma 4 E4B - CPU-Safe Mode          " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  KV Cache Type    : $KvCacheType"       -ForegroundColor Green
Write-Host "  Keep Alive       : $KeepAlive"          -ForegroundColor Green
Write-Host "  Max loaded models: 1"                   -ForegroundColor Green
Write-Host "  Models Path      : $_LLMsDir"              -ForegroundColor Green
Write-Host "  CPU affinity     : cores 2+3 (set below)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start Ollama in background, then set CPU affinity immediately
$ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$proc = Start-Process -FilePath $ollamaPath -ArgumentList "serve" -PassThru -WindowStyle Hidden
Start-Sleep 4

if ($proc -and !$proc.HasExited) {
    # Lock Ollama to cores 2 and 3 only (affinity mask = 0xC = 12)
    # Cores 0+1 remain free for Windows + user processes
    $proc.ProcessorAffinity = 12   # binary 1100 = cores 2 and 3
    $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
    Write-Host "[OK] Ollama PID $($proc.Id) | Affinity=cores2+3 | Priority=BelowNormal" -ForegroundColor Green
} else {
    Write-Host "[WARN] Could not set affinity - Ollama may not have started" -ForegroundColor Red
}

# Re-apply Modelfile with num_thread 2
Start-Sleep 3
Write-Host "Registering optimized Modelfile..." -ForegroundColor Yellow
& $ollamaPath create gemma4-e4b-opt -f "$PSScriptRoot\..\config\Modelfile.gemma4-e4b-opt"

Write-Host ""
Write-Host "Ollama ready. Test: ollama run gemma3:1b 'hi'" -ForegroundColor Cyan
Write-Host "Benchmark: python tests/benchmark_all_models.py" -ForegroundColor Cyan
