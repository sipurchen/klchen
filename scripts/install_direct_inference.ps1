# ============================================================================
# Install llama-cpp-python with CUDA 11.8 (GT 1030 / sm_61 compatible)
# Bypasses Ollama — uses GGUF blobs directly for Flash-MoE inference
# ============================================================================

# Resolve LLMs dir
$_pathsCfg = Join-Path (Split-Path $PSScriptRoot -Parent) "config\paths.ps1"
if (Test-Path $_pathsCfg) { . $_pathsCfg }
$_LLMsDir = if ($env:LLMS_DIR) { $env:LLMS_DIR } else { Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) "LLMmodel" }

Write-Host "=== Installing llama-cpp-python (CUDA 11.8) ===" -ForegroundColor Cyan

# Step 1: Try prebuilt CUDA 11.8 wheel (fastest, no compile needed)
$wheelUrl = "https://github.com/abetlen/llama-cpp-python/releases/latest/download"
Write-Host "[1/3] Attempting prebuilt CUDA 11.8 wheel..." -ForegroundColor Yellow

pip install llama-cpp-python `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118 `
    --force-reinstall `
    --no-cache-dir

if ($LASTEXITCODE -ne 0) {
    Write-Host "[1/3] Prebuilt failed. Trying source build with CUDA..." -ForegroundColor Yellow
    # Step 2: Build from source with CUDA (requires MSVC + CUDA toolkit)
    $env:CMAKE_ARGS = "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61"
    $env:FORCE_CMAKE = "1"
    pip install llama-cpp-python --no-cache-dir
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FALLBACK] Installing CPU-only llama-cpp-python..." -ForegroundColor Red
    $env:CMAKE_ARGS = ""
    pip install llama-cpp-python --no-cache-dir
}

# Step 3: Install supporting packages
pip install psutil numpy --quiet

Write-Host ""
Write-Host "=== Verifying Installation ===" -ForegroundColor Cyan
python -c "from llama_cpp import Llama; print('[OK] llama-cpp-python imported')"
python -c "
from llama_cpp import Llama, llama_supports_gpu_offload
print('[CUDA offload support]:', llama_supports_gpu_offload())
"

Write-Host ""
Write-Host "GGUF Model Paths:" -ForegroundColor Green
Write-Host "  Gemma4:E4b    -> $_LLMsDir\blobs\sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a"
Write-Host "  Gemma3:1b     -> $_LLMsDir\blobs\sha256-7cd4618c1faf8b7233c6c906dac1694b6a47684b37b8895d470ac688520b9c01"
Write-Host "  DeepSeek 1.5b -> $_LLMsDir\blobs\sha256-aabd4debf0c8f08881923f2c25fc0fdeed24435271c2b3e92c4af36704040dbc"
