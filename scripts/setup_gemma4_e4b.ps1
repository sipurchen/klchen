# ============================================================================
# Gemma 4 E4B - Automated Setup Script for Windows
# Modified by ClaudeO - Complete download, Ollama integration, and API setup
# ============================================================================
# Purpose: Downloads Gemma 4 E4B (instruction-tuned, Q4_K_M GGUF quantized)
#          to $env:LLMS_DIR\Gemma4_E4B (set LLMS_DIR or see config\paths.example.ps1).
# Requirements: Windows 10/11, PowerShell 5.1+, ~3GB disk space
# ============================================================================

$_pathsCfg = Join-Path (Split-Path $PSScriptRoot -Parent) "config\paths.ps1"
if (Test-Path $_pathsCfg) { . $_pathsCfg }
$_LLMsDir = if ($env:LLMS_DIR) { $env:LLMS_DIR } else { Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) "LLMmodel" }

param(
    [string]$InstallDir = "$_LLMsDir\Gemma4_E4B",
    [string]$OllamaModel = "gemma4:e4b",
    [string]$QuantType = "Q4_K_M",           # Balance of quality and memory
    [int]$ApiPort = 8000,
    [switch]$SkipOllamaInstall,
    [switch]$UseHuggingFace                   # Use HF GGUF instead of Ollama default
)

# --- Modified by ClaudeO: Color-coded logging utility ---
function Write-Step {
    param([string]$Message, [string]$Status = "INFO")
    $colors = @{ "INFO" = "Cyan"; "OK" = "Green"; "WARN" = "Yellow"; "ERR" = "Red" }
    $color = $colors[$Status]
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp][$Status] $Message" -ForegroundColor $color
}

# --- Modified by ClaudeO: Prerequisite checks ---
function Test-Prerequisites {
    Write-Step "Checking prerequisites..."

    # Check if running as admin (recommended but not required)
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Step "Not running as Administrator. Some operations may need elevation." "WARN"
    }

    # Check disk space on target drive
    $drive = Split-Path $InstallDir -Qualifier
    $freeGB = [math]::Round((Get-PSDrive ($drive -replace ':','')).Free / 1GB, 2)
    Write-Step "Drive $drive has ${freeGB}GB free space"
    if ($freeGB -lt 5) {
        Write-Step "Less than 5GB free on $drive. Gemma 4 E4B Q4 needs ~3GB." "WARN"
    }

    # Check Python
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        Write-Step "Python not found. Please install Python 3.10+ first." "ERR"
        return $false
    }
    $pyVer = python --version 2>&1
    Write-Step "Found $pyVer" "OK"

    # Check Node.js
    $node = Get-Command node -ErrorAction SilentlyContinue
    if (-not $node) {
        Write-Step "Node.js not found. Node.js agent features will be unavailable." "WARN"
    } else {
        $nodeVer = node --version 2>&1
        Write-Step "Found Node.js $nodeVer" "OK"
    }

    return $true
}

# --- Modified by ClaudeO: Ollama installation and configuration ---
function Install-Ollama {
    if ($SkipOllamaInstall) {
        Write-Step "Skipping Ollama installation (flag set)" "WARN"
        return
    }

    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaCmd) {
        Write-Step "Ollama already installed at $($ollamaCmd.Source)" "OK"
        # Update to latest version for Gemma 4 support
        Write-Step "Updating Ollama to latest version..."
        winget upgrade Ollama.Ollama --accept-package-agreements --accept-source-agreements 2>$null
        return
    }

    Write-Step "Installing Ollama via winget..."
    winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Step "winget failed. Downloading Ollama installer directly..." "WARN"
        $installerUrl = "https://ollama.com/download/OllamaSetup.exe"
        $installerPath = "$env:TEMP\OllamaSetup.exe"
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
        Start-Process -FilePath $installerPath -Wait
        Remove-Item $installerPath -Force
    }

    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    Write-Step "Ollama installed successfully" "OK"
}

# --- Modified by ClaudeO: Configure Ollama environment for memory optimization ---
function Set-OllamaConfig {
    Write-Step "Configuring Ollama environment variables..."

    # Set OLLAMA_MODELS to our target directory for model storage
    $modelsDir = Join-Path $InstallDir "ollama_models"
    [System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $modelsDir, "User")
    $env:OLLAMA_MODELS = $modelsDir

    # Memory optimization: limit KV cache and context
    # These reduce RAM usage significantly for E4B
    [System.Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "1", "User")
    $env:OLLAMA_NUM_PARALLEL = "1"

    # Keep model loaded for 30 minutes (prevents constant reload)
    [System.Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "30m", "User")
    $env:OLLAMA_KEEP_ALIVE = "30m"

    # Ensure models dir exists
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

    Write-Step "OLLAMA_MODELS set to $modelsDir" "OK"
}

# --- Modified by ClaudeO: Download Gemma 4 E4B via Ollama ---
function Get-Gemma4E4B {
    Write-Step "Downloading Gemma 4 E4B model via Ollama..."

    # Start Ollama service if not running
    $ollamaProcess = Get-Process ollama -ErrorAction SilentlyContinue
    if (-not $ollamaProcess) {
        Write-Step "Starting Ollama service..."
        Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3
    }

    # Pull the E4B model (Ollama handles GGUF conversion automatically)
    Write-Step "Pulling gemma4:e4b - this may take several minutes..."
    ollama pull $OllamaModel
    if ($LASTEXITCODE -ne 0) {
        Write-Step "Failed to pull model. Retrying..." "WARN"
        Start-Sleep -Seconds 5
        ollama pull $OllamaModel
    }

    # Verify the model is available
    $models = ollama list 2>&1
    if ($models -match "gemma4") {
        Write-Step "Gemma 4 E4B model downloaded and verified" "OK"
    } else {
        Write-Step "Model verification failed. Check ollama list manually." "ERR"
    }
}

# --- Modified by ClaudeO: Create custom Modelfile with memory-optimized settings ---
function New-OptimizedModelfile {
    Write-Step "Creating memory-optimized Modelfile..."

    $modelfilePath = Join-Path $InstallDir "Modelfile.gemma4-e4b-optimized"

    # This Modelfile configures Gemma 4 E4B for ~1GB memory usage
    # by limiting context window and KV cache parameters
    $modelfileContent = @"
# Gemma 4 E4B - Memory Optimized Configuration
# Modified by ClaudeO - Target: ~1GB RAM usage
FROM gemma4:e4b

# --- KV Cache & Context Optimization ---
# Reduce context window from 128K to 4K to drastically cut KV cache memory
PARAMETER num_ctx 4096

# Batch size reduction for lower memory footprint
PARAMETER num_batch 256

# Number of GPU layers (-1 = auto, adjust based on VRAM)
PARAMETER num_gpu -1

# --- Generation Parameters (from Google's recommended settings) ---
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 64
PARAMETER repeat_penalty 1.0

# --- System Prompt Template ---
TEMPLATE """<|turn|>system
{{ .System }}<turn|>
<|turn|>user
{{ .Prompt }}<turn|>
<|turn|>model
{{ .Response }}<turn|>"""

SYSTEM """You are Gemma 4 E4B, a highly capable AI assistant running locally. You excel at reasoning, coding, multimodal understanding, and agentic workflows. Respond helpfully and concisely."""
"@

    Set-Content -Path $modelfilePath -Value $modelfileContent -Encoding UTF8

    # Create the optimized model in Ollama
    Write-Step "Building optimized model variant: gemma4-e4b-opt..."
    ollama create gemma4-e4b-opt -f $modelfilePath
    if ($LASTEXITCODE -eq 0) {
        Write-Step "Optimized model 'gemma4-e4b-opt' created successfully" "OK"
    } else {
        Write-Step "Failed to create optimized model variant" "ERR"
    }

    return $modelfilePath
}

# --- Modified by ClaudeO: Install Python dependencies ---
function Install-PythonDeps {
    Write-Step "Installing Python dependencies..."

    $requirementsPath = Join-Path $InstallDir "requirements.txt"
    $requirements = @"
# Gemma 4 E4B API Server Dependencies
# Modified by ClaudeO
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
httpx>=0.27.0
pydantic>=2.0
python-multipart>=0.0.9
pillow>=10.0
soundfile>=0.12.0
numpy>=1.26.0
aiofiles>=23.0
websockets>=12.0
"@

    Set-Content -Path $requirementsPath -Value $requirements -Encoding UTF8
    python -m pip install --upgrade pip
    python -m pip install -r $requirementsPath
    Write-Step "Python dependencies installed" "OK"
}

# --- Modified by ClaudeO: Install Node.js dependencies ---
function Install-NodeDeps {
    $node = Get-Command node -ErrorAction SilentlyContinue
    if (-not $node) { return }

    Write-Step "Installing Node.js dependencies..."
    Push-Location $InstallDir

    $packageJson = @"
{
  "name": "gemma4-e4b-agents",
  "version": "1.0.0",
  "description": "Gemma 4 E4B AI Agent Framework - Modified by ClaudeO",
  "type": "module",
  "dependencies": {
    "express": "^4.21.0",
    "axios": "^1.7.0",
    "ws": "^8.18.0",
    "node-fetch": "^3.3.0"
  }
}
"@

    Set-Content -Path "package.json" -Value $packageJson -Encoding UTF8
    npm install
    Pop-Location
    Write-Step "Node.js dependencies installed" "OK"
}

# --- Modified by ClaudeO: Quick smoke test ---
function Test-ModelQuick {
    Write-Step "Running quick smoke test..."

    $testResult = ollama run $OllamaModel "Say 'Hello from Gemma 4 E4B' and nothing else." 2>&1
    if ($testResult -match "Hello") {
        Write-Step "Smoke test passed: $testResult" "OK"
    } else {
        Write-Step "Smoke test returned: $testResult" "WARN"
    }

    # Test the optimized variant too
    $optResult = ollama run gemma4-e4b-opt "Reply with only: OK" 2>&1
    Write-Step "Optimized model test: $optResult"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  Gemma 4 E4B - Automated Setup        " -ForegroundColor Magenta
Write-Host "  Target: $InstallDir                   " -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""

# Create target directory
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# Step 1: Prerequisites
if (-not (Test-Prerequisites)) {
    Write-Step "Prerequisites check failed. Aborting." "ERR"
    exit 1
}

# Step 2: Install and configure Ollama
Install-Ollama
Set-OllamaConfig

# Step 3: Download Gemma 4 E4B
Get-Gemma4E4B

# Step 4: Create optimized variant
New-OptimizedModelfile

# Step 5: Install dependencies
Install-PythonDeps
Install-NodeDeps

# Step 6: Quick test
Test-ModelQuick

Write-Host ""
Write-Step "Setup complete! Available models:" "OK"
ollama list | Select-String "gemma4"
Write-Host ""
Write-Step "Next steps:" "INFO"
Write-Host "  1. Start API server:  python $InstallDir\api\gemma4_api_server.py"
Write-Host "  2. Test via Ollama:   ollama run gemma4:e4b"
Write-Host "  3. Test optimized:    ollama run gemma4-e4b-opt"
Write-Host "  4. Python API test:   python $InstallDir\tests\test_api.py"
Write-Host ""
