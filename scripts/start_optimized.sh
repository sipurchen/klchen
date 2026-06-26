#!/bin/bash
# Gemma 4 E4B - Optimized Startup Script
# Applies: TurboQuant+ KV Cache + Flash Attention + MoE Streaming Experts

# Kill any existing Ollama instances
powershell.exe -Command "Get-Process | Where-Object Name -like '*ollama*' | Stop-Process -Force" 2>/dev/null
sleep 2

# === TurboQuant+ Environment Variables ===
# Flash Attention: reduces attention memory footprint, enables more GPU layers
export OLLAMA_FLASH_ATTENTION=true

# KV Cache Quantization (q8_0 = 50% memory vs f16, q4_0 = 75% reduction)
# q8_0 is safer quality/speed tradeoff; use q4_0 for more aggressive optimization
export OLLAMA_KV_CACHE_TYPE=q8_0

# === KV Cache: Memory Management ===
# Keep model loaded for 60 minutes (avoids repeated 9.6GB reload)
export OLLAMA_KEEP_ALIVE=60m

# Single parallel request to maximize memory for one inference
export OLLAMA_NUM_PARALLEL=1

# === Streaming Experts for MoE ===
# Allow Ollama to auto-manage GPU memory and stream expert tensors
# GPU overhead reserved for kernel operations
export OLLAMA_LOAD_TIMEOUT=20m        # KEY: allow 9.6GB model to fully load (was 5m)
export OLLAMA_GPU_OVERHEAD=134217728  # 128MB reserved (reduced to allow 12 GPU layers)

# Models path
export OLLAMA_MODELS="/path/to/LLMs"

echo "[INFO] Starting Ollama with TurboQuant+ optimizations..."
echo "  Flash Attention : $OLLAMA_FLASH_ATTENTION"
echo "  KV Cache Type   : $OLLAMA_KV_CACHE_TYPE"
echo "  Keep Alive      : $OLLAMA_KEEP_ALIVE"
echo "  Models Path     : $OLLAMA_MODELS"

ollama serve
