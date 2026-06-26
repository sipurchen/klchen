#!/bin/bash
# P6.1 Vulkan Backend Benchmark
# GT 1030 (Pascal sm_61) baseline vs CPU-only
# Usage: bash vulkan_bench.sh <model_path> [ctx_size]

MODEL=${1:-"model.gguf"}
CTX=${2:-2048}
LLAMA_BIN="$(dirname "$0")/../bin/llama-cpp/llama-bench.exe"

if [ ! -f "$LLAMA_BIN" ]; then
  LLAMA_BIN="llama-bench"
fi

echo "=== Vulkan Benchmark: GT 1030 ==="
echo "Model: $MODEL | CTX: $CTX"
echo ""

echo "[1/3] CPU-only baseline (threads=2, no GPU)"
"$LLAMA_BIN" \
  -m "$MODEL" \
  -ngl 0 \
  -t 2 \
  -c "$CTX" \
  -n 128 \
  --no-warmup \
  2>&1 | grep -E "pp|tg|tok/s"

echo ""
echo "[2/3] Vulkan partial offload (ngl=12)"
"$LLAMA_BIN" \
  -m "$MODEL" \
  -ngl 12 \
  -t 2 \
  -c "$CTX" \
  -n 128 \
  --no-warmup \
  2>&1 | grep -E "pp|tg|tok/s"

echo ""
echo "[3/3] Vulkan max offload (ngl=99)"
"$LLAMA_BIN" \
  -m "$MODEL" \
  -ngl 99 \
  -t 2 \
  -c "$CTX" \
  -n 128 \
  --no-warmup \
  2>&1 | grep -E "pp|tg|tok/s"

echo ""
echo "=== Done. VRAM target: < 1800MB for GT 1030 ==="
