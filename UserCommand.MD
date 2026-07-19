# 使用者指令手冊 / User Command Reference

> 本專案在各硬體平台上的常用指令速查表。  
> 主力平台: **Windows i5-4460 + GT 1030**  
> 其他平台規劃: 見 [Others_AI.MD](Others_AI.MD)

---

## 目錄

1. [Windows 主力平台](#1-windows-主力平台)
2. [Mac Mini M4 32GB](#2-mac-mini-m4-32gb)
3. [API 測試指令](#3-api-測試指令)
4. [診斷與除錯](#4-診斷與除錯)

---

## 1. Windows 主力平台

### 1.1 啟動 Ollama

```powershell
# 啟動 Ollama 服務 (背景)
ollama serve

# 確認運行
curl http://localhost:11434/api/tags

# 拉取模型
ollama pull gemma3:1b
ollama pull deepseek-r1:1.5b
```

### 1.2 啟動 llama-server (Gemma4 E4B)

> 首次使用需先下載 `bin\llama-cpp\` 二進制，見 [bin/README.md](bin/README.md)。

```powershell
# 純 CPU 模式 (GT 1030 VRAM 不足)
bin\llama-cpp\llama-server.exe `
  -m "<LLMS_DIR>\gemma4_textonly\google_gemma-4-E4B-it-Q4_K_M.gguf" `
  -ngl 0 -c 512 -t 4 -np 1 `
  --host 127.0.0.1 --port 8080

# 確認載入成功
curl http://127.0.0.1:8080/health
```

### 1.3 執行 Benchmark

```powershell
# 完整 benchmark (順序執行, CPU 安全)
cd <PROJECT_ROOT>
python tests/benchmark_all_models.py

# 生成比較報告
python tests/generate_report.py
# → 輸出至 docs/comparison_report.md
```

### 1.4 啟動 FastAPI 多代理伺服器

```powershell
# 確保 Ollama 已啟動後執行
python api/gemma4_api_server.py
# → 監聽 http://localhost:8000

# 端點:
#   POST /v1/chat/completions   (OpenAI 相容)
#   POST /api/agent/{agent_id}  (指定代理)
#   GET  /v1/models
```

### 1.5 直接測試 llama-server

```powershell
# 測試 /completion 端點 (繞過 thinking mode)
curl -X POST http://127.0.0.1:8080/completion `
  -H "Content-Type: application/json" `
  -d '{
    "prompt": "<bos><start_of_turn>user\n你好<end_of_turn>\n<start_of_turn>model\n",
    "n_predict": 100,
    "stop": ["<end_of_turn>", "<eos>"]
  }'
```

### 1.6 CPU 核心親和性設定 (防止螢幕黑屏)

```python
# 在 Python 腳本中設定 (psutil)
import psutil, os
proc = psutil.Process(os.getpid())
proc.cpu_affinity([0, 1])  # Python → 核心 0+1

# Ollama/llama-server → 核心 2+3 (由 benchmark 腳本自動設定)
# 或手動:
# Start-Process -FilePath "bin\llama-cpp\llama-server.exe" ... 後再設 affinity
```

---

## 2. Mac Mini M4 32GB

### 2.1 安裝環境

```bash
# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# llama.cpp (Metal 後端, 推薦)
brew install llama.cpp

# 或從源碼編譯最新版
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(sysctl -n hw.logicalcpu)

# Ollama (Metal 自動偵測)
brew install ollama

# Python 環境
pip3 install httpx fastapi uvicorn psutil
```

### 2.2 啟動 llama-server (Metal, Gemma4 E4B)

```bash
# 全 GPU Metal 模式 (32GB UMA 可容納)
llama-server \
  -m ~/LLMmodel/google_gemma-4-E4B-it-Q4_K_M.gguf \
  -ngl 999 \
  -c 4096 \
  -t 4 \
  -fa \
  --host 127.0.0.1 --port 8080

# 提供區域網路存取 (作為 API Server)
llama-server \
  -m ~/LLMmodel/google_gemma-4-E4B-it-Q4_K_M.gguf \
  -ngl 999 -c 4096 -t 4 -fa \
  --host 0.0.0.0 --port 8080
```

### 2.3 多模態 Gemma4 (原始 9.1 GB)

```bash
# 嘗試完整多模態 GGUF (需 ~10 GB 記憶體)
# 先確認 blob 路徑 (從 Ollama 安裝的)
ls ~/.ollama/models/blobs/

# 使用 llama-mtmd-cli (多模態)
llama-mtmd-cli \
  -m ~/.ollama/models/blobs/sha256-<hash> \
  --mmproj <vision_encoder_path> \
  -ngl 999 \
  --image test.jpg \
  -p "描述這張圖片"
```

### 2.4 Ollama (Metal)

```bash
# 啟動
ollama serve &

# 拉取模型
ollama pull gemma3:latest
ollama pull qwen3:8b
ollama pull qwen3:14b

# 測試
ollama run qwen3:8b "你好，請介紹一下你自己"
```

### 2.5 MLX (Apple 原生)

```bash
pip3 install mlx-lm

# 文字生成
python3 -m mlx_lm.generate \
  --model mlx-community/gemma-4-E4B-it-4bit \
  --prompt "<bos><start_of_turn>user\n你好<end_of_turn>\n<start_of_turn>model\n" \
  --max-tokens 200

# MLX 伺服器 (OpenAI 相容 API)
python3 -m mlx_lm.server \
  --model mlx-community/gemma-4-E4B-it-4bit \
  --port 8080
```

### 2.6 語音辨識 (Whisper Metal)

```bash
# 安裝 whisper.cpp
brew install whisper-cpp

# 或從源碼編譯 (Metal)
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make -j $(sysctl -n hw.logicalcpu) GGML_METAL=1

# 下載模型
./models/download-ggml-model.sh large-v3

# 執行語音辨識
./main -m models/ggml-large-v3.bin -l zh -f audio.wav
```

### 2.7 跨裝置連接 (從 Windows 連 Mac Mini)

```python
# Windows 端設定連接 Mac Mini 上的 llama-server
# 在 api/gemma4_api_server.py 修改:
LLAMA_SERVER_URL = "http://192.168.1.xxx:8080"  # 替換 Mac Mini IP

# 或設定環境變數
$env:LLAMA_SERVER_HOST = "192.168.1.xxx"
$env:LLAMA_SERVER_PORT = "8080"
```

---

## 3. API 測試指令

### 3.1 OpenAI 相容端點

```bash
# Chat Completions (Gemma4 格式)
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-e4b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 200
  }'

# 列出模型
curl http://127.0.0.1:8000/v1/models
```

### 3.2 專用代理端點

```bash
# 使用 coder 代理
curl -X POST http://127.0.0.1:8000/api/agent/coder \
  -H "Content-Type: application/json" \
  -d '{
    "message": "寫一個 Python hello world",
    "max_tokens": 300
  }'

# 代理清單: general, coder, analyst, translator, vision, planner
```

### 3.3 llama-server 直接端點

```bash
# /completion (繞過 chat template)
curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<bos><start_of_turn>user\n{USER_MESSAGE}<end_of_turn>\n<start_of_turn>model\n",
    "n_predict": 200,
    "temperature": 0.7,
    "stop": ["<end_of_turn>", "<eos>"]
  }'

# /tokenize (計算 token 數)
curl -X POST http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello world"}'

# 健康檢查
curl http://127.0.0.1:8080/health
```

---

## 4. 診斷與除錯

### 4.1 Windows 診斷

```powershell
# 確認 llama-server 版本
bin\llama-cpp\llama-server.exe --version

# 確認 GPU 資訊 (Vulkan)
bin\llama-cpp\llama-server.exe --list-devices

# 確認 Ollama 狀態
ollama list
ollama ps  # 目前載入的模型

# 記憶體使用
Get-Process | Where-Object {$_.Name -match "ollama|llama"} | `
  Select-Object Name, CPU, WorkingSet
```

### 4.2 Mac 診斷

```bash
# 確認 Metal 設備
system_profiler SPDisplaysDataType | grep "Metal"

# 統一記憶體使用
sudo memory_pressure
vm_stat

# llama.cpp Metal 層數確認
# 啟動時應看到: "Metal: loaded X kernels"

# 確認 Ollama Metal
ollama info  # 應顯示 Metal GPU backend
```

### 4.3 常見問題排查

| 問題 | 平台 | 解決方案 |
|------|------|---------|
| `螢幕黑屏/變暗` | Windows | CPU 核心親和性 → cores 2+3, 見 [README](README.md) |
| `alloc_tensor_range: failed` | Windows GT 1030 | 使用 `-ngl 0`（2 GB VRAM 不足） |
| `expected 2131, got 720` | Windows | 使用 bartowski 純文字 GGUF，非 Ollama blob |
| `回應全空白` | Windows/Mac | 使用 `/completion` 端點 + 手動 prompt 格式 |
| Flash Attention 停用 | Windows | 正常現象（CPU/GPU 混合），Mac 上不會發生 |
| `0.04 tok/s` Gemma4 | Windows | 換用 bartowski GGUF + llama-server |

---

## 快速參考卡 / Quick Reference Card

```
Windows (i5-4460 + GT 1030):
  啟動 Ollama:      ollama serve
  啟動 Gemma4:      bin\llama-cpp\llama-server.exe -m <gguf> -ngl 0 -c 512
  Benchmark:        python tests/benchmark_all_models.py
  API Server:       python api/gemma4_api_server.py
  速度上限:         ~2 tok/s (DDR3 17GB/s 瓶頸)

Mac Mini M4 32GB:
  啟動 Gemma4:      llama-server -m <gguf> -ngl 999 -c 4096 -fa
  Ollama:           ollama serve && ollama run qwen3:8b
  MLX:              python3 -m mlx_lm.generate --model ...
  速度預估:         ~19 tok/s (UMA 120GB/s)
  無需 CPU 親和性設定
```

---

*最後更新: 2026-05-14*  
*維護: SipurChen*
