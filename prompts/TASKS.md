# ============================================================================
# TASKS for Claude Sonnet 4.6 — Gemma4 E4B Project Gap Fixes
# Created by: Claude Opus 4.6 after audit review (2026-04-06)
# Priority: Fix gaps identified in comparison_report.md vs MASTER_PROMPT.md
# ============================================================================

---

## CONTEXT / 前情提要

硬體：i5-4460 (4C/4T Haswell AVX2) | GT 1030 2GB GDDR5 (48 GB/s, sm_61) | 34GB DDR3
系統：Windows 10 Pro | CUDA driver 13.0 (581.29) | Ollama v0.20.0
模型庫路徑：`\path\to\LLMs\blobs` (Ollama GGUF blob 格式)
專案路徑：`\path\to\project`
Git remote：`https://github.com/sipurchen/klchen.git` branch `BestSetupLLMs`

### 已完成
- [x] 三模型 benchmark（Gemma3:1b / DeepSeek-r1:1.5b / Gemma4:E4b）
- [x] 6 項測試（text/synonym/chat/vision/audio/TTS + video N/A）
- [x] CPU affinity 分離（Ollama cores 2+3, Python cores 0+1）
- [x] Modelfile num_thread=2 防止 CPU 過載
- [x] Git push 到 BestSetupLLMs branch
- [x] comparison_report.md 產生

### 待修復（本文件的 Tasks）
- [ ] 空回覆問題（4 個 status=ok 但 response=""）
- [ ] Markdown 表格排版錯亂
- [ ] MASTER_PROMPT.md 規格不準確（模型大小、架構）
- [ ] Phase 1-3 測試未執行（基礎設施 / Ollama direct / FastAPI）
- [ ] **核心目標：Gemma4:E4B 達到 10 tok/s**

---

## TASK 1: 修復空回覆問題
**Priority: HIGH**

4 個測試回傳 status=ok 但 response="" — 需逐一排查並重跑。

### 1.1 Gemma3:1b vision 空回覆 (0.7s)
- **原因推測**：test_image.png 是程式化產生的 100x100 漸層圖，太簡單/太小，模型回傳空
- **修復**：替換 `tests/assets/test_image.png` 為一張有實際內容的圖（例如含文字的截圖、風景照）
- 可以用 Python Pillow 產生含文字的 200x200 PNG：
```python
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGB', (200, 200), (30, 60, 120))
draw = ImageDraw.Draw(img)
draw.text((10, 80), "Hello AI", fill="white")
draw.rectangle([40, 130, 160, 170], outline="yellow", width=2)
img.save("tests/assets/test_image.png")
```
- 重新產生 `tests/assets/image_b64.txt`
- 重跑 Gemma3:1b vision 測試

### 1.2 DeepSeek-r1:1.5b synonym/chat 空回覆
- **原因推測**：DeepSeek-r1 使用 `<think>...</think>` reasoning tokens。所有 max_predict 額度可能被 thinking 消耗，visible output 為空
- **修復**：在 infer() 中對 DeepSeek 模型：
  - 增加 `max_predict` 到 300（讓 thinking + output 都有空間）
  - 或在 prompt 前加 `"Answer directly without thinking step by step."`
  - 或在 response 中解析 `<think>` 標籤，提取 `</think>` 之後的內容
- 重跑 DeepSeek synonym + chat

### 1.3 Gemma4:E4B vision 空回覆 (313.9s)
- **原因推測**：313.9s 幾乎全是 model loading + prompt eval，generation 未開始或僅生成 0 tokens
- **修復**：需先完成 Task 5（10 tok/s 優化），再重測
- 若仍為空：增加 timeout 到 1200s，減少 prompt 長度

---

## TASK 2: 修復報告 Markdown 表格
**Priority: MEDIUM**

`docs/comparison_report.md` 的 "Task Results" 表格欄位內含 `|` 和長文字，導致 Markdown 渲染錯亂。

**修復**：改寫 `tests/generate_report.py` 的 `fmt()` 函數：
- 限制每個 cell 在 40 字元內
- 移除 response 文字（改放到 Detailed Responses 區）
- 格式：`✅ 22.9s · 21.3 tok/s` 或 `❌ timeout 905s`

重新執行：`python tests/generate_report.py`

---

## TASK 3: 修正 MASTER_PROMPT.md 規格
**Priority: MEDIUM**

| 項目 | MASTER_PROMPT 寫的 | 實際值 | 修正 |
|---|---|---|---|
| Gemma4 E4B size | ~3 GB (Q4_K_M) | 9.1 GB | 更新為 9.1 GB |
| Architecture | Dense with PLE | MoE 43-layer, 64 experts | 更新為 MoE |
| RAM 需求 | ~3.3 GB | ~12+ GB | 更新 Memory 段落 |
| Effective params | 4.5B | 8B (MoE total, ~2.7B active/token) | 更新 |
| 專案路徑 | \path\to\LLMs\Gemma4_E4B | \path\to\project | 更新全文 |
| 模型路徑 | \path\to\LLMs\Gemma4_E4B\ollama_models | \path\to\LLMs | 更新 |

直接編輯 `prompts/MASTER_PROMPT.md`，修正上述所有數值。

---

## TASK 4: 補齊 Phase 1-3 測試
**Priority: MEDIUM**

原計畫 MASTER_PROMPT 定義 5 個 Phase（15+ 測試），但 benchmark 只覆蓋了 Phase 5 的子集。

### 4.1 Phase 1: Infrastructure
執行以下檢查並記錄結果：
```python
# 1.1 Ollama server reachable
httpx.get("http://localhost:11434")  # expect 200

# 1.2 gemma4:e4b model present
# curl http://localhost:11434/api/tags → check 'gemma4:E4b' in list

# 1.3 gemma4-e4b-opt exists
# same API → check 'gemma4-e4b-opt:latest'

# 1.4 API server healthy (need to start FastAPI first)
# python api/gemma4_api_server.py &
# httpx.get("http://localhost:8000/health")

# 1.5 Python deps
# python -c "import fastapi, httpx, uvicorn"
```

### 4.2 Phase 2: Ollama Direct
```python
# 2.1 Basic generation
# POST /api/chat model=gemma3:1b (use fast model)

# 2.2 System prompt JSON
# POST with system: "Reply only in JSON" → check { }

# 2.3 OpenAI-compatible
# POST /v1/chat/completions → check "choices" key

# 2.4 Optimized model
# POST /api/chat model=gemma4-e4b-opt (short prompt, num_predict=5)
```

### 4.3 Phase 3: Unified API
需要先啟動 FastAPI server (`python api/gemma4_api_server.py`)：
```python
# 3.1 Chat completion → valid response
# 3.2 Agent routing (coder) → contains code
# 3.3 All agents → 6 defaults
# 3.4 Custom agent registration
# 3.5 Model listing
# 3.6 Error handling → graceful fallback
```

將結果合併到 `tests/benchmark_results.json` 並重新產生報告。

---

## TASK 5: ★ 核心 — Gemma4:E4B 達到 10 tok/s ★
**Priority: CRITICAL**

### 理論基礎

Gemma4:E4B 是 MoE 架構（64 experts, top_k=2）。每個 token 只激活：
- 共享層（attention + embedding + router）≈ 2.5B 參數
- 活躍 expert（2/64 × 5.5B）≈ 0.17B 參數
- **每 token 實際讀取：2.67B params × 4.5 bits = 1.50 GB**

GT 1030 記憶體頻寬 48 GB/s → 1.50 GB / 48 = 31ms → **理論 32 tok/s**
若 GPU 放 1.0GB + CPU 放 0.5GB → 1/48 + 0.5/25.6 = 40ms → **理論 24.7 tok/s**

**即使打 5 折（KV cache overhead + kernel launch），仍有 12 tok/s > 10 tok/s 目標**

### 為何目前只有 0.04 tok/s

Ollama 的 `n_gpu_layers=12` 是 **layer-level** 分割，不是 **tensor-level** 分割。
12 個完整 layer（含 expert 權重）大約 2.5GB，但 VRAM 只有 2GB。
結果：Ollama 自動降到更少 GPU layers → 大部分計算在 CPU → 只用 2 threads → 0.04 tok/s。

**真正需要的是 tensor-level offloading**：把每一層的 attention 放 GPU、expert FFN 放 CPU。
這是 llama.cpp 的 `-ot` (override tensor) 功能，Ollama 不支援。

### 執行步驟

#### Step 5.1: 下載 llama.cpp 預編譯 Windows CUDA 二進制

```powershell
# 在 GitHub Releases 找到最新版 llama.cpp Windows + CUDA 12 binary
# 檔名類似: llama-bXXXX-bin-win-cuda-cu12.4-x64.zip
# 解壓到 \path\to\project\bin\llama-cpp\

# 驗證:
.\bin\llama-cpp\llama-cli.exe --version
.\bin\llama-cpp\llama-server.exe --version
```

> CUDA driver 581.29 支援 CUDA 13.0，所以 cu12.x 二進制完全相容。

#### Step 5.2: MoE Expert Offloading 啟動 llama-server

```powershell
# GGUF blob 路徑（直接用 Ollama 的 blob，它就是 GGUF）
$GGUF = "\path\to\LLMs\blobs\sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a"

# 核心指令：expert FFN 放 CPU，其餘放 GPU
.\bin\llama-cpp\llama-server.exe `
  -m $GGUF `
  -ngl 999 `
  -ot "blk\..*\.ffn_gate_exps\.weight=CPU" `
  -ot "blk\..*\.ffn_down_exps\.weight=CPU" `
  -ot "blk\..*\.ffn_up_exps\.weight=CPU" `
  -c 1024 `
  -b 256 `
  -t 3 `
  --mmap `
  --mlock `
  --host 0.0.0.0 `
  --port 8080
```

**解釋：**
- `-ngl 999`：嘗試把所有 layer 放 GPU
- `-ot "blk\..*\.ffn_*_exps\.weight=CPU"`：**但 expert FFN 強制留 CPU**
- 結果：attention tensors (~1.2GB) 在 GPU → 快速矩陣乘法
- Expert weights (~7.8GB) 在 CPU RAM via mmap → 只有活躍 expert 頁面被讀取
- `-c 1024`：小 KV cache 節省 VRAM
- `--mmap --mlock`：MLX 風格無 swap

#### Step 5.3: 進一步優化 — Q3_K_S 重量化（選用）

如果 Step 5.2 的 VRAM 不夠放所有 attention layers：

```powershell
# 用 llama-quantize 將 Q4_K_M 轉為 Q3_K_S（更小，更快）
.\bin\llama-cpp\llama-quantize.exe $GGUF \path\to\LLMs\gemma4-e4b-q3ks.gguf Q3_K_S

# 新模型大約 6-7 GB（vs 9.1GB）
# Attention layers at Q3: ~0.9GB → 完全 fit 進 2GB VRAM
# 重新啟動 llama-server 使用新 GGUF
```

理論速度：Q3_K_S active weights = 1.17 GB → 全放 GPU → **41 tok/s 理論，實際預期 12-20 tok/s**

#### Step 5.4: Benchmark 驗證

llama-server 提供 OpenAI-compatible API (`http://localhost:8080/v1/chat/completions`)。

```python
import httpx, time
t0 = time.time()
r = httpx.post("http://localhost:8080/v1/chat/completions", json={
    "model": "gemma4-e4b",
    "messages": [{"role": "user", "content": "List 3 colors."}],
    "max_tokens": 30, "temperature": 0.3
}, timeout=300)
elapsed = time.time() - t0
data = r.json()
tokens = data["usage"]["completion_tokens"]
print(f"{tokens} tokens in {elapsed:.1f}s = {tokens/elapsed:.1f} tok/s")
```

**目標：≥ 10 tok/s**

#### Step 5.5: 整合回 benchmark

修改 `tests/benchmark_all_models.py`：
- 新增 `LLAMA_SERVER = "http://localhost:8080"` 端點
- Gemma4:E4B 的 infer() 改呼叫 llama-server（不走 Ollama）
- Gemma3:1b / DeepSeek-r1:1.5b 仍走 Ollama（它們本來就夠快）

#### Step 5.6: CPU 佔用管理

```python
# 啟動 llama-server 後，立即設定 CPU affinity
import psutil
for p in psutil.process_iter(['pid','name']):
    if 'llama-server' in p.info['name'].lower():
        proc = psutil.Process(p.info['pid'])
        proc.cpu_affinity([1, 2, 3])   # 3 cores for llama-server
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
```

Ollama（服務 Gemma3 / DeepSeek）用 cores [2,3]，llama-server 用 cores [1,2,3]。
兩者不會同時跑（benchmark 是順序的）。

### 速度期待值總結

| 方案 | 量化 | GPU 內容 | CPU 內容 | 理論 tok/s | 預估實際 |
|------|------|----------|----------|-----------|---------|
| 現狀 (Ollama) | Q4_K_M | 12 full layers | 31 full layers | ~0.5 | **0.04** |
| llama-server -ot | Q4_K_M | attn (~1.2GB) | expert FFN (~7.8GB) | 24.7 | **10-15** |
| llama-server -ot | Q3_K_S | attn (~0.9GB) | expert FFN (~6GB) | 41.1 | **15-20** |
| llama-server -ot | IQ3_M  | attn (~0.8GB) | expert FFN (~5GB) | 48.0 | **18-25** |

---

## TASK 6: 最終報告更新 + Git Push
**Priority: LOW (after all above)**

1. 重跑所有 benchmark（含 10 tok/s 的 Gemma4）
2. `python tests/generate_report.py` → 更新 comparison_report.md
3. Git commit + push：
```bash
cd \path\to\project
git add -A
git commit -m "Fix empty responses, 10tok/s Gemma4 via llama-server -ot expert offloading"
git push origin BestSetupLLMs
```

---

## 執行順序建議

```
Task 1.1 → 1.2 → 2 → 3 → 4.1 → 4.2 → 4.3  （可快速完成，~30 min）
     ↓
Task 5.1 → 5.2 → 5.4 → 5.5                    （核心優化，~60 min）
     ↓
Task 5.3（選用：Q3_K_S 重量化如果 5.2 不夠快）
     ↓
Task 1.3 → 6                                    （重測 Gemma4 + 最終 push）
```
