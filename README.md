# Gemma4-E4B Local LLM Project

> **在低規格硬體上運行大型語言模型的完整調校指南**  
> Running Large Language Models on Budget Hardware — Full Optimization Guide

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![llama-server b8679](https://img.shields.io/badge/llama.cpp-b8679-green.svg)](https://github.com/ggml-org/llama.cpp)

---

## 硬體環境 / Hardware

| 元件 | 規格 |
|------|------|
| CPU | Intel Core i5-4460 (4 核心, 3.2 GHz, Haswell, 2013) |
| GPU | NVIDIA GeForce GT 1030 (2 GB GDDR5, Pascal sm_61) |
| RAM | 34 GB DDR3 (雙通道, ~17 GB/s 頻寬) |
| OS | Windows 10 Pro 22H2 (19045) |
| 推論引擎 | Ollama v0.20.0 + llama-server b8679 (Vulkan) |

---

## 專案目標 / Project Goals

1. 在 **i5-4460 + GT 1030** 這類 2013 年消費級硬體上，成功運行 3 個本地 LLM
2. 以 **Gemma4 E4B (7.5B 參數)** 為核心，達到可用的推論速度
3. 建立 CPU 過載防護機制，防止系統死機
4. 提供 OpenAI 相容的 FastAPI 多代理架構

---

## 模型清單 / Models

| 模型 | 大小 | 架構 | 後端 |
|------|------|------|------|
| `gemma3:1b` | 777 MB | Dense 29 層 | Ollama |
| `deepseek-r1:1.5b` | 1.04 GB | Dense 30 層 + 推理鏈 | Ollama |
| `Gemma4 E4B` (bartowski Q4_K_M) | 5.03 GB | Dense 42 層, 7.5B | llama-server |

---

## 最終測試結果 / Benchmark Results

> 測試日期：2026-04-08 · 硬體：i5-4460 + GT 1030 + 34 GB DDR3

### 速度比較

| 模型 | 摘要 tok/s | 同義詞 tok/s | 聊天 tok/s | **平均** |
|------|-----------|------------|-----------|---------|
| Gemma3 1B (Ollama) | 21.5 | 13.75 | 21.96 | **19.1** |
| DeepSeek-r1 1.5B (Ollama) | 14.98 | 11.51 | 11.13 | **12.5** |
| Gemma4 E4B (llama-server) | 1.42 | 2.19 | 2.09 | **1.9** |

### 功能支援

| 功能 | Gemma3 1B | DeepSeek-r1 1.5B | Gemma4 E4B |
|------|-----------|-----------------|-----------|
| 文字摘要 | ✅ 32.0s | ✅ 37.3s | ✅ 27.4s |
| 同義詞增強 | ✅ 12.5s | ✅ 39.6s | ✅ 36.5s |
| 一般對話 | ✅ 5.2s | ✅ 42.6s | ✅ 23.9s |
| 圖像分析 | ⚠️ 空回應 | ➖ 不支援 | ➖ 僅文字 GGUF |
| 語音辨識 | ➖ 不支援 | ➖ 不支援 | ➖ 僅文字 GGUF |
| TTS (語音合成) | ✅ Windows SAPI | ✅ Windows SAPI | ✅ Windows SAPI |
| 影片分析 | ➖ 不支援 | ➖ 不支援 | ➖ 不支援 |

### Gemma4 優化前後對比

| 方法 | 速度 | 回應品質 | 備註 |
|------|------|---------|------|
| Ollama (原始) | 0.04 tok/s | ⚠️ 幾乎逾時 | CPU 過載, 9.1 GB 多模態 blob |
| llama-server (本專案) | ~2 tok/s | ✅ 正常輸出 | **50× 提升** |
| 目標 (需 ≥6 GB GPU) | ~10 tok/s | ✅ 正常輸出 | RTX 3060 可達成 |

---

## 調校理論基礎 / Theoretical Foundation

### 1. 記憶體頻寬瓶頸 (Memory Bandwidth Bottleneck)

LLM 推論的**逐 token 生成階段**是記憶體頻寬瓶頸，不是計算瓶頸。

```
理論最大 tok/s = 記憶體頻寬 / 模型大小
DDR3 i5-4460   = 17 GB/s ÷ 5.03 GB = 3.4 tok/s (CPU 理論上限)
GT 1030 VRAM   = 48 GB/s ÷ 5.03 GB = 9.5 tok/s (GPU 理論上限)
RTX 3060       = 360 GB/s ÷ 5.03 GB = 71.6 tok/s
```

**參考來源：** Noam Shazeer, "Fast Transformer Decoding" (2019); Horace He, "Making Deep Learning Go Brrrr" (2022)

### 2. Q4_K_M 量化

將模型從 FP16 (16-bit) 量化至 4-bit，可在保留 ~99% 品質的同時減少 4× 記憶體使用。

```
原始 FP16:  7.5B × 2 bytes = 15 GB
Q4_K_M:    7.5B × 0.67 bytes = 5.03 GB  (含 KV 矩陣保持 6-bit)
```

**參考來源：** Dettmers et al., "GGML Q4_K Quantization" (2023); Tim Dettmers, "bitsandbytes" library

### 3. CPU 親和性 (CPU Affinity)

i5-4460 有 4 個核心。若 LLM 推論佔用所有核心，OS 排程器無法處理繪圖/IO，導致螢幕變黑。

```
策略：
  Python 進程    → 核心 0+1 (psutil.cpu_affinity([0,1]))
  Ollama/llama-server → 核心 2+3 (psutil.cpu_affinity([2,3]))
```

### 4. MoE 架構誤解的修正

初始假設 Gemma4 E4B 為 Mixture-of-Experts (64 experts, top_k=2)，理論上每個 token 只需載入 1.5 GB 的活躍參數。

**實際發現：** bartowski 的文字專用 GGUF 顯示 `n_expert = 0`，Gemma4 E4B 是**密集架構 (Dense)**，不是 MoE。"E4B" = Effective 4 Billion (有效 40 億), 非 Expert-based。

---

## 問題排查過程 / Troubleshooting Journey

### 問題 1：Ollama 造成 CPU 100% 過載導致螢幕黑屏

**根本原因：** Ollama 的 llama.cpp runner 預設使用所有 4 個核心，加上 Python benchmark 進程，總計超出系統容量。

**嘗試方案：**
| 方案 | 結果 |
|------|------|
| 降低 `num_thread=2` 在 Modelfile | ✅ 解決，但速度降低 |
| `psutil.cpu_affinity([2,3])` 固定 Ollama 至核心 2+3 | ✅ 根本解決 |
| `OLLAMA_MAX_LOADED_MODELS=1` | ✅ 防止雙模型同時載入 |

**最終設定：** Ollama 固定核心 2+3，Python 固定核心 0+1

---

### 問題 2：Gemma4 via Ollama 只有 0.04 tok/s

**根本原因：** Ollama 的 Gemma4 blob 是**合併多模態 GGUF**，包含：
- 文字轉換器：720 tensors
- 音訊編碼器 (Whisper-like)：720 tensors  
- 視覺編碼器 (SigLIP-like)：691 tensors
- **總計：2131 tensors, 9.1 GB**

Ollama 的內部修補版 llama.cpp 可以處理此格式，但標準 llama.cpp 無法。

**嘗試方案：**
| 方案 | 結果 | 原因 |
|------|------|------|
| `llama-server b8679 -m <blob>` | ❌ 失敗 | "expected 2131, got 720" — 架構只認識 720 個文字 tensors |
| `llama-gguf-split` 切分 blob 為 3 個檔案 | ❌ 失敗 | 分割後的檔案 header 仍標記需要 2131 tensors |
| `llama-mtmd-cli --mmproj <split>` | ❌ 失敗 | n_expert=0，自動尋找並合併 3 個分割檔，仍是 2131 tensors |
| **下載 bartowski 純文字 GGUF** | ✅ **成功** | 720 tensors，5.03 GB，正確格式 |

**關鍵指令：**
```
# 下載位置
https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF
檔案：google_gemma-4-E4B-it-Q4_K_M.gguf (5.03 GB)
```

---

### 問題 3：llama-server 載入時 Vulkan 記憶體分配失敗

**根本原因：** GT 1030 只有 2 GB VRAM。Gemma4 的 embedding table 需要 >2 GB 的單次記憶體分配，超出硬體限制。

```
embedding: 262144 vocab × 2560 embed × Q4_K_M ≈ 745 MB
但 Vulkan 嘗試一次分配整個連續區塊: 2,312,110,080 bytes ≈ 2.15 GB → 失敗
```

**嘗試方案：**
| 方案 | 結果 |
|------|------|
| `-ngl 999` (全 GPU) | ❌ `alloc_tensor_range: failed to allocate Vulkan0 buffer of size 2312110080` |
| `-ngl 12` (部分 GPU) | ❌ 同樣失敗，embedding table 仍超過 2 GB |
| `-ot "token_embd.weight=CPU" -ngl 999` | ❌ 仍分配 1 GB Vulkan buffer，失敗 |
| **`-ngl 0` (純 CPU)** | ✅ **成功載入並正常推論** |

**結論：** GT 1030 (2 GB VRAM) 無法承載此模型。需要 ≥6 GB VRAM 的 GPU。

---

### 問題 4：Gemma4 回應全為空白 (所有 token 進入隱藏思考模式)

**根本原因：** Gemma4 Instruct 的 chat template 預設包含 `<|think|>` token，啟用推理鏈 (Chain-of-Thought) 模式。在 `/v1/chat/completions` 端點中，所有推理 token 被歸類為 `reasoning_content` 而非 `content`，造成可見回應為空。

```
Server log: "init: chat template, thinking = 1"
生成 225 tokens：225 個推理 token + 0 個可見 token
```

**嘗試方案：**
| 方案 | 結果 |
|------|------|
| `'thinking': {'type': 'disabled'}` API 參數 | ❌ llama-server b8679 不支援此格式 |
| `--no-thinking` 啟動旗標 | ❌ 無效旗標 |
| `--chat-template-kwargs '{"thinking": false}'` | ❌ 仍顯示 `thinking = 1` |
| **`/completion` 端點 + 手動 prompt 格式** | ✅ **完全繞過 chat template** |

**解決方案 — 手動建構 Gemma4 Prompt：**
```python
prompt = (
    f"<bos><start_of_turn>user\n{user_message}<end_of_turn>\n"
    "<start_of_turn>model\n"
)
# 停止條件: ["<end_of_turn>", "<eos>"]
# 不包含 <|think|> → 直接輸出可見回應
```

---

### 問題 5：DeepSeek-r1 回應空白

**根本原因：** DeepSeek-r1 使用 `<think>...</think>` 標籤包裹推理過程。當 `max_tokens=150` 時，token budget 全被推理消耗完，可見輸出為空。

**解決方案：**
```python
# 1. DeepSeek 給予 3 倍 token budget
effective_max = max_tokens * 3 if "deepseek" in model else max_tokens

# 2. 清除 <think> 區塊
import re
def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
```

---

## 快速開始 / Quick Start

### 前置需求

```bash
# Python 環境
pip install httpx fastapi uvicorn psutil pillow

# Ollama (Windows)
# 下載: https://ollama.ai
ollama pull gemma3:1b
ollama pull deepseek-r1:1.5b

# llama-server (已包含於 bin/llama-cpp/)
# Gemma4 GGUF (需自行下載 5.03 GB)
# https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF
# 儲存至: E:\LLMmodel\gemma4_textonly\google_gemma-4-E4B-it-Q4_K_M.gguf
```

### 執行 Benchmark

```bash
cd E:\Gemma4_E4B_Project
python tests/benchmark_all_models.py
# 結果輸出至 tests/benchmark_results.json

python tests/generate_report.py
# 報告輸出至 docs/comparison_report.md
```

### 啟動 Gemma4 API Server (FastAPI)

```bash
# 確保 Ollama 已啟動
ollama serve

# 啟動 FastAPI (port 8000)
python api/gemma4_api_server.py
```

### 直接使用 llama-server (繞過 Ollama)

```bash
# 純 CPU 模式 (GT 1030 無法使用 GPU)
bin\llama-cpp\llama-server.exe \
  -m "E:\LLMmodel\gemma4_textonly\google_gemma-4-E4B-it-Q4_K_M.gguf" \
  -ngl 0 -c 512 -t 4 -np 1 \
  --host 127.0.0.1 --port 8080

# 測試推論
curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "<bos><start_of_turn>user\n你好<end_of_turn>\n<start_of_turn>model\n", "n_predict": 50}'
```

---

## 專案結構 / Project Structure

```
Gemma4_E4B_Project/
├── api/
│   ├── gemma4_api_server.py   # FastAPI OpenAI-compatible server (port 8000)
│   └── gemma4_client.py       # Python client library
├── bin/
│   └── llama-cpp/             # llama-server b8679 Vulkan Windows binary
│       ├── llama-server.exe
│       ├── llama-cli.exe
│       └── ggml-vulkan.dll    # Vulkan GPU backend
├── docs/
│   └── comparison_report.md   # 自動生成的比較報告
├── prompts/
│   ├── MASTER_PROMPT.md       # 專案規格文件
│   └── TASKS.md               # 優化任務清單
├── scripts/
│   └── direct_llm.py          # 直接 GGUF 推論腳本
├── tests/
│   ├── benchmark_all_models.py # 主 benchmark (CPU 安全 + 順序執行)
│   ├── generate_report.py      # 報告生成器
│   ├── benchmark_results.json  # 最新測試結果
│   ├── phase1_2_results.json   # Phase 1-2 基礎設施測試
│   ├── phase3_results.json     # Phase 3 API 測試
│   └── assets/
│       ├── test_image.png      # 256×256 視覺測試圖 (house scene)
│       ├── test_text.txt       # 摘要測試文字
│       ├── image_b64.txt       # Base64 圖像
│       └── audio_b64.txt      # Base64 音訊
└── README.md
```

---

## CPU 安全機制詳解 / CPU Safety Architecture

本專案的核心挑戰是在 **4 核心 CPU** 上同時運行 LLM 推論和控制進程，而不造成系統當機。

```
┌─────────────────────────────────────────┐
│              i5-4460 (4 cores)          │
│                                         │
│  Core 0+1          Core 2+3            │
│  ┌──────────┐      ┌──────────────┐    │
│  │  Python  │      │   Ollama /   │    │
│  │ Benchmark│      │ llama-server │    │
│  │ (控制層) │      │  (推論引擎)  │    │
│  └──────────┘      └──────────────┘    │
│                                         │
│  OLLAMA_MAX_LOADED_MODELS=1             │
│  num_thread=2 (Modelfile)               │
│  Sequential execution (no parallel)     │
└─────────────────────────────────────────┘
```

**為何不能並行執行：**
- 兩個模型同時在 RAM 中 = 6+ GB，加上 OS = 記憶體壓力過高
- 推論 CPU 使用率 ~80% + Python ~20% = 接近 100% → 熱節流
- DDR3 頻寬 (17 GB/s) 被兩個推論進程競爭

---

## 硬體升級建議 / Hardware Upgrade Path

| 目標速度 | 需要的 GPU | 預估 tok/s | 預算 |
|---------|-----------|-----------|------|
| 3 tok/s (現況) | GT 1030 (2 GB) | ~2 tok/s CPU-only | 現有 |
| 10 tok/s (目標) | RTX 3060 (12 GB) | ~71 tok/s 理論 | ~NT$6,000 |
| 20 tok/s | RTX 3070 (8 GB) | ~107 tok/s 理論 | ~NT$9,000 |
| GPU 需求 | **最少 6 GB VRAM** | embedding table > 2 GB | — |

**升級後的 llama-server 指令：**
```bash
# RTX 3060 (12 GB VRAM, CUDA)
# 下載 CUDA 版本的 llama-server
llama-server.exe -m google_gemma-4-E4B-it-Q4_K_M.gguf \
  -ngl 999 -c 2048 -t 4 \
  --host 127.0.0.1 --port 8080
```

---

## FastAPI 多代理架構 / Multi-Agent Architecture

`api/gemma4_api_server.py` 提供 OpenAI 相容的端點，管理 6 個專用代理：

| 代理 | 系統提示特化 | 用途 |
|------|------------|------|
| `general` | 通用助理 | 一般問答 |
| `coder` | 程式設計師 | 程式碼生成/除錯 |
| `analyst` | 資料分析師 | 分析與洞察 |
| `translator` | 多語翻譯員 | 語言翻譯 |
| `vision` | 視覺描述者 | 圖像分析 |
| `planner` | 任務規劃師 | 計畫制定 |

**API 端點：**
```
POST /v1/chat/completions   # OpenAI 相容
POST /v1/completions        # 原始補全
GET  /v1/models             # 列出可用模型
POST /api/agent/{agent_id}  # 指定代理推論
```

---

## Fork 指南 / Fork Guide

### 如果你有更好的 GPU (≥6 GB VRAM)

1. Fork 此 repository
2. 下載 Gemma4 GGUF：`bartowski/google_gemma-4-E4B-it-Q4_K_M.gguf` (5.03 GB)
3. 修改 `tests/benchmark_all_models.py` 中的 `start_llamaserver()`:
   ```python
   # 改為 GPU 模式
   "-ngl", "999",  # 全 GPU
   ```
4. 執行 benchmark 並更新結果

### 如果你想換其他模型

修改 `OLLAMA_SEQUENCE` 和 `GEMMA4_GGUF` 路徑，加入你的模型。

### 如果你想貢獻修正

已知的未解問題：
- [ ] Gemma3:1b 視覺回應空白 (模型可能不支援視覺)
- [ ] Gemma4 無法使用 GPU (2 GB VRAM 限制)
- [ ] 需要驗證 CUDA 版 llama-server 的效能

---

## 參考資料 / References

| 主題 | 來源 |
|------|------|
| LLM 推論瓶頸分析 | [Making Deep Learning Go Brrrr — Horace He (2022)](https://horace.io/brrr_intro.html) |
| Q4_K_M 量化原理 | [GGUF Format — ggml.ai](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) |
| llama.cpp 架構 | [llama.cpp GitHub — ggml-org](https://github.com/ggml-org/llama.cpp) |
| Gemma4 模型說明 | [Google Gemma 4 — ai.google.dev](https://ai.google.dev/gemma) |
| bartowski GGUF | [bartowski/google_gemma-4-E4B-it-GGUF — HuggingFace](https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF) |
| MoE 架構原理 | [Outrageously Large Neural Networks (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538) |
| CPU Affinity (Windows) | [SetProcessAffinityMask — Microsoft Docs](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setprocessaffinitymask) |
| Ollama Modelfile | [Ollama Modelfile Docs](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) |

---

## Changelog

### v1.0.0 (2026-04-08)
- ✅ llama-server b8679 Vulkan backend for Gemma4 E4B
- ✅ 50× speed improvement: 0.04 → 2 tok/s
- ✅ DeepSeek `<think>` stripping fix
- ✅ Gemma4 thinking-mode bypass via raw `/completion`
- ✅ CPU overload prevention (affinity + sequential execution)
- ✅ Phase 1-3 infrastructure tests
- ✅ FastAPI multi-agent server (6 agents)
- ✅ Automated benchmark pipeline

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

> 本專案由 Claude Sonnet 4.6 協助開發  
> Built with assistance from Claude Code (Anthropic)
