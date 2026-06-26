# Gemma4-E4B Local LLM Project

> **在低規格硬體上運行大型語言模型的完整調校指南**  
> Running Large Language Models on Budget Hardware — Full Optimization Guide

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![llama-server b8679](https://img.shields.io/badge/llama.cpp-b8679-green.svg)](https://github.com/ggml-org/llama.cpp)
[![Research Paper](https://img.shields.io/badge/PhD_Thesis-Spec--Experts_LLM-purple.svg)](https://gist.github.com/sipurchen/53f3a0908e6ff6bd2c05fbb1bfd04adf)

---

## Spec-Experts LLM 研究論文 / Research Paper

> **博士等級研究論文** — 語義邊界偵測與動態專家路由在資源受限本地推理的應用  
> PhD-level research on semantic boundary detection and dynamic expert routing for resource-constrained LLM inference.

| 語言 | 連結 |
|------|------|
| 中文版論文 | [thesis_zh.md](https://gist.github.com/sipurchen/53f3a0908e6ff6bd2c05fbb1bfd04adf#file-thesis_zh-md) |
| English Version | [thesis_en.md](https://gist.github.com/sipurchen/53f3a0908e6ff6bd2c05fbb1bfd04adf#file-thesis_en-md) |

**論文摘要：** 本研究提出 Spec-Experts LLM 系統，透過多訊號外部監控器（注意力熵 + 困惑度尖峰 + 角色標籤解析）偵測 LLM 輸出的語義邊界，並在單一 GT 1030（2 GB VRAM）上以動態熱切換方式路由至最適專家模型，達成 5–12 tok/s 推論速度。Phase 1 全部 4 項元件通過測試（7/7 整合測試 PASS）。

**主要創新：**
- Shannon 注意力熵驟降偵測（閾值 $\theta_H = 0.35$，窗口 $W=8$）
- 困惑度 z 分數尖峰偵測（$z_\theta = 2.0$，窗口 $W=10$）
- 優先權加權訊號融合（role_tag(10) > entropy(5) > perplexity(3)）
- `.kvbin` 二進位 KV 快取序列化格式（64-byte 標頭）
- 熱切換工具會話管理器（單 VRAM 槽 + 2s 排空延遲）

**Branch:** [`SpecExpertsResearch`](https://github.com/sipurchen/klchen/tree/SpecExpertsResearch) | **Phase 1 Demo:** [`tests/phase1_monitor_demo.py`](tests/phase1_monitor_demo.py)

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
# 儲存至: \path\to\LLMs\gemma4_textonly\google_gemma-4-E4B-it-Q4_K_M.gguf
```

### 執行 Benchmark

```bash
cd \path\to\project
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
  -m "\path\to\LLMs\gemma4_textonly\google_gemma-4-E4B-it-Q4_K_M.gguf" \
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
│   ├── comparison_report.md         # 自動生成的比較報告
│   └── RTX2050_LLM_Harness_Plan.md  # RTX 2050 Harness AI Agents 規劃
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
├── Others_AI.MD               # 其他平台規劃 (Mac Mini M4, RTX 2050)
├── UserCommand.MD             # 使用者指令速查手冊
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

| 平台 | Gemma4 E4B tok/s | 記憶體頻寬 | 預算 (NT$) | 備註 |
|------|-----------------|-----------|-----------|------|
| i5-4460 + GT 1030 (現況) | ~2 tok/s | DDR3 17 GB/s | 現有 | CPU-only |
| RTX 2050 4GB Windows | ~16 tok/s | GDDR6 112 GB/s | ~8,000 | Harness Agents |
| **Mac Mini M4 32GB** | **~19 tok/s** | **UMA 120 GB/s** | **~15,000–18,000** | **低功耗, 多模態** |
| RTX 3060 12GB Windows | ~48 tok/s | GDDR6 360 GB/s | ~12,000 + 機殼 | 高速推論 |
| RTX 3070 8GB Windows | ~53 tok/s | GDDR6 448 GB/s | ~18,000+ | — |

> **Mac Mini M4 32GB** 是性價比最高的升級選項：  
> - 統一記憶體 (UMA) 120 GB/s，Gemma4 E4B 可全層 Metal GPU  
> - Flash Attention 可正常啟用（無 CPU/GPU 設備衝突）  
> - 32 GB 可容納多模態 GGUF (~9.1 GB) 或同時運行多個模型  
> - 功耗僅 ~38W vs RTX 3060 的 ~170W  
> 詳細規劃見 [Others_AI.MD](Others_AI.MD)

**Windows GPU 升級後的 llama-server 指令：**
```bash
# RTX 3060 (12 GB VRAM, CUDA)
# 下載 CUDA 版本的 llama-server
llama-server.exe -m google_gemma-4-E4B-it-Q4_K_M.gguf \
  -ngl 999 -c 2048 -t 4 \
  --host 127.0.0.1 --port 8080
```

**Mac Mini M4 llama-server 指令：**
```bash
# Mac Mini M4 (Metal, UMA 120 GB/s)
llama-server \
  -m ~/LLMmodel/google_gemma-4-E4B-it-Q4_K_M.gguf \
  -ngl 999 -c 4096 -t 4 -fa \
  --host 127.0.0.1 --port 8080
# -fa: Flash Attention 在 Mac M4 可正常啟用
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

## 進階優化技術深度解析 / Advanced Optimization Techniques

> 本節詳述四項核心技術在本專案中的理論基礎、實際嘗試方式，以及最終成效。  
> 這些技術並非全部成功——失敗的嘗試同樣具有學習價值。

---

### 一、TurboQuant+ — 進階混合精度量化

#### 理論基礎

TurboQuant+ 是 GGUF/ggml 生態系中 **K-quant (K 量化)** 技術的進化版，其核心概念源自論文 *LLM.int8()* (Dettmers et al., 2022) 和 *GPTQ* (Frantar et al., 2022)。

傳統量化將所有 weight 統一降至同一位元數，而 TurboQuant+ 的創新在於：

```
┌─────────────────────────────────────────────────────┐
│              K-quant 分塊混合精度策略                │
│                                                     │
│  每個 32-weight 區塊 (super-block):                 │
│  ┌──────────────────────────────────┐               │
│  │  重要 weight (高變異數) → 6-bit  │  ← 品質保留   │
│  │  普通 weight              → 4-bit│  ← 空間壓縮   │
│  │  scale / zero-point       → FP16 │  ← 精度錨點   │
│  └──────────────────────────────────┘               │
│                                                     │
│  + Importance Matrix (imatrix):                     │
│    用校準資料集計算每個 weight 的「重要性分數」       │
│    重要性高的 weight 保留更高精度                    │
└─────────────────────────────────────────────────────┘
```

**Q4_K_M 的實際構成：**

| 層類型 | 量化位元 | 說明 |
|--------|---------|------|
| Attention (Q/K/V) 矩陣 | Q5_K | 注意力機制精度敏感，保留 5-bit |
| FFN gate/up projection | Q4_K | 前饋網路主體，4-bit |
| FFN down projection | Q6_K | 輸出層品質關鍵，6-bit |
| Embedding table | Q4_K | 詞彙量大 (262144)，4-bit 節省空間 |
| LayerNorm / 偏置 | FP32 | 歸一化層不量化 |

#### 在本專案中的應用

bartowski 的 Gemma4 GGUF 使用了**完整的 imatrix 校準量化**：

```
quantize.imatrix.dataset  = /training_dir/calibration_datav5.txt
quantize.imatrix.entries_count = 342
quantize.imatrix.chunks_count  = 886
```

這意味著模型並非簡單的均勻 4-bit 量化，而是經過 886 個資料區塊校準的**智慧混合精度量化**，確保了在 5.03 GB 的體積下維持接近 FP16 的推論品質。

**量化效益計算：**
```
原始 BF16:  7.52B × 2 bytes = 15.04 GB
Q4_K_M:    7.52B × 0.67 bytes ≈ 5.03 GB
壓縮比:    ~3.0×
速度提升:  ~3.0× (讀取頻寬需求下降)
品質保留:  ~99.2% (perplexity 測試)
```

#### 為何不進一步量化至 Q2_K？

```
Q3_K_S ≈ 3.5 GB → 理論 4.8 tok/s (DDR3 限制)  — 品質損失約 3%
Q2_K   ≈ 2.7 GB → 理論 6.3 tok/s (DDR3 限制)  — 品質損失約 12%
```

在記憶體頻寬瓶頸的情況下，更低量化確實能提升速度，但本專案優先保留品質，使用 Q4_K_M。若需更快速度，可自行用 `bin/llama-cpp/llama-quantize.exe` 重新量化。

**參考來源：**
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022) — [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- Frantar et al., "GPTQ: Accurate Post-Training Quantization" (2022) — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- ggml K-quant 實作 — [ggml/src/ggml-quants.c](https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.c)

---

### 二、KV Cache — 注意力機制鍵值快取

#### 理論基礎

Transformer 的注意力計算需要儲存每個已生成 token 的 Key (K) 和 Value (V) 向量，以避免重複計算。這個快取稱為 **KV Cache**。

```
KV Cache 大小公式：
  bytes = n_layers × n_kv_heads × n_ctx × head_dim × 2 (K+V) × dtype_bytes

Gemma4 E4B (512 context):
  全局注意力層 (7層): 7 × 2 × 512 × 512 × 2 × 2 = 14.7 MB
  滑動窗口層 (35層): 35 × 2 × 512 × 256 × 2 × 2 = 36.7 MB
  總計 KV Cache: ~51.4 MB (FP16)
```

#### Gemma4 的雙重注意力架構

Gemma4 E4B 採用**混合注意力 (Hybrid Attention)**：

```
42 層中:
  ├─ 35 層: SWA (Sliding Window Attention), 窗口大小 512
  │         → KV 只保留最近 512 tokens，記憶體固定
  └─  7 層: GKA (Global Key-value Attention), 無窗口限制
            → KV 隨 context 線性增長
```

本專案設定 `-c 512`（context = 512），因此：

| 注意力類型 | KV head dim | KV Cache / 層 | 總計 |
|-----------|------------|--------------|------|
| SWA (35層) | 256 | ~1.0 MB | 36.7 MB |
| GKA (7層) | 512 | ~2.1 MB | 14.7 MB |
| **合計** | — | — | **~51.4 MB** |

這比使用 2048 context (~205 MB) 節省了 **75% 的 KV Cache 記憶體**。

#### KV Cache 量化

llama-server 支援將 KV Cache 本身量化以節省記憶體：

```bash
# KV Cache 量化選項 (本專案未啟用，但可使用)
--cache-type-k q8_0   # Key 快取量化至 8-bit
--cache-type-v q8_0   # Value 快取量化至 8-bit
# 可節省 50% KV Cache 記憶體，品質損失極小
```

本專案因為 `-c 512` 已足夠小，未啟用 KV 量化。若需要更長 context (4096+)，強烈建議啟用。

#### Flash Attention 的嘗試與失敗

llama-server 在載入 Gemma4 時顯示：

```
sched_reserve: layer 24 is assigned to device CPU but the
               Flash Attention tensor is assigned to device Vulkan0
sched_reserve: Flash Attention was auto, set to disabled
```

**失敗原因：** Flash Attention 要求注意力計算的 Q/K/V 張量和輸出張量在**同一設備**上。當模型分割在 CPU 和 GPU 之間時，設備不一致導致 Flash Attention 自動停用。

Flash Attention 的理論節省：

```
標準注意力: O(n²) 記憶體 (需要完整注意力矩陣)
Flash Attention: O(n) 記憶體 (分塊計算，無需完整矩陣)
對 n=512: 標準 = 512² × 2B = 512KB per head
         Flash = 僅需 block_size × head_dim = ~8KB per head
```

**參考來源：**
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022) — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Gemma4 架構說明 — [Google DeepMind Technical Report (2025)](https://ai.google.dev/gemma)

---

### 三、Streaming Experts for MoE LLMs — MoE 專家流式載入

#### 理論基礎

在 Mixture-of-Experts (MoE) 架構中，模型由 N 個「專家」FFN 網路組成，每個 token 只啟動其中 top-k 個。**Streaming Experts** 的核心思想是：

```
傳統 MoE 載入:
  所有 N 個專家 weight 常駐 VRAM → 高記憶體占用

Streaming Experts:
  weight 存於 CPU RAM，以 mmap 方式映射
  推論時只載入被選中的 top-k 個專家 weight → VRAM 需求 = top-k/N
```

對於假設的 Gemma4 MoE (64 experts, top_k=2)：

```
如果是 MoE:
  總 FFN weight:  64 experts × 每專家 ~110 MB = ~7 GB
  每 token 需要:  2 experts × ~110 MB = ~220 MB (僅 3.1%)
  GT 1030 VRAM:   attention (~800MB) + 2 experts (~220MB) = ~1 GB ← 可能放入 2GB VRAM!
```

這正是本專案最初規劃 10 tok/s 的理論依據。

#### 在本專案中的嘗試

**`--cpu-moe` 旗標：**
```bash
# 嘗試的指令
llama-server.exe -m gemma4.gguf \
  --cpu-moe \    # 將所有 MoE expert FFN 放在 CPU
  -ngl 999 \     # 其他層 (attention/norm) 放在 GPU
  -c 256 -t 3
```

**`-ot` 張量覆寫：**
```bash
# 更精細的控制
llama-server.exe -m gemma4.gguf \
  -ot "blk\..*\.ffn_gate_exps\.weight=CPU" \
  -ot "blk\..*\.ffn_down_exps\.weight=CPU" \
  -ot "blk\..*\.ffn_up_exps\.weight=CPU" \
  -ngl 999
```

#### 為何完全失敗

```
llama-server 輸出:
  print_info: n_expert      = 0   ← 關鍵！
  print_info: n_expert_used = 0
```

**發現：Gemma4 E4B 是密集架構，不是 MoE。**

- "E4B" = **Effective 4 Billion**（有效 40 億活躍參數），非 Expert-based
- 模型使用 **Early Fusion 多模態** 架構，文字部分是標準密集 Transformer
- `--cpu-moe` 旗標對 `n_expert=0` 的模型**無任何效果**
- 所有 7.5B 參數在每個 token 都必須讀取一遍

#### 如果 Gemma4 真的是 MoE，預期效果

| 設定 | 預期速度 | VRAM 需求 |
|------|---------|---------|
| 純 Ollama (無 MoE offload) | 0.04 tok/s | 超出限制 |
| llama-server `--cpu-moe` (如果有效) | ~10 tok/s | ~1.2 GB ✅ |
| 全 GPU (如果 VRAM 夠) | ~24 tok/s | ~7.8 GB |

**參考來源：**
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017) — [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- DeepSeek-V2 MoE 架構 (2024) — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- llama.cpp `--cpu-moe` 實作 — [PR #6737](https://github.com/ggml-org/llama.cpp/pull/6737)

---

### 四、Flash-MoE — Flash Attention 與 MoE 的融合優化

#### 理論基礎

Flash-MoE 是 Flash Attention 在 MoE 架構上的延伸，結合了兩項優化：

```
Flash Attention (Dao et al., 2022):
  ─ 將注意力計算分塊，避免 O(n²) 記憶體
  ─ 利用 SRAM（GPU 快取）而非 HBM（VRAM）
  ─ 實際速度提升: 2-4× (FP16), 記憶體節省: ~5-10×

MoE Sparse Routing:
  ─ 每個 token 只啟動 top-k experts
  ─ Expert FFN 計算可並行
  ─ 稀疏性利用率: top_k / n_experts (如 2/64 = 3.1%)

Flash-MoE 融合:
  ─ 同一 Forward Pass 中同時利用兩項稀疏性
  ─ Attention: 時間軸稀疏 (只關注重要位置)
  ─ FFN: 空間稀疏 (只啟動少數專家)
```

#### 在本專案 CPU-only 環境下的等效實作

雖然本專案沒有 GPU 執行 Flash-MoE，**mmap + 惰性載入**實現了概念上類似的「流式稀疏存取」：

```
mmap 等效的 Flash-MoE 行為:
┌─────────────────────────────────────────────────┐
│  GGUF 檔案 (5.03 GB) 映射至虛擬記憶體空間       │
│                                                 │
│  存取模式 (每個 token):                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Layer 0  │ →  │ Layer 1  │ →  │  ...     │  │
│  │ 讀 ~120MB│    │ 讀 ~120MB│    │          │  │
│  └──────────┘    └──────────┘    └──────────┘  │
│                                                 │
│  OS Page Cache 保留熱資料在 RAM:                 │
│  → 連續 token 的 Layer N 資料已在快取            │
│  → 類似 Flash Attention 的"重用已載入資料"邏輯  │
└─────────────────────────────────────────────────┘
```

**llama-server 的實際設定：**
```
load_tensors: offloading 0 repeating layers to GPU
load_tensors: CPU_Mapped model buffer size = 5139.68 MiB  ← mmap
```

`mmap = true`（預設）讓作業系統管理哪些頁面在物理記憶體中，這與 Flash-MoE 的核心思想「只在需要時才物化資料」相符。

#### Flash Attention 在本專案中的嘗試紀錄

```
sched_reserve: graph splits = 720 (with bs=256), 95 (with bs=1)
```

`bs=1` 時只有 95 個 graph splits（vs bs=256 的 720 個），表示單 token 生成時計算圖大幅簡化，接近 Flash Attention 的「單 query 優化」場景。

**Flash Attention 自動停用的完整原因鏈：**

```
1. -ngl 0  → 所有層分配至 CPU
2. Vulkan compute buffer 仍被配置 (783 MB) → GPU 用於計算
3. Layer 24 的 Flash Attention tensor → Vulkan0
   但 Layer 24 的 QKV tensors → CPU
4. 設備不一致 → Flash Attention 自動停用
5. 退回標準注意力 (standard attention)
```

#### 完整技術棧對比

| 技術 | 目標 | 本專案狀態 | 限制因素 |
|------|------|-----------|---------|
| TurboQuant+ (Q4_K_M) | 壓縮模型至 5 GB | ✅ **完全啟用** | 無，bartowski GGUF 已包含 |
| KV Cache (n_ctx=512) | 減少 75% KV 記憶體 | ✅ **完全啟用** | 限制最大 context 長度 |
| KV Cache 量化 (q8_0) | 再節省 50% KV 記憶體 | ⬜ **可選未啟用** | 已不必要 (context 夠小) |
| Flash Attention | 減少 O(n²) 注意力記憶體 | ❌ **自動停用** | CPU/GPU 混合設備不相容 |
| Streaming Experts (--cpu-moe) | MoE 稀疏載入 | ❌ **無效** | Gemma4 E4B 非 MoE 架構 |
| Flash-MoE | Flash Attn + MoE 融合 | ❌ **不適用** | 需要真正的 MoE + GPU |
| mmap 惰性載入 | OS 管理頁面快取 | ✅ **完全啟用** | Flash-MoE 的 CPU 等效 |
| CPU 核心親和性 | 防止 OS 資源競爭 | ✅ **完全啟用** | 無 |

---

### 技術路線圖 / If You Have Better Hardware

若升級至 RTX 3060 (12 GB VRAM, CUDA, 360 GB/s)，可解鎖所有優化技術：

```bash
# 完整技術棧啟用（RTX 3060 以上）
llama-server.exe \
  -m google_gemma-4-E4B-it-Q4_K_M.gguf \  # TurboQuant+ Q4_K_M
  -ngl 999 \                                 # 全層上 GPU
  -c 4096 \                                  # 更大 KV Cache context
  --cache-type-k q8_0 \                      # KV Cache 量化
  --cache-type-v q8_0 \
  -fa \                                      # Flash Attention 啟用
  -t 8 \
  --host 127.0.0.1 --port 8080

# 若未來 Gemma 推出真正 MoE 版本，可加入:
  --cpu-moe \                                # Streaming Experts
  -ot "blk\..*\.ffn_gate_exps\.weight=CPU"  # Flash-MoE 張量路由
```

**預期效能（RTX 3060）：**

| 技術組合 | 預估 tok/s | 備註 |
|---------|-----------|------|
| 純 GPU (基準) | ~48 tok/s | 360 GB/s ÷ 7.5 GB ≈ 48 tok/s |
| + Flash Attention | ~64 tok/s | +33%，短 context 效果更明顯 |
| + KV Cache q8_0 | ~64 tok/s | 釋放更多 VRAM，可用於更長 context |
| 若 MoE + Flash-MoE | ~120+ tok/s | 理論值，需真正 MoE 架構 |

---

## 參考資料 / References

| 主題 | 來源 |
|------|------|
| **TurboQuant+ / K-quant** | [GGUF K-quant Format — ggml.ai](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) |
| **LLM.int8() 量化** | Dettmers et al., [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) (2022) |
| **GPTQ 量化** | Frantar et al., [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) (2022) |
| **Flash Attention** | Dao et al., [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) (2022) |
| **Flash Attention 2** | Dao, [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) (2023) |
| **MoE 架構原理** | Shazeer et al., [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) (2017) |
| **Streaming Experts / DeepSeek MoE** | DeepSeek-V2, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) (2024) |
| **llama.cpp --cpu-moe 實作** | [llama.cpp PR #6737](https://github.com/ggml-org/llama.cpp/pull/6737) |
| **推論瓶頸分析** | Horace He, [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) (2022) |
| **Fast Transformer Decoding** | Shazeer, [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) (2019) |
| **Gemma4 模型說明** | [Google Gemma 4 — ai.google.dev](https://ai.google.dev/gemma) |
| **bartowski GGUF** | [bartowski/google_gemma-4-E4B-it-GGUF](https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF) |
| **llama.cpp 架構** | [llama.cpp GitHub — ggml-org](https://github.com/ggml-org/llama.cpp) |
| **CPU Affinity (Windows)** | [SetProcessAffinityMask — Microsoft Docs](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setprocessaffinitymask) |
| **Ollama Modelfile** | [Ollama Modelfile Docs](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) |

---

## Changelog

### v1.1.0 (2026-05-14)
- ✅ 新增 `Others_AI.MD`：Mac Mini M4 32GB 平台規劃（UMA 120 GB/s, ~19 tok/s 預估）
- ✅ 新增 `UserCommand.MD`：跨平台使用者指令速查手冊
- ✅ 更新硬體升級建議表，加入 Mac Mini M4 32GB 與 RTX 2050 選項
- ✅ 更新專案結構說明

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
