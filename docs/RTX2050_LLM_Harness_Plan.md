# RTX 2050 4GB VRAM — Harness AI Agents 初始規劃
<!-- Claude Cowork BEGIN: RTX2050 Harness AI Agents Plan -->
<!-- Encoding: UTF-8 -->

## 目的 / Purpose

以 RTX 2050 4 GB VRAM 為目標硬體，結合 Harness AI Agents 框架核心，
透過 **lazy-loading / flash-loading** 動態管理多個 LLM Expert，
搭配 audreyt/pi-ds4 的 **Runtime Directional Steering** 去除中國強制答案，
達到即時可用的 AIVtuber 推理基礎。

---

## 硬體基線 / Hardware Baseline

| 項目 | 規格 |
|------|------|
| GPU | RTX 2050 (Ampere, 4 GB GDDR6) |
| 頻寬 | 112 GB/s @ 45W / 96 GB/s @ 35W |
| CUDA | 12.x |
| CPU | 目標: i5-4460 以上 |
| RAM | 目標: 32 GB DDR4 (CPU Offload 用) |
| 目前基線 | i5-4460 + GT 1030 2 GB |

**tok/s 估算公式（CUDA Ampere）:**
```
tok/s ≈ (bandwidth_GB/s ÷ model_VRAM_GB) × 0.65
```

---

## Harness AI Agents 架構 / Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Harness Coordinator                           │
│                   (Python Orchestrator)                          │
│                                                                  │
│  Context Router → Task Classifier → Agent Dispatcher            │
│       ↓                  ↓                  ↓                   │
│  [History]        [Intent Tags]        [Model Pool]             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┼────────────────────────────┐
           ↓               ↓                            ↓
   ┌───────────────┐ ┌───────────────┐        ┌────────────────────┐
   │  Gatekeeper   │ │  Flash-Load   │        │  CPU-Offload Agent │
   │  Agent        │ │  Agent        │        │  (Lazy-Load)       │
   │               │ │               │        │                    │
   │ Gemma3 1B     │ │ Qwen3 1.7B    │        │ Qwen3.6-27B        │
   │ 0.68 GB VRAM  │ │ 1.1 GB VRAM  │        │ 17 GB → CPU RAM    │
   │ ~107 tok/s    │ │ + 0.6B spec  │        │ ~5–8 tok/s         │
   │ [常駐 GPU]    │ │ ~140 tok/s   │        │ [依需求喚醒]       │
   │               │ │ [替換路由器] │        │                    │
   └───────────────┘ └───────────────┘        └────────────────────┘
           │                                           │
           └──────────────┬────────────────────────────┘
                          ↓
              ┌───────────────────────┐
              │  Directional Steering │
              │  Layer (audreyt)      │
              │                       │
              │  FFN output patch     │
              │  DS4_FFN = -1.5       │
              │  (去除中國強制答案)   │
              └───────────────────────┘
```

---

## Agent 類型與 VRAM 策略 / Agent Types & VRAM Strategy

### 1. Gatekeeper Agent（常駐 / Always Resident）

| 項目 | 值 |
|------|-----|
| 模型 | Gemma3 1B Q4_K_M |
| VRAM | 0.68 GB |
| tok/s | ~107 |
| 職責 | Intent routing, chat filtering, short replies |
| 載入策略 | 開機後常駐 GPU，不換出 |

```bash
llama-server -m gemma3-1b-q4km.gguf \
  -ngl 999 -fa --port 11434 \
  --ctx-size 4096 --parallel 2
```

### 2. Flash-Load Agent（瞬時替換 / Flash Swap）

| 項目 | 值 |
|------|-----|
| 模型 | Qwen3 1.7B Q4_K_M + 0.6B draft |
| VRAM | 1.1 + 0.4 = 1.5 GB |
| tok/s | ~140（投機解碼） |
| 職責 | Complex chat, bilingual responses |
| 載入策略 | 需要時從磁碟 flash-load → 替換 Gatekeeper → 用完卸載 |

```bash
llama-server -m qwen3-1.7b-q4km.gguf \
  --draft-model qwen3-0.6b-q4km.gguf \
  --draft-max 8 \
  -ngl 999 -fa --port 11435
```

> ⚠️ 投機解碼僅對 **Dense 模型** 有效；MoE 模型實測淨負收益

### 3. CPU-Offload Agent（懶載入 / Lazy-Load）

| 項目 | 值 |
|------|-----|
| 模型 | Qwen3.6-27B Q4_K_M 或 Gemma4 26B-A4B |
| VRAM | 4 GB（attention 層） |
| RAM | 13–17 GB（expert FFN 層） |
| tok/s | ~5–8 |
| 職責 | Deep reasoning, code review, knowledge queries |
| 載入策略 | 只在需要長推理時喚醒；idle 超過 120s 後自動卸載 |

```bash
# Qwen3.6-27B CPU offload (RTX 2050)
llama-server -m qwen3.6-27b-q4km.gguf \
  -ngl 18 --cpu-moe \
  -fa --merge-qkv \
  --ctx-size 8192 --port 11436

# Gemma4 26B-A4B MoE (bartowski GGUF)
llama-server -m gemma4-26b-a4b-q4km.gguf \
  -ngl 18 --cpu-moe \
  -fa --merge-qkv \
  --ctx-size 8192 --port 11437
```

---

## Runtime Directional Steering（去除中國強制答案）

來源：[audreyt/pi-ds4](https://github.com/audreyt/pi-ds4)

**核心機制：**
- 在推論時對 FFN 輸出層施加低秩激活向量
- 不修改模型權重，純推論期間調整激活分佈
- `DS4_DIR_STEERING_FFN=-1.5`：Q4_K_M 建議值
- `DS4_DIR_STEERING_FFN=-2.0`：imatrix GGUF 最佳值

**實作方式（每次 llama-server 啟動時注入）：**
```bash
# 設定環境變數後啟動
$env:DS4_DIR_STEERING_FFN = "-1.5"
llama-server -m qwen3.6-27b-q4km.gguf -ngl 18 --cpu-moe ...
```

**效果（對 Qwen3.6 系列驗證）：**
- 天安門、台灣、西藏等敏感議題：轉為平衡/開放回應
- 反事實糾正能力：提升
- 一般推理/coding 品質：無明顯下降

---

## LLM 全域比較表 / Global LLM Comparison Table (2026-05-13)

### 本機 RTX 2050 執行能力

| 模型 | 大小 | RTX 2050 執行 | tok/s | Steering | 備注 |
|------|------|--------------|-------|----------|------|
| Qwen3 0.6B Q4_K_M | 0.4 GB | ✅ 純 GPU | ~182 | ✅ | 分類/路由用 |
| Gemma3 1B Q4_K_M | 0.68 GB | ✅ 純 GPU | **~107** | ✅ | 推薦 Gatekeeper |
| Qwen3 1.7B + 0.6B spec | 1.5 GB | ✅ 純 GPU | **~140** ⭐ | ✅ | 最佳品質/速度比 |
| Qwen3 4B Q4_K_M | 2.2 GB | ✅ 純 GPU | ~33 | ✅ | 高品質對話 |
| Qwen3 4B + 0.5B spec | 2.6 GB | ✅ 純 GPU | ~45 | ✅ | 品質+投機加速 |
| Qwen3.6-27B Q4_K_M | 17 GB | ⚠️ CPU Offload | ~5–8 | ✅ best | 深度推理用 |
| Gemma4 26B-A4B MoE | 17 GB | ⚠️ CPU Offload | ~8–12 | ✅ | MoE 快於同大小 Dense |
| Qwen3.6-35B-A3B MoE | 22 GB | ⚠️ CPU Offload | ~6–9 | ✅ | 大 MoE，超出 RAM |
| Kimi-K2.5 (1T MoE) | 240–600 GB | ❌ 需 256 GB RAM | — | ⚠️ | 本機不可行 |
| DeepSeek V4 Flash | 87 GB | ❌ 需 128 GB RAM | — | ✅ | audreyt 使用方案 |
| ChatGPT GPT-4o | Cloud | ❌ API only | API | system prompt | OpenAI 管控 |
| Gemini 2.5 Flash | Cloud | ❌ API only | API | N/A | Google 管控 |
| Claude Sonnet 4.6 | Cloud | ❌ API only | API | N/A | Anthropic |

> **注意：** MoE 模型在 CPU Offload 時 expert FFN 在 CPU RAM，attention 仍在 GPU → 比同大小 Dense 快 ~30%

### 雲端 API 比較（參考用）

| 服務 | 模型 | 回應品質 | 延遲 | 中國審查 | 推薦用途 |
|------|------|---------|------|---------|---------|
| OpenAI | GPT-4o | ⭐⭐⭐⭐⭐ | ~1–2s | 依地區 | Coding, 複雜推理 |
| Anthropic | Claude Sonnet 4.6 | ⭐⭐⭐⭐⭐ | ~1s | 無 | 長文, 分析 |
| Google | Gemini 2.5 Flash | ⭐⭐⭐⭐ | <1s | 依地區 | 快速任務 |
| Groq | Llama 4 Scout 17B | ⭐⭐⭐ | 極低 | 無 | 本地替代 |
| Ollama Cloud | Gemma4 4B | ⭐⭐⭐ | <0.5s | 有部分 | 本地備援 |

---

## 各類別最佳 LLM 推薦 / Per-Category Recommendations

### Coding（程式開發）

| 優先級 | 模型 | 平台 | 理由 |
|--------|------|------|------|
| 1st | Claude Sonnet 4.6 (API) | 雲端 | 最強 long-context 程式分析 |
| 2nd | Qwen3.6-27B local | RTX 2050 CPU | 本機最強 coding，開源 |
| 3rd | Qwen3 4B Q4_K_M | RTX 2050 GPU | 33 tok/s，足夠快，離線可用 |

### Image 生成（圖像）

| 優先級 | 模型 | 平台 | 理由 |
|--------|------|------|------|
| 1st | Stable Diffusion (local) | CPU/GPU | GGUF 不適用；需專用框架 |
| 2nd | Stable Audio + SD3 | GPU | 已在 download_models.ps1 |
| Cloud | Firefly / DALL-E 3 | API | 最高品質，付費 |

> LLM 在 image 任務不直接生成；推薦用 SD 系列或 Firefly API

### Video 生成（影片）

| 優先級 | 模型 | 平台 | 理由 |
|--------|------|------|------|
| 1st | CogVideoX 5B (local) | GPU+CPU | 已下載，RTX 2050 可用 |
| 2nd | Wan2.1 T2V 14B | CPU | 高品質，但慢（CPU offload） |
| Cloud | RunwayML / Sora | API | 最高品質，付費 |

### Audio TTS / 音樂生成

| 優先級 | 模型 | 平台 | 理由 |
|--------|------|------|------|
| TTS 1st | CosyVoice2 0.5B | CPU | 中文高品質 TTS，已下載 |
| TTS 2nd | ChatTTS | CPU | 高變化性，適合歌唱 |
| Music 1st | MusicGen Large | CPU | 已下載，無歌詞限制 |
| Music 2nd | ACE-Step v1 3.5B | GPU | 新款，品質更佳 |
| STT | SenseVoice Small | CPU | 高準確多語言 STT，已下載 |

### Common Chat（日常對話 / AIVtuber）

| 優先級 | 模型 | 平台 | tok/s | 理由 |
|--------|------|------|-------|------|
| 1st | Qwen3 1.7B + spec | RTX 2050 | ~140 | 速度最快，品質可接受 |
| 2nd | Gemma3 1B | RTX 2050 | ~107 | 簡單對話 Gatekeeper |
| 3rd | Ollama Cloud Gemma4 4B | 雲端 | API | 低延遲，備援 |

---

## 總體表現最佳 LLM / Best Overall

| 場景 | 推薦 | 理由 |
|------|------|------|
| **RTX 2050 即時對話** | **Qwen3 1.7B + 0.6B spec** | ~140 tok/s，Dense，投機解碼有效 |
| **RTX 2050 高品質** | **Qwen3.6-27B CPU offload** | 最強本機推理，接受低速 |
| **Steering 最佳效果** | **Qwen3.6-27B + DS4_FFN=-1.5** | 最完整去審查方案 |
| **無限制雲端** | **Claude Sonnet 4.6** | 本 AI 使用中，無中國管控 |

---

## 實作路線圖 / Implementation Roadmap

```
Phase 1 (現在 → RTX 2050 到手):
  ├── 確認 llama.cpp CUDA 12.x build 正常
  ├── 測試 Gemma3 1B @ 107 tok/s
  ├── 測試 Qwen3 1.7B + spec @ 140 tok/s
  └── 驗證 DS4_DIR_STEERING_FFN=-1.5

Phase 2 (RTX 2050 穩定後):
  ├── 實作 Harness Coordinator Python 層
  ├── 實作 Agent Pool 管理 (load/unload/query)
  ├── 整合 klchen 的 Twitch chat pipeline
  └── 整合 audreyt 的 directional steering

Phase 3 (未來 Mac Mini M4):
  ├── 複製 .gguf 檔案 (跨平台)
  ├── 換 mlx-lm + Qwen3.6-27B (native Metal)
  ├── 或 llama.cpp Metal + Gemma4 26B-A4B
  └── 移除 --cpu-moe (32 GB 統一記憶體全載入)
```

---

## 參考資源 / References

- audreyt/pi-ds4: https://github.com/audreyt/pi-ds4
- klchen project: https://github.com/sipurchen/klchen
- bartowski GGUF: https://huggingface.co/bartowski
- llama.cpp: https://github.com/ggml-org/llama.cpp
- 詳細 Suggestions: `<PROJECT_ROOT>\CC_For_Codex_Suggestions.MD`

<!-- Claude Cowork END: RTX2050 Harness AI Agents Plan -->
