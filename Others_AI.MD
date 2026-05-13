# Others AI 平台規劃 / Other AI Platforms Planning

> 本文件記錄在**非主力硬體（i5-4460 + GT 1030）**以外，其他平台上部署本地 LLM 的規劃與分析。  
> 主力平台請參閱 [README.md](README.md) 及 [docs/RTX2050_LLM_Harness_Plan.md](docs/RTX2050_LLM_Harness_Plan.md)

---

## 平台清單 / Platform Index

| 平台 | 硬體 | 狀態 | 重點 |
|------|------|------|------|
| [Mac Mini M4 32GB](#mac-mini-m4-32gb) | Apple M4, 32 GB 統一記憶體 | 📋 規劃中 | Apple Silicon UMA, llama.cpp Metal |
| [RTX 2050 Windows](#rtx-2050-windows-ref) | RTX 2050 4 GB, i-series | 🔗 見獨立文件 | Harness AI Agents, Flash-Load |

---

## Mac Mini M4 32GB

### 硬體規格

| 項目 | 規格 |
|------|------|
| 晶片 | Apple M4 (3nm, 10 核心 CPU: 4P + 6E) |
| GPU | M4 GPU (10 核心, 3.6 TFLOPS FP32) |
| 統一記憶體 | 32 GB LPDDR5X (共用 CPU/GPU) |
| 記憶體頻寬 | **120 GB/s** |
| 儲存 | 256 GB / 512 GB NVMe SSD |
| TDP | ~20W (待機) / ~38W (峰值) |
| OS | macOS Sequoia 15.x |
| 推論引擎候選 | llama.cpp (Metal), Ollama (Metal backend), MLX |

---

### 理論推論速度 / Theoretical tok/s

```
理論最大 tok/s = 記憶體頻寬 / 模型大小

Mac Mini M4 32GB:
  Gemma4 E4B Q4_K_M (5.03 GB): 120 GB/s ÷ 5.03 GB ≈ 23.9 tok/s  ← 全 GPU 可達
  Gemma3 1B         (0.78 GB): 120 GB/s ÷ 0.78 GB ≈ 154 tok/s    ← 限 CPU 頻寬
  Qwen3-8B Q4_K_M  (~5.5 GB): 120 GB/s ÷ 5.5 GB  ≈ 21.8 tok/s
  Qwen3-14B Q4_K_M (~9.5 GB): 120 GB/s ÷ 9.5 GB  ≈ 12.6 tok/s
  Qwen3-30B Q4_K_M (~19 GB):  120 GB/s ÷ 19 GB   ≈  6.3 tok/s   ← 需 ~22 GB VRAM

M4 效率係數 (Metal backend 實測): ~0.75–0.85
實際預估 Gemma4 E4B: 23.9 × 0.80 ≈ 19 tok/s
```

**對比現有平台：**

| 平台 | Gemma4 E4B 速度 | 備註 |
|------|----------------|------|
| i5-4460 + GT 1030 (現有) | ~2 tok/s | CPU-only, DDR3 17 GB/s |
| RTX 2050 (目標 Windows) | ~15–18 tok/s | GDDR6 112 GB/s × 0.65 |
| **Mac Mini M4 32GB** | **~19 tok/s** | 統一記憶體, 120 GB/s |
| RTX 3060 12GB | ~48 tok/s | GDDR6 360 GB/s |

---

### UMA 統一記憶體架構優勢 / Unified Memory Architecture

Apple Silicon 的**統一記憶體 (Unified Memory Architecture)** 對 LLM 推論有結構性優勢：

```
傳統 x86 架構 (i5-4460 + GT 1030):
  CPU RAM (DDR3, 17 GB/s)  ←→  PCIe 4x  ←→  GPU VRAM (GDDR5, 48 GB/s)
  模型大於 2GB → 無法放入 GPU → 全跑 CPU → 瓶頸 17 GB/s

Apple M4 UMA:
  CPU 和 GPU 共享同一塊實體記憶體池 (LPDDR5X, 120 GB/s)
  模型放入統一記憶體 → CPU 和 GPU 都以 120 GB/s 存取
  → 無需 PCIe 搬運開銷
  → 32 GB 可容納最大 ~22 GB 的量化模型 (系統保留約 8–10 GB)
```

**關鍵優點：**
- Gemma4 E4B (5.03 GB) 完整放入 GPU, 頻寬 120 GB/s
- 無 VRAM 上限瓶頸（32 GB 共享，全部可用於 LLM）
- 支援 Gemma4 multimodal GGUF（可嘗試原始 9.1 GB blob，含視覺/音訊）
- 低功耗：~38W vs RTX 3060 的 ~170W

---

### 模型規劃 / Model Plan

| 模型 | 大小 | 預估速度 | 特色 | 優先度 |
|------|------|---------|------|-------|
| Gemma4 E4B Q4_K_M | 5.03 GB | ~19 tok/s | 主力文字 LLM | ⭐⭐⭐ |
| Gemma4 E4B multimodal | ~9.1 GB | ~10 tok/s | 含視覺/音訊 | ⭐⭐ |
| Qwen3-8B Q4_K_M | ~5.5 GB | ~18 tok/s | 中文能力強 | ⭐⭐⭐ |
| Qwen3-14B Q4_K_M | ~9.5 GB | ~12 tok/s | 更強推理 | ⭐⭐ |
| Gemma3 4B Q4_K_M | ~2.8 GB | ~32 tok/s | 快速回應 | ⭐⭐ |
| Whisper Large v3 | ~1.5 GB | 即時 | 語音辨識 (ASR) | ⭐⭐⭐ |

---

### 推論框架選擇 / Inference Backend

#### 選項 A：llama.cpp (Metal)

```bash
# macOS Metal 後端（推薦）
brew install llama.cpp
# 或從源碼編譯 (Metal 加速)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release

# 啟動 Gemma4 E4B
llama-server \
  -m ~/LLMmodel/google_gemma-4-E4B-it-Q4_K_M.gguf \
  -ngl 999 \        # 全層放入 Metal GPU
  -c 4096 \         # 32GB 可支援大 context
  -t 4 \
  -fa \             # Flash Attention (Metal 支援)
  --host 127.0.0.1 --port 8080
```

**Metal 特性：**
- `-ngl 999` 讓所有層使用 Metal GPU，頻寬完整 120 GB/s
- `-fa` Flash Attention 在 Metal 上正常運作（無 CPU/GPU 混合問題）
- UMA 不需要顯式的 VRAM 分配限制

#### 選項 B：Ollama (Metal backend)

```bash
# Ollama for macOS 自動使用 Metal
brew install ollama
ollama serve
ollama pull gemma3:latest
ollama pull qwen3:8b

# API 使用方式與 Windows 版相同
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "prompt": "你好，請介紹一下你自己"
}'
```

#### 選項 C：MLX (Apple 原生框架)

```bash
# Apple MLX — 針對 Apple Silicon 最佳化
pip install mlx-lm

# 直接從 HuggingFace 載入
python -m mlx_lm.generate \
  --model mlx-community/gemma-4-E4B-it-4bit \
  --prompt "Hello, who are you?"
```

**MLX 優點：** 專為 M 系列晶片設計，理論上最高效率

#### 推薦選擇

| 場景 | 推薦後端 | 原因 |
|------|---------|------|
| 本專案相容性 | llama.cpp Metal | 與 Windows llama-server 指令最接近 |
| 新專案/日常使用 | Ollama Metal | 操作簡單，自動管理 |
| 最高效能 | MLX | Apple 原生，專為 M4 優化 |
| 多模態 (視覺/音訊) | llama.cpp Metal | 可載入原始 9.1 GB multimodal GGUF |

---

### Flash Attention 在 Mac Mini M4 的優勢

與 GT 1030 不同，Mac Mini M4 的 Metal 後端可以正常啟用 Flash Attention：

```
Windows GT 1030 問題:
  - CPU/GPU 混合設備 → Flash Attention 自動停用
  - "sched_reserve: Flash Attention was auto, set to disabled"

Mac Mini M4:
  - UMA 統一記憶體: CPU 和 GPU 使用相同記憶體空間
  - 無設備邊界 → Flash Attention 可以正常啟用
  - 效果: context 4096 時記憶體節省 ~75%
```

---

### 多模態功能規劃 / Multimodal Features

由於 Mac Mini M4 有 32 GB 統一記憶體，可以嘗試：

| 功能 | 方案 | 模型大小需求 |
|------|------|------------|
| 文字生成 | llama.cpp / Ollama | 5–10 GB |
| **圖像分析** | Gemma4 multimodal GGUF | ~9.1 GB |
| **語音辨識** | Whisper.cpp Metal | ~1.5 GB (Large v3) |
| **語音合成** | Kokoro / Coqui TTS | ~1–2 GB |
| **影片理解** | 逐幀+Gemma4 vision | ~9.1 GB |
| 多模態同時運行 | 所有上述 | ~14 GB total — 32 GB 可承載 |

---

### 與現有專案整合 / Integration with Current Project

本專案的 `api/gemma4_api_server.py` 提供 OpenAI 相容 API，Mac Mini M4 可以：

```python
# 選項 1: Mac Mini 作為 API Server (本機或區域網路)
# 在 Mac Mini 上啟動 llama-server
llama-server -m gemma4.gguf -ngl 999 --host 0.0.0.0 --port 8080

# Windows i5-4460 上的 client 連接 Mac Mini
LLAMA_SERVER_URL = "http://192.168.1.xxx:8080"  # Mac Mini IP
```

```python
# 選項 2: 跨平台 benchmark 比較
# 在 Mac Mini 上執行同樣的 tests/benchmark_all_models.py
# 去除 CPU affinity 邏輯（Mac 不需要核心保護）
```

---

### CPU 安全機制調整 / macOS CPU Safety

Mac Mini M4 不需要 Windows 版本的 CPU 過載保護：

| 項目 | Windows i5-4460 | Mac Mini M4 |
|------|----------------|-------------|
| CPU 核心 | 4 核心，容易過載 | 10 核心 (4P+6E)，綽綽有餘 |
| CPU 親和性設定 | 必須，防止螢幕黑屏 | **不需要** |
| `OLLAMA_MAX_LOADED_MODELS=1` | 必須 | 可設 2–3 |
| 並行推論 | 禁止 | 可以嘗試 |
| 記憶體壓力 | 高（34 GB DDR3, 低頻寬） | 低（32 GB 120 GB/s） |

---

### 預期效能對比摘要 / Performance Summary

```
Mac Mini M4 32GB Gemma4 E4B 預期:
  速度:    ~19 tok/s  (vs 現有 2 tok/s = 9.5× 提升)
  context: 4096 tokens (vs 現有 512)
  Flash Attention: 啟用 (vs 現有停用)
  multimodal: 可嘗試 (vs 現有不支援)
  多模型並行: 支援 (vs 現有禁止)
  功耗: ~38W (vs RTX 3060 ~170W)
  性價比: ⭐⭐⭐⭐⭐ (約 NT$15,000–18,000)
```

---

## RTX 2050 Windows (ref)

> 詳細規劃請見 [docs/RTX2050_LLM_Harness_Plan.md](docs/RTX2050_LLM_Harness_Plan.md)

**摘要：** RTX 2050 4 GB VRAM + Harness AI Agents 框架，使用 Flash-Load 動態管理多 expert，搭配 Runtime Directional Steering。Gemma4 E4B 預計 ~15–18 tok/s。

---

## 平台對比總覽 / Full Platform Comparison

| 平台 | Gemma4 E4B | 最大模型 | 功耗 | 預算 (NT$) | 適用場景 |
|------|-----------|---------|------|-----------|---------|
| i5-4460 + GT 1030 | 2 tok/s | ~5 GB | ~65W | 現有 | 基礎實驗 |
| RTX 2050 Win | ~16 tok/s | ~4 GB VRAM | ~45W + CPU | ~8,000 | Harness Agents |
| **Mac Mini M4 32GB** | **~19 tok/s** | **~22 GB** | **~38W** | **~15,000–18,000** | **多模態, 日常使用** |
| RTX 3060 Win | ~48 tok/s | ~12 GB | ~170W | ~12,000 + build | 高速推論 |
| RTX 4090 Win | ~200 tok/s | ~24 GB | ~450W | ~50,000+ | 專業研究 |

---

*最後更新: 2026-05-14*  
*維護: LocalDeploy*
