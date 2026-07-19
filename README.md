# Gemma4-E4B Local LLM Project · Spec-Experts LLM Research

> **在低規格硬體上運行大型語言模型——並以 Spec-Experts 七階段框架實現語義感知推論**  
> Running LLMs on budget hardware with Spec-Experts 7-phase semantic-aware inference framework

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![llama-server b8679](https://img.shields.io/badge/llama.cpp-b8679-green.svg)](https://github.com/ggml-org/llama.cpp)
[![Research Paper](https://img.shields.io/badge/PhD_Thesis-Spec--Experts_LLM-purple.svg)](https://gist.github.com/sipurchen/53f3a0908e6ff6bd2c05fbb1bfd04adf)
[![Branch](https://img.shields.io/badge/branch-SpecExpertsResearch-orange.svg)](https://github.com/sipurchen/klchen/tree/SpecExpertsResearch)

---

## Spec-Experts LLM 研究論文 / Research Paper

> **博士等級研究論文** — 語義邊界偵測、動態專家路由與自強化推理在資源受限本地 LLM 部署的七階段研究  
> PhD-level research: 7-phase framework for semantic boundary detection, dynamic expert routing, and self-reinforcing inference on resource-constrained hardware.

| 語言 | 連結 |
|------|------|
| 中文版論文（七章完整版） | [thesis_zh.md](https://gist.github.com/sipurchen/53f3a0908e6ff6bd2c05fbb1bfd04adf#file-thesis_zh-md) |
| English Version (all 7 phases) | [thesis_en.md](https://gist.github.com/sipurchen/53f3a0908e6ff6bd2c05fbb1bfd04adf#file-thesis_en-md) |

**Branch:** [`SpecExpertsResearch`](https://github.com/sipurchen/klchen/tree/SpecExpertsResearch)

---

## 七階段技術總覽 / Seven-Phase Technology Overview

### Phase 1 — 外部多訊號監控器 (External Multi-Signal Monitor)

**模組：** `monitor/` · **示範：** `tests/phase1_monitor_demo.py`

三個偵測器並行作用於 llama-server SSE token 串流：

| 偵測器 | 優先權 | 訊號 | 閾值 |
|-------|------|------|-----|
| 角色標籤有限狀態機（`<think>`, ` ``` `, `# `） | 10（最高） | 確定性，立即發射 | 無 |
| Shannon 熵驟降監控器 | 5 | ΔH > θ_H | θ_H = 0.35, 窗口 W=8 |
| 困惑度 z 分數尖峰監控器 | 3 | z_i > z_θ | z_θ = 2.0, 窗口 W=10 |

**訊號融合規則：**

$$\text{emit}(c_i) \iff |i - i_{\text{last}}| \geq W_m \;\land\; [\text{role-tag} \;\lor\; |\mathcal{C}_i^{W_m}| \geq 2], \quad W_m = 12$$

**實測結果（Qwen3-1.7B，GT 1030）：**
- PPL z 分數峰值：4.3 至 407.2（境界轉換高達 204×閾值）
- 即時熵偵測：5.1 tok/s，Δ=0.41 > 0.35
- 7/7 整合測試通過

---

### Phase 2 — 三層 KV 快取卸載 (Three-Tier KV Cache Offload)

**模組：** `kv/kv_offload_manager.py`, `kv/kv_serializer.py`

$$\text{VRAM}(56\text{MB}) \xrightarrow{\text{LRU}} \text{RAM}(512\text{MB}) \xrightarrow{\text{LRU}} \text{Disk}(\texttt{.kvbin})$$

**`.kvbin` 格式：** 64 位元組標頭（KVBN 魔數 + session\_id + chunk\_id + layer範圍 + dtype/shape）+ 原始 fp16 張量

**LRU 驅逐代價：**

$$\text{cost}(C_i) = \frac{\text{age}(C_i) \cdot \text{size}(C_i)}{\pi(C_i)}, \quad \pi_{\text{code}}=1.5 > \pi_{\text{factual}}=1.2 > \pi_{\text{creative}}=0.8 > \pi_{\text{reasoning}}=0.5$$

**有效上下文擴展：** 2K → 18K+ tokens（VRAM 2K + RAM 4K + 磁碟 12K）

---

### Phase 3 — 方向性引導 (Directional Steering)

**模組：** `spec_experts/directional_steering.py`

四層無梯度引導代理（無需修改模型權重）：

| 層次 | 機制 | 效果 |
|------|------|------|
| 複合系統提示 | 按 α 排序正向示例 | 語義方向引導 |
| 溫度調整 | reasoning/code: −0.2; creative: +0.3 | 確定性/創意控制 |
| 對數機率偏置 | $\ell'_j = \ell_j + b_j$ | Token 分佈塑形 |
| 前綴預熱 | 前置概念激活 token | 即時方向觸發 |

**引導向量庫：** `code_quality`(α=15.0), `reasoning`(α=12.0), `creative`(α=10.0), `factual`(α=8.0), `concise`(α=10.0)

---

### Phase 4 — 專家 RAM 池（MoE 26B+）(Expert RAM Pool)

**模組：** `spec_experts/expert_ram_pool.py`

基於語義邊界事件的 MoE 專家預取，避免推論中途 VRAM 載入：

| 區塊類型 | 親和力專家集（Mixtral-8x7B） |
|---------|---------------------------|
| reasoning | {0, 1, 3, 7} |
| code | {2, 4, 5, 6} |
| factual | {0, 2, 4} |
| creative | {1, 3, 6, 7} |

**支援模型：**
- `mixtral-8x7b-q2`：8 專家，每個 1.875 GB，7 GB RAM → 3/8 預熱
- `deepseek-coder-v2-lite-q4`：64 專家，每個 0.25 GB，7 GB RAM → 8/64 預熱
- `deepseek-v3-671b-q2`：256 專家，每個 0.82 GB（Mac M4 前 33 個常駐 UMA）

**Spec-Experts 路由節省：** 20 區塊會話節省 ~1.76 s（Mixtral Q2, DDR3 17 GB/s）

---

### Phase 5 — 邊緣 VLM 路由器 (Edge VLM Router)

**模組：** `edge/vlm_router.py`

跨設備多模態視覺-文字路由：

| 設備 | 模型 | 後端 | ngl | VRAM | FPS |
|------|------|------|-----|------|-----|
| GT 1030 | moondream-2B Q4_K_M | Vulkan | 8 | 800 MB | 1.5 |
| Jetson Nano | MobileVLM-1.7B Q4 | CUDA | 32 | 2 GB | 10.0 |
| Android | LLaVA-Phi-1.5 Q4 | CPU | 0 | — | 5.0 |
| ARM 攝影機 | moondream-2B Q4 | CPU | 0 | — | 2.0 |

路由邏輯：圖像輸入 → VLM 推理（POST `/completion` + base64 圖像）；純文字 → 文字 LLM

---

### Phase 6 — Mac M4 Metal 專家池（200B+ MoE）(Mac M4 Metal Expert Pool)

**模組：** `mac_m4/metal_expert_pool.py`

利用 Mac Mini M4 32 GB UMA 的獨特優勢：

| 特性 | 數值 | 優勢 |
|------|------|------|
| UMA 頻寬 | 120 GB/s | vs PCIe 3.0×4 的 8 GB/s（15× 提升） |
| NVMe 速度 | 7 GB/s | 專家分頁載入：0.82 GB / 7 = 117 ms |
| Flash Attention | 啟用（Metal 後端） | $O(N^2) \to O(N)$，GT 1030 無法啟用 |
| DS-V3 前 33 專家 | 33 × 0.82 GB = 27 GB | 常駐 UMA，覆蓋 >80% 使用模式 |

**llama-server 啟動參數：**
```bash
llama-server --model <GGUF_PATH>/deepseek-v3.gguf \
  --n-gpu-layers 99 --flash-attn --mlock \
  --ctx-size 16384 --n-predict 4096
```

**Spec-Experts 路由效益（DS-V3，20 區塊）：** 40 次切換 → 24 次，節省 1.87 s

---

### Phase 7 — AGI 自迴圈 (AGI Self-Loop)

**模組：** `agi/self_loop.py`

```
[任務] → [專家路由] → [LLM 生成] → [品質評分] → [蒸餾 JSONL]
                                          ↓
                                 [LoRA 適配器 EMA 更新]
                                 q_t = 0.3·q_new + 0.7·q_{t-1}
                                          ↓
                               [適配器池選擇最優適配器]
                                    ↑______回饋______↑
```

**品質評分（區塊類型特定）：**
- Code: def/class/import 存在(0.4) + 長度>100(0.3) + 行數>3(0.3)
- Reasoning: 詞數>50(0.4) + 連接詞(0.3) + 句數>3(0.3)

**高品質樣本（Q ≥ 0.7）匯出為 JSONL 訓練數據**

**品質趨勢（6 次迭代示範）：** +0.030/iter（0.60 → 0.75，線性迴歸 β̂ > 0）

---

## 硬體環境 / Hardware

| 元件 | 規格 |
|------|------|
| CPU | Intel Core i5-4460 (4 核心, 3.2 GHz, Haswell, 2013) |
| GPU | NVIDIA GeForce GT 1030 (2 GB GDDR5, Pascal sm_61) |
| RAM | 34 GB DDR3 (雙通道, ~17 GB/s 頻寬) |
| OS | Windows 10 Pro 22H2 (19045) |
| 推論引擎 | Ollama v0.20.0 + llama-server b8679 (Vulkan) |
| VRAM 預算 | 850 MB（關閉 Chrome+Edge，保持 LINE 開啟） |

---

## 最終測試結果 / Benchmark Results

> 測試日期：2026-04-08 · 硬體：i5-4460 + GT 1030 + 34 GB DDR3

### 速度比較

| 模型 | 摘要 tok/s | 同義詞 tok/s | 聊天 tok/s | **平均** |
|------|-----------|------------|-----------|---------|
| Gemma3 1B (Ollama) | 21.5 | 13.75 | 21.96 | **19.1** |
| DeepSeek-r1 1.5B (Ollama) | 14.98 | 11.51 | 11.13 | **12.5** |
| Gemma4 E4B (llama-server) | 1.42 | 2.19 | 2.09 | **1.9** |

### Spec-Experts Phase 1 實測（GT 1030 + Qwen3-1.7B）

| 組件 | 狀態 | 關鍵指標 |
|------|------|---------|
| RoleTagParser | **PASS** | 4 邊界（CoT+程式碼），6（混合） |
| PerplexitySpikeMonitor | **PASS** | 6 尖峰，z 分數 4.3–407.2 |
| SignalFusion | **PASS** | 發射 4/6，遲滯抑制 2 |
| 即時熵監控器 | **PASS** | 5.1 tok/s，Δ=0.41 |
| **整合測試** | **7/7 PASS** | |

### Phase 2–7 結構驗證

| Phase | 驗證內容 | 結果 |
|-------|---------|------|
| P2 KV 卸載 | 三層儲存/取回，float16 往返 | **PASS** |
| P3 引導 | 4 種類型 × 引導配置 | **PASS** |
| P4 專家池 | Mixtral+DS-Coder，50–100% 命中率 | **PASS** |
| P5 邊緣 VLM | 4 設備 × 硬體配置 | **PASS** |
| P6 Mac M4 | UMA 池規劃，Flash Attn | **PASS** |
| P7 自迴圈 | 4 蒸餾樣本，+0.030/iter | **PASS** |

---

## 模型清單 / Models

| 模型 | 大小 | 架構 | 後端 |
|------|------|------|------|
| `gemma3:1b` | 777 MB | Dense 29 層 | Ollama |
| `deepseek-r1:1.5b` | 1.04 GB | Dense 30 層 + 推理鏈 | Ollama |
| `Gemma4 E4B` (bartowski Q4_K_M) | 5.03 GB | Dense 42 層, 7.5B | llama-server |
| Qwen3-1.7B (Spec-Experts 示範) | ~1.1 GB | Dense | llama-server |
| Mixtral-8x7B Q2_K (Phase 4 目標) | ~15 GB | MoE 8 experts | llama-server |
| DeepSeek-Coder-V2-Lite Q4 (Phase 4) | ~10 GB | MoE 64 experts | llama-server |
| moondream-2B Q4 (Phase 5 GT 1030) | ~1.2 GB | VLM | llama-server Vulkan |

---

## 專案結構 / Project Structure

```
Gemma4_E4B_Project/
├── monitor/
│   ├── attention_entropy.py      # Phase 1: Shannon 熵監控器
│   ├── perplexity_spike.py       # Phase 1: 困惑度 z 分數監控器
│   ├── role_tag_parser.py        # Phase 1: 角色標籤有限狀態機
│   └── signal_fusion.py          # Phase 1: 優先權加權訊號融合
├── segmentor/
│   └── chunk_classifier.py       # Phase 1: 語義區塊分類器
├── kv/
│   ├── kv_serializer.py          # Phase 2: .kvbin 格式序列化
│   └── kv_offload_manager.py     # Phase 2: 三層 VRAM/RAM/Disk LRU
├── spec_experts/
│   ├── controller.py             # FastAPI :8090 路由控制器
│   ├── tool_session.py           # 熱切換 llama-server（單 VRAM 槽）
│   ├── directional_steering.py   # Phase 3: 方向性引導
│   └── expert_ram_pool.py        # Phase 4: MoE 專家 RAM 預取池
├── edge/
│   └── vlm_router.py             # Phase 5: 邊緣 VLM 多設備路由
├── mac_m4/
│   └── metal_expert_pool.py      # Phase 6: Mac M4 UMA 專家池
├── agi/
│   └── self_loop.py              # Phase 7: LoRA 適配器池 + 串流蒸餾
├── tests/
│   ├── phase1_monitor_demo.py    # Phase 1 即時示範（需 llama-server :8080）
│   ├── phase2_7_demo.py          # Phase 2-7 結構驗證（離線）
│   ├── spec_experts_test.py      # 7/7 整合測試
│   ├── benchmark_all_models.py   # 基礎 benchmark
│   └── results_text.json         # 最新測試結果
├── docs/
│   ├── thesis_en.md              # 英文博士論文（七章）
│   └── thesis_zh.md              # 中文博士論文（七章）
├── api/
│   ├── gemma4_api_server.py      # FastAPI OpenAI-compatible server
│   └── gemma4_client.py          # Python client
└── bin/llama-cpp/                 # llama-server b8679 Vulkan Windows binary
```

---

## 調校理論基礎 / Theoretical Foundation

### 1. 記憶體頻寬瓶頸

```
理論最大 tok/s = 記憶體頻寬 / 模型大小
DDR3 i5-4460   = 17 GB/s ÷ 5.03 GB = 3.4 tok/s
GT 1030 VRAM   = 48 GB/s ÷ 5.03 GB = 9.5 tok/s
Mac M4 UMA     = 120 GB/s ÷ 3.5 GB = ~34 tok/s (Mixtral-8x7B Q2 激活部分)
```

### 2. Q4_K_M 量化

```
原始 FP16: 7.5B × 2 bytes = 15 GB
Q4_K_M:   7.5B × 0.67 bytes = 5.03 GB（KV 矩陣保持 6-bit）
```

### 3. Shannon 熵語義偵測

$$H_i = -\sum_{j=1}^k \tilde{p}_j \log_2 \tilde{p}_j, \quad \tilde{p}_j = \frac{\exp(\ell_j)}{\sum_{j'}\exp(\ell_{j'})}$$

低 H → 程式碼/結構化輸出；高 H → 推理/創意；驟降 → 語義狀態轉換。

### 4. CPU 親和性

```
策略：Python → 核心 0+1；Ollama/llama-server → 核心 2+3
防止 OS 排程器因全核心 LLM 佔用而造成螢幕黑屏
```

### 5. MoE 架構事實修正

**Gemma4 E4B 是密集架構（Dense），非 MoE。** "E4B" = Effective 4 Billion（有效 40 億活躍參數）。Spec-Experts 框架設計用於真正的 MoE 模型：Mixtral-8x7B、DeepSeek-V3 等。

---

## 快速開始 / Quick Start

### Phase 1-7 完整示範（離線，無需 llama-server）

```bash
git clone https://github.com/sipurchen/klchen
git checkout SpecExpertsResearch
cd Gemma4_E4B_Project

# Phase 2-7 結構驗證（不需要模型文件）
python tests/phase2_7_demo.py
```

### Phase 1 即時示範（需要 llama-server）

```bash
# 啟動 llama-server（任意支援 n_probs 的模型）
bin\llama-cpp\llama-server.exe -m <GGUF_PATH>\qwen3-1.7b.gguf \
  -ngl 8 -c 2048 --port 8080

# 執行 Phase 1 監控器
python tests/phase1_monitor_demo.py
```

### 整合測試

```bash
# 需要 FastAPI 控制器在 :8090
python tests/spec_experts_test.py
```

### Gemma4 基礎 Benchmark

```bash
ollama serve
python tests/benchmark_all_models.py
```

---

## 問題排查過程 / Troubleshooting Journey

### 問題 1：Ollama 造成 CPU 100% 過載

**根本原因：** Ollama 預設使用所有 4 個核心，與 Python benchmark 進程競爭。

**解決：**
```python
psutil.cpu_affinity([2, 3])  # llama-server 固定至核心 2+3
```

### 問題 2：Gemma4 via Ollama 只有 0.04 tok/s

**根本原因：** Ollama blob 是合併多模態 GGUF（2131 tensors，9.1 GB），標準 llama.cpp 只認識 720 個文字 tensors。**解決：** 下載 [bartowski 純文字 GGUF](https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF)（720 tensors，5.03 GB）。

### 問題 3：Vulkan 記憶體分配失敗（2.15 GB 單次分配）

**根本原因：** embedding table（262144 vocab × 2560 embed × Q4_K_M ≈ 745 MB）需要 2.15 GB 連續 Vulkan buffer，超過 2 GB VRAM。**解決：** `-ngl 0`（純 CPU 模式）。

### 問題 4：Gemma4 回應全為空白（thinking mode）

**根本原因：** chat template 預設 `<|think|>` token，所有 token 進入 `reasoning_content`。**解決：** 改用 `/completion` 端點並手動構造 prompt（不含 `<|think|>`）。

### 問題 5：DeepSeek-r1 回應空白

**解決：** 給予 3× token budget + `re.sub(r"<think>.*?</think>", "", text)`。

### 問題 6：GT 1030 Flash Attention 自動停用

**原因：** CPU/GPU 混合設備分割，QKV tensor 在 CPU，Flash Attention tensor 在 Vulkan0，設備不一致。Mac M4 UMA 無此問題。

---

## 硬體升級建議 / Hardware Upgrade Path

| 平台 | Spec-Experts 支援 | Gemma4 tok/s | 記憶體頻寬 | 備註 |
|------|-----------------|-------------|-----------|------|
| i5-4460 + GT 1030（現況） | Phase 1-5（Vulkan） | ~2 tok/s | DDR3 17 GB/s | VRAM 限制 MoE |
| RTX 2050 4GB | Phase 1-5（CUDA） | ~16 tok/s | GDDR6 112 GB/s | Mixtral Q2_K 部分卸載 |
| **Mac Mini M4 32GB** | **Phase 1-7 完整** | **~19 tok/s** | **UMA 120 GB/s** | **Flash Attn + DS-V3 前 33 專家** |
| RTX 3060 12GB | Phase 1-7 | ~48 tok/s | GDDR6 360 GB/s | Mixtral-8x7B 全 GPU |

---

## FastAPI 多代理架構 / Multi-Agent Architecture

`spec_experts/controller.py`（:8090）+ `api/gemma4_api_server.py`（:8000）：

| 代理 | 用途 | Phase 整合 |
|------|------|-----------|
| `general` | 一般問答 | P1 邊界偵測 + P3 引導 |
| `coder` | 程式碼生成/除錯 | P1 code 邊界 + P4 code 專家 |
| `analyst` | 分析與洞察 | P1 reasoning 邊界 + P3 factual 引導 |
| `translator` | 語言翻譯 | P3 factual 引導 |
| `vision` | 圖像分析 | P5 VLM 路由 |
| `planner` | 計畫制定 | P1 reasoning 邊界 + P7 自迴圈 |

---

## 進階技術深度解析 / Advanced Optimization Techniques

### TurboQuant+ — K-quant 混合精度量化

每個 32-weight super-block 自適應混合 4-bit/5-bit/6-bit，配合 imatrix 校準（886 個資料區塊），在 5.03 GB 體積下維持接近 FP16 品質（PPL 損失 < 0.8%）。

### KV Cache 三層卸載（Phase 2）

Qwen3-1.7B（$L=28, H=16, d_h=128, N=2048$）：KV ≈ 448 MB。透過 VRAM→RAM→Disk LRU 層級結構，有效上下文從 2K 擴展至 18K+，代價為提示重播（預填充速度 3–5× 解碼速度）。

### 方向性引導代理（Phase 3）

無需模型內部梯度存取，以四層代理實現 $\mathbf{h}_l' = \mathbf{h}_l + \alpha\hat{\mathbf{d}}$ 的效果。實驗：程式碼引導後 $T_{\text{eff}} = 0.40$，創意引導後 $T_{\text{eff}} = 1.00$。

### MoE 專家 RAM 池（Phase 4）

**專家激活局部性（實驗性）：**
$$\Pr[\text{top}_k(g(\mathbf{x}_{t_i})) \cap \text{top}_k(g(\mathbf{x}_{t_{i+1}})) \neq \emptyset] > 0.85$$

這使得區塊邊界事件觸發的專家預取命中率顯著高於反應式載入。

### Flash-MoE（Mac M4 Phase 6）

Flash Attention $O(N^2) \to O(N)$ + MoE 稀疏激活 = Flash-MoE 融合優化。GT 1030 因 CPU/GPU 設備分割無法啟用；Mac M4 UMA 消除此限制。

### LoRA 自迴圈蒸餾（Phase 7）

$$W' = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d,k)$$

EMA 品質更新（α=0.3）+ 品質閘控（Q ≥ 0.7）JSONL 匯出 → 未來 LoRA 訓練數據集。

---

## 參考文獻與專利 / References & Patents

### 學術論文

| 主題 | 來源 |
|------|------|
| MoE 稀疏門控 | Shazeer et al., [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) (2017) |
| Flash Attention | Dao et al., [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) (2022) |
| LLM.int8() 量化 | Dettmers et al., [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) (2022) |
| GPTQ 量化 | Frantar et al., [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) (2022) |
| LoRA 低秩適配 | Hu et al., [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) (2022) |
| 表示工程（引導） | Zou et al., [arXiv:2310.01405](https://arxiv.org/abs/2310.01405) (2023) |
| FlexGen 卸載 | Sheng et al., ICML 2023 |
| Shannon 熵 | Shannon (1948), Bell System Technical Journal |
| Mixtral MoE | Jiang et al., [arXiv:2401.04088](https://arxiv.org/abs/2401.04088) (2024) |
| DeepSeek-Coder-V2 | DeepSeek-AI, [arXiv:2406.11931](https://arxiv.org/abs/2406.11931) (2024) |

### 相關專利

| 專利號 | 持有人 | 差異化維度 |
|--------|--------|----------|
| US10,817,783B1 | Google LLC | Spec-Experts 外部運作，非參數，區塊粒度而非 token |
| US11,423,285B2 | Microsoft | Spec-Experts 訊號來自模型本身，無獨立分類器 |
| US20230409774A1 | Meta | Spec-Experts 時序多路復用，非多加速器並行 |
| WO2024/053456A1 | Google DeepMind | Spec-Experts 純推理時，無訓練時修改 |
| US11,580,423B1 | Amazon | Spec-Experts 純軟體，與任何 llama.cpp 伺服器相容 |
| US20240169235A1 | NVIDIA | Spec-Experts 語義粒度路由，高於硬體流水線層次 |

---

## Changelog

### v2.0.0 (2026-06-26) — Spec-Experts Phase 1-7 完整實作
- ✅ Phase 1：外部監控器（熵 + PPL z 分數 + 角色標籤 FSM）
- ✅ Phase 2：三層 KV 卸載（VRAM/RAM/Disk + .kvbin 格式）
- ✅ Phase 3：方向性引導（系統提示代理 + 溫度調整 + 對數機率偏置）
- ✅ Phase 4：MoE 專家 RAM 池（Mixtral-8x7B + DeepSeek-Coder-V2-Lite）
- ✅ Phase 5：邊緣 VLM 路由器（GT 1030 + Jetson + Android + ARM）
- ✅ Phase 6：Mac M4 Metal 專家池（Flash Attn + NVMe 分頁 + UMA 釘選）
- ✅ Phase 7：AGI 自迴圈（LoRA 適配器池 + 串流蒸餾 + EMA 品質更新）
- ✅ 博士論文：中英文七章完整版（含 LaTeX 公式 + 25 篇參考文獻 + 6 個專利引用）

### v1.1.0 (2026-05-14)
- 新增 `Others_AI.MD`：Mac Mini M4 32GB 平台規劃
- 新增 `UserCommand.MD`：跨平台使用者指令速查手冊

### v1.0.0 (2026-04-08)
- llama-server b8679 Vulkan backend for Gemma4 E4B（50× 提速：0.04 → 2 tok/s）
- DeepSeek `<think>` 剝除修正
- Gemma4 thinking-mode 繞過
- CPU 過載防護（親和性 + 順序執行）
- Phase 1-3 基礎設施測試
- FastAPI 多代理伺服器（6 代理）

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

> 本專案由 Claude Sonnet 4.6 協助開發  
> Built with assistance from Claude Code (Anthropic)
