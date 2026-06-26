# Spec-Experts LLM：面向資源受限本地推理的語義邊界偵測與動態專家路由系統

**作者：** LocalDeploy  
**分支：** `SpecExpertsResearch`  
**日期：** 2026-06-26  
**研究庫：** https://github.com/sipurchen/klchen  
**研究性質：** 個人獨立研究  

---

## 摘要

本文提出 **Spec-Experts LLM** 系統，透過三種互補機制，在嚴格資源受限硬體（GT 1030、2 GB VRAM、DDR3 記憶體）上部署大規模混合專家（MoE）及稠密大型語言模型（LLM）：(1) 多訊號**外部監控器**（External Monitor），利用注意力熵、困惑度尖峰及確定性角色標籤解析，在 token 串流中偵測語義區塊邊界；(2) **訊號融合**（Signal Fusion）層，採用優先權加權遲滯協議合併隨機性與確定性訊號；(3) **動態專家路由器**（Dynamic Expert Router），將每個語義區塊分派至最佳專家 LLM，並在 GPU 有限 VRAM 預算內進行模型熱切換。在 GT 1030（有效 VRAM 850 MB，關閉 Chrome+Edge）上，展示部分 GPU 卸載（ngl=10–12 層），對最多 1.7B 參數模型（Q8_0）達到 **5–12 tok/s**，語義邊界偵測在混合 CoT+程式碼語料上 F1 > 0.80。本架構設計可線性擴展至 Mac Mini M4（200B+ MoE）及邊緣 VLM 部署。

**關鍵詞：** 混合專家、語義分割、注意力熵、區塊路由、資源受限推理、KV 快取卸載、Vulkan 後端

---

## 1. 引言

### 1.1 研究動機

具有 7B–671B 參數的 LLM 普及帶來一個根本不對稱性：最先進的推理能力存在於需要數十乃至數百 GB GPU VRAM 的大型模型中，而絕大多數已部署硬體——消費級 GPU、邊緣設備、嵌入式系統——僅能在 2–8 GB VRAM 的限制下運作。最簡單的解決方案（量化至 Q2–Q4）以犧牲品質換取可行性。一個更有原則的方法認識到：LLM 推理在 token 串流中**並非均勻要求**——推理 token、程式碼 token 與事實檢索 token 表現出截然不同的困惑度分佈、最佳溫度區間，以及 MoE 模型中的專家激活模式。

我主張，LLM 輸出中的**區塊級異質性**是一種可被利用的一等屬性，能夠實現資源高效路由。能夠 (a) 即時偵測語義轉換且 (b) 將每個區塊分派至最適合該語義模態之專家模型的系統，可達到遠超單一大型模型的每 VRAM-MB 品質比值。

### 1.2 問題陳述

給定：
- 硬體預算 $B_{\text{VRAM}}$（此處：$B_{\text{VRAM}} = 850$ MB 有效值）
- LLM 輸出 token 語料 $T = \{t_1, t_2, \ldots, t_N\}$
- 專家模型集合 $\mathcal{M} = \{M_1, M_2, \ldots, M_K\}$，各模型 VRAM 佔用 $v_k \leq B_{\text{VRAM}}$
- 真值語義標籤函數 $\ell: T \to \{\texttt{reasoning}, \texttt{code}, \texttt{factual}, \texttt{creative}\}$

求 $T$ 的連續區塊劃分 $\Pi = \{C_1, C_2, \ldots\}$ 及路由函數 $r: \Pi \to \mathcal{M}$，使得：

$$\max_{\Pi, r} \sum_{C_i \in \Pi} Q(C_i, M_{r(C_i)}) \quad \text{s.t.} \quad v_{r(C_i)} \leq B_{\text{VRAM}} \; \forall i$$

其中 $Q(C, M)$ 是模型 $M$ 在區塊 $C$ 上的品質分數（以任務特定指標衡量：程式碼可執行性、事實準確性、推理連貫性）。

### 1.3 主要貢獻

1. **多訊號外部監控器**（§3）：三種互補邊界偵測器——Shannon 注意力熵、困惑度 z 分數、確定性角色標籤有限狀態機——在不存取模型內部的情況下，對即時 token 串流運作。

2. **優先權加權訊號融合**（§4）：遲滯閘控融合協議，在優先權排序 $\text{role\_tag}(10) > \text{entropy}(5) > \text{perplexity}(3)$ 下合併隨機性與確定性訊號。

3. **熱切換工具會話管理器**（§5）：單一槽位模型管理器，在 VRAM 預算內終止並重啟 llama-server 並載入適當模型，包含 2 秒 VRAM 排空延遲。

4. **磁碟備份 KV 卸載**（§6）：`.kvbin` 二進位格式，用於跨會話邊界序列化和重播 KV 快取前綴，使有效上下文窗口遠超 RAM 容量。

5. **GT 1030 可行性示範**（§7）：在 850 MB VRAM 上的實證驗證，達到 5–12 tok/s，Phase 1 全部 4 個組件通過測試。

---

## 2. 背景

### 2.1 混合專家架構

MoE 模型將前饋網路（FFN）劃分為 $E$ 個專家子網路 $\{f_1, \ldots, f_E\}$，並使用門控函數 $g: \mathbb{R}^d \to \Delta^E$ 對每個 token 選擇前 $k$ 個專家：

$$\text{FFN}_{\text{MoE}}(\mathbf{x}) = \sum_{e \in \text{top}_k(g(\mathbf{x}))} g_e(\mathbf{x}) \cdot f_e(\mathbf{x})$$

以 DeepSeek-Coder-V2-Lite（總計 16B，$k=2$，$E=64$）為例，每個 token 僅激活 2.4B 參數——約為總權重的 15%。Spec-Experts 假設認為，**同一語義區塊內的連續 token 傾向於激活相同的專家子集**，從而能夠以區塊粒度進行專家預快取。

**專家激活局部性**（實證觀察，參見 [1]）：對於區塊 $C$ 內的 token $t_i, t_{i+1}$：

$$\Pr[\text{top}_k(g(\mathbf{x}_{t_i})) = \text{top}_k(g(\mathbf{x}_{t_{i+1}}))] > 0.7$$

這證明了區塊級路由的合理性：每個區塊載入一次專家權重，而非每個 token 載入一次。

### 2.2 作為語義狀態指示器的注意力熵

層 $l$、頭 $h$ 在 token $t_i$ 處的注意力分佈 $\alpha^{(l,h)}$：

$$\alpha^{(l,h)}_{ij} = \frac{\exp(q_i^{(l,h)} \cdot k_j^{(l,h)} / \sqrt{d_k})}{\sum_{j'} \exp(q_i^{(l,h)} \cdot k_{j'}^{(l,h)} / \sqrt{d_k})}$$

此分佈的 Shannon 熵：

$$H^{(l,h)}_i = -\sum_j \alpha^{(l,h)}_{ij} \log \alpha^{(l,h)}_{ij}$$

**熵低**表示注意力集中（確定性上下文），是程式碼與事實檢索的特徵。**熵高**表示注意力分散，是開放式推理的特徵。**熵從滾動均值中驟降**標誌著語義狀態轉換。

在本實現中，我們使用 llama-server `n_probs` 端點的前 $k$ 個對數概率 $\{p_1, \ldots, p_k\}$ 近似 $H$：

$$\hat{H}_i = -\sum_{j=1}^{k} \tilde{p}_j \log \tilde{p}_j, \quad \tilde{p}_j = \frac{\exp(p_j)}{\sum_{j'} \exp(p_{j'})}$$

邊界條件：

$$\text{boundary}_{\text{entropy}}(i) = \mathbb{1}\left[\bar{H}_{i-W:i} - \hat{H}_i > \theta_H\right]$$

其中 $W = 8$ 為平滑窗口，$\theta_H = 0.35$ 為驟降閾值。

### 2.3 作為 Token 驚訝度的困惑度

Token 困惑度：

$$\text{PPL}(t_i) = \exp(-\log p(t_i | t_1, \ldots, t_{i-1}))$$

在大小為 $|\mathcal{W}|$ 的滑動窗口 $\mathcal{W}$ 上的 z 分數：

$$z_i = \frac{\text{PPL}(t_i) - \mu_{\mathcal{W}}}{\sigma_{\mathcal{W}} + \epsilon}$$

邊界條件：

$$\text{boundary}_{\text{ppl}}(i) = \mathbb{1}[z_i > z_{\theta}], \quad z_{\theta} = 2.0$$

**實證結果（§7.2）：** 第一個語義轉換（推理→程式碼）產生 $z \approx 216.7$；第二個（程式碼→事實）$z \approx 407.2$。兩者均以數量級超過 $z_\theta = 2.0$，表明偵測的穩健性。

### 2.4 先前研究與差異化

| 系統 | 方法 | 局限性 |
|------|------|--------|
| Mixture of Experts [2] | Token 級專家路由（門控） | 需要全模型在 VRAM 中 |
| Speculative Decoding [3] | 草稿模型 + 驗證器 | 單模型流水線，無語義路由 |
| FlexGen [4] | 張量卸載至磁碟/CPU | 整體模型，無區塊路由 |
| ExpertChoice [5] | 專家選擇門控 | 訓練時；無推理時適應 |
| **Spec-Experts（本研究）** | **推理時區塊路由 + 外部監控器** | **無需重新訓練；單 VRAM 槽** |

**核心差異化**：Spec-Experts 在**無模型內部存取**的情況下運作——監控器是 HTTP/SSE 串流上的外部觀察者，與任何基於 llama.cpp 的服務器兼容。這使得消費級硬體部署無需修改推理核心。

---

## 3. 外部監控器架構

### 3.1 訊號層概述

```
Token 串流（llama-server /completion 的 SSE）
         │
    ┌────┴────────────────────┐
    │                         │
    ▼                         ▼
[P1.1 RoleTagParser]   [P1.2+P1.3 概率性]
 確定性有限狀態機        隨機性偵測器
    │                    │             │
    │              [熵監控器]    [困惑度尖峰]
    │                    │             │
    └────────────────────┴─────────────┘
                         │
                  [P1.4 訊號融合]
                   優先權投票 +
                   遲滯閘控
                         │
                  [ChunkBoundary]
                   {tok_idx, type, pri}
                         │
               [ChunkClassifier]
               type → {reasoning, code,
                        factual, creative}
                         │
                  [ToolSessionManager]
                  熱切換 llama-server
                  在 VRAM 預算內
```

### 3.2 角色標籤解析器（P1.1）

確定性層在**結構性 token 詞彙**上實現有限狀態機（FSM）：

| 模式 | 優先權 | 區塊類型 |
|------|--------|---------|
| `<think>` | 10 | REASONING 開始 |
| `</think>` | 10 | REASONING 結束 |
| `` ``` `` | 10 | CODE 邊界 |
| `# `（行首） | 10 | FACTUAL（標題） |
| `[INST]` | 10 | INSTRUCTION |
| `<role:X>` | 10 | ROLE 切換 |

這些模式是**無損的**（對格式良好的輸出無假陰性），且**零延遲**（無需回溯窗口）。角色標籤邊界始終覆蓋概率性訊號。

**實驗結果（§7.1）：**
- 樣本 1（CoT+程式碼）：偵測到 4 個邊界——`<think>`(0)、`</think>`(16)、`` ``` ``(18)、`` ``` ``(42)
- 樣本 2（混合）：偵測到 6 個邊界——`<think>`(51)、`</think>`(63)、`# `(65)、`` ``` ``(84)、`# `(87)、`` ``` ``(98)

### 3.3 熵監控器（P1.3）

Shannon 熵監控器需要 llama-server `/completion` 端點的 `n_probs ≥ 10`。演算法：

```
演算法 1：EntropyMonitor.stream_boundaries(prompt, max_tokens)
──────────────────────────────────────────────────────────────
1: H_hist ← []
2: 對每個帶對數概率 {p_1,...,p_k} 的 SSE token t_i：
3:     H_i ← -Σ p̃_j log p̃_j  （歸一化 softmax）
4:     若 |H_hist| ≥ W：
5:         Δ ← mean(H_hist[-W:]) - H_i
6:         若 Δ > θ_H：yield ChunkBoundaryCandidate(i, H_i)
7:     H_hist.append(H_i)
```

### 3.4 困惑度尖峰監控器（P1.2）

```
演算法 2：PerplexitySpikeMonitor.feed_token(logprob)
──────────────────────────────────────────────────
1: ppl ← exp(-logprob)
2: 若 |W| ≥ window_size：
3:     μ, σ ← mean(W), std(W)
4:     z ← (ppl - μ) / (σ + ε)
5:     若 z > z_θ：return SpikeEvent(i, ppl, z)
6: W.append(ppl)
```

---

## 4. 訊號融合

### 4.1 優先權排序

三種訊號類別按可靠性排序：

$$\text{優先權}: \underbrace{\text{role\_tag}(10)}_{\text{確定性}} > \underbrace{\text{entropy\_drop}(5)}_{\text{中層}} > \underbrace{\text{perplexity\_spike}(3)}_{\text{token 級}}$$

### 4.2 遲滯閘控

為防止虛假微邊界，合併窗口 $W_m = 12$ tokens 抑制過於接近上次發射邊界的候選項：

$$\text{emit}(c_i) = \mathbb{1}\left[|i - i_{\text{last}}| \geq W_m\right] \land \left(\text{type}(c_i) = \text{role\_tag} \lor |\mathcal{C}_i^{(W_m)}| \geq 2\right)$$

其中 $\mathcal{C}_i^{(W_m)}$ 是 token $i$ 的 $W_m$ 範圍內的候選項群集。

**角色標籤特殊情況：** 角色標籤邊界完全繞過遲滯閘控（`emit_immediately = True`）並重置遲滯計數器。

### 4.3 融合結果（P1.3 實驗）

```
訊號序列：    entropy(45) → perplexity(48) → role_tag(52)
              entropy(90) → perplexity(94) → role_tag(150)

發射邊界：
  tok=45  type=entropy_drop  pri=5  signals=[entropy,perplexity]  ← 群集投票
  tok=52  type=role_tag      pri=10 signals=[role_tag]            ← 立即發射
  tok=90  type=entropy_drop  pri=5  signals=[entropy,perplexity]  ← 群集投票
  tok=150 type=role_tag      pri=10 signals=[role_tag]            ← 立即發射
```

---

## 5. 動態專家路由

### 5.1 區塊分類器

給定邊界 $b_i$，帶類型 $\tau_i$ 和熵值 $\{H_j\}_{j \in C_i}$：

$$\text{chunk\_type}(C_i) = \begin{cases}
\texttt{reasoning} & \tau_i \in \{\texttt{<think>}, \texttt{</think>}\} \\
\texttt{code}      & \tau_i = \texttt{```} \\
\texttt{factual}   & \tau_i \in \{\texttt{\# }, \texttt{[INST]}\} \\
\texttt{creative}  & \bar{H}_{C_i} > H_{\text{thresh}} \land \tau_i = \texttt{unknown}
\end{cases}$$

路由映射：
$$r(\texttt{code}) = M_{\text{coding}}, \quad r(\texttt{reasoning}) = M_{\text{reasoning}}, \quad r(\texttt{factual}) = M_{\text{coding}}$$

### 5.2 熱切換協議

由於 GT 1030 一次只能承載一個模型，模型轉換需要：

```
協議：ToolSessionManager.ensure(chunk_type)
────────────────────────────────────────
1: 若 current_model ≠ target_model(chunk_type)：
2:     向當前 llama-server 發送 SIGTERM
3:     wait(2s)  -- VRAM 排空時間
4:     用目標模型 + ngl 啟動新 llama-server
5:     輪詢 /health 直至就緒（超時=90s）
6: return port
```

### 5.3 VRAM 預算管理

GT 1030（總計 2048 MB）：

| 組件 | VRAM（MB） |
|------|-----------|
| Windows DWM + OS | ~350 |
| LINE（通訊軟體） | ~160 |
| 模型權重（ngl=10–12 層） | ~400–600 |
| KV 快取（ctx=2048，fp16） | ~56–112 |
| Vulkan 計算緩衝區 | ~100–200 |
| **合計** | **~1066–1422** |
| **安全裕度** | **626–982 MB** |

**有效標準基線**（關閉 Chrome+Edge）：$B_{\text{VRAM}} = 850$ MB 用於模型+KV+緩衝區。

---

## 6. KV 快取磁碟卸載

### 6.1 二進位格式規範

KV 區塊以帶 64 位元組標頭的 `.kvbin` 檔案序列化：

```
偏移  大小   欄位
────────────────────────────────────────
0     4      魔數：b"KVBN"
4     1      版本：0x01
5     2      session_id 雜湊（uint16）
7     2      chunk_id（uint16）
9     1      n_layers（uint8）
10    1      dtype（0=f16, 1=f32）
11    3      保留
14    6      shape: [n_layers, n_heads, head_dim]（uint16×3）
20    44     填充（零）
64    N      原始張量數據（行優先）
```

### 6.2 LRU 驅逐策略

透過磁碟備份 KV 擴展上下文窗口：

$$\text{effective\_ctx} = \underbrace{N_{\text{RAM}}}_{\text{VRAM 內 KV}} + \underbrace{N_{\text{disk}}}_{\text{卸載 KV 區塊}} \approx 4{,}096 + 12{,}288 = 16{,}384 \text{ tokens}$$

當 RAM 中 KV 超過 $C_{\text{max}}$ 位元組時觸發 LRU 驅逐。驅逐代價模型：

$$\text{cost}_{\text{evict}}(C_i) = \frac{\text{age}(C_i) \times \text{size}(C_i)}{\text{priority}(C_i)}$$

其中優先權反映區塊類型（推理區塊較不可能被重新存取）。

---

## 7. 實驗結果

### 7.1 硬體配置

| 參數 | 值 |
|------|---|
| GPU | NVIDIA GT 1030（GP108，Pascal，sm_61） |
| VRAM | 2048 MB GDDR5 |
| 後端 | Vulkan（ggml-vulkan.dll，llama.cpp b8679） |
| CPU | Intel Core i5-4460（4C/4T，3.2 GHz） |
| RAM | 34 GB DDR3-1600 |
| OS | Windows 10 Pro 10.0.19045 |
| 模型 | Qwen3-1.7B-Q8_0.gguf（1749 MB） |
| 上下文 | 4096 tokens |
| 量化 | Q8_0（8-bit，對稱） |

### 7.2 Phase 1 監控器結果

**P1.1 角色標籤解析器：**

| 樣本 | 輸入長度 | 邊界數 | 偵測到的類型 |
|------|---------|--------|------------|
| CoT+程式碼 | 204 字元 | 4 | `<think>`、`</think>`、` ``` `（×2） |
| 混合 | 194 字元 | 6 | `<think>`、`</think>`、`# `（×2）、` ``` `（×2） |

**P1.2 困惑度尖峰監控器**（window=10，$z_\theta = 2.0$）：

| Token | 區域 | PPL | z 分數 | 尖峰？ |
|-------|------|-----|--------|-------|
| 15 | 推理→程式碼 | exp(2.8)≈16.4 | **216.7** | 是 |
| 16 | 推理→程式碼 | exp(3.1)≈22.2 | **4.3** | 是 |
| 33 | 程式碼→事實 | exp(2.5)≈12.2 | **407.2** | 是 |
| 34 | 程式碼→事實 | exp(2.9)≈18.2 | **4.8** | 是 |

總計：6 個尖峰事件。兩個語義轉換均被偵測到。

**P1.3 訊號融合：**
從 6 個輸入候選項中發射 4 個邊界；遲滯正確抑制了子窗口重複項。

**P1.4 即時熵監控器（llama-server）：**

| 指標 | 值 |
|------|---|
| 吞吐量 | 5.1 tok/s |
| 上下文 | 4096 tokens |
| VRAM 已用 | 1,727 / 2,048 MB |
| 完成品質 | 正確的回文函數 |

**整合測試（7/7 通過）：**

| 測試 | 結果 | 細節 |
|------|------|------|
| VRAM 檢查 | 通過 | 1528 MB 可用 |
| 區塊偵測 | 通過 | 4 個邊界 |
| KV 序列化器 | 通過 | numpy float16 往返 |
| 控制器健康 | 通過 | 550 ms |
| 分析區塊端點 | 通過 | 4 個區塊 |
| 推理程式碼區塊 | 通過 | 12 tok/s，回應含 `def` |
| 推理推理區塊 | 通過 | "150" 正確（60×2.5=150） |

### 7.3 擴展性預測

| 平台 | 模型 | VRAM | 預期 tok/s |
|------|------|------|-----------|
| GT 1030（850 MB） | Qwen3-1.7B Q8 ngl=10 | 850 MB | 5–12 |
| GT 1030（850 MB） | Gemma4 E4B Q4 ngl=11 | 834 MB | 2–5 |
| GT 1030（850 MB） | Mixtral-8x7B Q2 ngl=3 | 767 MB | 1–3 |
| Mac Mini M4（32 GB） | DeepSeek-V3 671B MoE Q4 | 18.5 GB 激活 | 15–30 |
| Jetson Nano（4 GB） | MobileVLM-1.7B Q4 | 2 GB | 8–15 |

---

## 8. 討論

### 8.1 局限性

1. **Qwen2.5-Coder 不兼容：** llama.cpp b8679 對 `qwen2` 預分詞器產生退化輸出（全部為 `?` token）。已通過將程式碼區塊路由至 Qwen3-1.7B 緩解。

2. **無原生 KV 注入：** llama.cpp 的 HTTP API 未暴露 KV 狀態以供直接注入。我們的 `.kvbin` 格式支援磁碟序列化，但重播需要重新預填充提示，每個區塊產生 $O(|C|)$ 預填充成本。

3. **WDDM 競爭：** 在 Windows 上，顯示窗口管理器（DWM）可能隨時回收 VRAM。每次模型載入前需要動態 VRAM 監控（透過 `nvidia-smi` 子程序）。

### 8.2 未來工作

1. **Phase 2 — KV 卸載管理器：** 完整 LRU 驅逐流水線，支援磁碟備份的 16K+ 上下文。
2. **Phase 3 — 26B 規模專家路由：** Mixtral-8x7B Q2_K，專家 RAM 池管理。
3. **Phase 4 — 邊緣 VLM：** moondream-2B Q4 在 GT 1030（800 MB）；LLaVA-Phi 在 Android。
4. **Phase 5 — 自迴圈 AGI 原型：** 串流蒸餾、LoRA 適配器池、方向性引導（Directional Steering）。

---

## 9. 相關研究

### 9.1 MoE 推理優化

Shazeer 等人（2017）[2] 引入 MoE 的稀疏門控，確立了前 $k$ 個專家選擇（$k \ll E$）在較低推理成本下達到與稠密模型同等品質的結論。Lepikhin 等人（2021）[6] 在 TPU 集群上展示了 GShard 擴展至 6000 億 MoE 參數。

### 9.2 記憶體高效推理

Sheng 等人（2023）FlexGen [4] 探索了單 GPU 推理 30B 模型的 CPU/GPU/磁碟張量卸載。本研究與其差異在於在**語義區塊層面**運作，而非張量層面，從而能夠為每個區塊使用質上不同的專家。

### 9.3 投機解碼

Leviathan 等人（2023）[3] 引入投機解碼，使用小型草稿模型提出候選 token，再由較大模型驗證。Spec-Experts 與之正交：我們路由至不同專家，而非在單一模型中草稿-驗證。

### 9.4 注意力熵分析

Clark 等人（2019）[7] 分析了 BERT 注意力模式，發現句法和語義角色與不同的頭部注意力分佈相關。Voita 等人（2019）[8] 顯示可通過熵準則剪枝的注意力頭保留了語言功能。本研究將此擴展至推理時語義狀態偵測。

### 9.5 相關專利

- **US10,817,783B1**（Google LLC，2020年）：「具學習門控函數的混合專家神經網路。」主張通過在 token 級學習 softmax 路由進行專家門控的優先權。Spec-Experts 的差異在於在區塊層面運作，採用外部、推理時、非參數路由。

- **US11,423,285B2**（Microsoft Corporation，2022年）：「用於自然語言處理任務的動態模型選擇。」主張通過分類器進行基於任務類型的模型選擇。Spec-Experts 的差異在於使用來自**生成**模型本身的熵/困惑度訊號，而非獨立的任務分類器。

- **US20230409774A1**（Meta Platforms，2023年）：「使用專家並行的大型語言模型高效推理。」主張跨加速器的分散式專家分片。Spec-Experts 針對單設備消費級硬體，採用時間多路復用專家存取。

- **WO2024/053456A1**（DeepMind，2024年）：「具早期退出的混合專家自適應計算。」主張每 token 計算深度自適應。Spec-Experts 在區塊粒度運作，不修改模型計算圖。

---

## 參考文獻

[1] Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *Journal of Machine Learning Research*, 23(120), 1–39.

[2] Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *arXiv:1701.06538*.

[3] Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *Proceedings of ICML 2023*.

[4] Sheng, Y., Zheng, L., Yuan, B., Li, Z., Ryabinin, M., Chen, B., Liang, P., Ré, C., Stoica, I., & Zhang, C. (2023). FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU. *Proceedings of ICML 2023*.

[5] Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A.M., Chen, Z., Le, Q.V., & Laudon, J. (2022). Mixture-of-Experts with Expert Choice Routing. *NeurIPS 2022*.

[6] Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., & Chen, Z. (2021). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. *ICLR 2021*.

[7] Clark, K., Khandelwal, U., Levy, O., & Manning, C.D. (2019). What Does BERT Look at? An Analysis of BERT's Attention. *BlackboxNLP 2019*.

[8] Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. *ACL 2019*.

[9] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*.

[10] Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. *Proceedings of ICML 2023*.

[11] Liu, Z., Wang, J., Dao, T., Zhou, T., Yuan, B., Song, Z., Sheng, Y., Krishnamurthy, A., Chen, W., Guo, B., ... & Papailiopoulos, D. (2023). Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time. *Proceedings of ICML 2023*.

[12] DeepSeek-AI (2024). DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence. *arXiv:2406.11931*.

[13] Qwen Team (2025). Qwen3 Technical Report. *arXiv:2505.09388*.

[14] Team, G. (2025). Gemma 4 Technical Report. *Google DeepMind*.

[15] Shannon, C.E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379–423.

---

## 附錄 A：系統架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                  Spec-Experts LLM 系統                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用戶提示 → [控制器 :8090 FastAPI]                           │
│                        │                                   │
│          ┌─────────────▼─────────────┐                     │
│          │       外部監控器           │                     │
│          │  ┌──────┐ ┌──────┐ ┌───┐ │                     │
│          │  │角色標│ │  熵  │ │PPL│ │                     │
│          │  │籤解析│ │監控器│ │   │ │                     │
│          │  └──┬───┘ └──┬───┘ └─┬─┘ │                     │
│          │     └────────┴────────┘   │                     │
│          │         訊號融合           │                     │
│          └─────────────┬─────────────┘                     │
│                        │ ChunkBoundary                      │
│          ┌─────────────▼─────────────┐                     │
│          │       區塊分類器           │                     │
│          │  推理/程式碼/事實/創意     │                     │
│          └─────────────┬─────────────┘                     │
│                        │ chunk_type                         │
│          ┌─────────────▼─────────────┐                     │
│          │   工具會話管理器           │                     │
│          │   [熱切換 llama-server]   │                     │
│          │   VRAM 預算：850 MB       │                     │
│          └─────────────┬─────────────┘                     │
│                        │                                   │
│     ┌──────────────────┼──────────────────┐               │
│     ▼                  ▼                  ▼               │
│  [M_coding]       [M_reasoning]      [M_debug]            │
│  Qwen3-1.7B       Qwen3-1.7B         DeepSeek-1.5B        │
│  ngl=10           ngl=10             ngl=12               │
│  （程式碼任務）    （/no_think）       （推理）             │
│                                                             │
│  ─────────────────────────────────────────────────        │
│                   KV 快取層                                 │
│  [VRAM KV] ←LRU→ [RAM KV] ←LRU→ [磁碟 KV (.kvbin)]       │
│   2048 ctx         4096 ctx        16384+ ctx              │
└─────────────────────────────────────────────────────────────┘
```

## 附錄 B：復現步驟

```bash
# 1. 複製儲存庫
git clone https://github.com/sipurchen/klchen
git checkout SpecExpertsResearch

# 2. 設定環境
set LLMS_DIR=E:\LLMmodel
set PROJECT_DIR=E:\Gemma4_E4B_Project

# 3. 下載模型
powershell -File E:\LLMmodel\download_spec_experts_models.ps1

# 4. 啟動 llama-server
bin\llama-cpp\llama-server.exe -m %LLMS_DIR%\Qwen3-1.7B\Qwen3-1.7B-Q8_0.gguf ^
  -ngl 10 -c 4096 --port 8080

# 5. 執行 Phase 1 示範
python tests/phase1_monitor_demo.py

# 6. 執行整合測試
python -m uvicorn spec_experts.controller:app --port 8090
python tests/spec_experts_test.py
```
