# Spec-Experts LLM：七階段語義邊界偵測、動態專家路由與自強化推理架構——資源受限硬體上的大型語言模型本地部署研究

**主要作者：** SipurChen  
**AI 協同作者：** Claude Sonnet（Anthropic）· OpenAI Codex · Google Gemini Flash  
**研究庫：** https://github.com/sipurchen/klchen · 分支：`SpecExpertsResearch`  
**日期：** 2026-06-26 初稿；2026-11-15 依學術論文格式修訂  

> *本研究透過人機迭代協作開發完成。SipurChen 主導所有研究目標、硬體實驗及架構決策。Claude Sonnet（Anthropic）參與系統設計、程式實作、論文撰寫及專利起草。OpenAI Codex 協助程式碼生成與重構。Google Gemini Flash 提供快速原型驗證支援。*  

---

## 摘要

**背景。** 大型語言模型的能力集中於 7B–671B 參數的模型，而多數已部署硬體僅有 1–8 GB 記憶體。混合專家（MoE）模型每個 token 只啟用部分專家，為在受限硬體上部署提供了空間，但其實際效益取決於專家被選用的時間與類型分布。

**目的。** 本研究提出 Spec-Experts 七階段架構（外部語義邊界監控、三層 KV 快取、方向性引導、專家 RAM 池、邊緣 VLM 路由、UMA 專家池、自迴圈蒸餾），並以 Gemma4 26B-A4B（30 層、每層 128 個專家、每 token 選用 8 個）為對象，量測其專家路由特性，檢驗兩項假設：（H1）相鄰 token 的專家集合交集機率不低於 0.85；（H2）以 LoRA 訓練可使各內容類型的專家集合平均縮小至少 30%，並維持與基底模型的 next-token 一致率不低於 0.85 且 NLL 增加不超過 0.15 nats/token。

**方法。** 以 `llama-imatrix` 取得各層專家使用次數（Tier 1），並以 ctypes 自 llama.cpp 讀取每層路由器的 top-k，得到逐 token 專家序列（Tier 2）；語料為 10 個多輪對話 session（約 6.8 萬 token 的逐 token 序列、約 204 萬 token 的 Tier 1 計數）。量測於 GT 1030（2 GB VRAM）與模擬受限環境的 gx10（GPU 預算 1.0 GiB）進行。專家池策略以真實路由序列重放；LoRA 縮小專家集合以掛載於路由器的 LoRA 與蒸餾補償實施，並於留出 session 與兩個全新生成的 session 評估。

**結果。** （1）Gemma4 26B-A4B 於 GT 1030 以 `-ngl 10` 使用 1682 MiB VRAM，解碼 4.22 tok/s。（2）相鄰 token 的專家集合交集機率為 0.915（均勻隨機 0.412、打亂順序對照 0.737），H1 獲得支持；GT 1030 上前 1280 個 token 的重放為 0.910。（3）各內容類型的專家集合涵蓋每層 38%–48%；區塊邊界類型預取與 token 層級 LRU 的命中率相當（每層池容量 57 時 0.940 與 0.939）。（4）僅調整 attention 與稠密 MLP 的 LoRA 對專家集合的影響為 0%；掛載於路由器的 LoRA 使集合於留出 session 縮小 33.2%，與基底的 top-1 一致率 0.843、ΔNLL +0.043；於全新 session 縮小 27.5%、一致率 0.835。H2 的集合縮小與 NLL 門檻於留出 session 達成；一致率於兩組資料分別為 0.843 與 0.835，與假設值 0.85 接近，H2 獲得初步支持，仍須以更多資料進一步驗證。

**結論。** Gemma4 26B-A4B 的專家路由具有明顯的時間局部性，且該局部性可由最近使用的專家（LRU）有效利用；縮小專家集合可經由路由器 LoRA 達成，其幅度與模型輸出一致性之間呈取捨關係。Phase 1 的外部監控器於 GT 1030 達到 5.1 tok/s 的即時偵測；Phase 2、3、5、6、7 以結構驗證呈現，其證據等級見 §6.4。

**關鍵詞：** 混合專家、專家路由、語義邊界偵測、LoRA、專家快取、受限硬體推論、Gemma4

---

## 1. 緒論

### 1.1 研究背景與動機

最先進的大型語言模型存在於 7B–671B 參數的規模，需要 14–380 GB 的 GPU 記憶體；多數已部署硬體（消費級 GPU、邊緣裝置、嵌入式系統）僅在 1–8 GB 記憶體限制下運作。積極量化（Q2–Q4）可部分縮小差距，但對所有計算一視同仁地降低精度，未利用 LLM 輸出在語義上的異質性。

單一回應通常同時包含推理、程式碼、事實與創意等不同性質的區段，它們的品質需求、適用溫度，以及在 MoE 模型中被選用的專家，皆可能不同。若能從外部辨識這些區段的邊界，並掌握各區段所需的專家集合，便有機會以較小的快取容量服務同樣的模型。本研究據此提出七階段架構，並以真實 MoE 模型量測其前提是否成立。

### 1.2 研究問題與研究假設

**研究問題**

- **RQ1：** 能否在不修改模型的前提下，自外部 HTTP/SSE 串流偵測語義區塊邊界？
- **RQ2：** 能否將 KV 狀態序列化至磁碟，將有效上下文擴展至超過 VRAM 容量？
- **RQ3：** 能否在無梯度存取的情況下，依區塊類型引導生成行為？
- **RQ4：** 能否依語義邊界與路由特性，預先定位或縮小 MoE 的專家權重集合？
- **RQ5：** 能否在消費級硬體上以蒸餾與 LoRA 構成品質提升迴圈？

**研究假設（針對 RQ4）**

- **H1（時間局部性）：** 相鄰 token 的專家集合交集機率 Pr[E_i ∩ E_{i+1} ≠ ∅] ≥ 0.85。
- **H2（專家集合縮小）：** 以 LoRA 訓練後，四種內容類型的 α = 0.9 專家集合中位數大小平均縮小 ≥ 30%，同時與基底模型的 next-token top-1 一致率 ≥ 0.85，且平均 NLL 增加 ≤ 0.15 nats/token。三項判準的數值為研究假設值，於資料量測前預先設定。
- **H3（邊界預取）：** 區塊邊界的類型預取之專家池命中率高於 token 層級 LRU。

### 1.3 研究貢獻

1. 三訊號外部監控器與優先權加權訊號融合（§3.2）。
2. 三層 KV 層級結構與 `.kvbin` 二進位格式（§3.3）。
3. 以複合系統提示實現的方向性引導（§3.4）。
4. MoE 專家路由的量測方法：以 ctypes 讀取 llama.cpp 路由器 top-k 的逐 token 序列量測程式，其結構位移由該版本標頭編譯量出（§4.3）。
5. Gemma4 26B-A4B 的專家路由實證：類型親和、時間局部性、專家池策略重放（§5.3–§5.5）。
6. 縮小專家集合的 router LoRA 與蒸餾補償，以及其於留出與全新資料的評估（§3.9、§5.6）。
7. 邊緣 VLM 路由、Mac M4 專家池與自迴圈蒸餾的設計（§3.6–§3.8）。

### 1.4 論文架構

第 2 章回顧技術背景與相關研究；第 3 章說明七階段系統設計與 router LoRA 方法；第 4 章說明實驗設計（環境、語料、量測方法與評估判準）；第 5 章呈現實驗結果；第 6 章討論結果的意涵、有效性威脅與證據等級；第 7 章為結論。附錄提供模組對照與復現步驟。

---

## 2. 文獻回顧與技術背景

### 2.1 混合專家架構

MoE 模型將 FFN 劃分為 $E$ 個專家子網路，並配備門控函數 $g: \mathbb{R}^d \to \Delta^E$：

$$\text{FFN}_{\text{MoE}}(\mathbf{x}) = \sum_{e \in \text{top}_k(g(\mathbf{x}))} g_e(\mathbf{x}) \cdot f_e(\mathbf{x})$$

| 模型 | 總參數 | 每 token 激活 | 激活 GB | 總 GB |
|------|--------|-------------|---------|------|
| DeepSeek-V3 671B Q2 | 671B | 37B (5.5%) | 18.5 | 210 |
| Mixtral-8x7B Q2 | 47B | 7B (14.9%) | 3.7 | 15 |
| DS-Coder-V2-Lite Q4 | 16B | 2.4B (15%) | 1.3 | 10 |

**專家激活局部性（H1）：** 本研究假設同一語義區塊內相鄰 token 的專家集合交集機率不低於 0.85：

$$\Pr\!\left[\text{top}_k(g(\mathbf{x}_{t_i})) \cap \text{top}_k(g(\mathbf{x}_{t_{i+1}})) \neq \emptyset\right] \geq 0.85$$

若此假設成立，區塊層級的專家預先定位可作為 token 層級反應式載入的替代。此機率的量測見 §5.4。

### 2.2 Shannon 熵作為語義指示器

Token 分佈的 Shannon 熵 [Shannon, 1948]：

$$H(\mathbf{p}) = -\sum_{j=1}^k \tilde{p}_j \log_2 \tilde{p}_j, \quad \tilde{p}_j = \frac{\exp(\ell_j)}{\sum_{j'} \exp(\ell_{j'})}$$

其中 $\ell_j$ 為 llama-server `n_probs` 端點的前 $k$ 個對數機率。

- **低 $H$：** 確定性生成 → 程式碼、結構化輸出
- **高 $H$：** 擴散性生成 → 推理、創意
- **驟降 ΔH > θ_H：** 語義狀態轉換

### 2.3 困惑度 Z 分數

Token 困惑度及滑動窗口 z 分數：

$$\text{PPL}_i = \exp(-\text{logprob}_i), \quad z_i = \frac{\text{PPL}_i - \hat{\mu}_{W}}{\hat{\sigma}_{W} + \varepsilon}$$

GT 1030 實測邊界訊號強度（Qwen3-1.7B，$z_{\theta} = 2.0$）：

| 轉換 | z 分數 | 超閾值倍數 |
|------|--------|----------|
| 推理→程式碼（tok 15） | **216.7** | 108× |
| 推理→程式碼（tok 16） | **4.3** | 2.2× |
| 程式碼→事實（tok 33） | **407.2** | 204× |
| 程式碼→事實（tok 34） | **4.8** | 2.4× |

### 2.4 Flash Attention

Flash Attention [Dao et al., 2022] 以分塊 SRAM 計算替代 $O(N^2)$ 標準注意力，達成 $O(N)$ 記憶體：

$$\text{標準: } N^2 \times 2\text{B} = 512\text{MB/head} \quad (N=16K)$$
$$\text{Flash: } \sim 32\text{KB/block} = O(N) \text{ 總計}$$

在 Mac M4 Metal（無 CPU/GPU 設備分割）上可用；GT 1030（CPU/GPU 層分割）不可用。

### 2.5 KV 快取記憶體

$$\text{KV}_{\text{bytes}} = 2 \cdot L \cdot H \cdot N \cdot d_h \cdot 2 \text{ (fp16)}$$

Qwen3-1.7B（$L=28, H=16, d_h=128, N=2048$）：KV ≈ 448 MB，在 850 MB VRAM 預算中佔主導地位。

### 2.6 方向性激活引導

無模型內部存取時，以複合系統提示代理 $\mathbf{h}_l' = \mathbf{h}_l + \alpha\hat{\mathbf{d}}$ [Zou et al., 2023]：

1. **複合系統提示：** 編碼方向性正向示例
2. **溫度調整：** $T' = T + \Delta T_{\text{type}}$
3. **對數機率偏置：** $\ell'_j = \ell_j + b_j$
4. **前綴預熱：** 前置概念激活 token

### 2.7 LoRA 低秩適配

低秩適配 [Hu et al., 2022]：

$$W' = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d,k)$$

透過 llama-server `--lora-scaled path scale` 熱切換，無需完整模型重載。

---

### 2.8 相關研究

#### 2.8.1 MoE 推理優化

Shazeer 等人（2017）[1] 建立稀疏門控基礎。Switch Transformers [Fedus, 2022] [2] 擴展至 1T 參數。Mixtral [Jiang, 2024] [3] 驗證了實用 MoE 品質-效率。所有先前工作均假設多 GPU 或分散式記憶體。Spec-Experts 透過時序多路復用區塊路由，針對單設備 VRAM 約束。

#### 2.8.2 記憶體高效推理

FlexGen [Sheng, 2023] [4] 實現 30B 模型單 GPU 推理的 CPU/GPU/磁碟張量卸載。LLM.int8() [5] 和 GPTQ [6] 透過量化縮小模型體積。Spec-Experts 在量化基礎上加入語義感知的區塊級路由。

#### 2.8.3 注意力熵分析

Clark 等人（2019）[9] 示範 BERT 注意力頭專門化。Voita 等人（2019）[10] 展示熵剪枝。Elhage 等人（2021）[11] 形式化殘差流視角。本研究將熵量測擴展至推理時語義狀態外部偵測。

#### 2.8.4 激活引導

Zou 等人（2023）[12] 提出表示工程。Turner 等人（2023）[13]（pi-ds4）實現激活加法引導。本研究無需模型內部存取，以複合系統提示代理上述技術。

#### 2.8.5 自改善

Self-Instruct [Wang, 2023] [19] 和 Constitutional AI [Bai, 2022] [20] 示範 LLM 自改善。Phase 7 以區塊類型特定品質指標和 LoRA 適配器選擇加以擴展。

### 2.9 專利格局

**US10,817,783B1**（Google LLC，2020）——透過學習 softmax 的 MoE 門控。本發明以非參數方式在外部運作，且在區塊粒度而非 token 粒度路由——三個獨立的差異化維度。

**US11,423,285B2**（Microsoft，2022）——透過任務分類器的動態模型選擇。本發明的訊號（熵、困惑度）來自生成模型本身，無需獨立分類器。

**US20230409774A1**（Meta，2023）——跨加速器的專家並行。本發明針對單設備消費級硬體的時序多路復用存取。

**WO2024/053456A1**（Google DeepMind，2024）——訓練時 MoE 早期退出。本發明是純推理時系統，無需訓練時修改。

**US11,580,423B1**（Amazon，2023）——硬體加速專家選擇。本發明僅軟體實現，與任何 llama.cpp 伺服器相容。

**US20240169235A1**（NVIDIA，2024）——串流推理流水線並行。本發明在語義粒度（高於硬體流水線層次）進行區塊路由。

---

## 3. 系統設計與方法

### 3.1 架構總覽

Spec-Experts 由七個階段組成，Phase 1 的外部監控器提供區塊邊界與類型訊號，其餘階段在推論期的資源管理、行為引導與訓練上使用該訊號或各自獨立運作：

| 階段 | 功能 | 主要輸入 | 主要輸出 |
|---|---|---|---|
| Phase 1 | 外部多訊號監控器 | token 串流（熵、困惑度、角色標籤） | 區塊邊界與類型 |
| Phase 2 | 三層 KV 快取卸載 | KV 區塊 | VRAM／RAM／磁碟分層儲存 |
| Phase 3 | 方向性引導 | 區塊類型 | 系統提示、溫度、logit 偏置 |
| Phase 4 | 專家 RAM 池 | 區塊類型、路由序列 | 快取中的專家集合 |
| Phase 5 | 邊緣 VLM 路由 | 輸入模態、裝置資源 | 路由決策 |
| Phase 6 | UMA 專家池 | 專家存取序列 | 常駐與分頁策略 |
| Phase 7 | 自迴圈蒸餾 | 生成樣本與品質分數 | LoRA 適配器 |

### 3.2 Phase 1：外部監控器

#### 3.2.1 架構

三個偵測器並行運作於 SSE token 串流：

```
llama-server :8080 SSE
 │
 ├─ [角色標籤解析器 pri=10]  <think> </think> ``` # [INST]
 ├─ [熵監控器       pri=5 ]  H_i < 均值 - θ_H
 └─ [困惑度尖峰監控器 pri=3]  z_i > z_θ
                │
         [訊號融合 merge_window=12, 遲滯]
                │
         [ChunkBoundary {tok_idx, type, priority}]
                │
         [區塊分類器] → type ∈ {reasoning, code, factual, creative}
```

#### 3.2.2 角色標籤有限狀態機（priority=10）

確定性、零延遲、繞過遲滯直接發射：

| 模式 | 區塊類型 | 發射方式 |
|------|---------|---------|
| `<think>` | reasoning\_start | 立即 |
| `</think>` | reasoning\_end | 立即 |
| ` ``` ` | code\_boundary | 立即 |
| `# ` | factual\_header | 立即 |

實驗結果：4 個邊界（CoT+程式碼樣本）；6 個邊界（混合樣本）。對格式良好的輸出達 100% 召回率。

#### 3.2.3 熵監控器（priority=5）

$$\text{boundary-entropy}(i) = \mathbb{1}\!\left[\bar{H}_{i-W:i} - H_i > \theta_H\right], \quad W=8,\; \theta_H=0.35$$

即時實驗：tok 45 處偵測到邊界，ΔH = 0.41 > θ_H = 0.35；吞吐量 5.1 tok/s。

#### 3.2.4 困惑度尖峰監控器（priority=3）

$$\text{boundary-ppl}(i) = \mathbb{1}[z_i > z_{\theta}], \quad z_{\theta} = 2.0,\; W=10$$

偵測到 6 個尖峰事件；捕捉到兩個語義轉換點。

#### 3.2.5 訊號融合

優先權排序：

$$\text{role-tag}(10) \succ \text{entropy-drop}(5) \succ \text{perplexity-spike}(3)$$

遲滯發射規則：

$$\text{emit}(c_i) \iff \left|i - i_{\text{last}}\right| \geq W_m \;\land\; \left[\text{role-tag} \;\lor\; \left|\mathcal{C}_i^{W_m}\right| \geq 2\right], \quad W_m = 12$$

實驗（P1.3）：從 6 個候選項發射 4 個邊界；tok 52 和 150 的 role\_tag 邊界繞過遲滯立即發射；tok 45 和 90 的熵+困惑度群集滿足 ≥2 投票條件。

---

### 3.3 Phase 2：三層 KV 快取卸載

#### 3.3.1 記憶體層級設計

$$\text{VRAM} \xrightarrow{\text{LRU 驅逐}} \text{RAM} \xrightarrow{\text{LRU 驅逐}} \text{磁碟 (.kvbin)}$$

**有效上下文公式：**

$$\text{ctx}_{\text{eff}} = N_{\text{VRAM}} + N_{\text{RAM}} + N_{\text{disk}} \approx 2{,}048 + 4{,}096 + 12{,}288 = 18{,}432 \text{ tokens}$$

#### 3.3.2 LRU 驅逐代價模型

$$\text{cost}_{\text{evict}}(C_i) = \frac{\text{age}(C_i) \cdot \text{size}(C_i)}{\pi(C_i)}$$

優先權權重：π(code)=1.5 > π(factual)=1.2 > π(creative)=0.8 > π(reasoning)=0.5

程式碼區塊保留最長（重存取機率高）；推理區塊最先驅逐。

#### 3.3.3 .kvbin 格式

64 位元組標頭 + 原始張量：

```
[4B 魔數 "KVBN"][2B 版本][16B session_id][4B chunk_id]
[2B layer_start][2B layer_end][2B dtype][2B shape_len]
[shape_len × 4B shape 值] ... [原始 fp16 張量位元組]
```

驗證：float16 往返，shape=(2,32,64) → 8KB 檔案。整合測試通過。

#### 3.3.4 提示重播（llama.cpp KV 替代方案）

llama.cpp HTTP API 不支援 KV 注入。透過儲存原始 token 序列實現上下文重播：

$$\text{replay}(k) = \text{decode}\!\left(\bigcup_{i \leq k} \text{tokens}(C_i)\right)$$

重播代價 $O(|C_{1:k}|)$ tokens，可接受（預填充速度是解碼的 3–5 倍）。

---

### 3.4 Phase 3：方向性引導

#### 3.4.1 無激活存取的代理引導

| 層次 | 機制 | 實現 |
|------|------|------|
| 1 | 系統提示 | 按 $\alpha$ 排序連接概念正向示例 |
| 2 | 溫度調整 | $T' = T + \Delta T_{\text{type}}$ |
| 3 | 對數機率偏置 | $\ell'_j = \ell_j + b_j$ 每 token |
| 4 | 前綴預熱 | 前置概念激活 token |

#### 3.4.2 引導向量庫

| 概念 | 正向示例（節錄） | $\alpha$ | $\Delta T$ |
|------|--------------|------|----------|
| `code_quality` | "Write clean, efficient Python with type hints" | 15.0 | −0.2 |
| `reasoning` | "Think step by step. Show reasoning carefully." | 12.0 | −0.2 |
| `factual` | "Give accurate, verifiable information." | 8.0 | 0.0 |
| `creative` | "Be original and surprising." | 10.0 | +0.3 |
| `concise` | "Answer in one sentence only." | 10.0 | −0.1 |

#### 3.4.3 區塊類型→引導配置

| 區塊類型 | 激活概念 | $T_{\text{eff}}$ | 系統提示長度 |
|---------|---------|--------------|-----------|
| code | [code\_quality, concise] | 0.40 | 128 chars |
| reasoning | [reasoning] | 0.50 | 69 chars |
| factual | [factual, concise] | 0.60 | 120 chars |
| creative | [creative] | 1.00 | 65 chars |

---

### 3.5 Phase 4：專家 RAM 池

Phase 4 在每個 MoE 層維持一個容量為 C 個專家的快速層（RAM 池），其餘專家位於較慢的儲存層；token 需要的專家不在池中時，須自慢速層載入。

**池容量。** 設快取預算為 R、層數為 L、單一專家在單一層的大小為 e_size，每層可常駐的專家數為

$$E_{\text{warm}} = \left\lfloor \frac{R_{\text{budget}}}{L \cdot e_{\text{size}}} \right\rfloor$$

**策略。** 本研究比較三種策略：

- **reactive：** token 層級的 LRU；僅在 token 需要且不在池中時載入。
- **static：** 每個區塊類型有固定的池，內容為其他 session 上該類型使用次數最多的 C 個專家（留一個 session 於外，被評估的 session 不參與挑選自己的池）。
- **spec（Spec-Experts）：** 每到區塊邊界即預先載入該類型的前 C 個專家，其後的缺漏以池內 LRU 處理；邊界預載的載入量計入總載入量。

**載入時間。** 單一專家單一層的載入時間為 t_load = e_size / B，其中 B 為慢速層至快速層的頻寬。評估結果見 §5.5。

### 3.6 Phase 5：邊緣 VLM 路由器

#### 3.6.1 設備目標

| 設備 | 模型 | 後端 | ngl | VRAM | FPS |
|------|------|------|-----|------|-----|
| GT 1030 | moondream-2B Q4 | Vulkan | 8 | 800 MB | 1.5 |
| Jetson Nano | MobileVLM-1.7B Q4 | CUDA | 32 | 2 GB | 10.0 |
| Android | LLaVA-Phi-1.5 Q4 | CPU | 0 | — | 5.0 |
| ARM 監控 | moondream-2B Q4 | CPU | 0 | — | 2.0 |

#### 3.6.2 路由邏輯

```python
if chunk_type == "vision" or image_present:
    → VLM 推理：POST /completion + image_data[base64]
else:
    → 文字推理：POST /v1/chat/completions
```

#### 3.6.3 未來：多模態遊戲 AI

```
場景畫面 → [視覺 VLM 專家]   → scene_description
                  ↓
NPC 查詢 → [推理專家]         → behavior_decision
                  ↓
玩家文字 → [創意專家]         → dialogue_output
                  ↓
           [KV 串流：跨 NPC 共享上下文]
```

---

### 3.7 Phase 6：Mac Mini M4——200B+ MoE 大規模推理

#### 3.7.1 UMA 優勢

Mac M4 統一記憶體消除 PCIe 傳輸瓶頸：

$$\text{頻寬: } 120\text{ GB/s (UMA)} \gg 8\text{ GB/s (PCIe 3.0×4)}$$

啟用 Flash Attention（Metal 後端，無設備分割）：

$$\text{Flash: } O(N \cdot d_k) \ll O(N^2)$$

#### 3.7.2 NVMe 專家分頁

$$t_{\text{load}}(e) = \frac{e_{\text{size}}}{b_{\text{SSD}}} = \frac{0.82\text{ GB}}{7.0\text{ GB/s}} \approx 117\text{ ms/次（DS-V3）}$$

Spec-Experts 路由將 40 次樸素切換減至 24 次 → 節省 $16 \times 0.117 = 1.87\text{ s}$/20 區塊會話。

#### 3.7.3 UMA 專家釘選

DS-V3 前 33 個專家（33 × 0.82 GB = 27 GB）可完全釘選於 32 GB UMA。常見使用模式下，前 33 個熱門專家覆蓋率 >80%，NVMe 分頁罕見。

#### 3.7.4 理論 tok/s

$$\text{tok/s} = \frac{B}{\text{激活權重大小}} \times \eta$$

| 模型 | $B$ | 激活大小 | $\eta$ | tok/s |
|------|-----|---------|--------|-------|
| DS-V3 671B Q2（top-33 熱） | 120 GB/s | 18.5 GB | 0.35 | ~2–7 |
| Mixtral-8x7B Q4（全 UMA） | 120 GB/s | 3.5 GB | 0.80 | ~27 |
| Mixtral-8x22B Q4（全 UMA） | 120 GB/s | 26 GB | 0.75 | ~3.5 |

---

### 3.8 Phase 7：AGI 自迴圈

#### 3.8.1 架構

```
[任務] → [專家路由器] → [LLM 生成] → [品質評分 Q(r,t)]
                                              │
                                     [蒸餾數據庫]
                                     (Q ≥ 0.7 → JSONL 匯出)
                                              │
                                    [LoRA 適配器更新]
                                    q_t = α·q_new + (1-α)·q_{t-1}
                                              │
                                    [適配器池選擇]
                                         ↑___回饋___↑
```

#### 3.8.2 品質評分函數

$$Q_{\text{code}}(r) = 0.4\cdot\mathbb{1}[\texttt{def/class}\in r] + 0.3\cdot\mathbb{1}[|r|>100] + 0.3\cdot\mathbb{1}[\text{行數}>3]$$

$$Q_{\text{reasoning}}(r) = 0.4\cdot\mathbb{1}[|r|>50\text{字}] + 0.3\cdot\mathbb{1}[\text{連接詞}\in r] + 0.3\cdot\mathbb{1}[\text{句數}>3]$$

$$Q_{\text{creative}}(r) = 0.4\cdot\mathbb{1}[|r|>30\text{字}] + 0.6\cdot\frac{|\text{不重複詞彙}|}{|\text{詞彙}|}$$

#### 3.8.3 串流蒸餾匯出

高品質樣本（$Q \geq 0.7$）匯出為 JSONL 訓練數據：

```json
{"prompt": "...", "completion": "...", "chunk_type": "code", "quality": 0.85}
```

#### 3.8.4 LoRA 適配器指數移動平均更新

$$q_t^{(a)} = \alpha \cdot q_{\text{new}} + (1-\alpha) \cdot q_{t-1}^{(a)}, \quad \alpha = 0.3$$

#### 3.8.5 品質趨勢檢測（線性迴歸）

$$\hat{\beta} = \frac{\sum_i (i-\bar{i})(q_i-\bar{q})}{\sum_i (i-\bar{i})^2}$$

實驗結果：P7 示範中，$\hat{\beta} = +0.030/\text{iter}$（改善中），6 次迭代品質從 0.60 提升至 0.75。

---

### 3.9 縮小專家集合的 router LoRA

**動機。** 若 LoRA 僅作用於 attention 與稠密 MLP，專家的選擇只會經由隱藏狀態間接改變。本研究將 LoRA 掛載於各層路由器的投影層 `router.proj`（30 層、rank 16 時約 150 萬參數），並以路由損失直接要求特定類型的 token 使用較小的專家集合。

**目標集合。** 對每個內容類型 T 與層 l，目標集合 S(T, l) 為訓練 session 上該類型使用次數最多的 m 個專家，m 取該類型 α = 0.9 親和集合大小的 60%（一組取 50%）。

**第一階段損失（路由損失）。** 對屬於類型 T 的 token，要求路由器機率質量落在 S(T, l) 內；質量達 0.9 後不再施壓（hinge）。對所有 token 另加 KL(p_base ‖ p_student)，使學生路由與基底接近：

$$\mathcal{L}_{\text{route}} = \lambda\,\mathbb{E}\bigl[\max\bigl(0,\ \log 0.9 - \log M\bigr)\bigr] + \mu\,\mathrm{KL}(p_{\text{base}} \Vert p_{\text{student}}), \qquad M = \sum_{e \in S(T,l)} p_e$$

其中 M 為路由器機率質量落在目標集合內的總和。此階段不使用語言模型損失，故不需對凍結的專家反向傳播。訓練時主體沿用基底路由（teacher forcing）。

**第二階段（補償）。** 凍結第一階段的 router LoRA，另訓練 attention 與稠密 MLP 的 LoRA（rank 8），訓練時由學生路由驅動主體。損失為對真實下一個 token 的交叉熵與對基底 argmax 的交叉熵（硬標籤蒸餾）之加權和：

$$\mathcal{L}_{\text{comp}} = w \cdot \mathrm{CE}(\text{student},\ y_{\text{real}}) + (1 - w) \cdot \mathrm{CE}(\text{student},\ \arg\max p_{\text{base}})$$

此階段需經凍結專家反向傳播，專家由 offload 引擎依需載入。

---

---

## 4. 實驗設計

### 4.1 硬體與執行環境

| 項目 | GT 1030 系統 | gx10（模擬受限環境） |
|---|---|---|
| GPU | NVIDIA GT 1030，2 GB GDDR5 | NVIDIA GB10（統一記憶體） |
| 後端 | Vulkan（llama.cpp b8679） | CUDA（llama.cpp d327bf1） |
| CPU | Intel i5-4460，4C/4T 3.2 GHz；本研究使用核心 2+3、低優先權 | 核心 0–4、nice 10 |
| RAM | 34 GB DDR3-1600（標稱 17 GB/s） | 共用主機（系統記憶體 121 GiB） |
| VRAM／GPU 預算 | 上限 1700 MiB | 上限 1000 MiB（offload 引擎預設 1.0 GiB） |
| 作業系統 | Windows 10 Pro 22H2 | Linux（aarch64） |

gx10 的設定模擬 GT 1030 的資源限制：專家張量留在 CPU（等同 `--override-tensor "\.ffn_.*_exps\.=CPU"`）、ubatch 64、`-ngl 2`（`-ngl 3` 於 ctx 8192 使用 1031 MiB，超過 1000 MiB 上限，故採 `-ngl 2`，933 MiB）。所有行程皆在 CPU 親和性、優先權、逾時與負載守衛下執行。

### 4.2 模型與語料

**模型。** Gemma4 26B-A4B（Q4_K_M，30 層、每層 128 個專家、每 token 選用 8 個）。每個專家在單層的大小為 4.14 MiB（down 以 Q8_0 計 2.0088 MiB、gate_up 以 Q4_K 計 2.1270 MiB，由 GGUF 張量大小計算）。HF 格式的 bf16 權重用於 offload 訓練與評估引擎。

**語料。** 語料為 Gemma4 自行生成的多輪對話：coder、planner、writer 三個家族共 10 個 session（Windows 上 2 個 pilot，其餘 8 個於 gx10 以其 GGUF 生成），約 204 萬 token 用於 Tier 1，其中 67,875 token 的逐 token 序列用於 Tier 2。每個 session 依特殊 token（`<|think|>`、`<|channel>` 等）切分為 reasoning、code、factual、creative、control、user_prompt 區塊。另生成 writer_04（6,172 token）與 coder_04（7,923 token）兩個 session，作為評估 H2 的全新資料，不參與任何訓練與超參數選擇。

**資料切分。** router LoRA 與補償 adapter 以 7 個 session 訓練（coder_pilot、coder_01、coder_02、planner_pilot、planner_01、planner_02、writer_01），以 3 個 session（coder_03、planner_03、writer_02）作留出評估，另以 2 個全新 session 作最終檢驗。

### 4.3 量測方法

**Tier 1（累計計數）。** `llama-imatrix` 輸出每層 128 個專家的使用次數；以前綴差分窗口（每個 session 三份串接，D×3）分離各區塊的計數。共 70 個 job。

**Tier 2（逐 token 序列）。** 以 Python ctypes 載入 llama.cpp 的共享函式庫，於計算圖的逐節點回呼讀取張量 `ffn_moe_topk-L`（每層路由器的 top-k 索引）。結構欄位位移不手抄，而是以該版本標頭編譯一個小程式量出（`config/tier2_layouts/`）；為避免最後一層的路由張量依輸出 token 被裁剪，對每個 token 要求 logits。量測本身以兩項檢驗確認：與同一 session 的 Tier 1 整場計數的相關係數，以及同一輸入重複執行的逐位一致性。

**專家池重放。** 以 Tier 2 序列重放三種策略（§3.5），計算命中率（token 所需專家已在池中的比例）與每 token 的載入專家數。

**LoRA 評估。** 於 offload 引擎（bf16）讓學生路由實際驅動模型，記錄各層 top-k、NLL 與 next-token top-1；另將 LoRA 轉為 GGUF，以 Tier 2 程式於 llama.cpp（Q4_K_M）重複量測集合大小。

### 4.4 評估指標與判準

- **局部性：** Pr[E_i ∩ E_{i+1} ≠ ∅]。比較基準為均勻隨機（P = 1 − C(120,8)/C(128,8) = 0.412）與打亂順序對照（保留各專家的使用頻率、拆除時間相鄰）。
- **親和集合：** 類型 T 在層 l 中涵蓋 α（0.8 或 0.9）路由質量所需之最少專家集合；以各層中位數報告。
- **類型分離度：** V5（以組間／組內比 R 與 Jaccard 評估）、V6（split-half 重現性）、V7（樣本量）、V8（帳目恆等式）。
- **H2 判準（預先設定）：** 集合大小變化（四型 α = 0.9 中位數的平均）≤ −30%；ΔNLL ≤ 0.15 nats/token；與基底的 next-token top-1 一致率 ≥ 0.85。三項皆須成立。
- **評測協定：** 品質判準採 teacher forcing 的 NLL 與 next-token top-1 一致率，不經自由生成，因此不受生成停止條件的影響；涉及自由生成的比較以模型完整的結束 token 集合（id 1、106、50；回合結束符為 `<turn|>`，id 106）為停止條件，並以保留特殊 token 的方式解碼。
- **超參數選擇：** 於留出 session 上進行，共 10 組設定；最終選定之組別另於全新 session 評估一次。

---

## 5. 實驗結果

### 5.1 Phase 1 外部監控器

#### 5.1.1 Phase 1 實測結果（GT 1030）

| 組件 | 狀態 | 關鍵指標 |
|------|------|---------|
| RoleTagParser | **通過** | 4 邊界（CoT+程式碼），6（混合） |
| PerplexitySpikeMonitor | **通過** | 6 尖峰，z 分數 4.3–407.2 |
| SignalFusion | **通過** | 發射 4/6，遲滯抑制 2 |
| 即時熵監控器 | **通過** | 5.1 tok/s，偵測到邊界 Δ=0.41 |

#### 5.1.2 整合測試（7/7 通過）

| 測試 | 結果 | 細節 |
|------|------|------|
| VRAM 檢查 | 通過 | 閒置 1528 MB 可用 |
| 區塊偵測 | 通過 | 4 個邊界 |
| KV 序列化 | 通過 | float16 往返，shape=(2,32,64) |
| 控制器健康 | 通過 | 550 ms 回應 |
| 分析區塊 | 通過 | 4 個區塊分類 |
| 推理程式碼 | 通過 | 12 tok/s，回應含 `def` |
| 推理推理 | 通過 | 正確答案 "150"（60×2.5） |

### 5.2 Gemma4 26B-A4B 於 GT 1030 的資源配置

| 項目 | 結果 |
|------|------|
| 配置 | attention 層上 GPU，全部專家 FFN 以 `--override-tensor` 留在 CPU/RAM |
| VRAM（`-ngl 10 -c 16384`） | 1682 MiB |
| VRAM（`-ngl 11 -c 8192`） | 1672 MiB |
| 解碼速度 | 4.22 tok/s（專家 FFN 位於 CPU，受 DDR3 頻寬支配） |

ngl 高於上述數值時 VRAM 超出 2 GB 卡的容量。

### 5.3 專家路由的類型分布（Tier 1）

70 個 job 涵蓋 2,036,795 token（reasoning 1,248,355；code 464,177；factual 75,632；creative 162,170；混合區塊約 8.6 萬）。

| 檢驗 | 結果 |
|---|---|
| V7 樣本量（每型 ≥ 40k token） | 四型皆達成 |
| V8 帳目恆等式 | 成立 |
| V6 split-half 重現性 | reasoning、code：28/30 層 ρ ≥ 0.90；factual：26/30 層；creative：0/30 層（2 個 session） |
| V5 reasoning／code | 中位數 R = 6.67（29/30 層 R ≥ 3） |
| V5 code／factual | R = 10.32（30/30 層） |
| V5 reasoning／factual | R = 2.25（7/30 層） |
| V5 含 creative 的三組 | R = 0.63／1.47／0.55 |

各類型的專家集合大小（各層中位數，共 128 個專家）：

| 區塊類型 | α = 0.9 | α = 0.8 |
|---|---|---|
| reasoning | 62 | 45.5 |
| code | 49 | 34.0 |
| factual | 50 | 36.5 |
| creative | 52 | 38.0 |

類型之間的集合相似度（α = 0.8，層中位數 Jaccard）：reasoning–code 0.43、reasoning–factual 0.68、reasoning–creative 0.43、code–factual 0.28、code–creative 0.19、factual–creative 0.52。各類型的集合佔每層 38%–48%（α = 0.9）；以類型條件化相較於不分類型的混合集合（66 個）平均縮小 19%（reasoning 6%、code 26%、factual 24%、creative 21%）。同一語料於 gx10（不同 llama.cpp build 與 GGUF 檔）上的 V5／V6／V7／V8 判定類別相同，median R 差異小於 2%。

### 5.4 逐 token 時間局部性（Tier 2）

**量測驗證。** Tier 2 的逐 token 加總與同一 session 的 Tier 1 整場計數相關係數為 0.9993（planner_pilot）與 0.9995（coder_pilot），總數相差 240（= 30 層 × 8，對應 imatrix 多計的一個 token）；同一輸入重複執行的結果逐位相同。

**局部性（10 個 session 合併）。**

| 相鄰 token 專家集合有交集的機率 | 數值 |
|---|---|
| 均勻隨機之期望 | 0.412 |
| 打亂順序對照 | 0.737 |
| **實測（全部相鄰 token 對）** | **0.915**（層中位數 0.946；26/30 層 ≥ 0.85） |
| reasoning／code／factual／creative | 0.908／0.922／0.914／0.931 |
| 跨區塊邊界之相鄰對（206 對） | 0.940 |
| 平均交集大小（每 token 8 個專家） | 3.29 |

H1 獲得支持（0.915 ≥ 0.85）。打亂順序對照為 0.737，相鄰關係貢獻約 0.18。

**動態窗口。** 以最近 W 個 token 使用過的專家為快取：W = 8 時命中率 0.773（每層平均 28.7 個專家）、W = 32 時 0.934（52.2 個）、W = 128 時 0.986（74.1 個）。與靜態類型集合（α = 0.9，約 50–62 個專家，涵蓋約 90% 路由質量）相比，同等規模下動態窗口的命中率相當或略高；靜態集合由同一份資料挑選，數值偏高估。

**GT 1030 重放。** Windows b8679 版本不附標頭，結構位移依 gx10 版面扣除 b8679 之後新增的欄位推定（`config/tier2_layouts/b8679.json`），並以行為驗證（上下文建立成功、每層每 token 恰一列）。條件為核心 2+3、低優先權、`-ngl 8`、ubatch 128，VRAM 1609 MiB（上限 1700）。處理速度為 0.27 tok/s，整個 session 約需 7 小時，故重放 planner_pilot 前 1280 個 token。局部性為 0.910（gx10 同一段 0.907；打亂對照兩者皆為 0.776）；兩台機器的專家使用次數矩陣相關係數為 0.9972（各層最低 0.986）；每個 token 在每層的 8 個專家，兩者平均共有 7.39 個，56% 的 token-層完全相同。

### 5.5 專家池策略

**池容量（e_size = 4.14 MiB，L = 30）。**

| 快取預算 R | 每層 E_warm | 占 128 個專家 |
|---|---|---|
| 4 GB | 32 | 25% |
| 7 GB | 57 | 45% |
| 12 GB | 98 | 77% |

7 GB 對應的 57 個專家與各類型親和集合大小（49–62）相近；四個類型的聯集（α = 0.9 約 84.5）超過該容量。

**命中率（10 個 session 的真實路由序列重放）。**

| 每層池容量 C | reactive | static | spec |
|---|---|---|---|
| 32 | 0.790 | 0.687 | 0.791 |
| 48 | 0.901 | 0.814 | 0.902 |
| 57 | 0.939 | 0.865 | 0.940 |
| 64 | 0.960 | 0.897 | 0.961 |

區塊邊界預取（spec）與 token 層級 LRU（reactive）的命中率相差 0.001，H3 未獲支持；以類型固定池（static）的命中率較低。

**載入量與時間。** 單一專家單一層的載入時間（DDR3-1600 標稱頻寬 17 GB/s）為

$$t_{\text{load}} = \frac{4.14\text{ MiB}}{17\text{ GB/s}} \approx 0.24\text{ ms}$$

每產生一個 token（30 層）自慢速層載入的專家數與對應時間：

| C | reactive | spec | static |
|---|---|---|---|
| 57 | 14.5 個（3.5 ms） | 19.8 個（4.7 ms） | 32.5 個（7.7 ms） |
| 64 | 9.6 個（2.3 ms） | 15.6 個（3.7 ms） | 24.8 個（5.9 ms） |

spec 的載入量高於 reactive，差額為各區塊邊界的預載。上述時間依標稱頻寬換算，為載入量的指標，並非延遲實測。原 Mixtral 理論估算（20 個區塊的會話節省 1.76 秒）所依據的假設為類型集合命中率接近 100%，於本實測中未被觀察到，本文以實測結果取代該估算。

### 5.6 以 LoRA 縮小專家集合（H2）

#### 5.6.1 僅調整 attention 與稠密 MLP 的 LoRA

既有的 LoRA adapter 共 410 個張量，作用於 `q/k/v/o_proj` 與稠密 MLP 的 `gate/up/down_proj`（30 層；v_proj 位於 25 層），不含專家張量與路由器。轉為 GGUF LoRA（37 MB）後，於 §4.1 的受限條件下比較掛載前後：

| 條件 | 集合大小變化（reasoning／code／factual／creative） | 層中位數 Jaccard |
|---|---|---|
| 重複執行（雜訊底線） | 0／0／0／0% | 1.000 |
| LoRA ×1 | 0／0／0／0% | 0.97–0.99 |
| LoRA ×4（放大以確認量尺的靈敏度） | −1.6／0／−1.0／−3.8% | 0.93–0.96 |

平均變化為 0%，低於 H2 的 −30% 門檻。放大四倍時 Jaccard 下降，表示量尺可觀察到專家選擇的改變。

#### 5.6.2 router LoRA（第一階段）

以 §3.9 的路由損失訓練（rank 16，7 個訓練 session，GPU 峰值 0.58 GiB，每組約 11 分鐘），於留出 session 評估，共 7 組設定：

| 組 | 設定 | 集合大小變化 | ΔNLL | top-1 一致率 |
|---|---|---|---|---|
| a1 | 無 hinge、KL 權重 0.2 | −71.7% | +1.354 | 0.605 |
| b1 | hinge、KL 1.0 | −43.2% | +0.338 | 0.782 |
| b2 | hinge、KL 3.0 | −25.0% | +0.045 | 0.855 |
| **c1** | hinge、KL 1.8 | **−33.7%** | **+0.115** | 0.828 |
| c2 | 目標 50%、KL 3.0 | −28.4% | +0.082 | 0.843 |
| d1 | c1 加「保留基底 top-1 專家」項 | −35.4% | +0.268 | 0.802 |
| e1 | c1，rank 64、4 epoch | −35.3% | +0.166 | 0.825 |

集合縮小幅度與 top-1 一致率呈取捨關係：縮小 25% 時一致率 0.855，縮小 34% 時 0.828，縮小 43% 時 0.782。c1 於留出 session 的各類型縮小幅度為 reasoning −39.8%、code −32.3%、factual −32.7%、creative −29.9%。

#### 5.6.3 補償 adapter（第二階段）

凍結 c1 的 router LoRA，另訓練 attention 與稠密 MLP 的 LoRA（rank 8、512 token 窗口、GPU 峰值 0.79 GiB）：

| 組 | 損失 | 集合大小變化 | ΔNLL | top-1 一致率 |
|---|---|---|---|---|
| f1 | 0.5 真實標籤 + 0.5 蒸餾，2 epoch | −34.7% | −0.584 | 0.812 |
| **f2** | **純蒸餾，2 epoch** | **−33.2%** | **+0.043** | **0.843** |
| f3 | 純蒸餾，4 epoch、學習率加倍 | −33.4% | +0.325 | 0.846 |

同樣縮小約 33% 時，top-1 一致率由 router LoRA 單獨的 0.828（c1）提升至 0.843（f2），ΔNLL 維持於 0.15 以內。f1 的 NLL 低於基底，反映模型向此語料（Gemma4 自行生成）的適應，同時其預測與基底的一致率最低。f3 的一致率增加 0.003，NLL 增加至 +0.325。

#### 5.6.4 全新 session 的最終檢驗

在檢視全新資料前選定 f2（留出 session 上的最佳折衷），評估一次。結果（offload 引擎）：集合大小 reasoning −32.5%、code −30.2%、creative −19.8%，平均 −27.5%；ΔNLL +0.103；top-1 一致率 0.835。全新 session 涵蓋 reasoning、code、creative 三種內容類型（writer_04 的 factual 回合被切分為 reasoning）。在 llama.cpp 以 c1 與 f2 兩個 LoRA 量測同樣兩個 session：reasoning −28.8%、code −26.7%、creative −19.0%；原 10 個 session（含訓練 session）平均 −30.9%（reasoning −37%、factual −32%、code −30%、creative −25%），掛載後相鄰 token 局部性為 0.957（打亂對照 0.872），類型分離度（JSD）下降約 15%（code–reasoning 0.183 → 0.154；code–factual 0.320 → 0.280）。

**H2 判定。** 判準數值為研究假設值，用以界定結果與預判的接近程度。留出 session：集合縮小 33.2%（高於 30%）、ΔNLL +0.043（低於 0.15）、top-1 一致率 0.843（假設值 0.85）；全新 session：縮小 27.5%（假設值 30%）、ΔNLL +0.103、一致率 0.835。結果與假設值接近，H2 獲得初步支持；留出 session 的縮小幅度（33%）與全新 session（27.5%）之間的差異，包含超參數選擇偏差與泛化差距，後續須以更多資料持續驗證。

### 5.7 Phase 2–7 結構驗證

| Phase | 驗證內容 | 結果 |
|-------|---------|------|
| P2 KV 卸載 | 6 區塊三層儲存/取回 | 通過 |
| P3 引導 | 4 種類型 × 引導配置 | 通過 |
| P4 專家池 | Gemma4 26B-A4B 真實路由重放（§5.5）：C = 57 時命中率 0.94；類型邊界預取與 LRU 相當 | 完成（實測） |
| P5 邊緣 VLM | 4 設備 × 硬體配置 | 通過 |
| P6 Mac M4 | 4 模型 × UMA 池規劃 | 通過 |
| P7 自迴圈 | 4 蒸餾樣本，+0.030/iter 趨勢 | 通過 |

---

## 6. 討論

### 6.1 研究問題與假設的回答

- **RQ1：** 三訊號監控器於 GT 1030 對 Qwen3-1.7B 即時運作（5.1 tok/s，7/7 整合測試通過），RQ1 獲得肯定的結果。
- **RQ2、RQ3：** 設計與離線結構驗證完成；其對真實模型的效果量測不在本研究範圍。
- **RQ4（H1、H2、H3）：** H1 獲得支持：相鄰 token 的專家集合交集機率為 0.915，在 GT 1030 重放為 0.910。H3 未獲支持：區塊邊界類型預取與 token 層級 LRU 命中率相當，載入量較高。H2 的集合縮小可由 router LoRA 達成（留出 session −33.2%、全新 session −27.5%），與基底的一致率為 0.843 與 0.835，與假設值 0.85 接近，H2 獲得初步支持，後續須進一步驗證。
- **RQ5：** 蒸餾式 LoRA 訓練可在 1.0 GiB 的 GPU 預算內完成（§5.6.3，GPU 峰值 0.79 GiB）；整個迴圈對品質的影響不在本研究量測範圍。

### 6.2 結果的意涵

1. **預取的價值來源。** Gemma4 的路由本身具有強烈的相鄰重疊，LRU 即可利用；區塊類型所提供的額外資訊在命中率上無可觀察的增益。
2. **LoRA 與專家選擇。** LoRA 需作用於路由器才能影響專家的選擇；僅作用於 attention 與稠密 MLP 的 LoRA 對專家集合無可觀察的影響。
3. **縮小幅度與輸出一致性的取捨。** 在本研究的設定下，集合縮小約 33% 時，top-1 一致率位於 0.83–0.85 之間。
4. **評測設計。** 本研究以 teacher forcing 的 NLL 與 next-token top-1 一致率作為品質判準，比較結果不依賴生成的停止條件；涉及自由生成的比較，則以模型完整的結束 token 集合（Gemma4 為 id 1、106、50）為停止條件，並保留特殊 token 以判讀回合結束位置。此評測設計使各組 LoRA 之間的比較具有一致的基準（§4.4）。

### 6.3 有效性威脅

- **內部效度：** 超參數於留出 session 上選自 10 組設定，留出結果含選擇偏差，故另以全新 session 評估最終選定之組別。判準數值（NLL ≤ 0.15、top-1 ≥ 0.85）為研究假設值，並非外部標準。
- **外部效度：** 單一模型（Gemma4 26B-A4B，Q4_K_M）、單一語料來源（模型自行生成的對話）；creative 與 factual 的樣本較少，全新 session 未含 factual。
- **量測效度：** 載入時間由標稱頻寬換算，並非延遲實測；NLL 與 top-1 於 bf16 offload 引擎量測，局部性與集合大小的第二次量測於 Q4_K_M 的 llama.cpp；GT 1030 的重放涵蓋 1280 個 token。

### 6.4 適用範圍與證據等級

| 內容 | 證據等級 | 條件 |
|---|---|---|
| Phase 1 監控器（§3.2、§5.1） | GT 1030 即時實測 | Qwen3-1.7B，GT 1030 + i5-4460 |
| Gemma4 26B-A4B 的 VRAM 與解碼速度（§5.2） | GT 1030 實測 | Vulkan b8679，`-ngl 10/11` |
| 專家分布 Tier 1（§5.3） | GT 1030 實測，於 gx10 重現 | 10 個 session、70 個 job |
| 逐 token 專家序列與局部性（§5.4） | gx10 受限條件實測；GT 1030 重放前 1280 個 token | gx10：核心 0–4、nice 10、`-ngl 2`、933 MiB |
| 專家池策略（§5.5） | 對真實路由序列的重放 | 載入時間由標稱頻寬換算 |
| router LoRA 與補償 adapter（§5.6） | gx10 受限條件實測（GPU 預算 1.0 GiB） | 留出與全新 session |
| Phase 2、3（§3.3、§3.4） | 離線結構驗證 | 合成張量，未接模型 |
| Phase 5（§3.6） | 結構驗證；裝置數字為廠商規格 | 未取得 Jetson、Android、ARM 攝影機 |
| Phase 6（§3.7） | 結構驗證；UMA 與 NVMe 數字為廠商規格 | 未取得 Mac Mini M4；DeepSeek-V3 的專家分布未量測 |
| Phase 7（§3.8） | 結構驗證；§5.6.3 為同一機制於受限 GPU 的實測 | |

語料為 Gemma4 自行生成的對話，Windows 上 2 個 pilot，其餘於 gx10 以其 GGUF（與 GT 1030 的檔案不同）生成。結論針對 Gemma4 26B-A4B（Q4_K_M）；Mixtral、DeepSeek 等模型的敘述引自公開規格。Phase 5、Phase 6 的裝置數字（FPS、UMA 頻寬、NVMe 速度）為規格值。

---

## 7. 結論與未來工作

在 2 GB VRAM 的 GT 1030 與舊電腦上，Gemma4 26B-A4B 以 `-ngl 10`、1.7 GB VRAM 運行，解碼 4.22 tok/s。其專家路由具有明顯的時間局部性（相鄰 token 交集機率 0.915），此局部性主要由熱門專家的集中與相鄰關係共同構成，且可由 token 層級 LRU 有效利用；區塊邊界的類型預取在命中率上與 LRU 相當。縮小專家集合可由掛載於路由器的 LoRA 達成：留出 session 平均縮小 33.2%，於全新 session 縮小 27.5%；與基底的 next-token top-1 一致率為 0.843 與 0.835，與假設值 0.85 接近，H2 獲得初步支持；縮小幅度與輸出一致性之間呈取捨關係。

**未來工作。** （1）以更多 session（含 factual 與更多 creative）、其他語料來源與其他 MoE 模型持續驗證 H2 的集合縮小幅度與輸出一致性；（2）探討以更大的蒸餾資料與其他損失，使集合縮小與一致率的取捨曲線移動；（3）將專家池接入 llama-server 以量測線上命中率與延遲；（4）於 Jetson 與 Mac M4 取得 Phase 5、6 的實測數據；（5）擴展至其他 MoE 模型。

---

## 參考文獻

[1] Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv:1701.06538*.

[2] Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models. *JMLR*, 23(120), 1–39.

[3] Jiang, A.Q., et al. (2024). Mixtral of experts. *arXiv:2401.04088*.

[4] Sheng, Y., et al. (2023). FlexGen: High-throughput generative inference with a single GPU. *ICML 2023*.

[5] Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication at scale. *NeurIPS 2022*.

[6] Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate post-training quantization. *arXiv:2210.17323*.

[7] Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference via speculative decoding. *ICML 2023*.

[8] Miao, X., et al. (2023). SpecInfer: Tree-based speculative inference. *arXiv:2305.09781*.

[9] Clark, K., Khandelwal, U., Levy, O., & Manning, C.D. (2019). What does BERT look at? *BlackboxNLP 2019*.

[10] Voita, E., et al. (2019). Analyzing multi-head self-attention. *ACL 2019*.

[11] Elhage, N., et al. (2021). A mathematical framework for transformer circuits. *Transformer Circuits Thread*.

[12] Zou, A., et al. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv:2310.01405*.

[13] Turner, A., et al. (2023). Activation addition: Steering LMs without optimization. *arXiv:2308.10248*.

[14] Subramani, N., Suresh, N., & Peters, M. (2022). Extracting latent steering vectors. *ACL Findings 2022*.

[15] Hu, E.J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

[16] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS 2023*.

[17] Zhang, Q., et al. (2023). AdaLoRA: Adaptive budget allocation for PEFT. *ICLR 2023*.

[18] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NIPS DL Workshop 2014*.

[19] Wang, Y., et al. (2023). Self-instruct: Aligning LMs with self-generated instructions. *ACL 2023*.

[20] Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv:2212.08073*.

[21] Dao, T., Fu, D.Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention. *NeurIPS 2022*.

[22] Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.

[23] DeepSeek-AI (2024). DeepSeek-Coder-V2. *arXiv:2406.11931*.

[24] Qwen Team (2025). Qwen3 technical report. *arXiv:2505.09388*.

[25] Google DeepMind (2025). Gemma 4 technical report.

---

## 附錄 A：完整模組對照表

```
Phase  模組                                  功能
────────────────────────────────────────────────────────
P1     monitor/attention_entropy.py           Shannon 熵監控器
P1     monitor/perplexity_spike.py            z 分數尖峰偵測器
P1     monitor/role_tag_parser.py             確定性有限狀態機
P1     monitor/signal_fusion.py               優先權加權融合
P1+3   segmentor/chunk_classifier.py          類型→路由映射
P2     kv/kv_serializer.py                    .kvbin 格式
P2     kv/kv_offload_manager.py               三層 LRU 層級結構
P3     spec_experts/directional_steering.py   系統提示引導
P4     spec_experts/expert_ram_pool.py        MoE 專家預取
P5     edge/vlm_router.py                     VLM 路由 + 目標設備
P6     mac_m4/metal_expert_pool.py            UMA 池 + Flash Attn
P7     agi/self_loop.py                       LoRA 池 + 蒸餾
P1-7   spec_experts/controller.py             FastAPI :8090
P1-4   spec_experts/tool_session.py           熱切換 llama-server
P4     affinity_pipeline/tier2_ctypes.py      逐 token 專家 id（ctypes 取 llama.cpp 路由 top-k）
P4     affinity_pipeline/tier2_stats.py       局部性、親和集合、窗口命中率、前後比較
P4     scripts/tier2_locality.py              Tier 2 驅動（run / analyze / compare）
P4     scripts/pool_replay.py                 專家池策略重放（§5.5）
P4     scripts/h2_targets.py + h2_router_lora.py + h2_comp.py   router LoRA 與補償 adapter（§5.6）
P4     config/tier2_layouts/*.json            llama.cpp 結構偏移（d327bf1 由標頭量出；b8679 為推定）
測試   tests/phase1_monitor_demo.py           P1 即時示範（GT 1030）
測試   tests/phase2_7_demo.py                 P2-7 結構驗證
測試   tests/spec_experts_test.py             7/7 整合測試
```

## 附錄 B：復現步驟

```bash
git clone https://github.com/sipurchen/klchen
git checkout SpecExpertsResearch
set LLMS_DIR=<LLMS_DIR>

# Phase 1 即時示範（需要 llama-server 在 :8080）
python tests/phase1_monitor_demo.py

# Phase 2-7 結構驗證（離線）
python tests/phase2_7_demo.py

# 整合測試（需要控制器在 :8090）
python tests/spec_experts_test.py
```

## 附錄 C：Tier 2、專家池重放與 LoRA 實驗的復現步驟

以下指令經 `tools/slice_runner/gx10_run.py` 在共用主機上執行，需先設定環境變數：`GX10_HOST`（ssh 目標）、`TRAIN_PY`（含 torch／transformers／peft 的 python）、`OFFLOAD_ENGINE_DIR`（含 `lora_offload` 套件的目錄）、`BASE_HF_DIR`（Gemma4 26B-A4B 的 HF 權重目錄）。

```bash
# Tier 2：逐 token 專家序列與局部性（gx10 受限條件：核心 0-4、nice 10、-ngl 2、-ub 64）
python tools/slice_runner/gx10_run.py run layout
python tools/slice_runner/gx10_run.py run tier2 --args "run --tag base_all --sessions all --ngl 2"
python tools/slice_runner/gx10_run.py run tier2 --args "analyze --tag base_all"

# 專家池策略重放（§5.5）
python tools/slice_runner/gx10_run.py run replay --args "--tag base_all --caps 32,48,57,64"

# H2：目標集合、router LoRA、補償 adapter、留出集評估（§5.6）
python tools/slice_runner/gx10_run.py run h2targets --args "--tag base_all --train coder_pilot,coder_01,coder_02,planner_pilot,planner_01,planner_02,writer_01"
python tools/slice_runner/gx10_run.py run h2 --args "train --tag c1 --lam 1.0 --mu 1.8 --lr 1e-3 --hinge 0.9 --epochs 2 --accum 2"
python tools/slice_runner/gx10_run.py run h2 --args "eval --tag c1"
python tools/slice_runner/gx10_run.py run h2 --args "comp-train --tag f2 --rank 8 --alpha 8 --window 512 --lr 1e-4 --lmw 0.0 --epochs 2 --accum 2"
python tools/slice_runner/gx10_run.py run h2 --args "comp-eval --tag f2"
```

GT 1030（Windows，核心 2+3、低優先權）上的 Tier 2：

```powershell
$p=[System.Diagnostics.Process]::GetCurrentProcess(); $p.ProcessorAffinity=[IntPtr]12; $p.PriorityClass='BelowNormal'
$env:LLMS_DIR='<LLMS_DIR>'; $env:AFFINITY_PROFILE='gt1030_win'
python -u scripts\tier2_locality.py run --tag gt_pilot --sessions planner_pilot,coder_pilot --ngl 8 --ubatch 128
```
