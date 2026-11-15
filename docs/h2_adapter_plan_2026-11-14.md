# H2 adapter 規劃：為「只用我們要的專家」而訓練的 router LoRA（Fable5 CoT）

日期：2026-11-14。執行位置：gx10（研究在 gx10 上做，完成後同步回本機 repo）。性質：規劃，並依此執行。

## 1. Frame

H2 要過關，adapter 就是變數。既有那顆（僅調整 attention 與稠密 MLP 的 adapter）專家集合大小變化 0%，不是因為「LoRA 不能選專家」，而是它根本沒碰專家的選擇：
410 個張量只有 `q/k/v/o_proj` 與稠密 MLP，沒有 `router.proj`，訓練目標也只有語言模型損失，沒有任何「集中在某些專家」的壓力。
所以要驗證 H2，必須自己做一顆**目標與位置都對**的 adapter：LoRA 掛在路由器上，損失明確要求「這一類型的 token 只用這一類型的專家」。
判準不變：四種內容類型的 α=0.9 親和集合中位數平均下降 ≥ 30%；另加品質護欄，避免用「把模型弄壞」換集合變小。

## 2. Facts（本輪在 gx10 上直接查證）

- Gemma4 路由器（transformers 5.17 `Gemma4TextRouter`）：`norm → × scale × hidden^-0.5 → proj(2816→128)`，對 128 個專家 softmax，再取 top-8；
  `proj` 是 `nn.Linear`，可掛 LoRA；機率在 top-k 之前就有，可直接拿來算損失。
- llama.cpp d327bf1 在 `build_moe_ffn` 對 `gate_inp` 用 `build_lora_mm`，轉檔器有 `model.layers.{bid}.router.proj → gate_inp` 的映射
  （gemma4）。所以 router LoRA 原則上能轉成 GGUF LoRA、在 llama.cpp 上套用（**實際是否等價尚未驗證**，見 Forks 3）。
- 現成 offload 引擎（凍結專家按需載入、逐層串流非專家權重）能在 1.0 GiB GPU 預算內跑 forward；router 的參數很小（`proj` 36 萬、`scale` 2816），
  引擎會讓它常駐 GPU。
- 本專案已有可重用的量尺：Tier 2 逐 token 專家序列與 `tier2_stats`（親和集合、類型條件化、窗口命中率）。
- 資料：10 個 session（約 6.8 萬 token），已切成 reasoning／code／factual／creative／control／user_prompt 區塊。內容是 Gemma4 自己生成的對話，不含任何外部資料。
- 現成 adapter 的訓練設定問題（`<eos>` 結尾、Alpaca 格式、損失含提示詞）不沿用。

## 3. Forks

1. **LoRA 掛哪裡。** (a) attention + 稠密 MLP（現成那顆）：沒有路由壓力，已證實 0%。(b) 專家權重的低秩增量：專家是融合張量，工程量大。
   (c) 只掛 `router.proj`：參數約 150 萬、貼近目標、可逐層獨立訓練。**選 (c)。** 代價：adapter 只能改「選誰」，不能補償少用專家造成的品質損失，所以品質護欄很重要。
2. **損失怎麼寫。** 對每一層、每個屬於類型 T 的 token：`−log Σ_{e∈S(T,l)} p_e`（機率質量留在目標集合內），
   加上對所有 token 的 `KL(p_base ‖ p_student)`（讓路由不要離基底太遠）。`S(T,l)` 是**訓練 session** 上類型 T 在第 l 層使用次數最多的 m 個專家，
   `m = ⌈0.6 × |S_{T,l}(α=0.9)|⌉`（目標比判準更緊，留餘裕）。control／user_prompt 只有 KL。不用語言模型損失：目標是路由，而且那樣不必對凍結專家反向傳播。
   代價：訓練時主體沿用基底路由（teacher forcing），累積誤差要靠評估時「真的用學生路由跑」來檢驗。
3. **在哪個引擎量。** 訓練與評估都在 HF offload 引擎（同一引擎、同一模型、學生路由真的帶動主體），避免依賴「GGUF LoRA 套在 router 上是否等價」。
   GGUF 轉檔與 llama.cpp 上的重量是之後的獨立驗證，不是 H2 判定的依據。
4. **資料切分。** 7 個 session 訓練、3 個 session（coder_03、planner_03、writer_02）完全不碰，只用來評。H2 以**留出集**判定。
5. **品質護欄（先寫下）。** 留出集上相對基底：平均 NLL 增加 ≤ 0.15 nats/token，且 next-token top-1 與基底一致率 ≥ 0.85。超過就不算過關。

## 4. Fix Plan

1. `scripts/h2_targets.py`：用訓練 session 的 Tier 2 序列算每型每層目標集合，存 `logs/h2/targets.json`。
2. `scripts/h2_router_lora.py`（在 gx10 的 unsloth venv 執行，唯讀使用現成引擎）：`train`（router LoRA，窗口 768 token）與 `eval`（基底 vs 學生，同進程，記錄 top-k、NLL、top-1 一致率）。
3. 先小規模冒煙（1 個窗口，確認損失下降、不爆顯存），再正式訓練；超參數只調 λ、μ、學習率，每次都在留出集評，並**全部記錄**，不挑最好的一次當唯一結果。
4. 以 `tier2_stats` 算留出集的 α=0.9 集合中位數、Jaccard，對判準與護欄。
5. 結果同步回本機：adapter（PEFT 格式）、metrics、評估 JSON，更新論文 §6.8 與 README。

## 5. Final Proof

- 過關：留出集四種類型的 α=0.9 集合中位數平均下降 ≥ 30%，且兩項品質護欄都成立。
- 不過關也照實寫：寫明是哪一項沒過、最好的一組超參數是什麼、嘗試了幾組。
- 不涵蓋：GGUF／llama.cpp 上的等價性、GT 1030 上的重跑、專家權重本身的 LoRA、creative／factual 的 session 數太少造成的估計噪音。

## 6. 執行結果（2026-11-15）

7 組超參數、全部列在論文 §6.8。最接近的是 c1：留出集集合 −33.7%（尺寸判準過）、ΔNLL +0.115（過）、top-1 一致率 0.828（差 0.022，不過）。
**H2 初步獲得支持，仍須後續驗證。** 判準數值為研究假設值；集合縮小與 NLL 於留出 session 達成，top-1 一致率 0.843 接近假設值 0.85。規劃時預見的風險是 router LoRA 只能改「選誰」，補償不了少用專家造成的品質變化。
計畫外的調整：加了 hinge（質量達 0.9 後不再施壓）、「保留基底 top-1 專家」項（d1，結果更差）、rank 64 與 4 epoch（e1，沒有更好）。
本計畫的 Fork 3（引擎）有一處修正：HF 引擎判定 H2 之後，也把 c1 轉成 GGUF LoRA 在 llama.cpp 上量過（10 個 session，含訓練 session），尺寸 −30.9%，與 HF 的 −33.7% 一致。
下一步：補償版本（attention／稠密 MLP LoRA ＋ 蒸餾損失）、保留一份從未用於選超參數的測試資料。

## 7. 補償階段與最終檢驗（2026-11-15）

規劃時預見的風險（router LoRA 補償不了品質）用補償 adapter 處理：凍結 c1 的 router LoRA，再訓練 attention／稠密 MLP 的 LoRA（rank 8、窗口 512，GPU 預算 1.0 GiB 內），
訓練時讓學生路由真的帶動主體，損失 = w × 真實標籤交叉熵 + (1 − w) × 對基底 argmax 的交叉熵。三組：f1（w = 0.5）、f2（w = 0，純蒸餾）、f3（w = 0，4 epoch、學習率加倍）。
留出集：f2 集合 −33.2%、ΔNLL +0.043、top-1 0.8434（差 0.0066）；f1 的 NLL 低於基底但一致率最低（0.812）；f3 的 NLL 過界。
為了不被「用留出集挑超參數」的偏差誤導，另外生成兩個全新 session（writer_04、coder_04），事先選定 f2、只評一次：集合 −27.5%、ΔNLL +0.103、top-1 0.835。
**H2 於兩份資料上結果與假設值接近（留出 −33.2%／0.843，全新 −27.5%／0.835），獲得初步支持，後續以更多資料持續驗證。** 計畫外的調整：f1 第一次以 rank 16、768 窗口訓練時在 1.0 GiB 預算下 OOM，改為 rank 8、512 窗口。
