# S8：Tier 2 逐 token 專家序列與 H2（LoRA 縮小專家集合）

日期：2026-11-14。執行位置：gx10（共用主機）的預設受限條件：CPU 核心 0–4（taskset）、nice 10、load1 ≤ 6、一次一個行程、
環境標記 + PID 精準清理、只寫 `~/gemma4_affinity_sim`；模型 `gemma4-26b-Q4_K_M.gguf`（gx10 的檔，非 GT 1030 的）、llama.cpp `d327bf1`、
`-ngl 2 -ub 64`、專家張量留 CPU。

## 新增程式

| 檔案 | 作用 |
|---|---|
| `affinity_pipeline/tier2_ctypes.py` | ctypes 載入 libllama/libggml，裝 `cb_eval`，取 `ffn_moe_topk-L`，回傳 uint8 [token, 層, 8]；結構位移來自 layout JSON |
| `affinity_pipeline/tier2_stats.py` | 局部性（含打亂順序對照）、類型計數與親和集合、動態窗口命中率、類型條件化、掛 LoRA 前後比較 |
| `scripts/tier2_locality.py` | 驅動：`run`／`analyze`／`compare`；VRAM 用 llama.cpp 的 buffer 日誌檢查 |
| `config/tier2_layouts/d327bf1.json` | 該 build 的 sizeof/offsetof（由 `tools/slice_runner/remote/layout_probe.py` 用該 build 自己的標頭編譯量出） |
| `tools/slice_runner/remote/{lora_probe,s4_repro_inner,crosscheck_tier1}.py` | PEFT→GGUF LoRA 轉換、S4 逐 token 重現、與 Tier 1 的交叉驗證 |
| `tests/affinity_pipeline/test_tier2.py` | 14 個測試（統計、layout 檔、不載入 libllama 的 callback 邏輯） |

過程中修掉的真實問題：最後一個 MoE 層的路由張量在 llama.cpp 裡會先依「要輸出的 token」裁剪，`llama_batch_get_one` 只要求最後一個
token 的 logits，所以第 29 層只收到每個 ubatch 一列；改用 `llama_batch_init` 對每個 token 要求 logits 後解決。

## 量測驗證

- 與 Tier 1 交叉驗證（整個 session 的 imatrix 計數 vs Tier 2 逐 token 加總）：planner_pilot r = 0.99927、coder_pilot r = 0.99952
  （測試計畫 T6.3 門檻 0.90）；總數差 240 = 30 層 × 8（imatrix 多算一個 token）。
- 重跑一次（`base_ctl`）與 `base_all` 逐位相同：親和集合 Jaccard 1.000。
- VRAM：`-ngl 3` 為 1031 MiB（超過上限 1000，守衛拒絕執行）；`-ngl 2` 為 933.24 MiB（warn，在上限內）。

## 結果（10 個 session、67,875 token；原始輸出在本目錄 `logs/tier2/`）

局部性 `Pr[E_i ∩ E_{i+1} ≠ ∅]`：均勻隨機 0.412；打亂順序對照 0.737；實測 0.915（中位數層 0.946、26/30 層 ≥ 0.85）。
各類型 reasoning 0.908／code 0.922／factual 0.914／creative 0.931；跨區塊邊界 0.940（僅 206 對）。

動態窗口：W=8 命中 0.773（28.7 專家）、W=32 命中 0.934（52.2）、W=128 命中 0.986（74.1）。類型條件化平均縮小 19%（α=0.9）。

H2（`compare_base_all_vs_*.json`）：判準 = 四種內容類型 α=0.9 親和集合中位數平均下降 ≥ 30%（先寫下再看資料）。

| 條件 | 變化（reasoning／code／factual／creative） | 平均 | 層中位數 Jaccard |
|---|---|---|---|
| 重跑 | 0／0／0／0% | 0% | 1.000 |
| LoRA ×1（僅調整 attention／稠密 MLP 的既有 adapter） | 0／0／0／0% | 0% | 0.984 |
| LoRA ×4 | −1.6／0／−1.0／−3.8% | −1.6% | 0.93–0.96 |

**H2 不成立。** adapter 共 410 個張量：q/k/v/o 與稠密 MLP 的 LoRA，沒有專家張量、沒有路由器。

## 限制

- gx10 的 GGUF 不是 GT 1030 的；Windows b8679 的結構位移尚未量，所以沒在 GT 1030 重跑。
- H2 只測了一顆 adapter（POC 規模、不碰專家）；為選專家而訓練的 LoRA、路由遮罩後的品質測試沒做。
- 邊界 206 對、creative 只有 2 個 session。

## H2 的第二次量測：自己訓練的 router LoRA（2026-11-15）

規劃：`docs/h2_adapter_plan_2026-11-14.md`；程式：`scripts/h2_targets.py`、`scripts/h2_router_lora.py`、`tools/slice_runner/remote/h2_run.py`。
LoRA 只掛 `router.proj`（30 層、rank 16），損失 = hinge 的「機率質量留在該類型目標集合內」＋ KL(基底‖學生)；7 個 session 訓練、
3 個 session（coder_03、planner_03、writer_02）留出評估；GPU 峰值 0.58 GiB。證據：`h2/logs/h2/`（每組的 `eval_*.json`、`train_log.json`、
目標集合）。adapter 權重檔不收錄於 repo，可依論文附錄 C 重現。

| 組 | 集合大小變化 | ΔNLL | top-1 一致率 | 判定 |
|---|---|---|---|---|
| a1 無 hinge、KL 0.2 | −71.7% | +1.354 | 0.605 | 品質崩壞 |
| b1 hinge、KL 1.0 | −43.2% | +0.338 | 0.782 | 兩項護欄不過 |
| b2 hinge、KL 3.0 | −25.0% | +0.045 | 0.855 | 尺寸不足 |
| **c1 hinge、KL 1.8** | **−33.7%** | **+0.115** | 0.828 | 只差 top-1（0.85） |
| c2 目標 50%、KL 3.0 | −28.4% | +0.082 | 0.843 | 尺寸與 top-1 都差一點 |
| d1 c1＋保留 top-1 專家項 | −35.4% | +0.268 | 0.802 | 更差 |
| e1 c1，rank 64、4 epoch | −35.3% | +0.166 | 0.825 | 沒有更好 |

判準（訓練前寫下）：四型 α=0.9 集合中位數平均 ≤ −30%、ΔNLL ≤ 0.15、top-1 ≥ 0.85。**判準數值為研究假設值；三項結果與假設值接近，H2 初步獲得支持，仍須以更多資料持續驗證。**
c1 轉成 GGUF LoRA 後在 llama.cpp 用 Tier 2 程式量 10 個 session（含訓練 session）：平均 −30.9%（reasoning −37%、factual −32%、code −30%、creative −25%），
局部性 0.957（打亂對照 0.872），類型分離度 JSD 下降約 15%（`logs/tier2/compare_base_all_vs_h2_c1.json`）。
限制：用留出集挑了 7 組超參數（偏樂觀）；護欄數值是我自訂的；訓練為 teacher forcing。

## 專家池重放（thesis §6.3、§6.4）

`scripts/pool_replay.py`，對 10 個 session 的真實專家序列重放三種策略（reactive LRU、static 類型固定池、spec 邊界預取＋LRU）。命中率 C=32／48／57／64：
reactive 0.790／0.901／0.939／0.960；static 0.687／0.814／0.865／0.897；spec 0.791／0.902／0.940／0.961。C=57 時每 token 載入 reactive 14.5、spec 19.8、static 32.5 個專家。
每專家一層 4.14 MiB 由 GGUF 張量大小驗證（down Q8_0 2.0088 + gate_up Q4_K 2.1270）。

## 補償 adapter 與全新 session 檢驗

| 組 | 集合大小變化 | ΔNLL | top-1 一致率 |
|---|---|---|---|
| f1（0.5 真實標籤 + 0.5 蒸餾） | −34.7% | −0.584 | 0.812 |
| f2（純蒸餾） | −33.2% | +0.043 | 0.843 |
| f3（純蒸餾、4 epoch） | −33.4% | +0.325 | 0.846 |
| **f2 在全新 session（writer_04、coder_04）** | **−27.5%** | +0.103 | 0.835 |

llama.cpp（c1 + f2 兩個 LoRA）：10 個原 session −30.9%；兩個全新 session reasoning −28.8%、code −26.7%、creative −19.0%。證據：`h2/logs/h2/eval_f*.json`、`eval_f2_fresh.json`、`logs/tier2/compare_*`。
全新 session 沒有 factual 類型（writer_04 的 factual 回合被切成 reasoning）。**H2 初步獲得支持（結果與假設值接近），後續持續驗證。**

## GT 1030 重放

Windows b8679 沒有標頭、本機沒有 C 編譯器：`config/tier2_layouts/b8679.json` 的偏移由 gx10 版面推定，行為驗證通過。核心 2+3、BelowNormal、`-ngl 8`、ubatch 128、VRAM 1609 MiB。
速度 0.27 tok/s（整個 session 約 7 小時），所以只重放 planner_pilot 前 1280 個 token：局部性 0.910（gx10 同段 0.907）、專家使用矩陣相關 0.9972、每個 token-層平均 7.39/8 個專家相同。證據：`gt1030/`。
