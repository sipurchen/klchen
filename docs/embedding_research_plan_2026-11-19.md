# Embedding 輔助研究規劃：路由精準度（A、C）與回答精準度（B）（Fable5 CoT，共筆文件）

日期：2026-11-19。性質：研究規劃、預先登記的判準與各切片的驗收紀錄（Fable5 CoT 格式）。
實驗位置：gx10 模擬 GT 1030 的極度受限環境（理由：本機 CPU 受限且狀態不佳）。

## 1. Frame

目前的結果顯示，內容類型條件化能讓專家集合縮小 19%，但池重播中 Spec-Experts 預載（0.940）與 token 級 LRU（0.939）只差 0.001，
表示「以 4 種手工類型選專家池」在 C = 57 時沒有額外空間；類型也只有 4 種，混合內容落在邊界。同時，26B-A4B 在受限硬體上
只能小量化、小上下文，事實題的正確性是另一個瓶頸。Embedding 是同一個元件可以同時回應這兩件事的候選：

- **A（專家路由）**：用 chunk 的 embedding 取代手工類型，直接檢索「相似 chunk 的專家使用剖面」來決定專家池，預期在較小的池（C = 32、48）才看得到差異。
- **C（路由指紋）**：不加任何模型，用模型自己的前 n 個 token 的專家分佈、或前幾層的專家選擇，當作預測後續專家的 embedding。
- **B（回答精準度）**：用同一個 embedder 做檢索增強（RAG），提升事實題正確性。

三者共用「小型 embedder 在受限環境放得下、跑得動」這個前提，所以先把 embedder 的成本量清楚（切片 E02、E04），再分別驗證 A、C、B。

## 2. Facts（直接查證；來源附於括號）

1. 專案資料：12 個 session、約 8.2 萬 token、約 270 個 chunk（每 chunk 約 300 token）；型別分布 control／user_prompt／reasoning／code／factual／creative
   （本機 `data/routing_corpus_v2/sessions/*.spans.json` 逐檔統計）。內容為英文，Gemma4 自己生成的對話。
2. Tier 2 逐 token 專家序列（`uint8[token, 30, 8]`）在 gx10 `~/gemma4_affinity_sim/repo_run/logs/tier2/`：`base_all`（10 個 session）、`base_new`（2 個 session）；
   本機 `logs/tier2/` 只有 `gt_pilot`、`gt_smoke`。因此 A、C 的離線分析必須在 gx10 上跑（ssh 查證）。
3. 池重播基準（S8，C = 57）：reactive 0.939、static 0.865、spec 0.940；專家大小 4.14 MiB；`scripts/pool_replay.py` 已有 leave-one-session-out 流程（讀檔確認）。
4. gx10：load1 約 2.5；20 核；RAM 121 GB（free 約 45 GB）；venv 只有 `numpy 2.5.4` 與 `gguf`，**沒有** torch／sklearn／sentence-transformers；
   `~/llama.cpp/build/bin/llama-embedding` 存在；可連外（HTTP 200）；**未發現**任何 embedding 模型檔（ssh 查證）。
5. 受限規則（`tools/slice_runner/gx10_run.py`）：核心 0–4、nice 10、load1 ≤ 6、一次一個行程、只寫 `~/gemma4_affinity_sim`、只能跑白名單腳本、
   參數不得含 shell 字元與絕對路徑。受限預設：GPU 1.0 GiB、`-ngl 2 -ub 64`、專家在 CPU。
6. gx10 的存取只經 `tools/slice_runner/gx10_run.py` 的白名單與參數驗證（主機位址由執行環境的 `GX10_HOST` 提供，不寫入任何檔案）。
7. Embedding 模型：四個 Q8_0 GGUF 經使用者核准後下載至 gx10 與本機，sha256 與 Hugging Face 公布值核對相符。

未查證（要由切片補上）：span→文字的對應方式（spans 是 token 索引；文字在 `*.body.txt` 與 `*.turns.jsonl`）、embedder 在 2 核下的實際延遲與記憶體、
gx10 上 `-ngl 2` 生成的實際 tok/s。

## 3. 假設與預先登記的判準（先寫下，結果出來後不改）

| 編號 | 假設 | 判準 | 判讀措辭 |
|---|---|---|---|
| H4 | A：以 embedding 檢索專家剖面選池，優於手工類型 | C = 32 與 48 時，12 個 session 的 leave-one-session-out 平均：`static_emb` 命中率 − `static_type` ≥ +0.02，且逐 session 配對 bootstrap 95% 區間下界 > 0；且 `spec_emb` − `spec_type` ≥ +0.01 | 兩項皆達成＝支持；僅一項＝部分支持；皆未達成＝「尚未觀察到優勢」 |
| H4c | A 的成本可放進受限環境 | embedder 在 2 核（核心 0–1 內）每 chunk ≤ 1.0 s、常駐記憶體 ≤ 300 MiB、GPU 0 | 達成／未達成，附實測值 |
| H5 | C：模型自己的路由指紋可預測後續專家 | C1：n = 32 的指紋在 C = 57 對 chunk 後段 token 的池命中率 ≥ 標籤未知時的 `static_all` + 0.02；C2：以前幾層的選擇預測下一層 top-8，recall 相對該層熱門 top-8 基準提升 ≥ +0.15 | 同 H4 的三段式 |
| H6 | B：檢索增強提升事實題正確性且放得進預算 | 檢索 hit@3 ≥ 0.85 且比 BM25／TF-IDF 基準高 ≥ 0.05；端到端（N = 24）關鍵事實正確率 有檢索 − 無檢索 ≥ +0.20；每題新增 prompt ≤ 1024 token | 同上 |

資料量小（12 個 session）是已知限制：所有比較用逐 session 配對差值與 bootstrap，不用 token 級合併顯著性；報告必須把此限制寫進結果。

### 3.1 E03 預先登記補充（2026-11-19，在 E03 任何結果出現之前寫下）

- **執行者變更**：Codex 的使用額度暫時用盡，E03 由 Claude 自己實作，驗收改為「獨立重算 + 自檢」（見下）。
- **主要設定**（H4 的唯一判定依據）：表示法 = `bge-small-en-v1.5-q8_0`（語料為英文、體積最小）；來源 = 因果脈絡 `ctx.npy`（使用者提示 + 前一個 span）；
  池 = 訓練 span（其他 11 個 session）的專家使用剖面，以 cosine 相似度取 k = 8 個最近鄰、權重 = 相似度（負值截為 0），**每個訓練 span 的計數先除以其 token 數**，逐層取 top-C。
  `static_emb` = 該池固定使用；`spec_emb` = 每個 span 邊界以該池預載、之後在 C 內 LRU。
- 其餘表示法（all-MiniLM、multilingual-e5-small、Qwen3-Embedding-0.6B、tfidf_char）與非因果的 `chunk.npy`、軟類型版本，全部列為**次要結果**，不用於 H4 判定（避免多重比較）。
- 比較基準：`static_type`、`spec_type`、`reactive` 全用 E01 的定義（自檢：同一份程式內重跑 E01 的 `evaluate` 並要求逐位相同）。
- 探索性指標（不用於判定）：loads/token、混合 chunk 子集（內容型 span 且前一個內容型 span 類型不同，或軟類型最大權重 < 0.6）。
- 統計：12 個 session 的逐 session 配對差值，10,000 次 bootstrap（seed 0）。

## 4. Forks（真正的決策點與建議）

1. **要不要在 spec ≈ reactive 的情況下做 A？** 做。C = 57 沒有空間，但 C = 32／48 的 static 掉到 0.8 以下（待 E01 重現實測），
   那裡才是 embedding 能補的範圍；E03 同時報 C = 32、48、57、64，不只看 57。若 C = 32／48 也沒有差異，就如實記錄為「尚未觀察到優勢」。
2. **用哪些 embedder？** 三個層級：(i) 詞彙基準（TF-IDF／字元 n-gram，純 numpy，無需下載）當對照；(ii) 小型：`bge-small-en-v1.5` 或 `all-MiniLM-L6-v2` 的 GGUF（約 25–70 MB），
   是實際會放進 GT 1030 機器的候選；(iii) 參考上限：`Qwen3-Embedding-0.6B` 的 Q8_0 GGUF（約 0.6 GB），只用來看「品質上限」，不當部署候選。
   B 的檔案含中文，(ii) 若是純英文模型要在 E07B 加一個多語模型 `multilingual-e5-small` 做對照。四個模型已於使用者核准後下載（§6）。
3. **A 的 embedding 該看什麼文字？** 兩種都做：(a) chunk 本身的文字（生成後才有，非因果，當上限）；(b) 生成前就有的因果脈絡（使用者提示 + 前一個 chunk）。
   真正可部署的是 (b)；(a) 用來判斷 embedding 本身的資訊量。
4. **A 的剖面怎麼用？** 不訓練分類器，用 kNN：把訓練 chunk 的專家計數向量按相似度加權相加，取每層 top-C 當池（label-free）。
   另外跑「軟類型」版本（embedding 對 4 個類型質心的相似度當權重）當對照。理由：樣本只有約 270 個 chunk，非參數法最不容易過擬合，也最容易檢查。
5. **A 與 B 共用 embedder？** 是。部署上一個 embedder 常駐比兩個便宜；E02 的輸出同時被 E03（A）與 E07B（B）使用。代價：選型要同時滿足路由與檢索，E07B 會單獨回報檢索品質，不影響 H4 判定。
6. **B 的語料與題目。** 語料用本 repo 的 `docs/*.md`（中英混合、無外部資料），題目由執行 agent 寫 40 題並標註證據段落（20 中文、20 英文、含 10 題需跨兩段的）；Claude 抽查 12 題。
   端到端只取其中 24 題（中英各半）。不使用外部資料或其他專案資料。
7. **誰做哪個切片？** 實作者 ≠ 驗收者（驗收者是 Claude，並獨立重算關鍵數字）。需要執行指令、跑 gx10 的切片給 Codex；不需要 shell 的切片（架構規劃、題目撰寫）給 Antigravity。
   題目由 agy 寫、檢索評測由 Codex 跑，作者與評測者不同，避免題目貼合評測程式。

## 5. 切片（Fix Plan）

依賴關係：E01 → E03、E05；E02 → E03、E04、E07B；E07A、E07B → E08；E06、E07A 隨時可做（agy，不碰 gx10）。gx10 一次只跑一個行程，所以整體序列執行。

| 切片 | 內容 | 執行者 | 依賴 | 需下載 |
|---|---|---|---|---|
| E01 | 池重播框架化：`scripts/embed_pool_eval.py`（policy 可插拔；LOSO；C = 32/48/57/64），重現 S8 的 reactive／static／spec | codex | — | 否 |
| E05 | C：路由指紋（C1 前 n 個 token 的專家直方圖 kNN；C2 跨層預測下一層 top-8）離線分析 | codex | E01 | 否 |
| E06 | B：RAG 架構規劃文件（`docs/rag_architecture_2026-11-19.md`），含預算表與介面，不寫程式 | agy | — | 否 |
| E02 | span→文字對應、三層 embedder 的 chunk 向量抽取（`scripts/embed_chunks.py`），輸出 `logs/embed/<model>/<session>.npy` | codex | E01 | 是（閘門） |
| E04 | embedder 受限環境成本：2 核（核心 0–1）、記憶體、延遲、GPU 0 | codex | E02 | 否 |
| E03 | A 主實驗：`static_emb`／`spec_emb`（因果與非因果）／軟類型 vs `static_type`／`spec_type`／reactive，配對 bootstrap | codex | E01、E02 | 否 |
| E07A | B 題目集：讀 `docs/*.md`，寫 40 題與證據段落（`data/rag_eval/questions.jsonl`），不需 shell | agy | — | 否 |
| E07B | B 檢索評測：hit@k／MRR，embedder vs BM25／TF-IDF，中英分報 | codex | E02、E07A | 否（多語模型若要對照需下載） |
| E08 | B 端到端小樣本：24 題 有／無檢索 對 Gemma4 26B-A4B（受限預設）正確率與延遲 | codex | E07B | 否 |
| E09 | E08 補跑：top-2 配置（BM25 與 Qwen3 各取前 2 段），驗證新增 prompt ≤ 1024 token 並與 top-3 比較 | claude（Haiku 4.5） | E08 | 否 |
| E10 | `multilingual-e5-small` 英文題 hit@3 異常（0.10）的原因追查 | claude（Haiku 4.5） | E07B | 否 |

每個切片的共同驗收（Claude 逐項檢查，缺一不算完成）：

1. 每個切片留有報告（做了什麼、實際指令、產出檔、數字表、限制）；本 repo 收錄結果與證據檔（`docs/embedding_plan/evidence/`）。
2. 程式有測試（`tests/embedding/` 或 `tests/affinity_pipeline/`），本機 `.venv` 跑通；全專案測試不退步（目前基準：262 passed、1 skipped；計時測試單獨重跑可過者不算退步）。
3. 在 gx10 上的實跑透過 `gx10_run.py` 白名單；新增腳本時只增加自己的白名單項目與 `SHIP` 項。
4. Claude 在 gx10 上獨立重跑一次摘要指令，數字與報告一致（容差：命中率 ±0.001、延遲 ±15%）。
5. 脫敏：新增或修改的檔案不含主機名、使用者名、絕對個人路徑、其他專案的資料；以環境變數取代。
6. 措辭中性；判準照 §3，不得事後調整門檻。

各切片專屬驗收：

- **E01**：重現 S8 在 C = 57 的 reactive／static／spec（0.939／0.865／0.940，容差 0.001）；輸出多 C 值的表；static 的池只用「其他 session」計數（LOSO）。
- **E02**：`logs/embed/<model>/` 每個 session 一個 `[n_spans, d]` 向量檔 + 一個 `manifest.json`（模型檔 sha256、量化、維度、pooling、截斷長度）；每個 span 的文字來源與 token 範圍對得上（抽樣 20 個 span 由 Claude 比對）。
- **E04**：實測表（模型 × 核心數 1／2 × 每 chunk 延遲 p50／p95 × 峰值 RSS），並寫明是否達 H4c。
- **E03**：報告含 H4 的兩項判準各自的數值、配對差值、bootstrap 區間，以及「混合內容 chunk」（相鄰 span 類型不同或軟類型最大權重 < 0.6）的分項結果。

  資料用 12 個 session：`base_all`（10）與 `base_new`（2）兩個 tag 都要載入（E01 的 `embed_pool_eval.py` 一次只讀一個 tag，需擴充或自行合併，並保持 LOSO）。
- **E06**：含記憶體預算（向量庫、embedder、KV cache、專家池）在 GT 1030 的具體數字、與 Spec-Experts 的介面（檢索內容是否影響類型標籤）、失敗模式。
- **E07A**：`data/rag_eval/questions.jsonl` 40 行（20 中文、20 英文、其中 10 題需跨兩段證據）；每行 `{id, lang, question, evidence:[{file, heading, quote}], key_facts:[…], needs_two_passages}`；
  `quote` 必須逐字出現在該檔該節（Claude 以字串比對全數檢查）；題目不得直接複製證據句；Claude 另外人工抽查 12 題的可答性。
- **E07B**：以 E07A 的題目做檢索（chunk 單位：標題節，過長再切 ≤ 256 token）；指標表（hit@1/3/5、MRR）、與 BM25／TF-IDF 的配對差值、中英分報；測試覆蓋 chunk 切分與指標計算。
- **E08**：24 題 × 2 條件；評分規則在跑之前寫進報告（關鍵事實字串／正規式）；生成參數固定（temperature 0、max 新 token 96）；報告延遲（prefill 與生成）。
- **E09**（2026-11-22 新增，結果出現前預先登記）：`scripts/rag_e2e.py` 新增 `--top-k`（預設 3，只作用於 `bm25` 與 `emb` 條件；`gold` 不變）；
  以 `rage2e --args "--retriever Qwen3-Embedding-0.6B-Q8_0.gguf --top-k 2 --conditions none,bm25,emb --out logs/rag_e2e_top2"` 在 gx10 重跑同一批 24 題；
  報告：每條件的關鍵事實平均分、相對無檢索的差值與 bootstrap 區間、新增 prompt token 的中位數與最大值、前 2 段涵蓋完整證據的題數，並與 `docs/embedding_plan/evidence/rag_e2e/summary.json` 的 top-3 結果並列。
  判準（與 H6 端到端相同）：差值 ≥ +0.20 且新增 prompt 最大值 ≤ 1024；另報 top-2 相對 top-3 的平均分變化。結果檔放在 `docs/embedding_plan/evidence/rag_e2e_top2/`。測試：`tests/embedding/test_rag_e2e.py` 補 `--top-k` 的選段測試。
- **E10**（2026-11-22 新增）：新增 `scripts/e5_diagnose.py`（加入 gx10 白名單 `e5diag`），只使用 gx10 上既有的模型檔，不下載任何東西。至少回答下列問題並各附數字：
  (1) 以相同的 40 題與 498 個 chunk 重現 e5 的 hit@3（中／英）；(2) 變體對照：不加前綴、前綴對調、兩端都用 `query:`、輸入截斷改為 256 估計 token；
  (3) 檢索結果的語言分布：英文題的前 3 段中，中文為主的 chunk 占多少（與 bge、MiniLM、Qwen3 對照；chunk 的語言以其文字中 CJK 字元比例判定，閾值寫入報告）；
  (4) 句向量健全性：數組已知相近與不相近的英文句子對與中英對照句子對，報告餘弦相似度分布；(5) 結論：原因是否查明（是／否／部分）與證據，以及任何變體是否使英文題 hit@3 ≥ 0.40。
  結果檔放在 `docs/embedding_plan/evidence/e5_diagnose/`。不得修改 `scripts/eval_rag_retrieval.py` 的既有指標定義。

## 6. 閘門與約束

- **模型檔**：四個 Q8_0 GGUF（bge-small、all-MiniLM、multilingual-e5-small、Qwen3-Embedding-0.6B）下載至 gx10 的 `~/gemma4_affinity_sim/work/embed_models/`，並與 Hugging Face 公布的 sha256 核對。
- gx10：核心 0–4、nice 10、load1 ≤ 6、一次一個行程；只寫 `~/gemma4_affinity_sim`；**生成或訓練進行時不得 `sync`**（會清掉遠端已生成的內容）；
  E04 的 2 核量測使用核心 0–1（屬於 0–4 的子集合）。
- 本機：Python 只用專案 `.venv`；重運算都在 gx10。
- 公開 repo：新增檔案一律脫敏（見共同驗收 5）。`GX10_HOST` 只存在執行環境，不寫進檔案。

## 7. 驗收紀錄

- **E01 通過**（Claude，2026-11-19）。獨立重算：在 gx10 以舊 `replay` 與新 `embedreplay` 各跑 `--tag base_all --caps 32,57`，兩者逐位相同
  （C=32：reactive 0.790／static 0.687／spec 0.791；C=57：0.939／0.865／0.940；loads/token 50.4／75.2／53.2 與 14.5／32.5／19.8）；
  本機 `tests/affinity_pipeline` + `tests/embedding`：265 passed、1 skipped（執行者回報的 1 個失敗是其沙箱暫存目錄造成，非程式缺陷）；
  已清除其遺留的暫存目錄；`gx10_run.py` 僅新增 `embedreplay` 一項。
  **後續切片的修正**：`base_all` 只有 10 個 session，E03、E05 須同時載入 `base_new`（2 個 session）湊足 12 個；
  並注意 spec 相對 reactive 在 C = 32／48／57／64 都只多 0.001，A 的比較基準要用這兩條線。
- **E06 通過**（Claude，2026-11-19）。抽查：`kv/kv_offload_manager.py` L19/L20（56／512 MiB）、`spec_experts/controller.py` L69–97（`_detect_chunk_type(req.prompt)`）與文件所引完全相符；
  VRAM／KV 算術（單層全局 64 MiB、SWA 12 MiB、1536 cells）與 `docs/gemma4_26b_a4b_vram_benchmark.md` 一致；無主機／使用者／絕對路徑。
  文件把「檢索內容不得影響類型標籤」寫成四點邊界規則，可直接作為整合規格。
- **E07A 通過**（Claude，2026-11-19）。本機以腳本全數檢查：40 題（20 中／20 英）、10 題需兩段證據、`id` 無重複、80 則證據 `quote` 於其檔內逐字（空白正規化後）成立、標題存在、
  題目與證據句無互相包含；另外人工抽讀 6 題（Q01／Q04／Q05／Q10／Q21／Q35 等）皆可由所引段落作答。
  **解讀限制**：題目全部取自本專案私有文件，Gemma4 不可能預先知道，所以「無檢索」基準接近 0 是設計使然；E08 的 +0.20 判準測的是「有證據時能否據以作答」，不是一般問答能力。
  E07B 須在跑評測前凍結語料快照（檔案清單＋sha256）寫入報告，因為 `docs/` 在研究期間會持續變動。
- **E05 退回補件**（Claude，2026-11-19）。獨立重算：在 gx10 重跑 `fingerprint --tags base_all,base_new`，12 個 session 的 C1 差值 +0.094（CI [+0.065, +0.131]）、C2 差值 +0.214（CI [+0.204, +0.225]），與報告逐位一致；
  程式為 LOSO（鄰居與轉移計數都只用其他 session）。缺件：§5 規定「C1 的前 n 個 token 以 reactive LRU 計成本」，報告只量後段 token，沒有含前綴成本的整體比較；
  且 C1 的 0.914 是靜態池，對照 E01 的 reactive 0.939（全 token），需在**同一批 span** 上並列 reactive／spec_type／static_type／指紋池＋LRU 的命中率與 loads/token，才能判讀實用價值。
  H5 兩項預先登記門檻（+0.02、+0.15）依定義已達成，該結論保留，補件後再加上實用性判讀。
- **E05 通過**（Claude，2026-11-19，補件後）。補件結果在 gx10 已產生（執行者在佇列代理回覆前結束，數字由 Claude 自 JSON 補入報告）：113 個合格 span、C = 32／48／57。
  H5 兩項預先登記判準依定義達成（C1 +0.094，C2 +0.214）→「支持」；含前綴成本的實用性判讀：指紋池作為靜態池優於類型靜態池（C=57：0.914 vs 0.865），
  但加入 LRU 後與 `spec_type` 相當（hit rate 差 −0.0006）、loads/token 略多，token 級 LRU 的 loads/token 最低。
  **對 A 的預期**：E01 與 E05 都顯示 LRU 已佔滿可得的命中率，E03 的 `spec_emb − spec_type ≥ +0.01` 幾乎不可能達成；E03 除了預先登記的兩項判準外，
  另加探索性指標 loads/token（頻寬成本）與「不可補載（固定池）」情境，判準本身不改。
- **E02／E07B 的第一批嵌入結果作廢並重做**（Claude，2026-11-19 晚）。驗收時發現 E07B 的 bge-small、MiniLM 檢索 hit@3 只有 0.275（BM25 為 0.65），遠低於該類模型的常態，追查後確認兩個實作問題，
  都出在執行者呼叫 `llama-embedding` 的方式：(1) 該程式以換行切分 prompt 並對每一行回傳一個向量，而解析器只取第一個，所以**每段文字只有第一行被嵌入**（本機於 gx10 以 3 行文字實測回傳 3 個向量）；
  (2) 池化被強制為 mean，但 bge-small 的 GGUF 宣告 CLS（MiniLM、e5 為 mean，Qwen3 為 last）；另外超過模型視窗（512 token）的單行文字會使程式 abort。
  處置：新增 `affinity_pipeline/embedding_io.py`（私有分隔符、向量數檢查、token 預算截斷與失敗重試、採用 GGUF 宣告的池化），E02、E07B 與 E04 的腳本改用它並補 bge 的查詢指令前綴（模型文件建議用法），
  相關測試 4 項；舊的向量與 E07B 的 embedding 指標不得引用。A（E03）的 kNN 因此要用重新抽取的向量。
  教訓寫入規則：嵌入類切片的驗收必須包含「一段多行文字得到恰好一個向量」與「檢索基準合理性」兩項檢查。
- **E02 通過（修正後，附限制）**（Claude，2026-11-19）。以 `affinity_pipeline/embedding_io.py` 在 gx10 重新抽取：4 個 GGUF 模型 + `tfidf_char`、12 個 session、264 個 span，每個 session 一份 `chunk.npy`／`ctx.npy`
  與 manifest（模型 sha256 與下載時核對相符、池化採 GGUF 宣告值：bge = CLS、MiniLM／e5 = mean、Qwen3 = last）；形狀與 `spans.json` 的 span 數一致、範數 1.000、無 NaN。
  限制：超過約 480 個估計 token 的 span 文字被截斷（BERT 類視窗 512）；span→文字的 20 筆稽核已完成；第一批（只嵌入第一行）的向量已作廢。
- **E03 完成**（Claude 自行實作，2026-11-19；`scripts/a_embedding_pools.py`、gx10 `aembed`、結果 `logs/a_embedding_pools.json`）。自檢：同程式重算的 type／reactive 基準與 E01 `evaluate` **逐位相同**（四個 C 的最大差 0.0）；12 個 session、LOSO、10,000 次 bootstrap。
  **H4 判定（預先登記的主要設定：bge-small、因果脈絡 ctx、kNN k=8）：尚未觀察到優勢。**

  | C | `static_emb − static_type`（均值［95% 區間］） | `spec_emb − spec_type` |
  |---:|---|---|
  | 32 | +0.012［−0.009, +0.037］ | +0.0000 |
  | 48 | +0.025［+0.005, +0.049］ | −0.0000 |
  | 57 | +0.025［+0.008, +0.046］ | +0.0001 |
  | 64 | +0.021［+0.006, +0.038］ | +0.0003 |

  判準要求 C = 32 與 48 的固定池差值皆 ≥ +0.02 且區間下界 > 0，並且 `spec` 差值 ≥ +0.01：C = 32 未達、`spec` 未達。
  次要結果（不用於判定）：固定池情境下，C ≥ 48 時主要設定的差值 +0.02～+0.025（區間不含 0）；非因果的 chunk 文字向量為 +0.04～+0.06（四個模型相近，是資訊量上限）；
  `multilingual-e5-small` 的因果脈絡在四個 C 都為 +0.029～+0.036（區間不含 0）；**`tfidf_char` 因果脈絡為 +0.025～+0.030（區間不含 0），與 bge 相當**，表示因果脈絡情境下的增益並非 embedding 才有；
  軟類型版本比硬類型低 0.03～0.06；一旦池內允許 LRU 補載（`spec`），所有表示法的命中率都與 reactive 相差在 ±0.001 之內，loads/token 比 reactive 高 2.7～6.4（預載成本）。
  混合 chunk 子集：預先登記的定義（軟類型最大權重 < 0.6）在實測中退化（cosine 權重不夠尖，所有 span 都被判為混合），因此不採用；已追加「前一個內容 span 類型不同」的版本，結果見重跑的 `logs/a_embedding_pools_v2.json`。
- **E07B 結果**（Claude 自 gx10 讀取，2026-11-19；修正後）。498 個 chunk（11 個檔案；`CC_For_Codex_Suggestions.MD` 因副檔名大寫、Linux 的 `rglob("*.md")` 區分大小寫而未納入，四個模型使用同一份 chunk 集合），40 題：

  | 方法 | hit@1 | hit@3 | hit@5 | MRR | 中／英 hit@3 |
  |---|---:|---:|---:|---:|---|
  | BM25 | 0.475 | 0.650 | 0.750 | 0.592 | 0.55／0.75 |
  | tfidf_char | 0.250 | 0.425 | 0.475 | 0.357 | 0.20／0.65 |
  | bge-small（CLS） | 0.300 | 0.400 | 0.425 | 0.373 | 0.30／0.50 |
  | all-MiniLM（mean） | 0.325 | 0.425 | 0.500 | 0.400 | 0.20／0.65 |
  | multilingual-e5-small（mean） | 0.150 | 0.250 | 0.300 | 0.234 | 0.40／0.10 |
  | Qwen3-Embedding-0.6B（last） | 0.325 | 0.525 | 0.600 | 0.454 | 0.50／0.55 |

  H6 檢索判準（hit@3 ≥ 0.85 且比詞彙基準高 ≥ 0.05）：**四個模型都未達成**（最佳為 Qwen3 的 0.525，相對 BM25 −0.125［−0.275, +0.025］）。
  探索性（不用於判定）：BM25 + 稠密 RRF（k = 10）以 Qwen3 為最高，hit@3 = 0.70（+0.05 於 BM25）；其餘模型的融合不優於 BM25。以「含 ≥ 50% 關鍵事實的 chunk 即算命中」的答案導向指標重算，排序相同（BM25 0.60、Qwen3 0.525），
  顯示結論不是金標準只認單一檔案／標題造成的。`multilingual-e5-small` 英文題 hit@3 = 0.10 屬異常，未查明原因（GGUF 轉檔或前綴用法皆可能），不納入建議。
  **已知的資料問題**：本機 `data/rag_eval/chunks.jsonl`（545 個）與 gx10 實際嵌入的集合（498 個）不同；E08 與後續分析一律使用 gx10 的 `logs/rag_eval/chunks.jsonl`。
- **E07B 補充：可回答題數**（Claude，2026-11-19）。40 題中有 3 題（Q12、Q13、Q19，皆中文）的證據在 `docs/embedding_research_plan_2026-11-19.md`，而 E07B 依規劃把該檔排除在語料之外，
  所以這 3 題在所有方法下都不可能命中（上限 37／40 = 0.925，且只壓低中文）。以 37 個可回答題重算：BM25 hit@3 = 0.703、Qwen3 = 0.568、MiniLM = 0.459、bge = 0.432、e5 = 0.270，排序與結論不變（仍未達 0.85）。
  E08 只從可回答題中選題。
- **E04 結果**（Claude 自行實作並在 gx10 量測，2026-11-19；`scripts/embed_cost.py`，每模型 20 個真實 chunk、核心 0–1、GPU 0 的設定下 llama-embedding 一次一個行程）：

  | 模型 | 兩核冷啟動 p50／p95（秒） | 其中載入模型 | 暖機估計 p50（秒） | 峰值記憶體（MiB，行程自身 VmHWM） | H4c（≤ 1 s 且 ≤ 300 MiB） |
  |---|---|---:|---:|---:|---|
  | bge-small（37 MB） | 0.69／0.71 | 0.59 | 0.10 | 392 | 延遲達成、記憶體未達 |
  | all-MiniLM（25 MB） | 0.67／0.87 | 0.58 | 0.09 | 551 | 延遲達成、記憶體未達 |
  | multilingual-e5-small（132 MB） | 1.62／1.67 | 1.36 | 0.26 | 881 | 兩項皆未達 |
  | Qwen3-Embedding-0.6B（639 MB） | 3.36／3.54 | 3.14 | 0.21 | 5168 | 兩項皆未達 |

  單核與兩核差異很小（bge：0.77 對 0.69 s），表示成本以模型載入為主；常駐子行程（只付一次載入）時每 chunk 約 0.1–0.26 s。
  記憶體偏高的原因之一：gx10 上的 `llama-embedding` 連結了 CUDA 函式庫（`libggml-cuda`、`libcudart`、`libcublas`），即使 `-ngl 0` 也會初始化 CUDA 執行階段；縮小 `-c/-b/-ub` 為 512 後峰值不變（bge 392、MiniLM 543–553），所以不是緩衝大小。
  **解讀限制**：GT 1030 機器（Vulkan／CPU 版 llama.cpp）的實際常駐記憶體需在該機器重量；此處的記憶體值屬於 gx10 的 CUDA 版，只能當上界參考。H4c 在 gx10 模擬環境下的判定：延遲面 bge／MiniLM 達成，記憶體面未達成。
- **E08 結果**（Claude 自行實作並在 gx10 執行，2026-11-19；`scripts/rag_e2e.py`、`logs/rag_e2e/`）。Gemma4 26B-A4B Q4_K_M、受限預設（`-ngl 2 -ub 64`、專家在 CPU）、temperature 0、最多 128 個新 token；
  24 題（可回答題中，中英各 12，每種語言 4 題需兩段證據）；四種條件：無檢索、BM25 top-3、Qwen3-Embedding top-3、標準答案段落（上限 3 段）。評分為預先固定的自動規則（關鍵事實字串或其數字出現在答案中）。

  | 條件 | 關鍵事實平均分 | 中文 | 英文 | 全部事實皆答出的題數比 | 相對無檢索的差值［95% 區間］ | 新增 prompt token 中位數／最大 | 預填中位數 |
  |---|---:|---:|---:|---:|---|---|---:|
  | 無檢索 | 0.076 | 0.097 | 0.056 | 0.00 | — | — | 0.67 s |
  | BM25 top-3 | 0.611 | 0.833 | 0.389 | 0.46 | +0.535［+0.375, +0.694］ | 803／1029 | 3.7 s |
  | Qwen3 top-3 | 0.528 | 0.750 | 0.306 | 0.42 | +0.451［+0.285, +0.625］ | 854／1074 | 3.7 s |
  | 標準答案段落 | 0.632 | 0.917 | 0.347 | 0.46 | +0.556［+0.403, +0.708］ | 325／722 | 2.4 s |

  **H6 端到端判準**：差值 ≥ +0.20 在三個檢索條件都達成（區間下界 ≥ +0.285）；「每題新增 prompt ≤ 1024 token」在 top-3 配置下**最大值略超過**（BM25 的 Q23 為 1029、Qwen3 的 Q20 為 1074，中位數 803／854），標準答案段落條件達成。
  超出的兩題可由架構文件已規定的做法（超過上限時降為 top-2 或截斷）處理，本次未重跑。無檢索的答案多為編造的數字（例如 Q01 的「第 128 個 token、熵降 0.15」），檢索後逐項答對。
  **稽核與限制**：(1) 自動評分是保守的下界，對英文尤其低估：英文關鍵事實多為文字敘述，答案若改寫就不被計入；Claude 逐題閱讀 12 題英文標準答案條件的答案，人工判讀約 0.80 的事實被答出，自動分數為 0.35，
  中文的關鍵事實多為數字，兩者較一致（0.92）。因此英文的絕對值不可直接引用，三個檢索條件相對無檢索的方向與幅度不受影響（改寫只會抬高有檢索的條件）。(2) 標準答案段落條件只比 BM25 高 0.02，
  代表在這組題目上，**瓶頸不在檢索排名，而在模型如何從段落抽取事實**；BM25 的 top-3 涵蓋完整證據的題數為 17／24，Qwen3 為 13／24。(3) 128 個 token 的上限使 1–2 題的答案被截斷。
  (4) 速度是 gx10 模擬值（預填約 240 token／s、生成約 9–12 token／s），不是 GT 1030 的實測；架構文件對 GT 1030 的預填估計（25–278 秒）仍須在該機器驗證。

## 結果總表（2026-11-19，gx10 模擬受限環境）

| 假設 | 判定 | 主要數字 |
|---|---|---|
| H4（A：embedding 選專家池） | **尚未觀察到優勢**（預先登記判準未達成） | 固定池 C=48/57/64 +0.02～+0.025（區間不含 0），C=32 +0.012（含 0）；預載 + LRU 差值 ≤ +0.0003；TF-IDF 因果脈絡增益與 bge 相當 |
| H4c（A 的成本） | 延遲面達成、記憶體面未達成（gx10 CUDA 版量測，需在目標機重量） | bge 兩核 p95 0.71 s、392 MiB；MiniLM 0.87 s、551 MiB |
| H5（C：路由指紋） | **支持**（依預先登記判準） | C1 +0.094、C2 +0.214；含前綴成本後與 `spec_type` 相當、固定池優於類型池 |
| H6（B：RAG） | **部分支持** | 檢索 hit@3 未達（最佳稠密 0.525、BM25 0.650，門檻 0.85）；端到端差值 +0.45～+0.56 達成；top-3 新增 token 最大 1029～1074（略超 1024），**前 2 段配置最大 703、差值 +0.40～+0.43 達成**（平均分比 top-3 低 0.06～0.10，集中在需兩段證據的題）；e5 英文題異常為向量空間對比度過低，原因部分查明 |
- **E09 通過**（Claude，2026-11-22；執行：Claude Code Haiku 4.5）。獨立重算：以本機腳本由 `answers.json` 依預先固定的評分規則重算，BM25 前 2 段 0.510、Qwen3 前 2 段 0.472、無檢索 0.076 與 `summary.json` 一致；
  無檢索的 24 個答案與 E08 逐字相同（temperature 0 的重現性）；新增 prompt 最大 703 token（BM25、Qwen3 皆是），差值 +0.434（+0.271, +0.601）與 +0.396（+0.222, +0.576），H6 端到端兩項判準達成；
  與 top-3 比：平均分 −0.101（BM25，4 題變差）、−0.056（Qwen3，2 題變差），沒有任何題變好；前 2 段涵蓋完整證據 BM25 16／24（單段 14／16、兩段 2／8）、Qwen3 12／24（12／16、0／8）。
  驗收中修正：Haiku 的報告把單段／兩段題數寫成 20／4（實為 16／8）、涵蓋題數的分項也不正確，已由 Claude 重寫報告；它寫的測試以 try／except 吞掉失敗，已換成以暫存語料驗證 `top_k` 的測試；移除其多放的輔助腳本。
- **E10 完成（原因部分查明）**（Claude，2026-11-22）。Haiku 4.5 只寫出腳本、未執行 gx10 工作，並把「請使用者自行執行」當成結論；Claude 補完並執行（見 `E10.md`）。
  重現：base 英文題 hit@3 = 0.10（全部 0.25、中文 0.40），與 E07B 相同。已排除：前綴（base、前綴對調、兩端皆 `query:` 的各項指標完全相同；不加前綴英文題仍 0.10）、截斷（256 估計 token 英文題仍 0.10）、
  跨語言偏差（英文題前 3 段 100% 是英文 chunk，語料為英文 76.9%、混合 20.3%、中文 2.8%）、指標（答案導向 hit@3 英文 0.00–0.05）。
  觀察：e5 的 chunk 間餘弦均值 0.882（標準差 0.026），bge 為 0.649（0.073）、MiniLM 0.232（0.143）、Qwen3 0.412（0.105）；題目對 chunk 的相似度前 1 名與中位數只差 0.061（其他 0.197–0.431）；「embedding」與「vector」的餘弦為 1.000。
  結論：此檢索集上 e5 的相似度分布過於集中而不足以分辨證據；是 `multilingual-e5-small` 本身的性質，或此 GGUF 轉檔／量化與 llama.cpp 推論所致，需要參考實作的向量才能判明，本次未做，e5 不納入建議。

## 8. Final Proof（整體完成的檢查，非個別切片）

- H4／H4c／H5／H6 各有一行「支持／部分支持／尚未觀察到優勢」與其數值，數值可由報告中的指令在 gx10 重現。
- 全專案測試不退步；`dev` 的 git 歷史每個切片一個 commit，且 `git grep` 找不到主機名、使用者名、個人絕對路徑。
- 不在本計畫內驗證：embedder 在真實 GT 1030 機器上的延遲（只在 gx10 模擬）、跨模型（非 Gemma4）泛化、12 個 session 以外的資料；這些寫入各報告的限制欄。
