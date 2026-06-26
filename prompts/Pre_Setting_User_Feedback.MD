# Pre-Setting User Feedback
# 使用者對話紀錄與偏好設定
# Collected: 2026-04-06 | Reviewed by: Claude Opus 4.6

---

## 使用者身份
- 操作環境：Windows 10 Pro, i5-4460, GT 1030 2GB, 34GB RAM
- 目標：本地部署 Gemma4:E4B + 多模型比較測試 + 推送至 GitHub
- 語言偏好：中文為主，技術文件中英雙語
- Git repo：https://github.com/sipurchen/klchen.git (branch: BestSetupLLMs)

---

## 對話時間線與使用者指令

### Session 1（初始設定）

**U1:** `@\path\to\project\prompts\MASTER_PROMPT.md 請照著這個檔案進行`
> 指示 Claude 按照 MASTER_PROMPT.md 開始整個專案部署流程。

**U2:** `直接進行下一步操作`
> 不需要等待確認，直接往下做。

**U3:** `??` (多次)
> 催促繼續。使用者偏好：不要等待確認，直接執行。

**U4:** `B`
> 在 Gemma4:E4b 太慢的情況下，選擇先用 Gemma3:1b 測試。

**U5:** `現在調整回Gemma4_E4B，然後使用TurboQuant+ 、KV Cache與Streaming Experts for MoE LLMs等技術來進行調整，使時間能縮短到10分鐘以內`
> 核心需求：Gemma4:E4B 推論時間需在 10 分鐘內完成。
> 指定技術：TurboQuant+, KV Cache, Streaming Experts for MoE。

**U6:** `請以最佳配置重啟Ollama並測試以下部份：`
> 完整測試需求（6 項）：
> 1. 文字摘要
> 2. 同義詞增強
> 3. 影像分析
> 4. 音訊/語音辨識
> 5. TTS
> 6. 影片

> 額外要求：
> - 三個模型都要測（Gemma4:E4b, Gemma3:1b, DeepSeek-r1:1.5b）
> - 產生 Markdown 比較報告
> - Push 到 GitHub (BestSetupLLMs branch)
> - 包含環境設定（Windows + Ubuntu reference）
> - 不含個人資訊

### Session 1 續（測試執行）

**U7:** `完整測試結果？1. 不要同步進行以免造成資源過度佔用 2. 調整CPU佔用率以免當機`
> **關鍵反饋：**
> - ❌ 禁止平行測試（parallel execution 會導致資源過載）
> - ❌ CPU 必須被限制（否則系統會當機）
> - 必須順序（sequential）執行所有測試

### Session 2（優化嘗試）

**U8:** `1. 考慮跳過ollama，透過直接量化原LLMs來降低需求，參考Flash-MoE的邏輯概念，將LLMs量化到4 bits`
`2. 降低CPU佔用率及推論方式更動`
`3. 備用方案參考MLX的架構來運行LLMs，看能不能不要用到Swap`
> 三層策略：
> 1. 主方案：跳過 Ollama，直接量化到 4-bit，用 Flash-MoE 概念
> 2. 降低 CPU 佔用 + 改變推論方式
> 3. 備用方案：參考 MLX 架構避免使用 Swap

**U9:** `1. 每次只要呼叫Ollama就直接CPU過載 2. 過載結束Claude畫面變成空白，什麼結果都不知道`
> **嚴重問題回報：**
> - Ollama 一呼叫就 CPU 100% 過載
> - 過載後 Claude Code 畫面直接空白（context 丟失）
> - 結果完全看不到

### Session 3（審閱 + 規劃）

**U10:** `進行測試報告審閱比較是否符合原計畫`
> 要求比對 comparison_report.md vs MASTER_PROMPT.md，找出差異。

**U11:** `1. 將未符合原計畫的部份設定task給sonnet 4.6執行`
`2. 設定好task後將本對話上下文整理縮排，將我說過的話存於Pre_Setting_User_Feedback.MD中`
`3. 以能讓Gemma4:E4B在當前這台電腦可以執行為重點，除調整CPU佔用，利用1G VRAM GPU運作等基本需求外，透過Flash-MoE / MLX / KV cache / CPU affinity 等技術調整其運作狀態使期待值能達到10 tokens/s`
> 三項任務：
> 1. 產出 TASKS.md 給 Sonnet 4.6 執行
> 2. 將所有使用者訊息存為本檔案
> 3. **新目標：Gemma4:E4B 達到 10 tok/s**（從 0.04 tok/s 提升 250×）

---

## 使用者偏好摘要

| 類別 | 偏好 |
|------|------|
| 執行方式 | 不要等待確認，直接做。`??` 表示催促 |
| 測試方式 | **必須順序執行，禁止平行**。CPU 必須被節流 |
| CPU 管理 | Ollama 每次呼叫都會過載 → 必須設定 affinity + 低優先權 |
| 當機問題 | Claude Code 在 CPU 過載後會空白，需要 incremental save |
| 語言 | 中文溝通，技術文件中英雙語 |
| Git 規範 | 不含個人資訊，push 到指定 branch |
| 技術路線 | 偏好 Flash-MoE / MLX 概念，想跳過 Ollama（但 llama-cpp-python 編譯失敗） |
| 速度目標 | 初始：10 分鐘完成推論。最新：**10 tok/s** |
| 備用方案 | MLX 風格（mmap + mlock，no swap） |

---

## 已知問題與解決狀態

| 問題 | 狀態 | 解法 |
|------|------|------|
| Ollama CPU 100% 過載 | ✅ 已修 | psutil affinity [2,3] + num_thread=2 |
| Claude Code 空白 | ✅ 已修 | Incremental save (每測試後存檔) |
| OLLAMA_MODELS 路徑錯 | ✅ 已修 | 改為 \path\to\LLMs |
| OLLAMA_LOAD_TIMEOUT 5m 殺掉 cold start | ✅ 已修 | 改為 20m |
| llama-cpp-python 編譯失敗 (MSVC jinja) | ❌ 未解 | 改用預編譯 llama-server 二進制 |
| Gemma4 0.04 tok/s | ❌ 未解 | Task 5: llama-server -ot expert offloading |
| 空回覆 (4 個 tests) | ❌ 未解 | Task 1: 替換圖片 + DeepSeek think 解析 |
| Phase 1-3 測試未跑 | ❌ 未解 | Task 4: infrastructure + API tests |

---

## 技術決策紀錄

1. **Ollama vs llama-cpp-python vs llama-server**
   - Ollama: 可用但不支援 tensor-level expert offloading → 0.04 tok/s
   - llama-cpp-python: 編譯失敗（MSVC + jinja/utils.h C++20 問題）
   - **llama-server (預編譯二進制)**: 最可行路線，支援 `-ot` flag → 預期 10-15 tok/s

2. **MoE Expert Offloading 是關鍵**
   - Gemma4:E4B 有 64 experts, top_k=2
   - 每 token 只讀 2.67B params (vs 8B total) = 1.50 GB at Q4_K_M
   - GT 1030 48 GB/s 可以在 31ms 讀完 → 32 tok/s 理論值
   - Ollama 的 layer-level 分割無法利用這個特性

3. **CPU affinity 分離策略**
   - Core 0: OS 保留
   - Cores 1-3: 推論引擎（llama-server 或 Ollama）
   - 不同時跑多個推論引擎（sequential benchmark）
