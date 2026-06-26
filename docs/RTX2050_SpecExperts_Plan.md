# RTX 2050 + Spec-Experts LLM 部署規劃

> **分支：** `RTX2050-SpecExperts`  
> **基礎：** `BestSetupLLMs`（含 Phase 1-7 完整實作）  
> **目標：** 在 RTX 2050 4 GB VRAM + CUDA 環境驗證 Spec-Experts Phase 1-5

---

## 硬體規格

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA RTX 2050 4 GB GDDR6 (Turing TU117, CUDA sm_75) |
| 記憶體頻寬 | 112 GB/s |
| VRAM | 4 GB GDDR6 |
| CPU | i-series (搭配 RTX 2050 筆電/桌機) |
| 後端 | CUDA (llama.cpp CUDA build) |
| 有效 VRAM 預算 | ~3.2–3.5 GB（OS + DWM 保留 ~0.5–0.8 GB） |

---

## Spec-Experts Phase 支援計劃

| Phase | GT 1030（已驗證） | RTX 2050（本分支目標） | 差異 |
|-------|----------------|---------------------|------|
| P1 外部監控器 | ✅ Vulkan | ✅ CUDA | n_probs 端點相同，無需修改 |
| P2 KV 卸載 | ✅ 驗證 | ✅ 同架構 | GDDR6 112 GB/s，RAM 層更快 |
| P3 方向性引導 | ✅ 驗證 | ✅ 同架構 | Flash Attention 可啟用（純 GPU） |
| P4 專家 RAM 池 | ⚠️ 850 MB 限制 | ✅ 3.2 GB VRAM | Mixtral Q2_K 可部分 GPU 卸載 |
| P5 邊緣 VLM | ✅ moondream-2B Vulkan | ✅ moondream-2B CUDA | ngl 可從 8 提升至 32 |
| P6 Mac M4 Metal | ➖ 不適用 | ➖ 不適用 | 需 Apple Silicon |
| P7 AGI 自迴圈 | 結構驗證 | 品質評分+匯出 | LoRA 訓練需更多 VRAM |

---

## RTX 2050 特有優化

### Flash Attention 啟用

RTX 2050 純 CUDA（無 CPU/GPU 混合），Flash Attention 可正常啟用：

```bash
llama-server -m model.gguf \
  -ngl 999 \          # 全層上 CUDA GPU
  -fa \               # Flash Attention 啟用（GT 1030 無法）
  -c 4096 \           # 更大 context（GT 1030 只能 512）
  --host 127.0.0.1 --port 8080
```

### Mixtral-8x7B Q2_K 部分卸載（Phase 4）

GTX 1030 無法執行 Mixtral（15 GB 模型），RTX 2050 可部分卸載：

```bash
# 3.2 GB VRAM 預算：attention 層上 GPU，expert FFN 留 CPU
llama-server -m Mixtral-8x7B-v0.1-Q2_K.gguf \
  -ngl 8 \            # 部分層上 GPU（依 VRAM 調整）
  --cpu-moe \         # Expert FFN 留在 CPU RAM
  -c 2048 \
  --host 127.0.0.1 --port 8081
```

理論分析（RTX 2050，112 GB/s GDDR6）：

```
VRAM 3.2 GB 分配：
  Attention 層（8 層 GPU）: ~1.2 GB
  KV Cache（2048 ctx）:     ~0.8 GB
  可用 GPU 推論空間:         ~1.2 GB
Expert FFN（CPU RAM，17 GB/s DDR3 / DDR4）: 剩餘權重
預估 tok/s: ~4–8（混合 CPU/GPU）
```

### Spec-Experts 路由節省（RTX 2050 + Mixtral）

```
DDR4 3200（~51 GB/s） 下 Mixtral Q2_K 專家切換代價：
t_switch = 1.875 GB / 51 GB/s ≈ 36 ms

20 區塊會話：40 次切換 → 24 次（節省 16 × 36 ms = 0.58 s）
GDDR6 GPU 層切換（部分在 GPU）：更快，但 MoE expert 主要在 CPU
```

---

## 待辦測試清單

### 環境驗證
- [ ] 確認 CUDA llama-server build（sm_75 Turing）可正常載入
- [ ] 量測 RTX 2050 有效 VRAM（DWM + CUDA context overhead）
- [ ] `llama-server -ngl 999 -m qwen3-1.7b.gguf` 基準速度測試

### Phase 1 驗證（CUDA 環境）
- [ ] `tests/phase1_monitor_demo.py` 連接 CUDA llama-server
- [ ] 確認 `n_probs` 端點在 CUDA build 正常返回 top-k logprobs
- [ ] 測量 tok/s（預期 > GT 1030 的 5.1 tok/s）

### Phase 3 驗證（Flash Attention）
- [ ] `-fa` 旗標確認啟用（log 應顯示 `flash_attn = 1`）
- [ ] 比較 `-fa` 開/關的 context 4096 推論速度差異

### Phase 4 驗證（Mixtral 部分卸載）
- [ ] 量測 `--cpu-moe` + `-ngl 8` 下 Mixtral Q2_K tok/s
- [ ] `tests/phase2_7_demo.py` 在 RTX 2050 環境重新跑（驗證 P4 命中率）

### Phase 5 驗證（moondream CUDA）
- [ ] moondream-2B Q4 以 `-ngl 32 -fa` 啟動（CUDA，預期 3-5 FPS）
- [ ] VLM 路由測試：POST `/completion` + base64 圖像

---

## 與 ReadytoTestonRTX2050 舊分支的關係

`ReadytoTestonRTX2050`（現有舊分支）停在 `acf5c41`，缺少：
- Phase 1 外部監控器程式碼
- Phase 2-7 所有模組
- `.kvbin` 格式、KV 卸載管理器
- 方向性引導、專家 RAM 池
- 中英文博士論文

本分支（`RTX2050-SpecExperts`）從最新 `BestSetupLLMs` 分出，包含所有上述內容。
舊 `ReadytoTestonRTX2050` 分支保留作歷史記錄，可在硬體到位後廢棄。

---

*建立日期: 2026-06-26*  
*基礎 commit: 6137309*
