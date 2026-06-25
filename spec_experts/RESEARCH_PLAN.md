# Spec-Experts LLM — 本機測試推演計劃
> Branch: `SpecExpertsResearch`  
> Date: 2026-06-25  
> Base: Spec-Experts-LLM-ImplementPlan.md

---

## 環境標準（已修正）

**GT 1030 標準工作環境：**
- Chrome 關閉
- Edge 關閉
- LINE **保持開啟**（甲方溝通）
- DWM / Claude / Codex / Telegram 常駐

**實測 VRAM 數字：**
```
Total VRAM:              2048 MB
All apps running (live): ~489 MB free
STANDARD baseline:       ~850 MB free  (Chrome+Edge closed)
Conservative max:        ~1100 MB free (全關非必要)
```

**ngl 標準值（850MB 基準）：**

| 模型 | ngl | GPU% | VRAM使用 | CPU層 |
|------|-----|------|---------|------|
| Mixtral-8x7B Q2_K | 3 | 9% | 767MB | 29層 |
| DeepSeek-R1-7B Q4 | 3 | 10% | 817MB | 25層 |
| **Gemma4 E4B Q4 (MoE)** | **11** | **23%** | **834MB** | **35層** |
| Phi-4-mini Q4 (3.8B) | 10 | 31% | 806MB | 22層 |
| DeepSeek-R1-1.5B Q8 | 25 | 89% | 844MB | 3層 |
| Qwen2.5-14B Q2_K | 2 | 5% | 846MB | 38層 |

---

## 核心假設（升級難度版）

**挑戰：** 850MB VRAM 只能放 11-25 層，主體在 CPU+RAM 跑。
GPU 只作為「加速前段層」，後段全 CPU。

**Spec-Experts 角色因此更重要：**
1. 前 N 層放 GPU（速度提升）
2. 中間 attention 層 CPU 跑
3. Expert FFN 只載入被激活的 top-2 → RAM（不全載入）
4. KV cache → disk offload（突破 context 限制）
5. Monitor 減少無效 token 計算（chunk-level routing）

---

## 初期目標：GT 1030 850MB 環境，26B+ MoE/Dense 推論

**可行路徑：**
```
26B MoE (e.g. Mixtral-8x7B):
  Q2_K 總大小: ~15GB
  Active expert per token: 2/8 = 25% FFN
  Active 7B weight: ~3.5GB → 全在 RAM
  GPU: embedding + 前3層 (767MB)
  CPU: 剩餘 29層
  RAM pool: 常駐 top-4 experts (~7GB), 其餘 disk

  Spec-Experts 策略:
  - Monitor 標記 chunk 邊界 → 批次 expert routing
  - 同一 chunk 內 expert 不切換 → 減少 RAM→VRAM 搬移
  - KV prefix: chunk 內 reuse，跨 chunk disk streaming

  目標速度: 1-3 tok/s (DDR3 限制)
  目標 context: 16K+ (disk-backed KV)
```

**26B Dense (e.g. Qwen2.5-14B Q2_K 實測接近 26B 等效):**
```
  Q2_K: ~9GB RAM
  GPU: 前2層
  CPU: 38層
  速度: ~2 tok/s
  context: 4K (RAM) → 16K (disk KV)
```

---

## 中期目標：Mac Mini M4 (200B+ MoE)

**硬體：** 32GB unified memory, Metal backend, M4 Neural Engine

**ngl 等效：** unified memory → 全部 GPU-accessible，ngl=99 可行

**策略：**
```
DeepSeek-V3 671B MoE:
  Active 37B / token (top-2 / 256 experts)
  Q4 active: ~18.5GB → fit in 32GB
  Expert pool: top-32 in RAM, rest on SSD (M4 SSD ~7GB/s)
  Speed target: 15-30 tok/s
```

**多模態 + Metaverse Game AI：**
```
[Visual VLM Expert]   → 場景理解 chunk
[Reasoning Expert]    → 行為決策 chunk
[Creative Expert]     → 語言輸出 chunk
        ↓
[KV Streaming: 跨 NPC 共享 context]
```

---

## 最終目標：自循環 AGI 雛型

```
[感知] → Monitor → [Chunk 分割]
                         ↓
              [Specialist Expert 路由]
                         ↓
              [行動/輸出生成]
                         ↓
         [Streaming Distill → 自我強化]
                         ↑_______________|
                         (feedback loop)
```

**關鍵技術棧：**
1. Attention entropy → 語意狀態感知
2. KV prefix reuse → 長期記憶 (disk-backed)
3. LoRA adapter pool → 動態技能載入
4. Directional Steering → 目標導向行為控制

---

## 第二初期目標：邊緣視覺 VLM

| 裝置 | 模型 | VRAM | FPS |
|------|------|------|-----|
| Jetson Nano 4GB | MobileVLM-1.7B Q4 | 2GB | 8-15 |
| 舊手機 Android | LLaVA-Phi-1.5 Q4 | 1.5GB | 3-8 |
| 車載 (Qualcomm) | TinyLLaVA-3B Q4 | 2GB | 5-12 |
| 監控鏡頭 (ARM) | moondream-2B Q4 | 1GB | 2-5 |
| **GT 1030 (850MB)** | **moondream-2B Q4** | **~800MB** | **1-3** |

---

## 本機測試推演順序（修正版）

### Week 1 — 環境驗證（850MB 基準）
- [ ] 關閉 Chrome+Edge，確認 VRAM free ~850MB
- [ ] 跑 `gt1030_layer_calc.py` 確認數字
- [ ] 測試 Phi-4-mini Q4 (3.8B, ngl=10) → 最快起步
- [ ] 測試 Gemma4 E4B (ngl=11) → 主力模型
- [ ] 掛 monitor/attention_entropy.py 到 logprob stream

### Week 2 — KV Offload
- [ ] 實作 kv/kv_offload_manager.py (LRU eviction)
- [ ] prompt replay 模擬 KV prefix (繞道 API 限制)
- [ ] 量測 context 擴大效果: 4K → 16K

### Week 3 — Specialist Routing
- [ ] segmentor/chunk_classifier.py 整合 monitor 輸出
- [ ] per-chunk system prompt 切換
- [ ] Directional Steering 移植 (pi-ds4)

### Week 4 — 26B+ 測試
- [ ] Mixtral-8x7B Q2_K 下載測試 (ngl=3)
- [ ] Expert RAM pool 管理：top-4 常駐
- [ ] Monitor → chunk boundary → expert routing pipeline

---

## 關鍵風險（850MB 環境特有）

| 風險 | 說明 | 對策 |
|------|------|------|
| LINE 更新佔用更多 VRAM | 突發 VRAM spike | 監測 LINE VRAM；ngl 動態降級 |
| DWM VRAM 隨顯示複雜度變化 | 多視窗時 DWM 用更多 | 測試時最小化非必要視窗 |
| DDR3 bandwidth 瓶頸 | CPU 層數多 → 慢 | LRU expert cache，減少 RAM swap |
| ngl=3 GPU 效益極低 | Vulkan overhead 可能抵消收益 | 對比 ngl=0 benchmark，擇優 |
| WDDM 無法鎖定 VRAM | OS 可隨時拿回 VRAM | mlock RAM；Vulkan priority hint |

---

## 成功指標（修正）

| 階段 | 指標 | 目標值 |
|------|------|--------|
| GT 1030 初期 | Gemma4 E4B 穩定推論 | tok/s >= 2 |
| GT 1030 初期 | 26B MoE chunk routing | 成功分割 chunk + 路由 |
| GT 1030 初期 | context 有效長度 | >= 16K (disk KV) |
| GT 1030 初期 | Chunk boundary 精度 | >= 80% |
| Mac M4 中期 | 200B MoE 推論 | tok/s >= 15 |
| 邊緣 VLM | moondream on GT1030 | 可運行 |
| AGI 雛型 | 自我強化迴圈 | 3 round 後品質提升 |
