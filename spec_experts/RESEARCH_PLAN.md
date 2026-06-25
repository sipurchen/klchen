# Spec-Experts LLM — 本機測試推演計劃
> Branch: `SpecExpertsResearch`  
> Date: 2026-06-25  
> Base: Spec-Experts-LLM-ImplementPlan.md

---

## 核心假設

**Spec-Experts 機制：** 不載入整個模型，只動態路由並載入「被激活的專家層參數」到 VRAM/RAM，其餘留在 disk。

---

## 硬體分層目標

### 初期目標：GT 1030 (2GB VRAM) — 26B+ MoE/Dense 部份載入

**硬體限制：**
- VRAM: 2GB GDDR5 (Pascal sm_61, Vulkan OK)
- RAM: 34GB DDR3 (大量 offload 空間)
- CPU: i5-4460 (4C/4T, Haswell)
- Disk: HDD/SSD (KV offload 目標)

**可行策略：**
```
MoE 26B 範例 (Mixtral-8x7B, DeepSeek-V2-Lite 等):
  總參數: ~26B  
  Active 參數/token: ~7B (top-2 experts / 8)
  Active 7B @ Q4_K_M ≈ 3.5GB → 超出 VRAM

  解法 (Spec-Experts 機制):
  1. Embedding + 注意力層 → RAM offload (llama.cpp -ngl 部份)
  2. Expert FFN 層 → 只 prefetch top-2 expert weights to VRAM
  3. 非激活 experts → 留在 RAM (不佔 VRAM)
  4. KV cache → disk streaming (P3.x)

  實測目標:
  VRAM 使用: < 1.8GB (留 200MB buffer for Vulkan overhead)
  RAM 使用: 8-16GB (expert weights pool)
  速度目標: 3-8 tok/s (受 DDR3 bandwidth 限制)
```

**llama.cpp 參數推演：**
```bash
llama-server \
  --model model.gguf \
  --n-gpu-layers 12 \        # 只把前N層放 VRAM
  --tensor-split 0.6,0.4 \   # GPU:CPU split
  --ctx-size 4096 \
  --n-predict 512 \
  --threads 2 \              # i5-4460 保留 core 0,1 給 OS
  --mlock \                  # 鎖定 RAM 防 swap
  --vulkan                   # GT 1030 Vulkan backend
```

**Phase 1 本機測試計劃 (GT 1030):**
- [ ] P1-GT: 測試 Mixtral-8x7B Q2_K (最小量化) → 確認能啟動
- [ ] P1-GT: 測試 DeepSeek-R1-Distill-Qwen-7B Q4 → baseline 速度
- [ ] P1-GT: 測試 26B dense Q2 → 確認 RAM offload 能跑
- [ ] P1-GT: Monitor hook 掛載到 llama-server logprob stream
- [ ] P1-GT: 驗證 Vulkan sm_61 能跑 MoE routing

---

### 中期目標：Mac Mini M4 (200B+ MoE)

**硬體能力：**
- RAM: 32GB unified memory (兼作 VRAM)
- Neural Engine: ANE (加速 attention)
- Metal backend (pi-ds4 已有路徑)

**可行策略：**
```
DeepSeek-V3/V4 (671B MoE):
  Active 參數/token: ~37B (top-2 / 256 experts)
  Active 37B @ Q4 ≈ 18.5GB → 勉強 fit 32GB

  解法:
  1. Expert pool: 只保留最常激活的 top-32 experts in RAM
  2. 其餘 experts: disk streaming via SSD (M4 SSD 快)
  3. Directional Steering: per-chunk steering vector (pi-ds4 機制)
  4. 多模態: 視覺 encoder 獨立 specialist routing

  速度目標: 15-30 tok/s
```

**大型多模態 + Metaverse Game AI 架構：**
```
[遊戲世界狀態] → [Visual VLM Expert] → [場景理解 chunk]
                                              ↓
[NPC行為需求]  → [Reasoning Expert]   → [行為決策 chunk]  
                                              ↓
[對話生成]     → [Creative Expert]    → [語言輸出 chunk]
                                              ↓
                              [KV Streaming: 跨NPC共享 context]
```

---

### 最終目標：自循環 AGI 雛型

**機制：**
```
[外部世界感知] → Monitor (語意跳躍偵測)
                        ↓
              [Chunk 分割 + 分類]
                        ↓
         [Specialist Expert 路由]
                        ↓
              [行動/輸出生成]
                        ↓
         [Streaming Distill → 自我強化]
                        ↑__________________|
                        (feedback loop)
```

**關鍵技術棧：**
1. Attention entropy → 語意狀態感知
2. KV prefix reuse → 長期記憶 (disk-backed)
3. LoRA adapter pool → 動態技能載入
4. Directional Steering → 目標導向行為控制

---

### 第二初期目標：邊緣視覺 VLM

**目標裝置：** Jetson Nano / 舊手機 / 車載 / 監控鏡頭

**模型策略：**
```
VLM 分層 (解析度/精度 tradeoff):

裝置等級          模型                  VRAM    FPS
─────────────────────────────────────────────────
Jetson Nano 4GB   MobileVLM-1.7B Q4     2GB     8-15
舊手機 Android    LLaVA-Phi-1.5 Q4      1.5GB   3-8
車載 (Qualcomm)   TinyLLaVA-3B Q4       2GB     5-12
監控鏡頭 (ARM)    moondream-2B Q4        1GB     2-5
Mac Mini M4       LLaVA-34B Q4          20GB    real-time
```

**機器視覺應用：**
- 物件偵測 + 場景描述 (VLM specialist)
- 影像修復 (Real-ESRGAN 邊緣版 → 受攝像頭原始畫質限制)
- 異常偵測 (監控場景 attention entropy 觸發 alert)

---

## 本機測試推演順序

### Week 1 (GT 1030 基礎驗證)
1. 確認 llama-server Vulkan build 能跑 (已有 bin/)
2. 下載 Mixtral-8x7B-v0.1.Q2_K.gguf (~15GB) 或更小測試
3. 測試 --n-gpu-layers 調整 → 找到 VRAM 2GB 甜蜜點
4. 掛載 monitor/attention_entropy.py 到 logprob stream
5. 驗證 chunk boundary 偵測輸出

### Week 2 (KV Offload 機制)
1. 實作 kv/kv_serializer.py
2. 測試 prompt replay 模擬 KV prefix (繞道 API 不完整問題)
3. 量測 context 擴大效果 (4K → 16K effective)

### Week 3 (Specialist Routing)
1. 實作 segmentor/chunk_classifier.py
2. 測試 per-chunk system prompt 切換速度
3. Directional Steering 移植自 pi-ds4

---

## 關鍵風險 (GT 1030 特有)

| 風險 | 說明 | 對策 |
|------|------|------|
| Vulkan sm_61 不穩定 | GT 1030 Pascal 較舊 | 備用 CPU-only path |
| DDR3 bandwidth 瓶頸 | expert swap 慢 | LRU expert cache in RAM |
| MoE routing overhead | expert 切換延遲高 | batch routing，減少切換 |
| 2GB VRAM 碎片化 | 長跑後 OOM | 定期 VRAM defrag (重啟 context) |

---

## 成功指標

| 階段 | 指標 | 目標值 |
|------|------|--------|
| GT 1030 初期 | 26B MoE 可推論 | tok/s ≥ 3 |
| GT 1030 初期 | Chunk boundary 精度 | ≥ 80% vs 人工標注 |
| Mac M4 中期 | 200B MoE 可推論 | tok/s ≥ 15 |
| 邊緣 VLM | 物件偵測精度 | mAP ≥ 60% |
| AGI 雛型 | 自我強化迴圈 | 3 round 後輸出品質提升 |
