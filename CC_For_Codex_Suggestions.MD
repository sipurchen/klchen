# CC_For_Codex_Suggestions.MD
# klchen 專案提升建議案 — 供 Codex 參考實作
# 參考來源: audreyt/pi-ds4 調校方法論 + klchen 現有架構分析

**生成日期:** 2026-05-13  
**基礎倉庫:** https://github.com/sipurchen/klchen  
**參考倉庫:** https://github.com/audreyt/pi-ds4  
**硬體環境:** i5-4460 / GT 1030 2GB / 34 GB DDR3 / Windows 10  

---

## 核心發現：架構事實修正

在實施任何建議前，需先確認以下已知事實：

| 項目 | TASKS.md 假設 | 實際發現 | 影響 |
|------|-------------|---------|------|
| Gemma4 E4B 架構 | MoE (64 experts) | **Dense (n_expert=0)** | Task 5 的 `-ot ffn_*_exps` 方案**無效** |
| Gemma4 E4B blob | 多模態 (9.1 GB, 2131 tensors) | bartowski 純文字 GGUF (5.03 GB, 720 tensors) | llama-server 只能用 bartowski 版 |
| 10 tok/s 理論路徑 | MoE expert offloading | **CPU-only mmap (~2 tok/s 上限)** | 達成 10 tok/s 需 RTX 3060+ |

---

## 建議一：導入 Directional Steering（方向性激活調控）

**參考:** [audreyt/pi-ds4 #directional-steering](https://github.com/audreyt/pi-ds4#directional-steering)

### 問題
klchen 目前使用 system prompt 調整模型行為，但 system prompt 對於模型的**強記憶補全 (memorized completions)** 幾乎無效。例如，模型在被問及有爭議的主題時，會直接輸出訓練時記憶的單一答案，不管 system prompt 要求它保持中立。

### audreyt 的解法
使用 **低秩激活編輯 (low-rank activation edit)** 在 FFN 輸出層施加方向向量，使模型進入「不確定性模式」而非「斷言模式」：

```bash
# 核心參數: DS4_DIR_STEERING_FFN=-2
# 效果: 將模型推向"此為爭議問題"的回應暫存器 (response register)
# 原理: 不改變模型參數，只在推論時調整特定層的激活方向
```

### 對 klchen 的建議實作

**Step 1** — 為 Gemma4 bartowski GGUF 生成不確定性方向向量：

```python
# scripts/gen_steering_vector.py
# 需要: llama.cpp 的 llama-run 或支援 --prompt-cache 的版本
# 方法: 收集 50 個「已確定事實」vs 50 個「有爭議問題」的激活差值

import numpy as np

# 已確定 (settled): "台北是台灣的首都" / "水的沸點是100°C" 
# 爭議 (contested): "某政策是否正確" / "某事件的責任歸屬"
# 差值的平均即為「不確定性方向」
def build_uncertainty_vector(settled_activations, contested_activations):
    diff = np.array(contested_activations) - np.array(settled_activations)
    return diff.mean(axis=0)  # shape: [n_layers, hidden_dim]
```

**Step 2** — 在 `api/gemma4_api_server.py` 加入 steering 端點：

```python
# 在啟動 llama-server 時加入 --override-kv 參數
# 或透過 POST /v1/completions 的自訂 header 傳遞 steering scale
STEERING_FFN_SCALE = float(os.getenv("GEMMA4_STEERING_FFN", "0"))  # 0=停用

# 建議預設值: -1.5 (比 audreyt 的 -2 保守，因為本專案用 Q4_K_M 非 imatrix)
# 原因: audreyt 從 -3 降到 -2 是因為 imatrix 量化壓縮了激活分佈的頭空間
#       Q4_K_M 量化略有不同，建議從 -1.5 開始測試
```

**Step 3** — 在 `agents/gemma4_agent_node.mjs` 的 `AgentOrchestrator.autoRoute()` 中加入：

```javascript
// 偵測敏感/爭議類問題時，自動加入 steering 提示
const contestedKeywords = ['政策', '是非', '對錯', '應該', '責任', 'should', 'blame', 'right or wrong'];
const isContested = contestedKeywords.some(kw => lowerMsg.includes(kw));
if (isContested) {
    context.steeringScale = parseFloat(process.env.GEMMA4_STEERING_FFN || '-1.5');
}
```

**預期效益:** 減少模型在爭議性問題上的單一化斷言輸出，提升回應品質的多元性。

---

## 建議二：imatrix 校準量化（以降低模型大小並提升速度）

**參考:** [audreyt/ds4 cyberneurova IQ2XXS-w2Q2K imatrix GGUF](https://github.com/audreyt/pi-ds4#directional-steering)

### 問題
klchen 目前使用 `Q4_K_M`（5.03 GB），CPU-only 推論約 2 tok/s。  
audreyt 使用 `IQ2XXS-w2Q2K imatrix`（類似規模模型 ~87 GB，但這是 284B MoE，若縮放到 7.5B Dense 等比約 2.3 GB）。

### 對 klchen 的建議實作

使用 `llama-quantize` 搭配 imatrix 校準資料集，生成比 Q4_K_M 更小但品質相近的量化版本：

```powershell
# Step 1: 生成 imatrix 校準資料
# 使用 bartowski GGUF + 代表性繁體中文+英文資料集
.\bin\llama-cpp\llama-imatrix.exe `
  -m "E:\LLMmodel\gemma4_textonly\google_gemma-4-E4B-it-Q4_K_M.gguf" `
  -f scripts\calibration_dataset_zh_en.txt `
  -o bin\gemma4_e4b_imatrix.dat `
  -ngl 0 -t 3

# Step 2: 使用 imatrix 重新量化為 Q3_K_M
.\bin\llama-cpp\llama-quantize.exe `
  "E:\LLMmodel\gemma4_textonly\google_gemma-4-E4B-it-Q4_K_M.gguf" `
  "E:\LLMmodel\gemma4_textonly\google_gemma-4-E4B-it-Q3_K_M_imatrix.gguf" `
  Q3_K_M `
  --imatrix bin\gemma4_e4b_imatrix.dat
```

**預期效益:**
| 量化 | 大小 | 理論 tok/s (DDR3 17GB/s) | 品質損失 |
|------|------|------------------------|---------|
| Q4_K_M (現狀) | 5.03 GB | ~3.4 tok/s | 基準 |
| Q3_K_M + imatrix | ~3.9 GB | ~4.4 tok/s | ~1% (imatrix 補償) |
| Q2_K + imatrix | ~2.7 GB | ~6.3 tok/s | ~5% (可接受) |

**注意:** 達成 10 tok/s 仍需 GPU (≥6 GB VRAM)。但 imatrix Q2_K 可在現有硬體上讓速度翻倍。

---

## 建議三：Server Watchdog + Per-Process Lease 機制

**參考:** [audreyt/pi-ds4 ds4-watchdog.sh](https://github.com/audreyt/pi-ds4) — 當無活躍 client 時自動停止 server

### 問題
klchen 的 `api/gemma4_api_server.py` 啟動 llama-server 後沒有資源回收機制。在 4 核心 CPU + 34 GB RAM 環境下，llama-server 長時間佔用記憶體 + CPU affinity 會影響其他應用。

### 建議實作

```python
# api/gemma4_api_server.py 加入以下機制

import threading, time, psutil
from pathlib import Path

LEASE_DIR = Path("E:/Gemma4_E4B_Project/.leases")
LEASE_DIR.mkdir(exist_ok=True)
WATCHDOG_TIMEOUT = 300  # 5 分鐘無請求後停止 server

class ServerWatchdog(threading.Thread):
    """停止 llama-server 當 5 分鐘內無任何請求"""
    def __init__(self, server_proc):
        super().__init__(daemon=True)
        self.server_proc = server_proc
        self.last_request = time.time()

    def ping(self):
        self.last_request = time.time()

    def run(self):
        while True:
            time.sleep(30)
            if time.time() - self.last_request > WATCHDOG_TIMEOUT:
                if self.server_proc.poll() is None:
                    self.server_proc.terminate()
                    print("[Watchdog] llama-server stopped (idle timeout)")
                break

# 在每個 API 請求的 middleware 中呼叫 watchdog.ping()
```

**另：在 `agents/gemma4_agent_node.mjs` 的 Gemma4Client 加入心跳：**

```javascript
// 每 60s 發送 keepalive，防止 watchdog 誤殺正在等待的長請求
startKeepalive(intervalMs = 60000) {
    this._keepaliveTimer = setInterval(async () => {
        try { await axios.get(`${this.apiUrl}/health`, { timeout: 3000 }); }
        catch { /* ignore */ }
    }, intervalMs);
}

stopKeepalive() {
    if (this._keepaliveTimer) clearInterval(this._keepaliveTimer);
}
```

---

## 建議四：環境變數驅動的靈活設定

**參考:** audreyt/pi-ds4 的完整 env var 設計（DS4_SUPPORT_REPO, DS4_MPP, DS4_DIR_STEERING_FFN 等）

### 問題
klchen 的設定分散在多個檔案（`config/direct_inference.json`, `scripts/start_optimized.ps1`, hardcoded paths），難以在不同機器部署。

### 建議：統一設定入口 `config/.env`

```bash
# config/.env (加入 .gitignore)

# llama-server 路徑
LLAMA_SERVER_BIN=E:\Gemma4_E4B_Project\bin\llama-cpp\llama-server.exe
LLAMA_SERVER_HOST=127.0.0.1
LLAMA_SERVER_PORT=8080

# 模型路徑
GEMMA4_GGUF=E:\LLMmodel\gemma4_textonly\google_gemma-4-E4B-it-Q4_K_M.gguf

# Ollama 設定
OLLAMA_HOST=http://localhost:11434
OLLAMA_MAX_LOADED_MODELS=1

# CPU affinity (逗號分隔核心 ID)
LLAMA_CPU_AFFINITY=1,2,3
PYTHON_CPU_AFFINITY=0,1

# Steering (0 = 停用)
GEMMA4_STEERING_FFN=0

# Watchdog
WATCHDOG_TIMEOUT_SEC=300
```

在 `api/gemma4_api_server.py` 開頭加入：
```python
from dotenv import load_dotenv
load_dotenv("config/.env")
```

在 `agents/gemma4_agent_node.mjs` 開頭加入：
```javascript
import { config as dotenvConfig } from 'dotenv';
dotenvConfig({ path: 'config/.env' });
```

---

## 建議五：Agent 路由升級 — 從關鍵字到 Embedding 相似度

**參考:** audreyt/pi-ds4 的 uncertainty steering 概念：不同問題類型觸發不同的回應模式

### 問題
`AgentOrchestrator.autoRoute()` 目前使用硬碼關鍵字陣列匹配，對繁體中文輸入效果差（例如「幫我寫個函式」不匹配 `code` 關鍵字），且容易誤判。

### 建議實作：使用 Gemma3:1b 做快速意圖分類

```javascript
// 替換 autoRoute() 的關鍵字匹配
async autoRoute(message) {
    // 用最快的 Gemma3:1b 做 intent classification（單次小推論 ~0.3s）
    const intentPrompt = `Classify this request into ONE of: coder, analyst, translator, vision, planner, general.
Request: "${message.slice(0, 200)}"
Reply with only the category name.`;

    const intent = await this.client.chatDirect(
        [{ role: 'user', content: intentPrompt }],
        { model: 'gemma3:1b', numCtx: 256, temperature: 0.1 }
    );

    const agentId = intent.trim().toLowerCase();
    const target = this.agents.has(agentId) ? agentId : 'general';
    console.log(`[Orchestrator] Intent-routed to: ${target} (raw: "${intent.trim()}")`);
    return this.route(target, message);
}
```

**預期效益:** 中英文混合輸入的路由準確率從估計 40% 提升到 85%+，代價是 +0.3s 延遲。

---

## 建議六：Streaming 回應支援

**參考:** audreyt/pi-ds4 使用 HTTP streaming (`text/event-stream`) 提供即時輸出

### 問題
klchen 目前所有 API 回應都是等待完整生成後才返回。在 2 tok/s 的環境下，使用者需要等 10-30 秒才看到任何輸出，體驗極差。

### 建議實作

**Python FastAPI 端 (`api/gemma4_api_server.py`):**
```python
from fastapi.responses import StreamingResponse
import json

@app.post("/v1/chat/completions/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{LLAMA_SERVER_URL}/v1/chat/completions",
                json={**request.dict(), "stream": True},
                timeout=300
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        yield f"{line}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**Node.js 端 (`agents/gemma4_agent_node.mjs`):**
```javascript
async *chatStream(messages, options = {}) {
    const resp = await axios.post(
        `${this.apiUrl}/v1/chat/completions/stream`,
        { ...payload, stream: true },
        { responseType: 'stream', timeout: this.timeout }
    );

    for await (const chunk of resp.data) {
        const lines = chunk.toString().split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                const data = JSON.parse(line.slice(6));
                const token = data.choices?.[0]?.delta?.content;
                if (token) yield token;
            }
        }
    }
}
```

---

## 建議七：修正 TASKS.md 的錯誤技術假設

TASKS.md 的 Task 5 基於 Gemma4 是 MoE 的**錯誤假設**，以下修正：

### 已失效的技術路徑（不要實作）
- `-ot "blk\..*\.ffn_gate_exps\.weight=CPU"` — Gemma4 E4B **無 expert FFN tensors**
- `--cpu-moe` 旗標 — 對 `n_expert=0` 無效
- 10 tok/s 目標（現有硬體下不可達）

### 替代路徑：可實際改善速度的方案

| 方案 | 預估 tok/s | 可行性 |
|------|-----------|-------|
| imatrix Q2_K 重量化（建議二） | ~6 tok/s | 高（只需 llama-quantize） |
| `-c 256` 減少 KV Cache | ~2.5 tok/s | 高（短對話場景適用） |
| `--cache-type-k q8_0 --cache-type-v q8_0` | ~2.2 tok/s | 高（需 llama-server b8679+） |
| 升級至 RTX 3060 12GB | ~48 tok/s | 需硬體（最終解法） |

---

## 建議八：測試覆蓋補強

參考 audreyt/pi-ds4 的 idempotent 啟動設計，klchen 需要加入以下測試：

```python
# tests/test_server_lifecycle.py
def test_llama_server_restarts_cleanly():
    """llama-server crash 後能自動重啟"""
    ...

def test_watchdog_stops_idle_server():
    """5 分鐘無請求後 server 自動停止"""
    ...

def test_cpu_affinity_applied():
    """確認 llama-server 被限制在指定 CPU cores"""
    import psutil
    for proc in psutil.process_iter(['name', 'cpu_affinity']):
        if 'llama-server' in proc.info['name']:
            assert set(proc.info['cpu_affinity']) == {1, 2, 3}
```

---

## 優先實作順序（供 Codex 參考）

```
P0 (立即修正，影響正確性):
  → 建議七: 移除 TASKS.md 的錯誤 MoE 假設，更新速度目標

P1 (高效益，低風險):
  → 建議四: 統一 config/.env 設定入口
  → 建議六: Streaming 回應（使用者體驗最大改善）

P2 (中效益，中複雜度):
  → 建議二: imatrix Q3_K_M / Q2_K 重量化（速度 +30-100%）
  → 建議五: 使用 Gemma3:1b 做 intent routing

P3 (進階，需更多研究):
  → 建議一: Directional steering（需生成方向向量）
  → 建議三: Watchdog + lease 機制
  → 建議八: 測試補強
```

---

## 附錄：audreyt/pi-ds4 可直接借用的工程模式

| 模式 | 說明 | 對應檔案 |
|------|------|---------|
| Per-process lease | `~/.pi/ds4/clients/<pid>.json` — 追蹤活躍 client | `ds4-watchdog.sh` |
| Idempotent startup | 第二次啟動跳過已下載的 GGUF | `index.ts:startup()` |
| HF resumable download | `--resume-download` 自動續傳 | `download_model.sh` |
| Env var override chain | 每個設定都有 env var 覆蓋 | `index.ts:constants` |
| OpenAI-compatible shim | `http://127.0.0.1:8000/v1` | `index.ts:BASE_URL` |

klchen 已實作後三項（HF 續傳在 LLMmodel/download_models.ps1、OpenAI shim 在 gemma4_api_server.py）；建議補充前兩項（lease + idempotent startup）。

---

*本文件由 Claude Sonnet 4.6 (claude-sonnet-4-6) 基於 audreyt/pi-ds4 代碼審查及 klchen 架構分析生成。*
