# ============================================================================
# GEMMA 4 E4B — Master Prompt & Project Plan for Claude Sonnet 4.6
# Modified by ClaudeO
# Purpose: Complete bilingual (EN/ZH) reference for Cowork & Claude Code
# ============================================================================

---

## 📋 PROJECT OVERVIEW / 專案概述

### English
This project deploys Google's Gemma 4 E4B (released April 2, 2026) as a local
AI inference engine on Windows, accessed through Ollama, a Python FastAPI
server, and a Node.js agent framework. The E4B variant has 4.5B effective
parameters using Per-Layer Embeddings (PLE), supports native audio + vision
input, 128K context, 140+ languages, and runs under Apache 2.0 license.

### 中文
本專案將 Google 於 2026/4/2 發布的 Gemma 4 E4B 部署為本地 AI 推論引擎（Windows），
透過 Ollama、Python FastAPI 伺服器、Node.js Agent 框架提供服務。E4B 使用
Per-Layer Embeddings (PLE) 技術，有效參數 4.5B，原生支援音訊+視覺輸入、
128K 上下文窗口、140+ 語言，採用 Apache 2.0 授權。

---

## 🏗️ ARCHITECTURE / 系統架構

```
┌─────────────────────────────────────────────────────┐
│                    User Applications                 │
│          Python Scripts / Node.js Apps / CLI          │
└──────────┬──────────────────┬───────────────────────┘
           │                  │
    ┌──────▼──────┐   ┌──────▼──────┐
    │  FastAPI     │   │  Node.js    │
    │  Port 8000   │   │  Port 8001  │
    │  (Unified    │   │  (Agent     │
    │   API)       │   │  Framework) │
    └──────┬──────┘   └──────┬──────┘
           │                  │
           └────────┬─────────┘
                    │
           ┌────────▼────────┐
           │  Ollama Server  │
           │  Port 11434     │
           │  (gemma4:e4b)   │
           │  (gemma4-e4b-   │
           │   opt)          │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │  Gemma 4 E4B    │
           │  GGUF Model     │
           │  (Q4_K_M ~3GB)  │
           │  <LLMS_DIR>\Gemma4_E4B  │
           └─────────────────┘
```

---

## 📁 FILE STRUCTURE / 檔案結構

```
<PROJECT_ROOT>\
├── scripts/
│   └── setup_gemma4_e4b.ps1      # PowerShell setup: download + Ollama + deps
├── api/
│   ├── gemma4_api_server.py       # FastAPI unified server (port 8000)
│   └── gemma4_client.py           # Python client SDK
├── agents/
│   └── gemma4_agent_node.mjs      # Node.js agent framework (port 8001)
├── tests/
│   └── test_gemma4_full.py        # 5-phase comprehensive test suite
├── config/
│   └── Modelfile.gemma4-e4b-opt   # Ollama Modelfile (memory-optimized)
├── prompts/
│   └── MASTER_PROMPT.md           # This file (Sonnet 4.6 reference)
├── ollama_models/                 # Ollama model storage
├── requirements.txt               # Python dependencies
└── package.json                   # Node.js dependencies
```

---

## 🚀 SETUP STEPS FOR SONNET 4.6 / 設置步驟

### Step 1: Initial Setup / 初始設置

```
PROMPT FOR CLAUDE CODE:

You are setting up a Gemma 4 E4B local AI deployment on Windows.

Execute the following steps in order:
1. Open PowerShell as Administrator
2. Navigate to <PROJECT_ROOT>\scripts\
3. Run: .\setup_gemma4_e4b.ps1
4. Verify: ollama list (should show gemma4:e4b and gemma4-e4b-opt)
5. Test:  ollama run gemma4:e4b "Hello, respond briefly."

If Ollama is not installed:
  winget install Ollama.Ollama
If the model pull fails:
  Ensure ollama serve is running, then retry: ollama pull gemma4:e4b

---
你正在 Windows 上設置 Gemma 4 E4B 本地 AI 部署。
按順序執行以下步驟：
1. 以管理員身份開啟 PowerShell
2. 切換到 <PROJECT_ROOT>\scripts\
3. 執行：.\setup_gemma4_e4b.ps1
4. 驗證：ollama list（應顯示 gemma4:e4b 和 gemma4-e4b-opt）
5. 測試：ollama run gemma4:e4b "你好，簡短回覆。"
```

### Step 2: Start Services / 啟動服務

```
PROMPT FOR CLAUDE CODE:

Start the Gemma 4 E4B service stack:

Terminal 1 (Ollama):
  ollama serve

Terminal 2 (Python API):
  cd <LLMS_DIR>\Gemma4_E4B
  python api/gemma4_api_server.py

Terminal 3 (Node.js Agents - optional):
  cd <LLMS_DIR>\Gemma4_E4B
  node agents/gemma4_agent_node.mjs

Verify all services:
  curl http://localhost:11434        # Ollama
  curl http://localhost:8000/health  # Python API
  curl http://localhost:8001/health  # Node.js Agents

---
啟動 Gemma 4 E4B 服務堆疊：
終端 1：ollama serve
終端 2：python api/gemma4_api_server.py
終端 3：node agents/gemma4_agent_node.mjs（可選）
驗證：分別 curl 三個端口確認 healthy
```

### Step 3: Run Tests / 執行測試

```
PROMPT FOR CLAUDE CODE:

Run the comprehensive test suite:
  cd <LLMS_DIR>\Gemma4_E4B
  python tests/test_gemma4_full.py

Expected: 5 phases, 15+ tests, all PASS.
If Phase 2 fails: check ollama serve is running.
If Phase 3 fails: check python API server on port 8000.
If Phase 4 (memory) fails: verify Modelfile context settings.

---
執行完整測試套件：
  python tests/test_gemma4_full.py
預期：5 個階段、15+ 項測試全部通過。
Phase 2 失敗 → 檢查 ollama serve
Phase 3 失敗 → 檢查 API 伺服器 port 8000
Phase 4 失敗 → 檢查 Modelfile 的 context 設定
```

---

## 🤖 AGENT SYSTEM PROMPTS / Agent 系統提示詞

### General Assistant / 通用助手
```
You are Gemma 4 E4B, a helpful local AI assistant. Respond concisely
and accurately. Use structured formatting when it aids clarity.
---
你是 Gemma 4 E4B 本地 AI 助手。簡明準確地回覆。需要時使用結構化格式。
```

### Code Assistant / 程式助手
```
You are an expert programming assistant running locally via Gemma 4 E4B.
Write clean, well-documented code. Always explain your approach briefly.
Support Python, JavaScript, TypeScript, and shell scripting.
When debugging, show the root cause first, then the fix.
---
你是本地運行的 Gemma 4 E4B 程式專家。寫乾淨、有文件的程式碼。
簡述方法後提供解決方案。支援 Python/JS/TS/Shell。
除錯時先顯示根本原因，再給出修復。
```

### Data Analyst / 資料分析師
```
You are a data analysis agent powered by Gemma 4 E4B. Analyze data
precisely. Present findings with clear structure. When given images
of charts or tables, extract key insights. Always cite specific numbers.
---
你是 Gemma 4 E4B 驅動的資料分析 Agent。精確分析資料。
結構化呈現發現。處理圖表影像時提取關鍵洞察。始終引用具體數字。
```

### Vision Analyst / 視覺分析師
```
You are a visual analysis agent running Gemma 4 E4B. Analyze images
with precision: describe content, extract text (OCR), interpret charts,
identify objects. Provide structured analysis: 1) Summary 2) Details
3) Extracted text.
---
你是 Gemma 4 E4B 視覺分析 Agent。精確分析影像：
描述內容、提取文字(OCR)、解讀圖表、識別物件。
結構化輸出：1) 摘要 2) 細節 3) 提取的文字。
```

### Task Planner / 任務規劃師
```
You are an agentic task planner powered by Gemma 4 E4B. Break complex
requests into actionable steps. For each step, specify: the action,
required inputs, expected outputs, and any tool calls needed.
Use function calling format when appropriate. Think step by step.
---
你是 Gemma 4 E4B Agent 任務規劃師。將複雜請求分解為可執行步驟。
每步指定：動作、輸入、輸出、工具呼叫。
適時使用 function calling 格式。逐步思考。
```

### Multilingual Translator / 多語翻譯師
```
You are a professional translator powered by Gemma 4 E4B with native
support for 140+ languages. Translate accurately while preserving tone,
idioms, and cultural nuances. Always specify source and target languages.
---
你是 Gemma 4 E4B 專業翻譯師，原生支援 140+ 語言。
準確翻譯，保留語氣、慣用語和文化語境。始終標明來源語和目標語。
```

---

## 💾 MEMORY OPTIMIZATION / 記憶體優化

### Target: ~1GB Runtime Memory / 目標：運行時約 1GB

```
CRITICAL CONFIGURATION FOR MEMORY OPTIMIZATION:

ACTUAL MEMORY PROFILE (i5-4460, GT 1030 2GB VRAM, 34GB RAM):
  Gemma4:E4B is MoE architecture — model file is 9.1GB at Q4_K_M.
  It does NOT fit in 2GB VRAM. Use expert offloading strategy:

  Strategy A — Ollama (simple, ~0.04 tok/s):
    num_ctx 1024, num_thread 2, num_gpu -1
    Ollama does layer-level split only → very slow on CPU

  Strategy B — llama-server -ot (recommended, ~10-15 tok/s):
    Tensor-level split: attention on GPU, expert FFN on CPU RAM
    llama-server -m <gguf> -ngl 999
      -ot "blk\..*\.ffn_gate_exps\.weight=CPU"
      -ot "blk\..*\.ffn_down_exps\.weight=CPU"
      -ot "blk\..*\.ffn_up_exps\.weight=CPU"
      -c 1024 -t 3 --mmap --mlock

Memory breakdown (Strategy B, Q4_K_M):
  Attention/shared tensors (GPU):  ~1.2 GB  ← fits in 2GB VRAM
  Expert FFN weights (CPU mmap):   ~7.8 GB  ← only active 2/64 read/token
  KV cache (1K ctx):               ~0.1 GB
  TOTAL RAM:                       ~9.5 GB  ← well within 34GB

CPU Safety (CRITICAL on i5-4460):
  Set llama-server affinity to cores [1,2,3]; reserve core 0 for OS.
  Without this: 100% CPU saturation crashes Claude Code interface.

---
記憶體優化關鍵配置：
1. Q4_K_M 量化：模型從 ~10GB 縮至 ~3GB
2. Modelfile 限制 context：num_ctx 4096（KV cache 從 ~2GB 降至 ~64MB）
3. OLLAMA_NUM_PARALLEL=1（單請求模式）
4. num_batch 256（降低批次大小）
5. E4B 的 PLE 架構本身就減少活動參數

極端優化（接近 1GB）：
  使用 IQ2_M 量化（模型 ~1.2GB）+ 2K context → 總計 ~1.5GB
  注意：更低量化 = 更低品質。生產環境建議最低 Q4_K_M。
```

---

## 📡 API USAGE REFERENCE / API 使用參考

### Python Quick Start

```python
# 最簡用法 / Simplest usage
from gemma4_client import Gemma4Client

client = Gemma4Client()

# Basic chat / 基本對話
response = client.chat("Explain quantum computing in 3 sentences.")
print(response)

# Agent-based / 使用 Agent
code = client.chat("Write a merge sort in Python.", agent_id="coder")
print(code)

# Vision / 視覺分析
analysis = client.analyze_image("photo.jpg", prompt="Describe this scene.")
print(analysis)

# Streaming / 串流回覆
for chunk in client.chat_stream("Tell me about Taiwan's history."):
    print(chunk, end="", flush=True)

# Custom system prompt / 自訂系統提示
reply = client.chat(
    "Summarize this text.",
    system_prompt="You are a concise summarizer. Maximum 2 sentences.",
)

# With thinking mode / 啟用思考模式
answer = client.chat(
    "What is 127 * 389?",
    enable_thinking=True,  # Activates Gemma 4's reasoning chain
)
```

### Direct Ollama (curl)

```bash
# Basic chat / 基本對話
curl http://localhost:11434/api/chat -d '{
  "model": "gemma4:e4b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'

# With system prompt / 帶系統提示
curl http://localhost:11434/api/chat -d '{
  "model": "gemma4-e4b-opt",
  "messages": [
    {"role": "system", "content": "Reply only in JSON."},
    {"role": "user", "content": "List 3 colors."}
  ],
  "stream": false,
  "options": {"num_ctx": 4096, "temperature": 0.3}
}'

# OpenAI-compatible / OpenAI 相容端點
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "gemma4:e4b",
  "messages": [{"role": "user", "content": "Hi"}]
}'
```

### Node.js Quick Start

```javascript
import { Gemma4Client, createDefaultOrchestrator } from './agents/gemma4_agent_node.mjs';

// Direct client / 直接客戶端
const client = new Gemma4Client();
const response = await client.chat(
  [{ role: 'user', content: 'Hello from Node.js!' }],
  { agentId: 'general' }
);
console.log(response);

// Multi-agent orchestrator / 多 Agent 編排器
const orchestrator = createDefaultOrchestrator();

// Auto-route based on intent / 根據意圖自動路由
const result = await orchestrator.autoRoute('Write a REST API in Express');
console.log(result.agentId); // → 'coder'
console.log(result.response);

// Register custom agent / 註冊自訂 Agent
orchestrator.createAgent({
  agentId: 'custom_qa',
  name: 'QA Agent',
  systemPrompt: 'You review code for bugs. List issues as numbered items.',
  temperature: 0.2,
});
```

---

## 🧪 TESTING PLAN / 測試計劃

### Phase 1: Infrastructure / 基礎設施 (5 tests)
| # | Test | Command | Expected |
|---|------|---------|----------|
| 1.1 | Ollama server reachable | `curl localhost:11434` | 200 OK |
| 1.2 | gemma4:e4b model present | `ollama list` | Contains gemma4 |
| 1.3 | Optimized variant exists | `ollama list` | gemma4-e4b-opt |
| 1.4 | API server healthy | `curl localhost:8000/health` | status: healthy |
| 1.5 | Python deps installed | `python -c "import fastapi"` | No error |

### Phase 2: Ollama Direct / Ollama 直連 (4 tests)
| # | Test | Expected |
|---|------|----------|
| 2.1 | Basic text generation | Non-empty response |
| 2.2 | System prompt (JSON output) | Response contains { } |
| 2.3 | OpenAI-compatible endpoint | Has "choices" key |
| 2.4 | Optimized model responds | Non-empty response |

### Phase 3: Unified API / 統一 API (6 tests)
| # | Test | Expected |
|---|------|----------|
| 3.1 | Chat completion | Valid response text |
| 3.2 | Agent routing (coder) | Contains code |
| 3.3 | All agents registered | 6 default agents |
| 3.4 | Custom agent registration | status: registered |
| 3.5 | Model listing | >0 models |
| 3.6 | Error handling | Graceful fallback |

### Phase 4: Memory / 記憶體 (3 tests)
| # | Test | Expected |
|---|------|----------|
| 4.1 | Reduced context (2K) | Correct answer |
| 4.2 | Model disk size | < 5 GB |
| 4.3 | Concurrent requests | ≥ 1 success |

### Phase 5: Multimodal / 多模態 (3 tests)
| # | Test | Expected |
|---|------|----------|
| 5.1 | Vision endpoint | Returns analysis |
| 5.2 | Audio endpoint | Responds (200/422) |
| 5.3 | TTS info | Lists engines |

### Run all tests / 執行所有測試:
```bash
python tests/test_gemma4_full.py
```

---

## 🔧 COWORK PROMPT TEMPLATES / Cowork 提示詞模板

### Template 1: Bug Fix / 除錯
```
[EN] Using the Gemma 4 E4B API at http://localhost:8000, the following test
is failing: {paste test output}. Diagnose the root cause and fix it.
Check both the API server code and the Ollama model configuration.

[ZH] 使用 http://localhost:8000 的 Gemma 4 E4B API，以下測試失敗：
{貼上測試輸出}。診斷根本原因並修復。
同時檢查 API 伺服器程式碼和 Ollama 模型配置。
```

### Template 2: Add New Agent / 新增 Agent
```
[EN] Add a new agent to the Gemma 4 E4B system. The agent should:
- ID: {agent_id}
- Purpose: {description}
- System prompt: {prompt}
- Temperature: {temp}
Register it in both the Python API (gemma4_api_server.py default agents)
and the Node.js framework (gemma4_agent_node.mjs factory function).
Add a test case in test_gemma4_full.py Phase 3.

[ZH] 在 Gemma 4 E4B 系統中新增 Agent：
- ID：{agent_id}
- 用途：{description}
- 系統提示：{prompt}
- 溫度：{temp}
在 Python API 和 Node.js 框架中都要註冊。
在 test_gemma4_full.py Phase 3 增加測試案例。
```

### Template 3: Memory Tuning / 記憶體調整
```
[EN] The Gemma 4 E4B system is using too much memory ({current_mb}MB).
Target is {target_mb}MB. Adjust these parameters in order of impact:
1. num_ctx in Modelfile (lower = less KV cache)
2. Quantization level (Q4 → Q3 → IQ2)
3. num_batch (lower = less activation memory)
4. OLLAMA_NUM_PARALLEL (set to 1)
Rebuild the optimized model after changes:
  ollama create gemma4-e4b-opt -f Modelfile.gemma4-e4b-optimized

[ZH] Gemma 4 E4B 系統記憶體使用過多（{current_mb}MB）。
目標是 {target_mb}MB。按影響順序調整：
1. Modelfile 的 num_ctx（越低=越少 KV cache）
2. 量化等級（Q4 → Q3 → IQ2）
3. num_batch（越低=越少活化記憶體）
4. OLLAMA_NUM_PARALLEL 設為 1
修改後重建模型：
  ollama create gemma4-e4b-opt -f Modelfile.gemma4-e4b-optimized
```

### Template 4: Multimodal Integration / 多模態整合
```
[EN] Integrate Gemma 4 E4B's vision capabilities into {application}.
The E4B model natively supports:
- Image input: variable aspect ratio, configurable token budget
- Audio input: up to 30 seconds (WAV/MP3/FLAC)
- For multimodal prompts: place media BEFORE text
- For OCR: use high visual token budget (560 or 1120)
Use the /v1/vision/analyze endpoint or pass images as base64 in chat.

[ZH] 將 Gemma 4 E4B 的視覺能力整合到 {application} 中。
E4B 模型原生支援：
- 影像輸入：可變長寬比、可配置 token 預算
- 音訊輸入：最長 30 秒（WAV/MP3/FLAC）
- 多模態提示：將媒體放在文字之前
- OCR：使用高視覺 token 預算（560 或 1120）
使用 /v1/vision/analyze 端點或在 chat 中傳入 base64 影像。
```

### Template 5: Full Deployment Check / 完整部署驗證
```
[EN] Perform a full deployment verification of the Gemma 4 E4B system:
1. Run: python tests/test_gemma4_full.py
2. Check all 5 phases pass
3. Verify memory usage: tasklist /fi "imagename eq ollama*" (Windows)
4. Test each agent: curl POST to /v1/chat/completions with each agent_id
5. Test vision: upload a test image to /v1/vision/analyze
6. Verify OpenAI compatibility:
   curl http://localhost:11434/v1/chat/completions
7. Report results with pass/fail counts per phase

[ZH] 執行 Gemma 4 E4B 系統的完整部署驗證：
1. 執行：python tests/test_gemma4_full.py
2. 確認 5 個階段全部通過
3. 驗證記憶體使用：tasklist /fi "imagename eq ollama*"
4. 測試每個 Agent：用不同 agent_id POST /v1/chat/completions
5. 測試視覺：上傳測試影像到 /v1/vision/analyze
6. 驗證 OpenAI 相容性
7. 報告每階段的通過/失敗數
```

---

## ⚙️ CLAUDE CODE SPECIFIC INSTRUCTIONS / Claude Code 專用指令

```
SYSTEM CONTEXT FOR CLAUDE CODE:

Project: Gemma 4 E4B Local AI Deployment
Location: <LLMS_DIR>\Gemma4_E4B
Tech Stack: Ollama + FastAPI (Python) + Express (Node.js)
Model: gemma4:e4b (base) / gemma4-e4b-opt (memory-optimized)

Key Files:
  api/gemma4_api_server.py  — Main API server (modify for new endpoints)
  api/gemma4_client.py      — Python SDK (modify for new client features)
  agents/gemma4_agent_node.mjs — Node.js agents (modify for new agents)
  tests/test_gemma4_full.py — Test suite (add tests for new features)

Coding Conventions:
  - UTF-8 encoding for all files
  - Mark all modifications with "Modified by ClaudeO" comments
  - English comments and documentation
  - Before modifying a function: check for dependencies and similar functions
  - Do not modify unrelated files without explicit permission
  - Files > 1000 lines: propose modularization

API Endpoints:
  GET  /health                    — Health check
  POST /v1/chat/completions       — OpenAI-compatible chat
  POST /v1/vision/analyze         — Image analysis (multipart)
  POST /v1/audio/transcribe       — Audio transcription (multipart)
  POST /v1/audio/tts              — TTS info (placeholder)
  GET  /v1/agents                 — List agents
  POST /v1/agents                 — Register agent
  DELETE /v1/agents/{agent_id}    — Remove agent
  GET  /v1/models                 — List Ollama models

Memory Optimization Levers (ordered by impact):
  1. num_ctx → controls KV cache size (biggest impact)
  2. Quantization → controls model weight size
  3. num_batch → controls activation memory
  4. OLLAMA_NUM_PARALLEL → controls concurrent request overhead

---
專案：Gemma 4 E4B 本地 AI 部署
位置：<LLMS_DIR>\Gemma4_E4B
技術棧：Ollama + FastAPI (Python) + Express (Node.js)

編碼規範：
  - 所有檔案使用 UTF-8
  - 修改處標記 "Modified by ClaudeO"
  - 英文註解和文件
  - 修改前檢查相依性
  - 勿修改未要求的檔案
  - 超過 1000 行的檔案：建議模組化
```

---

## 📊 GEMMA 4 E4B TECHNICAL SPECS / 技術規格

| Spec | Value |
|------|-------|
| Total Parameters | 8B (MoE, ~2.7B active/token) |
| Effective Parameters | ~2.7B per token (top_k=2 of 64 experts) |
| Architecture | MoE (Mixture of Experts), 43 layers, 64 experts |
| Context Window | 128K tokens |
| Languages | 140+ |
| Modalities (Input) | Text, Image, Audio |
| Modalities (Output) | Text |
| Audio Max Duration | 30 seconds |
| Video Max Duration | 60 seconds (1 fps) |
| License | Apache 2.0 |
| Quantization (recommended) | Q4_K_M (GGUF) |
| Disk Size (Q4_K_M) | ~9.1 GB (MoE — larger than dense 8B) |
| RAM (Q4, 1K ctx) | ~12 GB total (model + KV cache + OS) |
| RAM (Q3_K_S, 1K ctx) | ~8 GB total |
| Thinking Mode | Configurable via `<\|think\|>` token |
| System Prompts | Native support (system role) |
| Function Calling | Native support |
| Structured Output | JSON mode supported |
| GPU Offload Strategy | Use llama-server -ot for MoE expert offloading |
| Expected tok/s (optimized) | 10-15 tok/s with llama-server expert split |

---

## 🔄 MAINTENANCE COMMANDS / 維護指令

```bash
# Update Ollama / 更新 Ollama
winget upgrade Ollama.Ollama

# Re-pull model (latest version) / 重新拉取模型
ollama pull gemma4:e4b

# Rebuild optimized variant / 重建優化變體
ollama create gemma4-e4b-opt -f <PROJECT_ROOT>\Modelfile.gemma4-e4b-optimized

# Check memory usage / 檢查記憶體使用
tasklist /fi "imagename eq ollama*"

# Clean unused models / 清理未使用模型
ollama rm <model_name>

# View model details / 查看模型詳情
ollama show gemma4:e4b

# Restart everything / 重啟所有服務
taskkill /im ollama.exe /f
ollama serve
python api/gemma4_api_server.py
```

---

*Generated by Claude Opus 4.6 — Modified by ClaudeO*
*Last updated: 2026-04-04*
