# LLM Comparison Report

> Generated: 2026-04-08 03:22  
> Inference: **Ollama v0.20.0** (Gemma3, DeepSeek) + **llama-server b8679** (Gemma4)  
> Hardware: i5-4460 CPU · GT 1030 2GB VRAM · 34GB DDR3 RAM  
> Gemma4: bartowski text-only Q4_K_M GGUF (5GB) · CPU-only inference · raw /completion (no thinking)

## Model Overview

| Model | Size | Architecture | GPU Layers | Context |
|-------|------|--------------|------------|---------|
| **Gemma3 1B** | 777 MB | Dense 29-layer | -1 (fully GPU via Ollama) | 2048 |
| **DeepSeek-r1 1.5B** | 1.04 GB | Dense 30-layer + reasoning | 20/30 (partial via Ollama) | 2048 |
| **Gemma4 E4B** | 5.0 GB | Dense 42-layer (7.5B, Q4_K_M) | 0/43 CPU-only (2GB VRAM too small) | 512 |

## Task Results

| Task | **Gemma3 1B** | **DeepSeek-r1 1.5B** | **Gemma4 E4B** |
|------|------|------|------|
| Text Summarization | ✅ 32.0s · 21.5 tok/s | ✅ 37.3s · 14.98 tok/s | ✅ 27.4s · 1.42 tok/s |
| Synonym Enhancement | ✅ 12.5s · 13.75 tok/s | ✅ 39.6s · 11.51 tok/s | ✅ 36.5s · 2.19 tok/s |
| General Chat | ✅ 5.2s · 21.96 tok/s | ✅ 42.6s · 11.13 tok/s | ✅ 23.9s · 2.09 tok/s |
| Image Analysis | ✅ 0.9s | ➖ N/A | ➖ N/A |
| Audio / STT | ➖ N/A | ➖ N/A | ➖ N/A |
| TTS Output | ✅ 2.0s | ✅ 2.0s | ✅ 2.0s |
| Video Analysis | ➖ N/A | ➖ N/A | ➖ N/A |

## Detailed Responses

### Gemma3 1B

**Text Summarization** ✅
- Time: 32.0s | Speed: 21.5 tok/s
- Response: Artificial intelligence is rapidly advancing across various industries through machine learning, allowing computers to learn and make decisions without direct human input. This technology is now impacting areas like healthcare and transportation, becoming more integrated into our daily routines.

**Synonym Enhancement** ✅
- Time: 12.5s | Speed: 13.75 tok/s
- Response: Here are a few options for rewriting "AI is changing the way we work and live" with more sophisticated vocabulary, each with a slightly different nuance:

**Option 1 (Emphasis on transformative impact):**

“Artificial intelligence is fundamentally reshaping the contours of both professional and pers

**General Chat** ✅
- Time: 5.2s | Speed: 21.96 tok/s
- Response: Here are three benefits of using local LLMs:

1.  **Enhanced Privacy:** Your data stays within your own infrastructure, eliminating the need to send it to a third-party server.
2.  **Reduced Latency:** Local processing minimizes reliance on internet connectivity, leading to faster responses.
3.  **C

**Image Analysis** ✅
- Time: 0.9s | Speed: 0 tok/s
- Response: (empty)

**Audio / STT** ➖
- Not supported for this model/task combination.

**TTS Output** ✅
- Time: 2.0s | Speed: 0 tok/s
- Response: WAV file created via Windows SAPI

**Video Analysis** ➖
- Not supported for this model/task combination.

### DeepSeek-r1 1.5B

**Text Summarization** ✅
- Time: 37.3s | Speed: 14.98 tok/s
- Response: Artificial intelligence is rapidly transforming modern industries by enabling computers to learn from data and identify patterns with minimal human intervention, leading to advancements across healthcare diagnostics, autonomous vehicles, and beyond.

**Synonym Enhancement** ✅
- Time: 39.6s | Speed: 11.51 tok/s
- Response: AI is fundamentally altering the very essence of human life and work.

**General Chat** ✅
- Time: 42.6s | Speed: 11.13 tok/s
- Response: The top three benefits of local large language models (LLMs) can be summarized as follows:

1. **Faster Processing**: Local LLMs benefit from being trained on data specific to a particular region or area, which reduces their computational requirements and

**Image Analysis** ➖
- Not supported for this model/task combination.

**Audio / STT** ➖
- Not supported for this model/task combination.

**TTS Output** ✅
- Time: 2.0s | Speed: 0 tok/s
- Response: WAV file created via Windows SAPI

**Video Analysis** ➖
- Not supported for this model/task combination.

### Gemma4 E4B

**Text Summarization** ✅
- Time: 27.4s | Speed: 1.42 tok/s
- Response: Artificial intelligence is rapidly transforming modern industries by enabling computers to learn from data and make decisions autonomously. Its sophisticated applications are becoming increasingly pervasive in daily life, ranging from healthcare diagnostics to autonomous vehicles.

**Synonym Enhancement** ✅
- Time: 36.5s | Speed: 2.19 tok/s
- Response: Here are several options, depending on the specific nuance you wish to convey:

**More Formal/Academic:**

* "Artificial intelligence is fundamentally reshaping the paradigms of our professional and personal existence."
* "The integration of artificial intelligence is profoundly transforming the mod

**General Chat** ✅
- Time: 23.9s | Speed: 2.09 tok/s
- Response: 1. Enhanced data privacy and security by keeping data on-premises.
2. Reduced latency and improved inference speed due to local processing.
3. Full control over the model and deployment environment for customization.

**Image Analysis** ➖
- Not supported for this model/task combination.

**Audio / STT** ➖
- Not supported for this model/task combination.

**TTS Output** ✅
- Time: 2.0s | Speed: 0 tok/s
- Response: WAV file created via Windows SAPI

**Video Analysis** ➖
- Not supported for this model/task combination.

## Speed Comparison (tok/s)

| Model | Summarize | Synonym | Chat | Avg tok/s |
|-------|-----------|---------|------|-----------|
| Gemma3 1B | 21.5 | 13.75 | 21.96 | **19.07** |
| DeepSeek-r1 1.5B | 14.98 | 11.51 | 11.13 | **12.54** |
| Gemma4 E4B | 1.42 | 2.19 | 2.09 | **1.9** |

## Optimization Notes

### Gemma4 E4B: llama-server (bypassing Ollama)

Ollama's Gemma4 blob is a combined multimodal GGUF (2131 tensors: text + audio + vision)
that Ollama's patched runner handles internally. Standard llama.cpp b8679 cannot load it.
Solution: use **bartowski/google_gemma-4-E4B-it-GGUF** (text-only, 720 tensors, 5.03 GB)
via llama-server b8679 with CPU-only inference.

| Setting | Value | Reason |
|---------|-------|--------|
| `-ngl 0` | CPU-only | GT 1030 VRAM (2GB) too small for embedding table (>2GB alloc) |
| `-t 4` | 4 threads | All cores for Gemma4 when running alone |
| Raw `/completion` | Bypass chat template | Gemma4 instruct adds `<\|think\|>` → all tokens go to hidden reasoning |
| CPU affinity 2+3 | Cores 2+3 only | Python on 0+1; prevents CPU overload/screen blank |

**Speed**: ~2 tok/s (hardware-bound: 5GB model × 17 GB/s DDR3 ≈ 3.4 tok/s theoretical max)
**vs Ollama**: 0.04 tok/s → 2 tok/s = **50× improvement**

### CPU Safety (i5-4460 overload prevention)

| Technique | Effect |
|-----------|--------|
| Python: cores 0+1 | LLM runner can't steal OS scheduling |
| Ollama/llama-server: cores 2+3 | Isolated from Python process |
| Sequential model loading | No two models in RAM simultaneously |
| Results saved per-test | Crash-safe; partial results preserved |

## Hardware & Environment

```
CPU:   Intel Core i5-4460 (4 cores, 3.2GHz base)
GPU:   NVIDIA GeForce GT 1030 (2GB GDDR5, sm_61 Pascal)
RAM:   34GB DDR3
OS:    Windows 10 Pro 10.0.19045
CUDA:  11.8

Ollama:        v0.20.0  (Gemma3 1B, DeepSeek-r1 1.5B)
llama-server:  b8679 Vulkan/CPU  (Gemma4 E4B text-only)
Gemma4 GGUF:   bartowski/google_gemma-4-E4B-it-Q4_K_M.gguf (5.03 GB)
Ollama blobs:  E:\LLMmodel\blobs
```
