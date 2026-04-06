# LLM Comparison Report

> Generated: 2026-04-06 14:46  
> Inference: **Ollama API**  
> Hardware: i5-4460 CPU | GT 1030 2GB VRAM | 34GB RAM  
> Optimizations: Q4_K_M quant · KV cache q8_0 · Ollama CPU affinity (cores 2+3) · num_thread=2 · sequential unload between models
> **Gemma4 E4B finding**: 9.1GB MoE on i5-4460 (2 threads) = ~25 sec/token. Text tasks timeout at 900s. Vision responds in 313s (empty). Practically usable only with num_predict ≤ 5.

## Model Overview

| Model | Size | Architecture | GPU Layers | Context |
|-------|------|--------------|------------|---------|
| **Gemma3 1B** | 777 MB | Dense 29-layer | -1 (fully GPU) | 2048 |
| **DeepSeek-r1 1.5B** | 1.04 GB | Dense 30-layer | 20/30 (partial) | 2048 |
| **Gemma4 E4B** | 9.1 GB | MoE 43-layer | 12/43 (attn→GPU, experts→RAM) | 1024 |

## Task Results

| Task | **Gemma3 1B** | **DeepSeek-r1 1.5B** | **Gemma4 E4B** |
|------|------|------|------|
| Text Summarization | ✅ 22.9s | 21.35 tok/s — _Artificial intelligence is rapidly advancing across various industries through m_ | ✅ 29.1s | 12.87 tok/s — _Artificial intelligence is transforming industries by enabling machines to learn_ | ❌ 905.4s — _timed out_ |
| Synonym Enhancement | ✅ 10.4s | 21.27 tok/s — _Here are a few options for rewriting "AI is changing the way we work and live" w_ | ✅ 17.3s | 12.5 tok/s — __ | ❌ 934.5s — _timed out_ |
| General Chat | ✅ 4.2s | 21.15 tok/s — _Here are the top 3 benefits of running LLMs locally:  1.  **Privacy:** Your data_ | ✅ 15.2s | 14.21 tok/s — __ | ❌ 905.4s — _timed out_ |
| Image Analysis | ✅ 0.7s — __ | ➖ N/A | ✅ 313.9s — __ |
| Audio / STT | ➖ N/A | ➖ N/A | ❌ 900.4s — _timed out_ |
| TTS Output | ✅ 2.0s — _WAV generated via Windows SAPI_ | ✅ 2.0s — _WAV generated via Windows SAPI_ | ✅ 2.0s — _WAV generated via Windows SAPI_ |
| Video Analysis | ➖ N/A | ➖ N/A | ➖ N/A |

## Detailed Responses

### Gemma3 1B

**Text Summarization** ✅
- Time: 22.9s | Speed: 21.35 tok/s
- Response: Artificial intelligence is rapidly advancing across various industries through machine learning, allowing computers to learn and make decisions without direct human input. This technology is now impacting areas like healthcare and transportation, becoming increasingly integrated into our daily routi

**Synonym Enhancement** ✅
- Time: 10.4s | Speed: 21.27 tok/s
- Response: Here are a few options for rewriting "AI is changing the way we work and live" with a more sophisticated vocabulary, each with a slightly different nuance:

**Option 1 (Focus on transformative impact):**

“Artificial Intelligence is fundamentally reshaping the contours of both professional and perso

**General Chat** ✅
- Time: 4.2s | Speed: 21.15 tok/s
- Response: Here are the top 3 benefits of running LLMs locally:

1.  **Privacy:** Your data stays private.
2.  **Speed & Control:** Faster responses and no reliance on cloud services.
3.  **Cost Savings:** Avoid API fees. 

Do you want to know more about any of these benefits?

**Image Analysis** ✅
- Time: 0.7s | Speed: 0.0 tok/s
- Response: 

**Audio / STT** ➖
- Not supported for this model/task combination.

**TTS Output** ✅
- Time: 2.0s | Speed: 0 tok/s
- Response: WAV generated via Windows SAPI

**Video Analysis** ➖
- Not supported for this model/task combination.

### DeepSeek-r1 1.5B

**Text Summarization** ✅
- Time: 29.1s | Speed: 12.87 tok/s
- Response: Artificial intelligence is transforming industries by enabling machines to learn from data with minimal human intervention

**Synonym Enhancement** ✅
- Time: 17.3s | Speed: 12.5 tok/s
- Response: 

**General Chat** ✅
- Time: 15.2s | Speed: 14.21 tok/s
- Response: 

**Image Analysis** ➖
- Not supported for this model/task combination.

**Audio / STT** ➖
- Not supported for this model/task combination.

**TTS Output** ✅
- Time: 2.0s | Speed: 0 tok/s
- Response: WAV generated via Windows SAPI

**Video Analysis** ➖
- Not supported for this model/task combination.

### Gemma4 E4B

**Text Summarization** ❌
- Error: timed out

**Synonym Enhancement** ❌
- Error: timed out

**General Chat** ❌
- Error: timed out

**Image Analysis** ✅
- Time: 313.9s | Speed: 0.0 tok/s
- Response: 

**Audio / STT** ❌
- Error: timed out

**TTS Output** ✅
- Time: 2.0s | Speed: 0 tok/s
- Response: WAV generated via Windows SAPI

**Video Analysis** ➖
- Not supported for this model/task combination.

## Speed Comparison (tok/s)

| Model | Summarize | Synonym | Chat | Avg tok/s |
|-------|-----------|---------|------|-----------|
| Gemma3 1B | 21.35 | 21.27 | 21.15 | **21.26** |
| DeepSeek-r1 1.5B | 12.87 | 12.5 | 14.21 | **13.19** |
| Gemma4 E4B | 0 | 0 | 0 | **0** |

## Optimization Notes

### Flash-MoE Expert Routing (Gemma4 E4B)

Gemma4 E4B uses a Mixture-of-Experts (MoE) architecture. Only a subset of expert
FFN layers activate per token. By setting `n_gpu_layers=12`, attention layers stay
in GPU VRAM (fast) while expert weight tensors reside in CPU RAM. Combined with
`use_mmap=True`, only the *active* expert pages are loaded from disk — mimicking
sparse Flash-MoE behavior without custom kernels.

### MLX-style No-Swap Architecture

| Technique | Effect |
|-----------|--------|
| `use_mmap=True` | GGUF mapped to virtual address space; unused expert pages stay on disk |
| `use_mlock=True` | Active model pages pinned in RAM; prevents eviction to swap |
| `n_ctx=1024` | KV cache ~250MB vs ~1GB at 8192; frees VRAM for more GPU layers |
| `type_k/v=q8_0` | KV cache quantized to 8-bit; further reduces VRAM pressure |
| Ollama affinity [2,3] | Ollama locked to cores 2+3 via `psutil`; cores 0+1 free for OS |
| Python affinity [0,1] | Benchmark process on cores 0+1; no CPU contention with Ollama |
| `num_thread=2` in Modelfile | llama.cpp worker cap; was 8 which caused 100% on all 4 cores |

### Q4_K_M Quantization (already applied)

All models are already quantized to 4-bit (Q4_K_M) in their GGUF blobs. This means:
- Weight precision: 4 bits per parameter
- Quality retention: ~99% vs FP16 for most benchmarks
- Memory savings: ~4× vs FP32 baseline

Further quantization options (not applied, require `llama-quantize`):
- Q3_K_S: ~7GB, +15% faster, minor quality loss
- Q2_K: ~5GB, +35% faster, noticeable quality loss

## Hardware & Environment

```
CPU:   Intel Core i5-4460 (4 cores, 3.2GHz base)
GPU:   NVIDIA GeForce GT 1030 (2GB GDDR5, sm_61 Pascal)
RAM:   34GB DDR3
OS:    Windows 10 Pro 10.0.19045
CUDA:  11.8

Ollama: v0.20.0 (fallback path)
llama-cpp-python: direct GGUF path (preferred)
Model store: E:\LLMmodel\blobs (Ollama blob format = raw GGUF)
```
