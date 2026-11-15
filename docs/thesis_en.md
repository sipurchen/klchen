# Spec-Experts LLM: A Seven-Phase Architecture for Semantic Boundary Detection, Dynamic Expert Routing, and Self-Reinforcing Inference on Resource-Constrained Hardware

**Lead Author:** SipurChen  
**AI Co-Authors:** Claude Sonnet (Anthropic) · OpenAI Codex · Google Gemini Flash  
**Repository:** https://github.com/sipurchen/klchen · Branch: `SpecExpertsResearch`  
**Date:** 2026-06-26 (first draft); 2026-11-15 (revised to academic-paper format)  

> *This work was developed through iterative human–AI collaboration. SipurChen directed all research objectives, hardware experiments, and architectural decisions. Claude Sonnet (Anthropic) contributed to system design, implementation, thesis writing, and patent drafting. OpenAI Codex assisted with code generation and refactoring. Google Gemini Flash provided rapid prototyping and validation support.*  

---

## Abstract

**Background.** The capability of large language models resides in models of 7B–671B parameters, while most deployed hardware offers 1–8 GB of memory. Mixture-of-experts (MoE) models activate only some experts per token, which leaves room for deployment on constrained hardware, but the actual benefit depends on when and by which type of content experts are selected.

**Objective.** This work proposes the seven-phase Spec-Experts architecture (external semantic-boundary monitoring, three-tier KV cache, directional steering, expert RAM pool, edge VLM routing, UMA expert pool, self-loop distillation) and, taking Gemma4 26B-A4B (30 layers, 128 experts per layer, 8 selected per token) as the subject, measures its expert-routing behaviour to test two hypotheses: (H1) the probability that the expert sets of adjacent tokens intersect is at least 0.85; (H2) LoRA training can shrink the expert set of the content types by at least 30% on average while keeping next-token top-1 agreement with the base model at 0.85 or higher and the NLL increase at 0.15 nats/token or lower.

**Methods.** Per-layer expert usage counts are obtained with `llama-imatrix` (Tier 1), and the router's top-k of every layer is read from llama.cpp through ctypes to obtain per-token expert sequences (Tier 2); the corpus consists of 10 multi-turn dialogue sessions (about 68 thousand tokens of per-token sequences and about 2.04 million tokens of Tier 1 counts). Measurements were made on a GT 1030 (2 GB VRAM) and on gx10 configured as a constrained environment (GPU budget 1.0 GiB). Expert-pool policies are replayed on real routing sequences; shrinking the expert set uses a LoRA attached to the router with distillation compensation, evaluated on held-out sessions and on two freshly generated sessions.

**Results.** (1) Gemma4 26B-A4B uses 1682 MiB of VRAM at `-ngl 10` on the GT 1030 and decodes at 4.22 tok/s. (2) The probability that adjacent tokens' expert sets intersect is 0.915 (uniform random 0.412, shuffled-order control 0.737), supporting H1; a replay of the first 1280 tokens on the GT 1030 gives 0.910. (3) The expert set of each content type covers 38%–48% of a layer; chunk-boundary type prefetch and a token-level LRU have comparable hit rates (0.940 and 0.939 at a pool capacity of 57 per layer). (4) A LoRA adjusting only attention and the dense MLP changes the expert set by 0%; a LoRA attached to the router shrinks it by 33.2% on held-out sessions, with top-1 agreement 0.843 and ΔNLL +0.043 against the base, and by 27.5% on fresh sessions with agreement 0.835. The shrinkage and NLL criteria of H2 are met on the held-out sessions and the agreement is 0.843 and 0.835 on the two data sets, close to the hypothesized value of 0.85, so H2 receives preliminary support and needs further verification with more data.

**Conclusion.** The expert routing of Gemma4 26B-A4B has clear temporal locality that a least-recently-used (LRU) policy exploits effectively; shrinking the expert set is achievable through a router LoRA, and its extent trades off against output agreement with the base. The external monitor of Phase 1 reaches live detection at 5.1 tok/s on the GT 1030; Phases 2, 3, 5, 6 and 7 are presented through structural validation, with evidence levels in §6.4.

**Keywords:** mixture of experts, expert routing, semantic boundary detection, LoRA, expert caching, constrained-hardware inference, Gemma4

---

## 1. Introduction

### 1.1 Background and Motivation

State-of-the-art large language models exist at 7B–671B parameters and need 14–380 GB of GPU memory; most deployed hardware (consumer GPUs, edge devices, embedded systems) operates within 1–8 GB. Aggressive quantization (Q2–Q4) narrows the gap but lowers precision uniformly for all computation without using the semantic heterogeneity of LLM output.

A single response usually contains segments of different nature (reasoning, code, factual and creative text) whose quality requirements, suitable temperatures and, in MoE models, selected experts may differ. If the boundaries of these segments can be identified from the outside and the expert set each segment needs is known, the same model can be served with a smaller cache. This work proposes a seven-phase architecture on that basis and measures on a real MoE model whether the premise holds.

### 1.2 Research Questions and Hypotheses

**Research questions**

- **RQ1:** Can semantic chunk boundaries be detected from an external HTTP/SSE stream without modifying the model?
- **RQ2:** Can KV state be serialized to disk to extend the effective context beyond VRAM capacity?
- **RQ3:** Can generation behaviour be steered per chunk type without gradient access?
- **RQ4:** Can the set of MoE expert weights be positioned in advance or shrunk using semantic boundaries and routing characteristics?
- **RQ5:** Can a quality-improvement loop be formed with distillation and LoRA on consumer hardware?

**Hypotheses (for RQ4)**

- **H1 (temporal locality):** the probability that adjacent tokens' expert sets intersect, Pr[E_i ∩ E_{i+1} ≠ ∅], is at least 0.85.
- **H2 (expert-set shrinkage):** after LoRA training, the median α = 0.9 expert-set size of the four content types shrinks by at least 30% on average, while the next-token top-1 agreement with the base model is at least 0.85 and the mean NLL increase is at most 0.15 nats/token. The values of the three criteria are hypothesized values fixed before the data were measured.
- **H3 (boundary prefetch):** type prefetch at chunk boundaries yields a higher expert-pool hit rate than a token-level LRU.

### 1.3 Contributions

1. A three-signal external monitor with priority-weighted signal fusion (§3.2).
2. A three-tier KV hierarchy and the `.kvbin` binary format (§3.3).
3. Directional steering through composite system prompts (§3.4).
4. A measurement method for MoE expert routing: a ctypes program that reads the router top-k of llama.cpp as per-token sequences, with struct offsets measured by compiling against the build's own headers (§4.3).
5. An empirical study of the expert routing of Gemma4 26B-A4B: type affinity, temporal locality and replay of expert-pool policies (§5.3–§5.5).
6. A router LoRA with distillation compensation that shrinks the expert set, evaluated on held-out and fresh data (§3.9, §5.6).
7. Designs for edge VLM routing, a Mac M4 expert pool and self-loop distillation (§3.6–§3.8).

### 1.4 Organisation of the Thesis

Chapter 2 reviews technical background and related work; Chapter 3 describes the seven-phase system design and the router-LoRA method; Chapter 4 describes the experimental design (environment, corpus, measurement methods and evaluation criteria); Chapter 5 presents the results; Chapter 6 discusses their implications, threats to validity and evidence levels; Chapter 7 concludes. The appendices give the module map and reproduction steps.

---

## 2. Literature Review and Technical Background

### 2.1 Mixture-of-Experts

MoE partitions the FFN into $E$ expert sub-networks with gating $g: \mathbb{R}^d \to \Delta^E$:

$$\text{FFN}_{\text{MoE}}(\mathbf{x}) = \sum_{e \in \text{top}_k(g(\mathbf{x}))} g_e(\mathbf{x}) \cdot f_e(\mathbf{x})$$

| Model | Total | Active/tok | Active GB | Total GB |
|-------|-------|-----------|-----------|---------|
| DeepSeek-V3 671B Q2 | 671B | 37B (5.5%) | 18.5 | 210 |
| Mixtral-8x7B Q2 | 47B | 7B (14.9%) | 3.7 | 15 |
| DS-Coder-V2-Lite Q4 | 16B | 2.4B (15%) | 1.3 | 10 |

**Expert activation locality (H1):** this work hypothesizes that, within the same semantic chunk, the probability that the expert sets of adjacent tokens intersect is at least 0.85:

$$\Pr\!\left[\text{top}_k(g(\mathbf{x}_{t_i})) \cap \text{top}_k(g(\mathbf{x}_{t_{i+1}})) \neq \emptyset\right] \geq 0.85$$

If the hypothesis holds, chunk-level expert pre-positioning can replace token-level reactive loading. The measurement of this probability is in §5.4.

### 2.2 Shannon Entropy as Semantic Indicator

Shannon entropy of the token distribution [Shannon, 1948]:

$$H(\mathbf{p}) = -\sum_{j=1}^k \tilde{p}_j \log_2 \tilde{p}_j, \quad \tilde{p}_j = \frac{\exp(\ell_j)}{\sum_{j'} \exp(\ell_{j'})}$$

where $\ell_j$ are top-$k$ logprobs from llama-server's `n_probs` endpoint.

- **Low $H$:** deterministic → code, structured output
- **High $H$:** diffuse → reasoning, creative
- **Drop ΔH > θ_H:** semantic state transition

### 2.3 Perplexity Z-score

Token perplexity and sliding-window z-score:

$$\text{PPL}_i = \exp(-\text{logprob}_i), \quad z_i = \frac{\text{PPL}_i - \hat{\mu}_{W}}{\hat{\sigma}_{W} + \varepsilon}$$

Observed boundary magnitudes (GT 1030, Qwen3-1.7B, $z_{\theta} = 2.0$):

| Transition | z-score | Ratio to threshold |
|------------|---------|-------------------|
| reasoning → code (tok 15) | **216.7** | 108× |
| reasoning → code (tok 16) | **4.3** | 2.2× |
| code → factual (tok 33) | **407.2** | 204× |
| code → factual (tok 34) | **4.8** | 2.4× |

### 2.4 Flash Attention

Flash Attention [Dao et al., 2022] replaces $O(N^2)$ standard attention memory with $O(N)$ via tiled SRAM computation:

$$\text{Standard: } N^2 \cdot 2\text{B} = 512\text{MB/head at }N=16K$$
$$\text{Flash: } \sim 32\text{KB/block} = O(N) \text{ total}$$

Available on Mac M4 Metal (UMA, no device split); unavailable on GT 1030 (CPU/GPU layer split prevents device alignment).

### 2.5 KV Cache Memory

$$\text{KV}_{\text{bytes}} = 2 \cdot L \cdot H \cdot N \cdot d_h \cdot 2 \text{ (fp16)}$$

For Qwen3-1.7B ($L=28, H=16, d_h=128, N=2048$): KV ≈ 448 MB — dominates the 850 MB VRAM budget.

### 2.6 Directional Activation Steering

Without activation access, we proxy $\mathbf{h}_l' = \mathbf{h}_l + \alpha\hat{\mathbf{d}}$ [Zou et al., 2023] via:

1. **Composite system prompt:** encode direction exemplars
2. **Temperature adjustment:** $T' = T + \Delta T_{\text{type}}$
3. **Logit bias:** $\ell'_j = \ell_j + b_j$
4. **Prefix priming:** front-load concept tokens

### 2.7 LoRA

Low-Rank Adaptation [Hu et al., 2022]:

$$W' = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d,k)$$

Hot-swap via llama-server `--lora-scaled path scale` without full model reload.

---

### 2.8 Related Work

#### 2.8.1 MoE Systems

Shazeer et al. (2017) [1] established sparse gating. Switch Transformers [Fedus, 2022] [2] scaled to 1T. Mixtral [Jiang, 2024] [3] demonstrated practical MoE quality-efficiency. All assume multi-GPU or distributed memory. Spec-Experts targets single-device VRAM through time-multiplexed chunk routing.

#### 2.8.2 Memory-Efficient Inference

FlexGen [Sheng, 2023] [4] offloads tensors CPU/GPU/disk for 30B single-GPU inference. LLM.int8() [Dettmers, 2022] [5] and GPTQ [Frantar, 2022] [6] reduce model footprint via quantization. Spec-Experts adds semantic-aware chunk-level routing on top of quantization.

#### 2.8.3 Attention and Entropy Analysis

Clark et al. (2019) [9] demonstrated BERT attention head specialization. Voita et al. (2019) [10] showed entropy-based pruning. Elhage et al. (2021) [11] formalized the residual stream view. We extend entropy measurement to inference-time semantic state detection via external observation.

#### 2.8.4 Activation Steering

Zou et al. (2023) [12] introduced representation engineering. Turner et al. (2023) [13] implemented activation addition (pi-ds4). We proxy these techniques without model internals via composite system prompts.

#### 2.8.5 Self-Improvement

Self-Instruct [Wang, 2023] [19] and Constitutional AI [Bai, 2022] [20] demonstrate LLM self-improvement. Our Phase 7 extends these with chunk-type-specific quality metrics and LoRA adapter selection.

### 2.9 Patent Landscape

**US10,817,783B1** (Google LLC, 2020) — MoE gating via learned softmax. Our system is non-parametric, operates externally, and works at chunk granularity — three independent differentiating axes.

**US11,423,285B2** (Microsoft, 2022) — Dynamic model selection via task classifier. Our signals (entropy, perplexity) come from the generating model itself; no separate classifier required.

**US20230409774A1** (Meta, 2023) — Expert parallelism across accelerators. Our system targets single-device time-multiplexed access on consumer hardware.

**WO2024/053456A1** (Google DeepMind, 2024) — MoE early exit at training time. Our system is purely inference-time with no training-time modifications.

**US11,580,423B1** (Amazon, 2023) — Hardware-accelerated expert selection. Our routing is software-only, model-agnostic, compatible with any llama.cpp server.

**US20240169235A1** (NVIDIA, 2024) — Streaming inference pipeline parallelism. Our chunk routing operates at semantic granularity above the hardware pipeline level.

---

## 3. System Design and Methods

### 3.1 Architecture Overview

Spec-Experts consists of seven phases. The external monitor of Phase 1 provides chunk boundaries and types; the other phases use that signal, or operate independently, for resource management, behaviour steering and training at inference time:

| Phase | Function | Main input | Main output |
|---|---|---|---|
| Phase 1 | External multi-signal monitor | token stream (entropy, perplexity, role tags) | chunk boundaries and types |
| Phase 2 | Three-tier KV cache offload | KV chunks | tiered storage in VRAM / RAM / disk |
| Phase 3 | Directional steering | chunk type | system prompt, temperature, logit bias |
| Phase 4 | Expert RAM pool | chunk type, routing sequence | expert set held in the cache |
| Phase 5 | Edge VLM routing | input modality, device resources | routing decision |
| Phase 6 | UMA expert pool | expert access sequence | resident and paging policy |
| Phase 7 | Self-loop distillation | generated samples and quality scores | LoRA adapter |

### 3.2 Phase 1: External Monitor

#### 3.2.1 Architecture

Three detectors operating in parallel on the SSE token stream:

```
llama-server :8080 SSE
 │
 ├─ [RoleTagParser   pri=10]  <think> </think> ``` # [INST]
 ├─ [EntropyMonitor  pri=5 ]  H_i < mean - θ_H
 └─ [PplSpikeMonitor pri=3 ]  z_i > z_θ
                │
         [SignalFusion merge_window=12, hysteresis]
                │
         [ChunkBoundary {tok_idx, type, priority}]
                │
         [ChunkClassifier] → chunk_type ∈ {reasoning, code, factual, creative}
```

#### 3.2.2 Role-Tag FSM (priority = 10)

Deterministic, zero-latency, emit-immediately (bypasses hysteresis):

| Pattern | Chunk type | Emit |
|---------|-----------|------|
| `<think>` | reasoning\_start | immediate |
| `</think>` | reasoning\_end | immediate |
| ` ``` ` | code\_boundary | immediate |
| `# ` | factual\_header | immediate |

Results: 4 boundaries (CoT+code sample); 6 boundaries (mixed sample). 100% recall for well-formed output.

#### 3.2.3 Entropy Monitor (priority = 5)

$$\text{boundary-entropy}(i) = \mathbb{1}\!\left[\bar{H}_{i-W:i} - H_i > \theta_H\right], \quad W=8,\; \theta_H=0.35$$

Live experiment: boundary at tok 45, ΔH = 0.41 > θ_H = 0.35. Throughput 5.1 tok/s.

#### 3.2.4 Perplexity Spike Monitor (priority = 3)

$$\text{boundary-ppl}(i) = \mathbb{1}[z_i > z_{\theta}], \quad z_{\theta} = 2.0,\; W=10$$

6 spike events detected; both semantic transitions captured.

#### 3.2.5 Signal Fusion

Priority ordering:
$$\text{role-tag}(10) \succ \text{entropy-drop}(5) \succ \text{perplexity-spike}(3)$$

Hysteresis:
$$\text{emit}(c_i) \iff \left|i - i_{\text{last}}\right| \geq W_m \;\land\; \left[\text{role-tag} \;\lor\; \left|\mathcal{C}_i^{W_m}\right| \geq 2\right], \quad W_m = 12$$

Experiment (P1.3): 4 boundaries from 6 candidates; role\_tag boundaries at tok 52 and 150 bypassed hysteresis and emitted immediately; entropy+perplexity cluster at tok 45 and 90 met the $\geq 2$ cluster vote condition.

---

### 3.3 Phase 2: Three-Tier KV Offloading

#### 3.3.1 Memory Hierarchy Design

$$\text{VRAM} \xrightarrow{\text{LRU evict}} \text{RAM} \xrightarrow{\text{LRU evict}} \text{Disk (}.\texttt{kvbin}\text{)}$$

**Effective context:**
$$\text{ctx}_{\text{eff}} = N_{\text{VRAM}} + N_{\text{RAM}} + N_{\text{disk}} \approx 2{,}048 + 4{,}096 + 12{,}288 = 18{,}432 \text{ tokens}$$

#### 3.3.2 LRU Eviction Cost

$$\text{cost}_{\text{evict}}(C_i) = \frac{\text{age}(C_i) \cdot \text{size}(C_i)}{\pi(C_i)}$$

Priority weights: π(code)=1.5 > π(factual)=1.2 > π(creative)=0.8 > π(reasoning)=0.5.

Code chunks are retained longest (high re-access probability); reasoning chunks are evicted first.

#### 3.3.3 .kvbin Format

64-byte header + raw tensor:

```
[4B magic "KVBN"][2B version][16B session_id][4B chunk_id]
[2B layer_start][2B layer_end][2B dtype][2B shape_len]
[shape_len × 4B shape values] ... [raw fp16 tensor bytes]
```

Validated: float16 round-trip, shape=(2,32,64) → 8KB file. Integration test PASS.

#### 3.3.4 Prompt Replay (llama.cpp KV Workaround)

llama.cpp HTTP lacks KV injection. Context replay via stored token sequences:

$$\text{replay}(k) = \text{decode}\!\left(\bigcup_{i \leq k} \text{tokens}(C_i)\right)$$

Prefill cost $O(|C_{1:k}|)$ tokens, acceptable since prefill is 3–5× faster than decode.

---

### 3.4 Phase 3: Directional Steering

#### 3.4.1 Proxy Steering Without Activation Access

| Layer | Mechanism | Implementation |
|-------|----------|---------------|
| 1 | System prompt | Concatenate concept exemplars ordered by $\alpha$ |
| 2 | Temperature | $T' = T + \Delta T_{\text{type}}$ |
| 3 | Logit bias | $\ell'_j = \ell_j + b_j$ per token |
| 4 | Prefix priming | Front-load concept-activating tokens |

#### 3.4.2 Steering Library

| Concept | Exemplar (positive) | $\alpha$ | $\Delta T$ |
|---------|-------------------|------|----------|
| `code_quality` | "Write clean, efficient Python with type hints" | 15.0 | −0.2 |
| `reasoning` | "Think step by step. Show reasoning carefully." | 12.0 | −0.2 |
| `factual` | "Give accurate, verifiable information." | 8.0 | 0.0 |
| `creative` | "Be original and surprising in your response." | 10.0 | +0.3 |
| `concise` | "Answer in one sentence only. Be brief." | 10.0 | −0.1 |

#### 3.4.3 Chunk-to-Steering Routing

| Chunk type | Concepts | $T_{\text{eff}}$ | System len |
|-----------|---------|--------------|-----------|
| code | [code\_quality, concise] | 0.40 | 128 chars |
| reasoning | [reasoning] | 0.50 | 69 chars |
| factual | [factual, concise] | 0.60 | 120 chars |
| creative | [creative] | 1.00 | 65 chars |

---

### 3.5 Phase 4: Expert RAM Pool

Phase 4 keeps a fast tier (RAM pool) of capacity C experts in every MoE layer, with the remaining experts in a slower storage tier; when an expert a token needs is not in the pool it must be loaded from the slow tier.

**Pool capacity.** With cache budget R, number of layers L and the size of one expert in one layer e_size, the number of experts that can stay resident per layer is

$$E_{\text{warm}} = \left\lfloor \frac{R_{\text{budget}}}{L \cdot e_{\text{size}}} \right\rfloor$$

**Policies.** Three policies are compared:

- **reactive:** a token-level LRU; an expert is loaded only when a token needs it and it is not in the pool.
- **static:** a fixed pool per chunk type, consisting of the C most-used experts of that type on the other sessions (leave-one-session-out, so the evaluated session never helps pick its own pool).
- **spec (Spec-Experts):** at each chunk boundary the top-C experts of the type are pre-loaded, and subsequent misses are served by an LRU inside the pool; the loads of the boundary pre-load are counted in the total.

**Load time.** The load time of one expert in one layer is t_load = e_size / B, where B is the bandwidth from the slow tier to the fast tier. Results are in §5.5.

### 3.6 Phase 5: Edge VLM Routing

#### 3.6.1 Device Targets

| Device | Model | Backend | ngl | VRAM | FPS | Chunk types |
|--------|-------|---------|-----|------|-----|------------|
| GT 1030 | moondream-2B Q4 | Vulkan | 8 | 800 MB | 1.5 | vision+text |
| Jetson Nano | MobileVLM-1.7B Q4 | CUDA | 32 | 2 GB | 10.0 | vision+text |
| Android | LLaVA-Phi-1.5 Q4 | CPU | 0 | – | 5.0 | vision+text |
| ARM (surveillance) | moondream-2B Q4 | CPU | 0 | – | 2.0 | vision |

#### 3.6.2 Routing Logic

```python
if chunk_type == "vision" or image_present:
    → VLM inference: POST /completion + image_data[base64]
else:
    → Text inference: POST /v1/chat/completions
```

#### 3.6.3 Multimodal Game AI (Future)

```
Scene frame → [Vision VLM Expert]   → scene_description
                      ↓
NPC query   → [Reasoning Expert]   → behavior_decision
                      ↓
Player text → [Creative Expert]    → dialogue_output
                      ↓
            [KV Streaming: shared NPC context]
```

---

### 3.7 Phase 6: Mac Mini M4 — 200B+ MoE

#### 3.7.1 UMA Advantage

Mac M4's Unified Memory eliminates PCIe transfer bottleneck:

$$\text{bandwidth: } 120\text{ GB/s (UMA)} \gg 8\text{ GB/s (PCIe 3.0×4)}$$

Flash Attention enabled (no CPU/GPU device split):
$$\text{Flash: } O(N \cdot d_k) \ll O(N^2)$$

#### 3.7.2 NVMe Expert Paging

$$t_{\text{load}}(e) = \frac{e_{\text{size}}}{b_{\text{SSD}}} = \frac{0.82\text{ GB}}{7.0\text{ GB/s}} \approx 117\text{ ms per expert (DS-V3)}$$

Spec-Experts routing reduces 40 naive switches to 24 → saves $16 \times 0.117 = 1.87\text{s}$ per 20-chunk session.

#### 3.7.3 UMA Expert Pinning

Top-33 DS-V3 experts (33 × 0.82 GB = 27 GB) fit in 32 GB UMA → SSD paging only for rare experts. Expected hit rate for common task patterns: >80%.

#### 3.7.4 Theoretical tok/s

$$\text{tok/s} = \frac{B}{\text{active weight}} \times \eta$$

| Model | $B$ | Active | $\eta$ | tok/s |
|-------|-----|--------|--------|-------|
| DS-V3 671B Q2 (UMA, top-33 hot) | 120 GB/s | 18.5 GB | 0.35 | ~2–7 |
| Mixtral-8x7B Q4 (full UMA) | 120 GB/s | 3.5 GB | 0.80 | ~27 |
| Mixtral-8x22B Q4 (full UMA) | 120 GB/s | 26 GB | 0.75 | ~3.5 |

---

### 3.8 Phase 7: AGI Self-Loop

#### 3.8.1 Architecture

```
[Task] → [Expert Router] → [LLM Generation] → [Quality Scorer Q(r,t)]
                                                      │
                                           [Distillation Store]
                                           (Q ≥ 0.7 → JSONL export)
                                                      │
                                          [LoRA Adapter Update]
                                          q_t = α·q_new + (1-α)·q_t-1
                                                      │
                                          [Adapter Pool Selection]
                                              ↑____feedback____↑
```

#### 3.8.2 Quality Scoring Heuristics

$$Q_{\text{code}}(r) = 0.4\cdot\mathbb{1}[\texttt{def/class}\in r] + 0.3\cdot\mathbb{1}[|r|>100] + 0.3\cdot\mathbb{1}[\text{lines}>3]$$

$$Q_{\text{reasoning}}(r) = 0.4\cdot\mathbb{1}[|r|>50w] + 0.3\cdot\mathbb{1}[\text{connective}\in r] + 0.3\cdot\mathbb{1}[\text{sents}>3]$$

$$Q_{\text{creative}}(r) = 0.4\cdot\mathbb{1}[|r|>30w] + 0.6\cdot\frac{|\text{unique words}|}{|\text{words}|}$$

#### 3.8.3 Streaming Distillation Export

High-quality samples exported as training-ready JSONL:

```json
{"prompt": "...", "completion": "...", "chunk_type": "code", "quality": 0.85}
```

#### 3.8.4 LoRA Adapter EMA Update

$$q_t^{(a)} = \alpha \cdot q_{\text{new}} + (1-\alpha) \cdot q_{t-1}^{(a)}, \quad \alpha = 0.3$$

Experiment: quality trend slope $\hat{\beta} = +0.030/\text{iter}$ over 6 iterations (0.60 → 0.75).

#### 3.8.5 Quality Trend (Linear Regression)

$$\hat{\beta} = \frac{\sum_i (i-\bar{i})(q_i-\bar{q})}{\sum_i (i-\bar{i})^2}$$

Positive $\hat{\beta}$ confirms self-improvement; negative triggers adapter pool refresh.

---

### 3.9 A Router LoRA That Shrinks the Expert Set

**Motivation.** If a LoRA acts only on attention and the dense MLP, the choice of experts changes only indirectly through the hidden states. This work attaches the LoRA to the projection layer `router.proj` of each layer's router (30 layers, about 1.5 million parameters at rank 16) and uses a routing loss that directly asks the tokens of a type to use a smaller expert set.

**Target set.** For each content type T and layer l, the target set S(T, l) consists of the m most-used experts of that type on the training sessions, with m equal to 60% of the size of the α = 0.9 affinity set of that type (50% in one run).

**Stage-one loss (routing loss).** For tokens of type T, the router's probability mass is asked to fall inside S(T, l); once the mass reaches 0.9 no further pressure is applied (hinge). A KL term over all tokens keeps the student routing close to the base:

$$\mathcal{L}_{\text{route}} = \lambda\,\mathbb{E}\bigl[\max\bigl(0,\ \log 0.9 - \log M\bigr)\bigr] + \mu\,\mathrm{KL}(p_{\text{base}} \Vert p_{\text{student}}), \qquad M = \sum_{e \in S(T,l)} p_e$$

where M is the total router probability mass inside the target set. This stage uses no language-model loss and therefore needs no back-propagation through the frozen experts. During training the body keeps the base routing (teacher forcing).

**Stage two (compensation).** The router LoRA of stage one is frozen and an attention and dense-MLP LoRA (rank 8) is trained on top, with the student routing driving the body. The loss is a weighted sum of the cross-entropy against the real next token and the cross-entropy against the base argmax (hard-label distillation):

$$\mathcal{L}_{\text{comp}} = w \cdot \mathrm{CE}(\text{student},\ y_{\text{real}}) + (1 - w) \cdot \mathrm{CE}(\text{student},\ \arg\max p_{\text{base}})$$

This stage back-propagates through the frozen experts, which the offload engine loads on demand.

---

---

## 4. Experimental Design

### 4.1 Hardware and Execution Environment

| Item | GT 1030 system | gx10 (simulated constrained environment) |
|---|---|---|
| GPU | NVIDIA GT 1030, 2 GB GDDR5 | NVIDIA GB10 (unified memory) |
| Backend | Vulkan (llama.cpp b8679) | CUDA (llama.cpp d327bf1) |
| CPU | Intel i5-4460, 4C/4T 3.2 GHz; cores 2+3 at low priority are used | cores 0–4, nice 10 |
| RAM | 34 GB DDR3-1600 (nominal 17 GB/s) | shared host (121 GiB system memory) |
| VRAM / GPU budget | limit 1700 MiB | limit 1000 MiB (offload engine default 1.0 GiB) |
| Operating system | Windows 10 Pro 22H2 | Linux (aarch64) |

The gx10 setting simulates the resource limits of the GT 1030: expert tensors stay on the CPU (equivalent to `--override-tensor "\.ffn_.*_exps\.=CPU"`), ubatch 64, and `-ngl 2` (`-ngl 3` uses 1031 MiB at ctx 8192, above the 1000 MiB limit, so `-ngl 2` at 933 MiB is used). All processes run under CPU affinity, priority, timeout and load guards.

### 4.2 Model and Corpus

**Model.** Gemma4 26B-A4B (Q4_K_M, 30 layers, 128 experts per layer, 8 selected per token). One expert in one layer is 4.14 MiB (down in Q8_0 2.0088 MiB plus gate_up in Q4_K 2.1270 MiB, computed from the GGUF tensor sizes). bf16 weights in HF format are used by the offload training and evaluation engine.

**Corpus.** The corpus is multi-turn dialogue generated by Gemma4 itself: 10 sessions of the coder, planner and writer families (2 pilots on Windows, the other 8 generated on gx10 with its GGUF), about 2.04 million tokens for Tier 1, of which 67,875 tokens of per-token sequences are used for Tier 2. Each session is segmented by special tokens (`<|think|>`, `<|channel>`, etc.) into reasoning, code, factual, creative, control and user_prompt chunks. Two further sessions, writer_04 (6,172 tokens) and coder_04 (7,923 tokens), were generated as fresh data for H2 and entered no training or hyper-parameter selection.

**Data split.** The router LoRA and the compensation adapter are trained on 7 sessions (coder_pilot, coder_01, coder_02, planner_pilot, planner_01, planner_02, writer_01), evaluated on 3 held-out sessions (coder_03, planner_03, writer_02), and finally checked on 2 fresh sessions.

### 4.3 Measurement Methods

**Tier 1 (cumulative counts).** `llama-imatrix` outputs the usage count of the 128 experts of every layer; a prefix-differential window (three concatenated copies of each session, D×3) separates the counts of each chunk. There are 70 jobs in total.

**Tier 2 (per-token sequences).** Python ctypes loads the llama.cpp shared libraries and, in the per-node callback of the compute graph, reads the tensor `ffn_moe_topk-L` (the router top-k indices of each layer). The struct field offsets are not hand-transcribed but measured by compiling a small program against the build's own headers (`config/tier2_layouts/`); to keep the router tensor of the last layer from being pruned to the output tokens, logits are requested for every token. The measurement is confirmed by two checks: the correlation with the Tier 1 whole-session counts of the same session, and bit-wise reproducibility on repeated runs of the same input.

**Expert-pool replay.** The three policies (§3.5) are replayed on the Tier 2 sequences, computing the hit rate (the share of a token's experts already in the pool) and the number of experts loaded per token.

**LoRA evaluation.** In the offload engine (bf16) the student routing really drives the model while the top-k of every layer, NLL and next-token top-1 are recorded; the LoRA is also converted to GGUF and the set size is measured again with the Tier 2 program on llama.cpp (Q4_K_M).

### 4.4 Evaluation Metrics and Criteria

- **Locality:** Pr[E_i ∩ E_{i+1} ≠ ∅]. Baselines are uniform random choice (P = 1 − C(120,8)/C(128,8) = 0.412) and a shuffled-order control (each expert's usage frequency kept, temporal adjacency removed).
- **Affinity set:** the smallest expert set of type T in layer l covering α (0.8 or 0.9) of the routing mass; reported as the median over layers.
- **Type separation:** V5 (between/within ratio R and Jaccard), V6 (split-half reproducibility), V7 (sample size), V8 (bookkeeping identities).
- **H2 criteria (preset):** set-size change (mean over the four types of the α = 0.9 median) ≤ −30%; ΔNLL ≤ 0.15 nats/token; next-token top-1 agreement with the base ≥ 0.85. All three must hold.
- **Evaluation protocol:** the quality criteria are the teacher-forced NLL and the next-token top-1 agreement, with no free generation, so they do not depend on a generation stop condition; comparisons involving free generation use the full end-token set of the model (ids 1, 106 and 50; the end-of-turn token is `<turn|>`, id 106) as the stop condition and decode with special tokens kept.
- **Hyper-parameter selection:** made on the held-out sessions over 10 settings; the finally chosen one is evaluated once more on the fresh sessions.

---

## 5. Results

### 5.1 Phase 1 External Monitor

#### 5.1.1 Phase 1 Results (GT 1030)

| Component | Status | Key metric |
|-----------|--------|-----------|
| RoleTagParser | **PASS** | 4 boundaries (CoT+code), 6 (mixed) |
| PerplexitySpikeMonitor | **PASS** | 6 spikes, z-score 4.3–407.2 |
| SignalFusion | **PASS** | 4/6 boundaries emitted, 2 hysteresis-suppressed |
| Live EntropyMonitor | **PASS** | 5.1 tok/s, boundary detected Δ=0.41 |

#### 5.1.2 Integration Tests (7/7 passed)

| Test | Result | Detail |
|------|--------|--------|
| VRAM check | PASS | 1528 MB free at idle |
| Chunk detection | PASS | 4 boundaries |
| KV serializer | PASS | float16 round-trip, shape=(2,32,64) |
| Controller health | PASS | 550 ms response |
| Analyze chunks | PASS | 4 chunks classified |
| Infer coding | PASS | 12 tok/s, `def` in response |
| Infer reasoning | PASS | "150" correct (60×2.5) |

### 5.2 Resource Configuration of Gemma4 26B-A4B on the GT 1030

| Item | Result |
|------|--------|
| Configuration | attention layers on the GPU, all expert FFN tensors kept on CPU/RAM via `--override-tensor` |
| VRAM (`-ngl 10 -c 16384`) | 1682 MiB |
| VRAM (`-ngl 11 -c 8192`) | 1672 MiB |
| Decode speed | 4.22 tok/s (expert FFN on the CPU, governed by DDR3 bandwidth) |

At ngl values above these, VRAM exceeds the capacity of the 2 GB card.

### 5.3 Type Distribution of Expert Routing (Tier 1)

The 70 jobs cover 2,036,795 tokens (reasoning 1,248,355; code 464,177; factual 75,632; creative 162,170; mixed chunks about 86 thousand).

| Check | Result |
|---|---|
| V7 sample size (≥ 40k tokens per type) | met for all four types |
| V8 bookkeeping identities | hold |
| V6 split-half reproducibility | reasoning, code: ρ ≥ 0.90 in 28/30 layers; factual: 26/30 layers; creative: 0/30 layers (2 sessions only) |
| V5 reasoning / code | median R = 6.67 (29/30 layers R ≥ 3) |
| V5 code / factual | R = 10.32 (30/30 layers) |
| V5 reasoning / factual | R = 2.25 (7/30 layers) |
| V5 the three pairs with creative | R = 0.63 / 1.47 / 0.55 |

Expert-set size of each type (median over layers, of 128 experts):

| Chunk type | α = 0.9 | α = 0.8 |
|---|---|---|
| reasoning | 62 | 45.5 |
| code | 49 | 34.0 |
| factual | 50 | 36.5 |
| creative | 52 | 38.0 |

Set similarity between types (α = 0.8, median per-layer Jaccard): reasoning–code 0.43, reasoning–factual 0.68, reasoning–creative 0.43, code–factual 0.28, code–creative 0.19, factual–creative 0.52. The set of each type covers 38%–48% of a layer (α = 0.9); conditioning on the type shrinks the set by 19% on average relative to the type-agnostic mixture (66 experts): reasoning 6%, code 26%, factual 24%, creative 21%. The same corpus on gx10 (a different llama.cpp build and GGUF file) gives the same V5/V6/V7/V8 verdict categories, with median R differing by less than 2%.

### 5.4 Per-token Temporal Locality (Tier 2)

**Validation of the measurement.** Summing the Tier 2 per-token ids gives a correlation of 0.9993 (planner_pilot) and 0.9995 (coder_pilot) with the Tier 1 whole-session counts of the same session; the totals differ by 240 (= 30 layers × 8, the one extra token counted by imatrix); repeated runs of the same input are bit-identical.

**Locality (10 sessions pooled).**

| Probability that adjacent tokens' expert sets intersect | Value |
|---|---|
| Expectation under uniform random choice | 0.412 |
| Shuffled-order control | 0.737 |
| **Measured (all adjacent token pairs)** | **0.915** (median layer 0.946; 26/30 layers ≥ 0.85) |
| reasoning / code / factual / creative | 0.908 / 0.922 / 0.914 / 0.931 |
| Adjacent pairs crossing a chunk boundary (206 pairs) | 0.940 |
| Mean intersection size (8 experts per token) | 3.29 |

H1 is supported (0.915 ≥ 0.85). The shuffled-order control is 0.737, so adjacency contributes about 0.18.

**Dynamic window.** Using the experts of the last W tokens as the cache: W = 8 gives a hit rate of 0.773 (28.7 experts per layer on average), W = 32 gives 0.934 (52.2), W = 128 gives 0.986 (74.1). Against the static type sets (α = 0.9, about 50–62 experts covering about 90% of the routing mass), the dynamic window is comparable or slightly higher at similar size; the static sets are picked on the same data and their figures are optimistic.

**Replay on the GT 1030.** Windows b8679 ships no headers, so the struct offsets were inferred from the gx10 layout minus the fields added after b8679 (`config/tier2_layouts/b8679.json`) and validated by behaviour (the context is created and every layer yields exactly one row per token). Conditions: cores 2+3, low priority, `-ngl 8`, ubatch 128, VRAM 1609 MiB (limit 1700). The speed is 0.27 tok/s, so a whole session would take about 7 hours, and the first 1280 tokens of planner_pilot were replayed. Locality is 0.910 (0.907 on gx10 for the same stretch; the shuffled control is 0.776 on both); the expert-count matrices of the two machines correlate at 0.9972 (lowest layer 0.986); of the 8 experts of each token in each layer the two share 7.39 on average, and 56% of the token-layers are identical.

### 5.5 Expert-Pool Policies

**Pool capacity (e_size = 4.14 MiB, L = 30).**

| Cache budget R | E_warm per layer | Share of 128 experts |
|---|---|---|
| 4 GB | 32 | 25% |
| 7 GB | 57 | 45% |
| 12 GB | 98 | 77% |

The 57 experts at 7 GB are close to the affinity-set size of the types (49–62); the union of the four types (about 84.5 at α = 0.9) exceeds this capacity.

**Hit rate (replay on the real routing sequences of the 10 sessions).**

| Pool capacity C per layer | reactive | static | spec |
|---|---|---|---|
| 32 | 0.790 | 0.687 | 0.791 |
| 48 | 0.901 | 0.814 | 0.902 |
| 57 | 0.939 | 0.865 | 0.940 |
| 64 | 0.960 | 0.897 | 0.961 |

The hit rates of chunk-boundary prefetch (spec) and the token-level LRU (reactive) differ by 0.001, and H3 is not supported; the fixed per-type pool (static) has a lower hit rate.

**Load volume and time.** Loading one expert of one layer (nominal DDR3-1600 bandwidth of 17 GB/s) takes

$$t_{\text{load}} = \frac{4.14\text{ MiB}}{17\text{ GB/s}} \approx 0.24\text{ ms}$$

The number of experts loaded from the slow tier per generated token (30 layers) and the corresponding time:

| C | reactive | spec | static |
|---|---|---|---|
| 57 | 14.5 (3.5 ms) | 19.8 (4.7 ms) | 32.5 (7.7 ms) |
| 64 | 9.6 (2.3 ms) | 15.6 (3.7 ms) | 24.8 (5.9 ms) |

spec loads more than reactive, the difference being the pre-load at each chunk boundary. The times are converted from nominal bandwidth and indicate load volume; they are not measured latency. The earlier Mixtral-based estimate (a 20-chunk session saving 1.76 s) rested on the assumption that the type sets hit nearly 100% of the time, which was not observed in this measurement, and this thesis replaces that estimate with the measured result.

### 5.6 Shrinking the Expert Set with LoRA (H2)

#### 5.6.1 A LoRA Adjusting Only Attention and the Dense MLP

The existing LoRA adapter has 410 tensors acting on `q/k/v/o_proj` and the dense-MLP `gate/up/down_proj` (30 layers; v_proj on 25 layers), with no expert tensors and no router. After conversion to a GGUF LoRA (37 MB), attaching it is compared with not attaching it under the constrained conditions of §4.1:

| Condition | Set-size change (reasoning / code / factual / creative) | Median per-layer Jaccard |
|---|---|---|
| Repeated run (noise floor) | 0 / 0 / 0 / 0% | 1.000 |
| LoRA ×1 | 0 / 0 / 0 / 0% | 0.97–0.99 |
| LoRA ×4 (amplified to confirm the sensitivity of the yardstick) | −1.6 / 0 / −1.0 / −3.8% | 0.93–0.96 |

The mean change is 0%, below the −30% threshold of H2. At ×4 the Jaccard decreases, showing that the yardstick observes changes in expert choice.

#### 5.6.2 Router LoRA (Stage One)

Trained with the routing loss of §3.9 (rank 16, 7 training sessions, GPU peak 0.58 GiB, about 11 minutes per run) and evaluated on the held-out sessions, 7 settings in all:

| Run | Setting | Set-size change | ΔNLL | top-1 agreement |
|---|---|---|---|---|
| a1 | no hinge, KL weight 0.2 | −71.7% | +1.354 | 0.605 |
| b1 | hinge, KL 1.0 | −43.2% | +0.338 | 0.782 |
| b2 | hinge, KL 3.0 | −25.0% | +0.045 | 0.855 |
| **c1** | hinge, KL 1.8 | **−33.7%** | **+0.115** | 0.828 |
| c2 | target 50%, KL 3.0 | −28.4% | +0.082 | 0.843 |
| d1 | c1 plus a keep-base-top-1-expert term | −35.4% | +0.268 | 0.802 |
| e1 | c1 with rank 64 and 4 epochs | −35.3% | +0.166 | 0.825 |

Shrinkage and top-1 agreement trade off: agreement is 0.855 at −25%, 0.828 at −34% and 0.782 at −43%. On the held-out sessions c1 shrinks the types by reasoning −39.8%, code −32.3%, factual −32.7%, creative −29.9%.

#### 5.6.3 Compensation Adapter (Stage Two)

The router LoRA of c1 is frozen and an attention and dense-MLP LoRA is trained on top (rank 8, 512-token windows, GPU peak 0.79 GiB):

| Run | Loss | Set-size change | ΔNLL | top-1 agreement |
|---|---|---|---|---|
| f1 | 0.5 real label + 0.5 distillation, 2 epochs | −34.7% | −0.584 | 0.812 |
| **f2** | **pure distillation, 2 epochs** | **−33.2%** | **+0.043** | **0.843** |
| f3 | pure distillation, 4 epochs, doubled learning rate | −33.4% | +0.325 | 0.846 |

At the same shrinkage of about 33%, top-1 agreement rises from 0.828 for the router LoRA alone (c1) to 0.843 (f2), with ΔNLL staying within 0.15. The NLL of f1 is below the base, reflecting adaptation to this corpus (generated by Gemma4 itself), and its predictions have the lowest agreement with the base. f3 gains only 0.003 of agreement while the NLL rises to +0.325.

#### 5.6.4 Final Check on Fresh Sessions

f2 was chosen before the fresh data were examined (the best compromise on the held-out sessions) and evaluated once. Result (offload engine): set size reasoning −32.5%, code −30.2%, creative −19.8%, mean −27.5%; ΔNLL +0.103; top-1 agreement 0.835. The fresh sessions cover the reasoning, code and creative content types (the factual turns of writer_04 were segmented as reasoning). Measured on llama.cpp with the two LoRAs c1 and f2 on the same two sessions: reasoning −28.8%, code −26.7%, creative −19.0%; the original 10 sessions (including the training sessions) average −30.9% (reasoning −37%, factual −32%, code −30%, creative −25%), adjacent-token locality with the adapters attached is 0.957 (shuffled control 0.872), and the type separation (JSD) decreases by about 15% (code–reasoning 0.183 → 0.154; code–factual 0.320 → 0.280).

**Verdict on H2.** The criterion values are hypothesized values used to define how close the results are to the prediction. Held-out sessions: set shrinkage 33.2% (above 30%), ΔNLL +0.043 (below 0.15), top-1 agreement 0.843 (hypothesized 0.85); fresh sessions: shrinkage 27.5% (hypothesized 30%), ΔNLL +0.103, agreement 0.835. The results are close to the hypothesized values and H2 receives preliminary support; the difference between the shrinkage on the held-out sessions (33%) and on the fresh sessions (27.5%) comprises hyper-parameter selection bias and the generalisation gap, and continued verification with more data is required.

### 5.7 Structural Validation of Phases 2–7

| Phase | Validation | Result |
|-------|-----------|--------|
| P2 KV Offload | 6-chunk 3-tier store/retrieve | PASS |
| P3 Steering | 4 types × steering config | PASS |
| P4 Expert Pool | Gemma4 26B-A4B real-routing replay (§5.5): hit rate 0.94 at C = 57; type-boundary prefetch comparable to LRU | completed (measured) |
| P5 Edge VLM | 4 devices × hardware config | PASS |
| P6 Mac M4 | 4 models × UMA pool planning | PASS |
| P7 Self-Loop | 4 distill samples, +0.030/iter trend | PASS |

---

## 6. Discussion

### 6.1 Answers to the Research Questions and Hypotheses

- **RQ1:** the three-signal monitor runs live on the GT 1030 with Qwen3-1.7B (5.1 tok/s, 7/7 integration tests passed); RQ1 receives an affirmative result.
- **RQ2, RQ3:** the designs and offline structural validation are complete; measuring their effect on a real model is outside the scope of this work.
- **RQ4 (H1, H2, H3):** H1 is supported: the probability that adjacent tokens' expert sets intersect is 0.915, and 0.910 in the replay on the GT 1030. H3 is not supported: chunk-boundary type prefetch has a hit rate comparable to a token-level LRU and a higher load volume. The set shrinkage of H2 is achievable with the router LoRA (held-out sessions −33.2%, fresh sessions −27.5%), with agreement against the base of 0.843 and 0.835, close to the hypothesized value of 0.85, so H2 receives preliminary support and needs further verification.
- **RQ5:** distillation-style LoRA training completes within a 1.0 GiB GPU budget (§5.6.3, GPU peak 0.79 GiB); the effect of the whole loop on quality is outside the scope of this work.

### 6.2 Implications

1. **Source of the prefetch value.** The routing of Gemma4 itself has strong adjacent overlap that an LRU exploits; the chunk type provides no observable additional gain in hit rate.
2. **LoRA and expert selection.** A LoRA must act on the router to influence which experts are selected; a LoRA acting only on attention and the dense MLP has no observable effect on the expert set.
3. **Trade-off between shrinkage and output consistency.** Under the setting of this work, at about 33% shrinkage the top-1 agreement lies between 0.83 and 0.85.
4. **Evaluation design.** This work uses the teacher-forced NLL and the next-token top-1 agreement as quality criteria, so comparisons do not depend on a generation stop condition; comparisons involving free generation use the full end-token set of the model (ids 1, 106 and 50 for Gemma4) as the stop condition and keep special tokens to read the end-of-turn position. This design gives the comparisons among the LoRAs a consistent basis (§4.4).

### 6.3 Threats to Validity

- **Internal validity:** hyper-parameters were chosen from 10 settings on the held-out sessions, so the held-out results contain selection bias; the finally chosen setting was therefore also evaluated on fresh sessions. The criterion values (NLL ≤ 0.15, top-1 ≥ 0.85) are hypothesized values of the researcher, not external standards.
- **External validity:** a single model (Gemma4 26B-A4B, Q4_K_M) and a single corpus source (dialogue generated by the model itself); creative and factual samples are fewer and the fresh sessions contain no factual type.
- **Measurement validity:** load times are converted from nominal bandwidth and are not measured latency; NLL and top-1 were measured in the bf16 offload engine while the second measurement of locality and set size used Q4_K_M llama.cpp; the replay on the GT 1030 covers 1280 tokens.

### 6.4 Scope and Evidence Levels

| Item | Evidence level | Conditions |
|---|---|---|
| Phase 1 monitor (§3.2, §5.1) | live measurement on the GT 1030 | Qwen3-1.7B, GT 1030 + i5-4460 |
| VRAM and decode speed of Gemma4 26B-A4B (§5.2) | measured on the GT 1030 | Vulkan b8679, `-ngl 10/11` |
| Expert distribution, Tier 1 (§5.3) | measured on the GT 1030, reproduced on gx10 | 10 sessions, 70 jobs |
| Per-token expert sequences and locality (§5.4) | measured on gx10 under constrained conditions; GT 1030 replay of the first 1280 tokens | gx10: cores 0–4, nice 10, `-ngl 2`, 933 MiB |
| Expert-pool policies (§5.5) | replay on real routing sequences | load times converted from nominal bandwidth |
| Router LoRA and compensation adapter (§5.6) | measured on gx10 under constrained conditions (GPU budget 1.0 GiB) | held-out and fresh sessions |
| Phases 2, 3 (§3.3, §3.4) | offline structural validation | synthetic tensors, no model attached |
| Phase 5 (§3.6) | structural validation; device numbers are vendor specifications | no Jetson, Android or ARM camera obtained |
| Phase 6 (§3.7) | structural validation; UMA and NVMe numbers are vendor specifications | no Mac Mini M4 obtained; the expert distribution of DeepSeek-V3 was not measured |
| Phase 7 (§3.8) | structural validation; §5.6.3 is a measurement of the same mechanism on a constrained GPU | |

The corpus is dialogue generated by Gemma4 itself, with 2 pilots on Windows and the others generated on gx10 with its GGUF (a different file from the GT 1030's). The conclusions concern Gemma4 26B-A4B (Q4_K_M); statements about Mixtral, DeepSeek and similar models cite public specifications. The device numbers of Phase 5 and Phase 6 (FPS, UMA bandwidth, NVMe speed) are specification values.

---

## 7. Conclusion and Future Work

On the 2 GB VRAM GT 1030 and an older PC, Gemma4 26B-A4B runs with `-ngl 10` at 1.7 GB VRAM and decodes at 4.22 tok/s. Its expert routing has clear temporal locality (adjacent-token intersection probability 0.915), which is formed jointly by the concentration on popular experts and by adjacency and can be exploited effectively by a token-level LRU; chunk-boundary type prefetch has a comparable hit rate to an LRU. Shrinking the expert set is achievable with a LoRA attached to the router: −33.2% on average on the held-out sessions and −27.5% on the fresh sessions; the next-token top-1 agreement with the base is 0.843 and 0.835, close to the hypothesized value of 0.85, so H2 receives preliminary support, and shrinkage and output consistency trade off.

**Future work.** (1) Continue verifying the shrinkage and output consistency of H2 with more sessions (including factual and more creative), other corpus sources and other MoE models; (2) explore larger distillation data and other losses to move the trade-off curve between shrinkage and agreement; (3) connect the expert pool to llama-server to measure online hit rate and latency; (4) obtain measured data for Phases 5 and 6 on a Jetson and a Mac M4; (5) extend to other MoE models.

---

## References

[1] Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv:1701.06538*.

[2] Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models. *JMLR*, 23(120), 1–39.

[3] Jiang, A.Q., et al. (2024). Mixtral of experts. *arXiv:2401.04088*.

[4] Sheng, Y., et al. (2023). FlexGen: High-throughput generative inference with a single GPU. *ICML 2023*.

[5] Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication at scale. *NeurIPS 2022*.

[6] Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate post-training quantization. *arXiv:2210.17323*.

[7] Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference via speculative decoding. *ICML 2023*.

[8] Miao, X., et al. (2023). SpecInfer: Tree-based speculative inference. *arXiv:2305.09781*.

[9] Clark, K., Khandelwal, U., Levy, O., & Manning, C.D. (2019). What does BERT look at? *BlackboxNLP 2019*.

[10] Voita, E., et al. (2019). Analyzing multi-head self-attention. *ACL 2019*.

[11] Elhage, N., et al. (2021). A mathematical framework for transformer circuits. *Transformer Circuits Thread*.

[12] Zou, A., et al. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv:2310.01405*.

[13] Turner, A., et al. (2023). Activation addition: Steering LMs without optimization. *arXiv:2308.10248*.

[14] Subramani, N., Suresh, N., & Peters, M. (2022). Extracting latent steering vectors. *ACL Findings 2022*.

[15] Hu, E.J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

[16] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS 2023*.

[17] Zhang, Q., et al. (2023). AdaLoRA: Adaptive budget allocation for PEFT. *ICLR 2023*.

[18] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NIPS DL Workshop 2014*.

[19] Wang, Y., et al. (2023). Self-instruct: Aligning LMs with self-generated instructions. *ACL 2023*.

[20] Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv:2212.08073*.

[21] Dao, T., Fu, D.Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention. *NeurIPS 2022*.

[22] Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.

[23] DeepSeek-AI (2024). DeepSeek-Coder-V2. *arXiv:2406.11931*.

[24] Qwen Team (2025). Qwen3 technical report. *arXiv:2505.09388*.

[25] Google DeepMind (2025). Gemma 4 technical report.

---

## Appendix A: Module Map

```
Phase  Module                              Purpose
─────────────────────────────────────────────────────────────
P1     monitor/attention_entropy.py        Shannon entropy monitor
P1     monitor/perplexity_spike.py         z-score spike detector
P1     monitor/role_tag_parser.py          deterministic FSM
P1     monitor/signal_fusion.py            priority-weighted fusion
P1+3   segmentor/chunk_classifier.py       type → route mapping
P2     kv/kv_serializer.py                 .kvbin format
P2     kv/kv_offload_manager.py            3-tier LRU hierarchy
P3     spec_experts/directional_steering.py system prompt steering
P4     spec_experts/expert_ram_pool.py     MoE expert prefetch
P5     edge/vlm_router.py                  VLM routing + targets
P6     mac_m4/metal_expert_pool.py         UMA pool + Flash Attn
P7     agi/self_loop.py                    LoRA pool + distillation
P1-7   spec_experts/controller.py          FastAPI :8090
P1-4   spec_experts/tool_session.py        hot-swap llama-server
P4     affinity_pipeline/tier2_ctypes.py      per-token expert ids (ctypes, router top-k from llama.cpp)
P4     affinity_pipeline/tier2_stats.py       locality, affinity sets, window hit rate, before/after comparison
P4     scripts/tier2_locality.py              Tier 2 driver (run / analyze / compare)
P4     scripts/pool_replay.py                 expert-pool policy replay (§5.5)
P4     scripts/h2_targets.py + h2_router_lora.py + h2_comp.py   router LoRA and compensation adapter (§5.6)
P4     config/tier2_layouts/*.json            llama.cpp struct offsets (d327bf1 measured from headers; b8679 inferred)
Tests  tests/phase1_monitor_demo.py        P1 live demo (GT 1030)
Tests  tests/phase2_7_demo.py              P2-7 structural validation
Tests  tests/spec_experts_test.py          7/7 integration tests
```

## Appendix B: Reproduction

```bash
git clone https://github.com/sipurchen/klchen
git checkout SpecExpertsResearch
set LLMS_DIR=<LLMS_DIR>

# Phase 1 live (requires llama-server on :8080)
python tests/phase1_monitor_demo.py

# Phase 2-7 structural (offline)
python tests/phase2_7_demo.py

# Integration tests (requires controller on :8090)
python tests/spec_experts_test.py
```

## Appendix C: Reproducing Tier 2, the Expert-Pool Replay and the LoRA Experiments

The commands below run through `tools/slice_runner/gx10_run.py` on a shared host and need these environment variables: `GX10_HOST` (ssh target), `TRAIN_PY` (python with torch / transformers / peft), `OFFLOAD_ENGINE_DIR` (directory containing the `lora_offload` package) and `BASE_HF_DIR` (HF checkpoint directory of Gemma4 26B-A4B).

```bash
# Tier 2: per-token expert sequences and locality (gx10 restricted: cores 0-4, nice 10, -ngl 2, -ub 64)
python tools/slice_runner/gx10_run.py run layout
python tools/slice_runner/gx10_run.py run tier2 --args "run --tag base_all --sessions all --ngl 2"
python tools/slice_runner/gx10_run.py run tier2 --args "analyze --tag base_all"

# Expert-pool policy replay (§5.5)
python tools/slice_runner/gx10_run.py run replay --args "--tag base_all --caps 32,48,57,64"

# H2: target sets, router LoRA, compensation adapter, held-out evaluation (§5.6)
python tools/slice_runner/gx10_run.py run h2targets --args "--tag base_all --train coder_pilot,coder_01,coder_02,planner_pilot,planner_01,planner_02,writer_01"
python tools/slice_runner/gx10_run.py run h2 --args "train --tag c1 --lam 1.0 --mu 1.8 --lr 1e-3 --hinge 0.9 --epochs 2 --accum 2"
python tools/slice_runner/gx10_run.py run h2 --args "eval --tag c1"
python tools/slice_runner/gx10_run.py run h2 --args "comp-train --tag f2 --rank 8 --alpha 8 --window 512 --lr 1e-4 --lmw 0.0 --epochs 2 --accum 2"
python tools/slice_runner/gx10_run.py run h2 --args "comp-eval --tag f2"
```

Tier 2 on the GT 1030 (Windows, cores 2+3, low priority):

```powershell
$p=[System.Diagnostics.Process]::GetCurrentProcess(); $p.ProcessorAffinity=[IntPtr]12; $p.PriorityClass='BelowNormal'
$env:LLMS_DIR='<LLMS_DIR>'; $env:AFFINITY_PROFILE='gt1030_win'
python -u scripts\tier2_locality.py run --tag gt_pilot --sessions planner_pilot,coder_pilot --ngl 8 --ubatch 128
```
