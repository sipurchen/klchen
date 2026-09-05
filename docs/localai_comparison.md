# LocalAI vs Spec-Experts — Research Comparison

> Generated: 2026-09-05
> Target: [mudler/LocalAI](https://github.com/mudler/LocalAI) (upstream project, not forks)
> Method: web research (Fable 5.1 agent), compared against Spec-Experts 7-phase framework in this repo

---

## 1. LocalAI Architecture Summary

**What it is:** Go control-plane exposing OpenAI/Anthropic/ElevenLabs/Ollama-compatible HTTP APIs, dispatching to **60+ backends** (llama.cpp, vLLM, SGLang, transformers, diffusers, whisper.cpp, MLX/MLX-VLM, plus their own `vllm.cpp`, `parakeet.cpp`).

**Backend abstraction:** each backend is a separate **gRPC service** (`LoadModel`, `Predict`, `PredictStream`, `Embedding`, `Rerank`, `TokenizeString`, `GetMetrics`, `Health`), packaged as an OCI image, pulled on demand via a "backend gallery" carrying hardware profiles (cuda12/13, rocm, sycl, vulkan, metal, l4t, cpu-avx). The llama.cpp backend (`backend/cpp/llama-cpp/grpc-server.cpp`) is a gRPC re-skin of upstream `tools/server/server.cpp`, so it inherits llama-server features ~1:1. External backends attach via `--external-grpc-backends BACKEND_NAME:URI`.

**GGUF / offload / quant (per-model YAML):**
- `gpu_layers` — default disables llama.cpp auto-fit (`-1` re-enables it; currently buggy, see [issue #8562](https://github.com/mudler/LocalAI/issues/8562))
- `f16`, `mmap`, `mmlock`, `low_vram`, `numa`, `threads`, `context_size` (`-1` reads GGUF metadata)
- `tensor_split`, `main_gpu`
- `cache_type_k` / `cache_type_v` (`f16/f32/q8_0/q4_0/q4_1/q5_0/q5_1`, maps to `-ctk/-ctv`)
- `draft_model` / `n_draft` (+ `spec_type`: `ngram-*`, `draft-eagle3`, `draft-mtp`)
- `rope_freq_base/scale/scaling`, `mmproj` + `mmproj_use_gpu`
- anything else passes through via `options: ["--flag", "--flag:value"]`
- quant selection is **manual** (pick the GGUF) — no auto quant picking

**MoE-specific:** `cpu_moe: true` (`--cpu-moe`), `n_cpu_moe: N` (`--n-cpu-moe`), `override_tensor: "<regex>=<buft>,..."` (`-ot`). That's the full extent — static tensor placement, no dynamic/semantic-aware expert prefetch. Separately, mudler maintains [apex-quant](https://github.com/localai-org/apex-quant): MoE-aware mixed-precision GGUF recipe (routed-expert / shared-expert / attention tensor classes; first/last MoE blocks kept at Q5_K/Q6_K, middle layers compressed harder) via stock `llama-quantize --tensor-type-file`.

**KV / memory management:** model-level, not token-level. Server-side RAM prompt cache on by default since v4.3 (`--cache-ram`, `--kv-unified`, `--cache-idle-slots`). Whole-model LRU eviction (`--max-active-backends=N`), `LOCALAI_VRAM_BUDGET=80%|12GB`, `concurrency_groups` (anti-affinity), idle/busy watchdogs (`LOCALAI_WATCHDOG_IDLE_TIMEOUT`, `..._BUSY_TIMEOUT`, `--watchdog-interval`). Docs admit VRAM estimation is unreliable across backends.

**P2P / federated:** libp2p + EdgeVPN, shared token. **federated** mode (`--p2p --federated`) routes whole requests to one node = load balancer. **worker** mode wraps llama.cpp `rpc-server`, splits layers of a *single* dense model across machines by memory. Separate enterprise distributed mode (PostgreSQL + NATS, prefix-cache-aware routing).

**Monitoring:** `/metrics` OpenMetrics endpoint — request latency/errors only. **No token-level signal analysis** (no entropy, no perplexity, no boundary detection).

**Multimodal:** GPT-Vision API on `/v1/chat/completions` (base64 or URL image), llama.cpp `mmproj` projector (`mmproj_use_gpu:false` keeps projector on CPU), `known_input_modalities: [text,image,audio,video]` capability declaration. Speculative decoding auto-disabled when mmproj active.

**Agents / tools:** backend-native function-calling parsers, `logit_bias` map + `grammar` file per model, `reasoning:` block (`disable`, `strip_reasoning_only`, `thinking_start_tokens`, `tag_pairs`), MCP via `mcp: {remote:, stdio:}`, `agent.max_iterations` (default 10).

**Low-spec guidance:** lower `gpu_layers`/`context_size`, Q4_K_M, `threads: 1` when fully GPU-offloaded, `--max-active-backends=1`, watchdog on, `LOCALAI_FORCE_META_BACKEND_CAPABILITY=default` for CPU-only. Nothing specific to 2GB-class GPUs or MoE-on-tiny-VRAM beyond "use `cpu_moe`".

---

## 2. Feature-by-Feature Comparison

| Area / our phase | LocalAI | Spec-Experts (ours) |
|---|---|---|
| P1 — token-stream monitor | `/metrics`: request-level latency/errors only | Role-tag FSM + Shannon entropy + perplexity z-score fusion — **unique** |
| P2 — KV tiering | llama-server RAM prompt cache (`--cache-ram`), KV quant (`-ctk/-ctv`); whole-model LRU eviction; no disk KV tier | VRAM→RAM→Disk `.kvbin` per-token tiering — **unique** |
| P3 — steering | Static per-model `logit_bias`, `grammar`, `temperature`, `reasoning` tag handling; no per-boundary dynamics | Boundary-type-triggered temp/logit-bias/prefix warmup — **unique** |
| P4 — expert RAM pool | Static placement: `cpu_moe`, `n_cpu_moe`, `override_tensor` | Semantic-affinity expert prefetch — **unique**; LocalAI's baseline is exactly our `-ot` starting point |
| P5 — edge VLM router | Single-process mmproj; `known_input_modalities`; federated mode routes whole requests but not by modality | Modality/device-aware routing — **unique**, though federated mDNS node-discovery is a comparable pattern |
| P6 — Mac M4 expert pool | Metal backend image; no NVMe expert paging | NVMe expert paging — **unique** |
| Config | YAML per model, `options:` passthrough, gallery `overrides` merge | Custom (this repo) |
| API | OpenAI + Anthropic + Ollama + ElevenLabs compat | llama-server native |

---

## 3. Worth Adopting

1. **`--cpu-moe` / `--n-cpu-moe` as the baseline before regex `-ot`.** Our 10 tok/s plan uses `-ot` regexes directly; upstream `--n-cpu-moe N` (keep first N layers' experts on CPU) is the cleaner knob to sweep first for Gemma4 E4B on GT 1030 — LocalAI treats it as the primary MoE lever. Reserve `-ot` for the fine-grained placement Phase-4's expert pool actually needs.
2. **apex-quant recipe** (`llama-quantize --tensor-type-file`) — edge-layers-high / middle-low per-tensor precision for MoE. Directly shrinks active-expert bytes for 2GB VRAM; citable baseline, and smaller experts = more fit in the RAM tier.
3. **llama-server's built-in RAM prompt cache** (`--cache-ram`, `--kv-unified`, `--cache-idle-slots`, `--slot-save-path`). Benchmark Phase-2's `.kvbin` tiering **against** this stock path — need the number showing custom disk format beats upstream slot save/restore.
4. **KV-cache quantization (`-ctk q8_0 -ctv q8_0` or `q4_0`)** — trivially doubles/quadruples resident tokens in the 56MB VRAM KV tier before spilling to RAM.
5. **`gpu_layers=-1` auto-fit conflicts with `tensor_buft_override`** (issue #8562) — our `-ot` plan must pin `-ngl` explicitly, don't rely on auto-fit.
6. **Idle/busy watchdog pattern** (`--watchdog-interval`, busy timeout) — cheap sidecar behavior for Phase-1: a stalled SSE stream on the i5-4460 (the screen-blank scenario in memory) could be detected and the server bounced.
7. **`concurrency_groups` / `LOCALAI_VRAM_BUDGET` idea** — declare a VRAM ceiling in config so the KV tier and expert pool contend against one explicit budget instead of each assuming the full 2GB.
8. **`reasoning.tag_pairs` / `thinking_start_tokens` config shape** — compact declarative way to define the role-tag FSM's boundaries per model instead of hardcoding tags.
9. **`known_input_modalities` declaration per model** — clean capability descriptor Phase-5's router could read to decide routing.

---

## 4. Not Relevant — Skip

OCI-per-backend packaging and backend gallery; 60-backend gRPC abstraction; multi-user API keys, quotas, RBAC, PII redaction, cosign signing; PostgreSQL+NATS distributed mode; P2P/federated (no single-machine use case — worker mode only shards one dense model across machines, orthogonal to intra-machine expert paging); Anthropic/ElevenLabs/Ollama API shims; web UI, agent hub, MCP tool loop; diffusers/TTS/ASR backends; `vllm.cpp` (paged KV for batched multi-tenant throughput, not tiny-VRAM single-user).

---

## 5. Gaps in Our Project That LocalAI Highlights

- **No stock-baseline benchmark.** LocalAI gets far with just `cpu_moe` + `-ctk q8_0` + RAM prompt cache. Every phase needs a "vs. plain llama-server flags" row or the contribution is unclear to reviewers.
- **No declarative per-model config.** LocalAI's YAML (`gpu_layers`, `n_cpu_moe`, `override_tensor`, `cache_type_*`, `options:`) is a good schema to mirror so one file drives Gemma4-E4B/GT1030, Mixtral/RAM, and DeepSeek/M4 configs.
- **No request-level metrics endpoint.** Our monitor has richer signals than LocalAI's `/metrics` but no export for graphs/thesis plots.
- **No hardware auto-detection.** LocalAI probes GPU capability and picks a backend; we still hand-pin CPU affinity (0xC) and thread counts.
- **Quantization treated as fixed.** apex-quant shows MoE-aware quant is itself a lever; our expert pool assumes fixed expert sizes.

---

## Sources

- [LocalAI README](https://github.com/mudler/LocalAI)
- [Model Configuration](https://localai.io/docs/advanced/model-configuration/)
- [VRAM Management](https://localai.io/docs/advanced/vram-management/)
- [Distributed/P2P](https://localai.io/docs/features/distribute/)
- [GPU Acceleration](https://localai.io/docs/features/gpu-acceleration/)
- [Runtime Settings](https://localai.io/docs/features/runtime-settings/)
- [CLI Reference](https://localai.io/docs/reference/cli-reference/)
- [Functions](https://localai.io/docs/features/openai-functions/)
- [MCP](https://localai.io/docs/features/mcp/)
- [Vision](https://localai.io/docs/features/gpt-vision/)
- [Issue #8562 — auto-fit vs tensor_buft_override](https://github.com/mudler/LocalAI/issues/8562)
- [v4.3.0 release notes](https://github.com/mudler/LocalAI/discussions/9971)
- [apex-quant](https://github.com/localai-org/apex-quant)
- [llama-cpp-backend.md](https://github.com/mudler/LocalAI/blob/master/.agents/llama-cpp-backend.md)
