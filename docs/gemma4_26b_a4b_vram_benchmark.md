# Gemma4 26B-A4B MoE — Real VRAM/ngl/tok-s Benchmark (GT 1030 2GB)

> Measured: 2026-09-05, branch `VRAM-Opt-StreamingExperts`
> Hardware: i5-4460 (4-core) · GT 1030 2GB VRAM (Vulkan, driver reports 452MB baseline used by desktop) · 34GB DDR3
> llama-server b8679 (Vulkan backend)
> Model: `google_gemma-4-26B-A4B-it-Q4_K_M.gguf` (17.0 GB file) — **real sparse MoE**, confirmed via GGUF metadata, not the dense `gemma4_textonly` E4B (5.4GB) tested previously.

## Why this model, not the "E4B textonly" one

Two different GGUFs exist in `E:\LLMmodel` under the "Gemma4" name — they are architecturally unrelated:

| | `gemma4_textonly` (old `project_10toks` memory) | `gemma4-26B-A4B-it-GGUF` (this benchmark) |
|---|---|---|
| `general.architecture` | `gemma4` | `gemma4` |
| `expert_count` | **0 (dense)** | **128** |
| `expert_used_count` | 0 | **8** |
| Params | 7.5B dense | 25.23B total, ~4B active/token |
| block_count | 42 | 30 |
| Prior finding | 2 tok/s, CPU-only, 10 tok/s ruled impossible | this doc |

The old "10 tok/s impossible" conclusion was correct **for the dense model** — it doesn't apply here. `26B-A4B` is the actual MoE checkpoint the expert-offload strategy in `config/direct_inference.json` was designed for.

## Real architecture (from GGUF header, not assumed)

- 30 transformer blocks, `embedding_length`=2816, `expert_count`=128, `expert_used_count`=8/token, `expert_feed_forward_length`=704
- **Sliding-window pattern, ratio 5:1** — confirmed via `n_head_kv` per layer: 25 "local" layers (SWA, window=1024 tokens, richer KV: 8 kv-heads/2048-dim) + 5 "global" layers (full context, leaner KV: 2 kv-heads/1024-dim) at indices 5,11,17,23,29
- `-ngl N` in this llama.cpp build offloads the **last N layers** (closest to output) first, not the first N
- SWA cache is fixed at `n_swa + n_ubatch` = 1024+512 = **1536 cells regardless of `-c`** — local-layer KV cost does NOT scale with `--ctx-size`. Only the 5 global layers' KV scales with ctx.
- Default `--parallel`(`-np`) is **4** — quadruples KV cache for a use case that only ever needs 1 sequence. **Always pass `-np 1`.**

## Method

`llama-server -ngl N --override-tensor "\.ffn_.*_exps\.=CPU" -c <ctx> -np 1 --no-warmup`, all 128-expert FFN tensors forced to CPU/RAM regardless of `-ngl` (matches LocalAI's `cpu_moe` pattern, see [localai_comparison.md](localai_comparison.md)); only attention + norms + embedding get GPU-offloaded per layer. VRAM figures below are the sum of the three `Vulkan0` buffers llama.cpp itself reports at load (model weights + KV(non-SWA)+KV(SWA) + compute graph) — measured directly from server logs, not `nvidia-smi` (which is contaminated by ~200-300ms driver release lag between consecutive test runs).

## Results — ctx = 16384

| ngl | model buf | KV (global+local) | compute buf | **total** | fits <1.7GB? |
|-----|-----------|--------------------|--------------|-----------|--------------|
| 6   | 791.09 MiB | 64+48 MiB | 527.41 MiB | **1430.5 MiB** | ✅ comfortable |
| 9   | 915.42 MiB | 128+72 MiB | 517.50 MiB | **1632.9 MiB** | ✅ |
| 10  | 952.60 MiB | 128+84 MiB | 517.50 MiB | **1682.1 MiB** | ✅ (58 MiB margin) |
| 11  | 994.74 MiB | 128+96 MiB | 517.50 MiB | **1736.2 MiB** | ⚠️ borderline (~5 MiB under 1.7GB) |
| 12+ | — | — | — | OOM | ❌ (confirmed empirically) |

**Recommended: `-ngl 10` at ctx=16384** (1682 MiB, safe margin under 1.7GB target).

## Results — ctx = 8192

| ngl | model buf | KV (global+local) | compute buf | **total** | fits <1.7GB? |
|-----|-----------|--------------------|--------------|-----------|--------------|
| 11  | 994.74 MiB | 64+96 MiB | 517.50 MiB | **1672.2 MiB** | ✅ |
| 13+ | — | — | — | OOM | ❌ (confirmed empirically) |

Global-layer KV halves at ctx=8192 (scales linearly with ctx; local/SWA KV is flat), which buys exactly **+1 ngl layer** (11 vs 10) versus ctx=16384 before hitting the same VRAM ceiling.

**Recommended: `-ngl 11` at ctx=8192.**

## Real measured decode speed — the 25 tok/s target is NOT met

Actual `/completion` benchmark, `-ngl 10 -c 16384 -t 4`, short prompt:

```
prompt:    34 tokens,  3.68 tok/s  (271.5 ms/token — prefill, CPU-bound MoE routing)
decode:    13 tokens,  4.22 tok/s  (237.1 ms/token)
```

**~4.2 tok/s measured, vs. 25 tok/s target — a ~6× gap.** This is a hard CPU/RAM-bandwidth ceiling, not a config problem:
- Only attention (a small fraction of FLOPs) runs on GPU; all 8-of-128 active experts per token, every layer, run on CPU because they don't fit in 2GB VRAM.
- i5-4460 has 4 cores and DDR3 (~17 GB/s dual-channel) — the CPU-side expert compute is both bandwidth- and core-count-bound.
- The previously-recorded "24.7 tok/s theoretical" figure in memory (`project_10toks.md` index line) does not apply to this workload — that number assumed full GPU residency, impossible here since experts alone (128 × ~6M params/expert × 30 layers) can't fit in 2GB.

**Untested:** IQ2_S quant (10.2GB file) — first attempt stalled >120s past `initializing slots`, likely cold mmap page-fault on first touch (file had never been read in this session). Worth a retry with the file pre-warmed (`cat` the file once to prime OS page cache) before timing it; smaller active-expert bytes could meaningfully help RAM-bandwidth-bound decode.

## Practical takeaways for the 25 tok/s goal

1. **25 tok/s is not reachable on this MoE model on this hardware** with the current architecture (CPU-bound expert compute). Options, roughly ordered by effort:
   - Re-baseline the target against measured ~4.2 tok/s, or
   - Test IQ2_S/IQ2_M (smaller active bytes/token → less RAM traffic) — needs a clean warm-cache retest,
   - Move MoE inference to a GPU with enough VRAM to hold active experts + KV entirely (per [[project_10toks]] dense-model math and the RTX2050 plan in [RTX2050_LLM_Harness_Plan.md](RTX2050_LLM_Harness_Plan.md), a 4GB+ card changes this completely),
   - Reduce active-expert-per-token cost via a smaller/different MoE checkpoint for the "coder"/"planner" experts (see split-experts discussion) rather than routing through all 128 experts of one big model.
2. **`-np 1` is a free win** — quarters KV cache vs. the llama-server default of 4 parallel slots, for zero cost in a single-user setup.
3. **`-ngl 10` @ ctx=16384 / `-ngl 11` @ ctx=8192** are the validated, empirically-safe operating points under the 1.7GB budget on this card as configured today (452 MiB already consumed by the desktop compositor at idle — closing other GPU-accelerated apps before a real run would recover more headroom).
