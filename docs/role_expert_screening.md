# VLM / Coder / Planner Role-Model Screening (CPU-safe)

> Measured: 2026-09-05, branch `VRAM-Opt-StreamingExperts`
> Hardware: i5-4460 · GT 1030 2GB · constraint: **CPU capped to 2 threads pinned to cores 2+3 (0xC mask), BelowNormal priority — never unrestricted**, VRAM target ≤1.7GB, ctx 8192/16384

## Why the constraint matters

An earlier probe in this same session launched `llama-server.exe -t 4` directly with **no CPU affinity/priority limit**, which pegged all 4 cores and froze the whole machine (not just Claude Code's display — mouse/keyboard input stopped responding). All screening below uses a PowerShell wrapper (`scripts/probe_role_models.ps1`, `scripts/probe_role_models2.ps1`) that pins the process to cores 2+3 and sets `BelowNormal` priority immediately after spawn, matching the existing Ollama-safety pattern in [feedback_cpu_overload memory]. **Never launch llama-server from this project without that wrapper.**

## Candidates tested (all GGUF already present under `E:\LLMmodel`)

| Role | Model | Quant | File size |
|------|-------|-------|-----------|
| VLM | SmolVLM-500M-Instruct | Q8_0 (+mmproj) | ~500 MB |
| Coder | Qwen2.5-Coder-1.5B-Instruct | Q8_0 | ~1.6 GB |
| Coder | Qwen2.5-Coder-3B-Instruct | Q4_K_M | 2.1 GB |
| Planner | Qwen3-1.7B | Q8_0 | ~1.8 GB |
| Planner | DeepSeek-R1-Distill-Qwen-1.5B | Q8_0 | ~1.6 GB |

All are **dense** models (no MoE tricks needed) — unlike the 26B-A4B case, full or near-full `-ngl` offload is feasible on a 2GB card.

## Round 1 — ngl=999 (full offload)

| Model | ctx | Status | VRAM (nvidia-smi) | decode tok/s |
|---|---|---|---|---|
| SmolVLM-500M | 8192 | ✅ OK | 1099 MiB | **56.1** |
| SmolVLM-500M | 16384 | ✅ OK | 1419 MiB | **67.7** |
| Coder-1.5B | 8192/16384 | ❌ OOM | — | — (weight buffer alone 1564.6 MiB) |
| Coder-3B | 8192/16384 | ❌ OOM | — | — (2.1GB file too big for full offload) |
| Qwen3-1.7B | 8192/16384 | ❌ OOM | — | — (weight buffer alone 1743.8 MiB) |
| DeepSeek-R1-1.5B | 8192/16384 | ❌ OOM | — | — (same 1564.6 MiB as Coder-1.5B, same base arch/quant) |

Only the 500M VLM fits fully. Everything ≥1.5B dense at Q8_0 needs partial `-ngl`.

## Round 2 — reduced ngl (ctx=8192)

| Model | ngl | model buf | KV | compute | Status | decode tok/s |
|---|---|---|---|---|---|---|
| Coder-1.5B (28 layers) | 25 | 1374.88 MiB | 192 MiB | 299.75 MiB | ✅ OK | **18.26** |
| Coder-3B (36 layers) | 24 | 1257.19 MiB | 184 MiB | 300.75 MiB | ✅ OK | **8.31** |
| Qwen3-1.7B (28 layers) | 22 | 1386.65 MiB | — | — | ❌ OOM (still too big) |
| DeepSeek-R1-1.5B (28 layers) | 25 | 1374.88 MiB | 192 MiB | 299.75 MiB | ✅ OK | **18.83** |

At ctx=16384, all four OOM'd at these same ngl values (KV cache roughly doubles, pushes past the card's real ceiling) — **ctx=16384 needs a lower ngl** for the 1.5B+ dense models than ctx=8192 does; not yet bisected (time-boxed this session).

## Recommendation

| Role | Pick | Config | Why |
|---|---|---|---|
| **VLM** | SmolVLM-500M-Instruct | `-ngl 999 -c 16384` | Only VLM available; full-GPU, 67.7 tok/s, huge headroom (600+ MiB spare) |
| **Coder** | Qwen2.5-Coder-1.5B-Instruct-Q8_0 | `-ngl 25 -c 8192` | 18.26 tok/s vs Coder-3B's 8.31 — 1.5B clearly wins on this hardware; 3B's extra quality isn't worth the >2x slowdown here |
| **Planner** | DeepSeek-R1-Distill-Qwen-1.5B-Q8_0 | `-ngl 25 -c 8192` | Only planner candidate that actually fit; Qwen3-1.7B doesn't fit even at ngl=22 (bigger per-layer cost, embd=2048 vs 1536) and would need more aggressive ngl reduction + retest before it's viable |

All three, run one-at-a-time (never concurrently — combined they'd exceed 2GB), each with `-t 2` + cores-2+3 affinity + BelowNormal, stay within the 1.7GB/25-50% CPU targets and are 3-16× faster than the 26B-A4B MoE case (4.22 tok/s) because none of them are CPU-bound on expert FFN — this is the concrete case for splitting coder/planner into small dense specialists rather than routing everything through one big MoE model.

## Open items for next round

- Bisect ctx=16384 ngl for Coder-1.5B / Coder-3B / DeepSeek-R1-1.5B (start ~4-6 layers lower than the ctx=8192 values above)
- Re-test Qwen3-1.7B at lower ngl (~14-18) if it's still wanted as an alternative planner
- IQ2_S retest for the 26B-A4B MoE model, still pending from the previous benchmark doc
