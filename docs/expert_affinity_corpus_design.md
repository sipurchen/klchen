# Expert-Affinity Corpus Design v2 — Gemma4 26B-A4B (128 experts, top-8)

> Status: **design document only** — nothing in this file has been executed against the model. Branch `VRAM-Opt-StreamingExperts`, 2026-09-05.
> Supersedes the corpus-building part (§2.4) of `docs/expert_routing_visibility_plan.md`. The Tier-1 mechanism itself (`llama-imatrix.exe` writing `blk.N.ffn_down_exps.weight.counts`) is validated and unchanged; **only the corpus methodology is replaced.**
> Path placeholders follow `bin/README.md`: `<PROJECT_ROOT>` = this repo, `<LLMS_DIR>` = the local GGUF store. All real paths in this doc are written with placeholders; the scripts hold the concrete paths.

---

## 0. TL;DR

| Item | v1 (first attempt, this session) | v2 (this design) |
|---|---|---|
| Profiling axis | task **role** (coder / planner / control) | **chunk type** (reasoning / code / factual / creative) per thesis §5.3 / §6.1; roles are a *weighted combination* of the four (§4.5) |
| Corpus unit | 12–16 unrelated single-turn Q&A snippets, `n_predict=200`, ≈2 048–3 072 tokens per role after imatrix truncation | long, scripted, **multi-turn sessions** of 4 000–8 191 body tokens each, thinking enabled, ≥ 20 000 tokens per chunk type (§3, §6.7) |
| Chat markers | `<start_of_turn>` / `<end_of_turn>` — **not in this model's vocabulary** (tokenized as literal text) | the model's real template tokens `<|turn>` (105) / `<turn|>` (106), `<|think|>` (98), `<|channel>` (100) / `<channel|>` (101) — read from the GGUF header this session (§2.1) |
| Reasoning content | **empty** thought channel (`<|channel>thought\n<channel|>`) because `<|think|>` was never sent — i.e. v1 never profiled any reasoning at all | `<|think|>` system turn → genuine `<|channel>thought … <channel|>` spans, verified by an automated check (§6.2) |
| imatrix chunking | `-c 1024` over a concatenation of unrelated snippets: every 1 024-token window sliced snippets in half and prepended unrelated ones; 30 % of the coder corpus (the tail) silently dropped | each imatrix invocation = **exactly one causal document** (`file = body + <bos> + body`, `-c = |body|+1`, `--chunks 1`); per-chunk-type attribution by **prefix-differential** runs (§5.4) — no sample ever straddles a chunk boundary, by construction and by an automated assertion (§6.3) |
| Validation | none | pilot + 9 checks with numeric pass/fail (§6) |
| Per-expert size used for pool budget | 2.1 MiB (assumed) | **4.14 MiB** per expert per layer (gate_up Q4_K 2.127 MiB + down Q8_0 2.009 MiB, from tensor byte counts, §2.4) |

**Single most important correction:** v1 profiled *the wrong thing* three times over — wrong turn tokens, no reasoning content, and imatrix windows that spliced unrelated snippets — so its `.counts` histograms are not a measurement of anything the thesis defines. v2 makes every imatrix chunk a single, complete, causally-coherent session (or session prefix), and attributes counts to chunk types by differencing prefixes, so that every profiled token has exactly the context it had at generation time.

---

## 1. What was verified this session (facts, not assumptions)

Everything below was obtained by reading files/binaries on this machine — GGUF metadata via `gguf` 0.18.0, `--help` output, `grep -a` on the shipped binaries, and the v1 logs/outputs in `logs/`. **No inference was run.**

### 1.1 The model's real chat template and special tokens (from `tokenizer.chat_template` in the GGUF header)

| Token text | id | type | Role in the template |
|---|---|---|---|
| `<bos>` | 2 | control | emitted once at the start (`{{- bos_token -}}`) |
| `<|turn>` | 105 | control | opens a turn: `<|turn>system\n`, `<|turn>user\n`, `<|turn>model\n` |
| `<turn|>` | 106 | control (**EOG**) | closes a turn: `<turn|>\n`. Listed by the loader as an EOG token — this is what stops generation |
| `<|think|>` | 98 | control | thinking switch: when `enable_thinking` is true the template emits `<|turn>system\n<|think|>\n…<turn|>\n` **at the very top** |
| `<|channel>` | 100 | user-defined | opens the thought channel: the model writes `<|channel>thought\n` |
| `<channel|>` | 101 | user-defined | closes the thought channel |
| `<start_of_turn>`, `<end_of_turn>`, `<think>`, `</think>` | — | **not in vocab** | v1's markers. They tokenized as ordinary text (`<`, `start`, `_of`, …) |

Two template behaviours matter for corpus design:

1. **Thinking off ⇒ the template itself pre-emits an empty channel.** With `add_generation_prompt` and `enable_thinking` false the template appends `<|turn>model\n<|channel>thought\n<channel|>` — the model then just answers. v1 never sent `<|think|>`, so the model reproduced the empty channel on its own (`data/routing_corpus/planner/05.txt` line 4–5: `<|channel>thought\n<channel|>To find out …`). **v1 contains zero reasoning-channel content.** The `coder/01.txt` sample (`<|channel>thoughtjoin` followed directly by a code fence, no `<channel|>`) is a malformed turn caused by the wrong turn markers.
2. **History strips thoughts.** The template's `strip_thinking` macro removes every `<|channel>…<channel|>` block from *previous* assistant messages when a multi-turn prompt is rendered. In production (`/v1/chat/completions`) earlier reasoning is therefore **not** in context. §3.4 decides how the corpus handles this.

`<|channel>`/`<channel|>` are *user-defined* tokens, so llama-server renders them in `/completion` output even without `-sp`; `<turn|>` is a *control* token and is **dropped** from output unless `--special` (`-sp`) is set — which is why v1's script had to append an end marker by hand (and appended a non-existent one).

### 1.2 `llama-imatrix.exe` (b8679) chunk mechanics — static evidence

From `--help` and `grep -a` on the binary (`bin/llama-cpp/llama-imatrix.exe`):

- Flags that exist: `-c/--ctx-size`, `-b`, `-ub`, `--chunks N` (max chunks, default -1 = all), `--chunk/--from-chunk N` (start offset), `--parse-special`, `--no-ppl`, `--output-format {gguf,dat}`, `-o`, `-ofreq/--output-frequency N`, `--save-frequency N`, `--process-output`, `-fit [on|off]`, `--in-file`. **There is no `--chunk-size`**: the chunk length *is* `-c`.
- Strings in the binary: `you need at least %d tokens for a context of %d tokens`, `computing over %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d`, `partial data (%.2f%%)`, `add_bos`, and the import **`llama_memory_clear`**.
- v1 logs confirm: `-c 1024` on a 1 767-token file → refused (`you need at least 2048 tokens`); the coder file → `computing over 2 chunks` (so 2 048 of its ≈2 900 tokens were used and the tail was **dropped**); `imatrix.chunk_size = 1024`, `imatrix.chunk_count = 2` written into the output GGUF; every layer's `counts` sums to **16 384 = 8 × 2 048** — i.e. `.counts` is a per-token-slot count (each token adds 1 to each of its 8 selected experts), shape `(128,)`.

What this establishes about the implementation (this is the same `compute_imatrix` lineage as `llama-perplexity`; the observable evidence is consistent with it in every detail):

| Property | Evidence | Confidence |
|---|---|---|
| Input file is tokenized once as a single stream; `n_chunks = floor(n_tokens / n_ctx)`; the remainder is never processed | `computing over 2 chunks` on a ≈2 900-token file at `-c 1024`; `imatrix.chunk_count` | **certain** (observed) |
| Refuses unless `n_tokens ≥ 2·n_ctx` | observed error message | **certain** |
| Chunks are processed as **independent causal sequences**: KV/memory is cleared before each chunk | `llama_memory_clear` import; `--from-chunk N` (starting at an arbitrary chunk would be meaningless if state carried across chunks) | high (static); **confirmed by experiment V3b before any result is trusted** |
| The **first token of every chunk is overwritten with BOS** when the model's `add_bos_token` is true (it is: `tokenizer.ggml.add_bos_token = true`) | `add_bos` string; perplexity/imatrix lineage | high (static); **confirmed by pilot check P2** |
| Tokenization uses `add_special = true`, so imatrix **prepends its own BOS** to the file, and `llama.dll` will happily produce **two** BOS tokens if the text also starts with `<bos>` | `llama.dll` contains the warning `…the prompt also starts with a BOS token. So now the final prompt starts with 2 BOS tokens…` | high; **confirmed by pilot check P1** (exact token count read from the 2× error message) |
| `-c` is used **as given** (not padded to 256) | no `padded` string in `llama.dll`; the only rounding message is `n_ctx is not divisible by n_seq_max` (irrelevant at `-np 1`) | medium; **confirmed by pilot check P0** (`n_ctx=` in the `computing over …` line with an odd `-c`) |

Consequence (the core of Flaw 3): a token at file position `p` is evaluated with attention over exactly the tokens `[floor(p/c)·c, p)` of the file, with token `floor(p/c)·c` replaced by BOS. Nothing else. A sample that starts at position `s` and ends at `e` with `floor(s/c) ≠ floor(e/c)` has its tail evaluated with **no attention to its own head** and with whatever unrelated text preceded it in that window. The v1 corpus had this at every one of its 1 024-token boundaries.

### 1.3 `llama-server.exe` (b8679) — reasoning-related flags (from `--help`)

- `--reasoning-format {none, deepseek, deepseek-legacy}` (default `auto`): controls whether thought tags are **extracted from chat-completion responses** into `message.reasoning_content`. This is chat-endpoint (`/v1/chat/completions`) parsing only.
- `--reasoning [on|off|auto]` (`LLAMA_ARG_REASONING`): sets the template's `enable_thinking` for the chat endpoint (`auto` = detect from template).
- `--reasoning-budget N`, `--reasoning-budget-message`: chat-endpoint thinking budget (injects the end-of-thinking tag).
- `--chat-template-kwargs '{"enable_thinking":true}'`: alternative way to set the template variable for the chat endpoint.
- `-sp, --special`: render control tokens in output text (default off).
- Binary strings present: `enable_thinking` (4×), `reasoning_content` (19×), `stop_type`, `stopping_word`, `tokens_predicted`, `tokens_evaluated`, `generation_settings`, `truncated`, `n_predict`, `return_t…` (fragment — `return_tokens` support is **to be confirmed in P5**).

**Decision:** corpus generation uses the raw **`/completion`** endpoint with a prompt we render ourselves from the real template tokens (§4.1), plus `-sp`. Reason: imatrix later re-tokenizes *text*, so the saved text must be the raw token stream including the thought channel and control tokens — exactly what the chat endpoint's reasoning parser would strip out. `--reasoning-format` / `--reasoning-budget` are therefore **not used**; they are irrelevant to `/completion`.

### 1.4 What v1's own `.counts` output tells us about sample size

`logs/imatrix_coder.gguf` (2 048 tokens, 2 chunks), read with `gguf`:

| Statistic (30 layers × 128 experts) | Value |
|---|---|
| Layer × expert cells with **zero** observations | **543 / 3 840 = 14.1 %** (per layer: 92–123 of 128 experts ever selected; worst layer 10: 92/128) |
| Cells with fewer than 20 observations | 1 391 / 3 840 = 36 % |
| Median per-expert count | ≈ 40 (⇒ selected on ≈ 2 % of tokens) |
| Largest per-expert count | ≈ 1 500 (⇒ selected on ≈ 73 % of tokens — quasi-shared experts exist) |

With 14 % of cells at zero and a third below 20 counts, no per-expert frequency estimate from v1 is stable; §6.7 turns this into a minimum corpus size.

### 1.5 Per-expert byte sizes (for the pool budget, thesis §6.2)

From tensor byte counts in the model GGUF (`n_bytes / 128`): `blk.N.ffn_gate_up_exps.weight` Q4_K = 272.2 MiB/layer → **2.127 MiB/expert**; `blk.N.ffn_down_exps.weight` Q8_0 = 257.1 MiB/layer → **2.009 MiB/expert**. So `e_size ≈ 4.14 MiB per expert per layer` (all expert tensors: 14 429 MiB total, mean 3.76 MiB/expert-layer because a few layers use other quant types). The 2.1 MiB figure in `expert_routing_visibility_plan.md` §2.4 counted only one of the two tensors and must be replaced when §6.2 is re-computed.

---

## 2. Definitions used throughout

- **Body**: the text of one session (or session prefix) **without** any leading `<bos>`. Its tokenization with `llama-tokenize.exe --no-bos --parse-special` has length `L` ("body tokens").
- **Document** (imatrix chunk content): `[BOS] + body` — `L + 1` tokens. The BOS comes from imatrix's own `add_special`, never from the text.
- **Span**: a maximal token range `[s, e)` of one chunk type inside a body, from the span index (§4.3). Types: `user_prompt`, `reasoning`, `code`, `factual`, `creative`, `control` (turn-marker tokens) — the last two of the thesis's four are turn-labelled (§4.3).
- **Cut point** `b_k`: a body-token index at which a prefix-differential run is made (§5.4); `b_0 = 0 < b_1 < … < b_m = L`.
- **`counts_X[l, e]`**: the `blk.l.ffn_down_exps.weight.counts` tensor from run X (only `ffn_down_exps`; `ffn_gate_up_exps` carries identical ids and would double-count). Invariant: `Σ_e counts_X[l, e] = 8 · (tokens processed in run X)` for every `l`.
- **`p_T[l, :]`** = `counts_T[l, :] / Σ_e counts_T[l, e]` — the per-layer routing distribution of chunk type `T`.
- **Affinity set** `S_T,l(α)`: the smallest set of experts, taken in descending `p_T[l, ·]` order, whose mass is ≥ α (α = 0.80 and 0.90 reported).
- **JSD** = Jensen–Shannon divergence, base 2, in [0, 1]. **Jaccard**(A, B) = |A∩B| / |A∪B|.

---

## 3. Flaw 1 — corpus depth: from isolated snippets to full-depth sessions

### 3.1 Why v1's snippets could not represent production routing

Production envelope (`docs/gemma4_26b_a4b_vram_benchmark.md`): `ctx = 8 192` (`-ngl 11`) or `16 384` (`-ngl 10`), single slot. Real sessions accumulate thousands of tokens of code, discussion and reasoning before most generated tokens. The router input at position `p` is the residual stream after attention over `[0, p)`, so a token's routing is a function of its whole context. v1 profiled tokens whose context was ≤ 300 tokens, i.e. a *different input distribution* to the router than production's.

One architectural fact bounds how strong this effect can be, and it must be stated rather than hand-waved: **25 of the 30 layers use sliding-window attention with `n_swa = 1 024`** (`gemma4.attention.sliding_window = 1024`, pattern `[T,T,T,T,T,F,…]`); only the 5 global layers (5, 11, 17, 23, 29) attend beyond 1 024 tokens. Deep-context influence on routing therefore flows through those five layers' residual contributions to all subsequent layers. It is real but its magnitude is an **empirical question** — so v2 measures it directly (V4, §6.4) instead of assuming either answer. If V4 shows no measurable depth effect, the session length can be cut to ≈2 048 tokens and the profiling cost drops ~4×; if it does, the long sessions are mandatory. Either result is recorded.

### 3.2 Session families and scripted turn sequences

Sessions are generated by the model itself, driven by a scripted list of user turns per family, stored as JSON in `data/routing_corpus_v2/turns/{family}_{nn}.json` (committed; small). Every turn carries a **turn label** used by the segmenter (§4.3) for prose that no marker distinguishes.

| Family | Sessions (initial) | Turn script shape (8–14 user turns) | Chunk types it feeds |
|---|---|---|---|
| `coder` | 3 | start a small project (spec) → implement module → add tests → "here is a failing test output, fix it" → refactor → add a CLI → extend with a second module → review/explain → performance question → final summary. Each turn's user text quotes the previous code back in part (as a real user would), so the context is genuinely accumulated. | code (heavy), reasoning (thought preambles), factual (explanations) |
| `planner` | 3 | multi-step migration/benchmark/incident planning → dependency ordering → estimation with arithmetic → trade-off matrix → "the constraint changed, re-plan" → risk register → step-by-step word problems → decision write-up. | reasoning (heavy), factual, small code (tables/config) |
| `writer` | 2 | brief → outline → draft a story/announcement/product copy → revise tone → continue the story → summarize → write the FAQ. Turns labelled `creative` except the FAQ/summary (`factual`). | creative (heavy), reasoning, factual |

Targets: body length `L` ∈ [4 000, 8 191] tokens per session (upper bound = `-c 8192` document, §5.3); initial total ≈ 8 sessions ≈ 50–60 k body tokens. §6.7 sets the per-type minimum (20 k) and the stopping rule for generating more sessions.

### 3.3 Generation parameters

- Server: `llama-server.exe -m <LLMS_DIR>\gemma4-26B-A4B-it-GGUF\google_gemma-4-26B-A4B-it-Q4_K_M.gguf -ngl 0 --override-tensor "\.ffn_.*_exps\.=CPU" -c 8192 -np 1 -t 2 -tb 2 -sp --swa-full --no-webui --no-warmup --port 8260` under the safety wrapper (§7). **Amended 2026-10-01**: `-ngl 9` was the original choice, but this SWA model's default windowed SWA cache makes `cache_prompt` fall back to "forcing full prompt re-processing" on every turn (confirmed empirically — log line present), which would have made an 8-14 turn session cost hours instead of minutes. `--swa-full` fixes the caching but needs a full-ctx-sized SWA KV cache; that only fits in 2GB VRAM at `-ngl 0` (tested: `-ngl 9` + `--swa-full` OOMs on the compute buffer; `-ngl 0` + `--swa-full` reaches "server is listening" with ~1.1GB compute buffer, no weights on GPU). Generation speed is secondary here — profiling itself (`llama-imatrix.exe`, §5) never uses this server or `--swa-full` and keeps the validated `-ngl 7-11` operating points.
- Per turn: `POST /completion` with `{"prompt": <rendered transcript>, "n_predict": 3072, "cache_prompt": true, "seed": <session seed>, "temperature": 1.0, "top_k": 64, "top_p": 0.95, "return_tokens": true}`. **Amended 2026-10-01**: the original 1536 budget was empirically too small — a real planner turn 1 ("migrate MySQL to PostgreSQL, zero downtime") produced a genuine, well-structured 1536-token reasoning span that was still mid-sentence (`stop_type: "limit"`) on two separate seeds, never reaching the answer. Raised to 3072. Sampling values are the model's own recommendations (`general.sampling.temp = 1.0`, `top_k = 64`, `top_p = 0.95` in the GGUF header); v1's 0.2 produced repetitive, low-entropy text that under-represents production. A fixed seed per session makes regeneration reproducible (modulo the numeric caveat in V3).
- `cache_prompt: true` makes each turn prefill only the new tokens (the transcript is append-only, §3.4), so a session costs ≈ (generated tokens ÷ 4 tok/s) + (user tokens ÷ ~20 tok/s) ≈ **25–35 min** for a 6 000-generated-token session.
- Turn acceptance: `stop_type == "eos"` (the model closed the turn itself) **and** the §6.2 CoT checks pass. A turn that hits `n_predict` (`stop_type == "limit"`) is regenerated once with `seed + 1`; if it fails again the session is **truncated before that turn** (never kept with a dangling channel).

### 3.4 Decision: thoughts are retained in the accumulated transcript

Production strips earlier thoughts from history (§1.1). For profiling, v2 **keeps** each turn's full raw output (thought channel included) in the transcript that the next turn is generated from. Reasons:

1. It makes each session a **single linear causal stream**: the text fed to imatrix *is* what the model saw at every generation step. That is what makes §5's boundary arithmetic exact and the prefix-differential subtraction valid (a prefix of the stream is exactly the model's context at that point).
2. It halves profiling cost versus the stripped variant, where every turn is a different document sharing no exact prefix with the next (each turn would need its own differential series).
3. The decode-time context of turn `t` (what Phase 4's pool serves) is identical in both variants except for earlier turns' thoughts, which are far from the decode position and visible only to the 5 global layers.

This is a deliberate deviation from the template and is **tested, not assumed**: V4b (§6.4) regenerates one planner session in the stripped variant and compares the per-type histograms; if the difference exceeds the split-half noise, the final run switches to the stripped variant (cost ×2) and this section is amended. The pilot also checks that turns ≥ 2 still open and close the channel correctly with thoughts in history (P3).

---

## 4. Flaw 2 — profiling axis: chunk types, not roles

### 4.1 Prompt rendering (real template tokens)

First turn of a session (thinking on, optional family system text `S`):

```
<|turn>system
<|think|>
{S}<turn|>
<|turn>user
{U1}<turn|>
<|turn>model
```

(`<bos>` is **not** written: the server's `/completion` tokenizer adds it — P1 confirms — exactly as imatrix does later, so the generation-time and profiling-time token streams agree.) The model then produces `<|channel>thought\n{reasoning}<channel|>{answer}` and stops on `<turn|>`. The script appends `<turn|>\n` (P4 establishes whether `-sp` already includes the stop token in `content`; it must not be doubled), then `<|turn>user\n{U2}<turn|>\n<|turn>model\n`, and so on. Rendering is done in the script, not via `/apply-template`, because the transcript must retain thoughts (§3.4) and the template would strip them.

### 4.2 Elicitation of genuine reasoning

- `<|think|>` in the system turn is the only switch the template defines. No prompt-engineering ("think step by step") is used — it would pollute the answer text with meta-instructions.
- Reasoning-heavy turns (planner family, and "fix this failing test" turns in coder) are designed to *require* multi-step work so the channel is non-trivial; §6.2 enforces a minimum channel length per labelled turn and a session-level reasoning fraction.
- If P3 shows the model emits an empty channel even with `<|think|>` (template variant mismatch), the fallback is `--chat-template-kwargs '{"enable_thinking":true}'` through the chat endpoint with `--reasoning-format none` (keeps thoughts inline in `content`) and `-sp`; the transcript is then reconstructed from `content`. This is a fallback only — it loses control over exact prompt tokens.

### 4.3 Segmentation into chunk-type spans (`scripts/segment_chunk_types.py`, new)

Input: `sessions/{id}.body.txt` (raw transcript, no leading `<bos>`) and `sessions/{id}.turns.jsonl` (per-turn: role, label, char offsets, `tokens_predicted`, `stop_type`, and the server's generated ids if `return_tokens` works).

1. Tokenize the whole body **once** with the profiling tokenizer: `llama-tokenize.exe -m <model> -f body.txt --no-bos --parse-special` (piece mode) and again with `--ids`. Build the char→token map by accumulating piece byte lengths. This is the same `common_tokenize` path imatrix uses, so span indices are in imatrix's coordinates.
2. Mark spans, in priority order, on token indices:
   - `control`: `<|turn>role\n` header tokens and each `<turn|>\n`; the system turn (`<|think|>` line and `S`).
   - `user_prompt`: everything inside a user turn.
   - `reasoning`: from the `<|channel>` token (id 100) through the `<channel|>` token (id 101) inclusive, inside a model turn. Both are single tokens, so these boundaries are exact.
   - `code`: from the token containing an opening ``` to the token containing the matching closing ``` inclusive (fence detection on text: a line starting with ```; the snap from char offset to token index is recorded and must be ≤ 1 token — §6.3).
   - Remaining model-turn tokens: `factual` or `creative` **according to the turn label**. Markers cannot separate these two (thesis §3's `ChunkClassifier` uses heuristics); in a scripted corpus the label is ground truth and is used instead.
3. Output `sessions/{id}.spans.json`: `[{"start": s, "end": e, "type": T, "turn": t}]`, covering `[0, L)` exactly with no gaps or overlaps (asserted).
4. Consistency: if `return_tokens` works, compare the server's generated ids for each model turn against the re-tokenized ids of the same text; report the mismatch rate (P6). Re-tokenization of generated text is not guaranteed to reproduce the generated token sequence (BPE non-uniqueness); the rate is expected to be < 1 % and is recorded as a known approximation of Tier 1 (Tier 2 profiles live decode and has none).

### 4.4 Outputs: one accumulator per chunk type

Per-type accumulators `counts_T[l, e]` for `T ∈ {reasoning, code, factual, creative}` plus `user_prompt`, `control`, and `mixed` (windows that fail the purity rule in §5.4) — built by `scripts/aggregate_expert_affinity.py` from the differential runs (§5.4). These are the direct analogue of thesis §6.1's `chunk_type → affinity experts` table, for 30 layers × 128 experts. There are **no separate per-role imatrix runs**: the role histograms are derived (§4.5) and cross-checked against whole-session runs (V8).

The "per-type corpus files" requested (`reasoning.txt`, `code.txt`, …) are **not** produced as imatrix inputs: concatenating extracted spans of one type would recreate Flaw 3 (span *k* from session A followed by span *j* from session B, sliced by the chunk window). They are produced only as *human-readable exports* of the span index (`exports/{type}.txt`, each span prefixed with its session/turn id) for inspection — and the design must never feed them to imatrix.

### 4.5 From chunk types to the product's coder/planner sets

Phase 4 uses affinity at two moments: **(a)** session start / role selection — a static pre-warm; **(b)** each chunk boundary emitted by the Phase 1 monitor — a dynamic refinement to the chunk type's set. (b) uses `S_T,l(α)` directly. (a) is a mixture: for role `R` with measured chunk-type token fractions `w_R,T` (from the span index of `R`'s own sessions, not assumed):

```
p_R[l, :] = Σ_T  w_R,T · p_T[l, :]        (T over reasoning, code, factual, creative, user_prompt)
S_R,l(α)  = smallest α-mass set of p_R[l, :]
```

Expected shape (to be replaced by measured `w`): coder ≈ {code 0.4–0.5, reasoning 0.2–0.3, factual 0.2, user_prompt 0.1}; planner ≈ {reasoning 0.4–0.5, factual 0.3, code 0.05, user_prompt 0.15}. V8 checks that `p_R` computed this way equals the whole-session histogram of `R`'s sessions (an identity if the bookkeeping is right). The RAM-pool budget per layer for a role is `|S_R,l(0.9)| × 4.14 MiB`; the union `|∪_T S_T,l(0.9)|` bounds what a pool must hold to serve all chunk types of that role without reload.

---

## 5. Flaw 3 — chunk-boundary causal corruption

### 5.1 The problem, stated precisely

With the §1.2 mechanics, imatrix at `-c c` evaluates file token `p` with context `[c·⌊p/c⌋, p)`, first token of that range replaced by BOS. For a sample occupying `[s, e)`:

- if `⌊s/c⌋ = ⌊(e−1)/c⌋`: every token of the sample sees `[c·⌊s/c⌋, p)`, which contains the sample's own head **plus whatever unrelated text preceded it inside the window** (v1: the previous snippets). The router input is contaminated by foreign context.
- if not: the tokens after the boundary see *none* of the sample's own earlier tokens and a BOS where a mid-sentence token was. Their routing is computed on a context that never occurs in production — not "less representative", but wrong.

In v1 both happened at every window. The dismissal of this as minor was itself wrong: the corrupted tokens are exactly the ones deepest into each sample, i.e. the ones Flaw 1 says matter most.

Distinguishing case (b) of the task statement: when a *single continuous document* is windowed, the tokens after a boundary still lose their own prefix (they start from BOS mid-document). That is milder than v1 (no *foreign* text enters), but it is still a context truncation and it directly contradicts Flaw 1's requirement of deep context. It is therefore **not** acceptable for v2's main runs either; windowing is used only where the window *is* the whole document.

### 5.2 Options evaluated

| Option | Mechanism | Verdict |
|---|---|---|
| (a) one invocation per sample, `-c` = that sample's length | needs `≥ 2·c` tokens per file → duplicate the sample (`body + <bos> + body`) and `--chunks 1`; cost = one pass over the sample; no filler, no repetition inside a chunk | **adopted** as the building block (`D×2` construction, §5.3) |
| (b) long continuous documents, windowed at `-c` | tokens after each internal boundary lose their prefix (see above); also cannot attribute counts to chunk types inside a window | rejected for main runs; the depth test V4 uses the *shallow* variant deliberately as a contrast |
| (c) pad/align sample boundaries to multiples of `-c` | padding tokens are routed too (their counts pollute the histogram) and a shared `-c` forces all samples to the same length | rejected; alignment is achieved by (a) instead, which needs no padding |

Additional constraint discovered: attribution to chunk types needs per-**span** counts, but a chunk yields only a whole-chunk histogram. (a) alone gives whole-session histograms; the **prefix-differential** construction (§5.4) extends (a) to spans without any padding or boundary crossing.

### 5.3 The `D×2` construction (exact arithmetic) — amended to D×3, see note below

> **Amended 2026-10-01**: empirically, 2 copies left near-zero margin against imatrix's `>= 2*ctx tokens`
> gate — retokenizing the same prefix text as part of a longer file can retokenize a few dozen-to-hundred
> tokens differently right at the `<bos>` separator boundary than it does in isolation (a BPE merge-
> boundary effect), which pushed the largest pilot sessions' D×2 files just under the threshold (observed:
> a 6634-token body's nominal `2L+1=13269` fell 1 short of the required `13270`; a direct retokenization of
> the actual on-disk D×2 file read **13409–13410**, not the predicted 13269 — the second copy is not
> byte-for-byte identically tokenized to the first once concatenated). This does not affect correctness:
> `--chunks 1` only ever processes chunk 0 (`[BOS] + prefix_text`), so retokenization drift in the
> second/third copy is invisible to the profiled counts — it only has to be long enough to clear the gate.
> The construction below now triplicates (`body + <bos> + body + <bos> + body`) for comfortable margin.

For a body of `L` tokens (measured by `llama-tokenize.exe --no-bos --parse-special --show-count`):

```
file text   = body_text + "<bos>" + body_text
imatrix run = -c (L+1)  --chunks 1  --parse-special
file tokens = [BOS_auto] + body(L) + [<bos>=2] + body(L)   = 2L + 2 = 2·(L+1)   ✓ satisfies the 2× rule exactly
chunk 0     = tokens[0 : L+1] = [BOS] + body                 (token 0 already BOS → the BOS overwrite is a no-op)
chunk 1     = tokens[L+1 : 2L+2] = [<bos>→BOS] + body        (identical; skipped by --chunks 1)
```

Every profiled token sees exactly `[BOS] + body[0 : p)` — its true generation-time context. Requirements: the body text must not contain a literal `<bos>` (else a double BOS shifts everything — P1 checks the count); `-c` must be honoured un-padded (P0); `-c ≤ 8192` keeps the run inside the validated VRAM envelope (§7). Worked example, session `coder_01` with `L = 7 900`: `-c 7901`, file = 15 802 tokens, one pass ≈ 7 901 / 20 tok/s ≈ 6.6 min + load.

The whole-session run of a session is the `D×2` run of its full body.

### 5.4 Prefix-differential attribution to chunk types

Because prefill is causal, `counts` accumulated over the first `b` tokens of a chunk do not depend on tokens after `b`. Hence for cut points `b_0 = 0 < b_1 < … < b_m = L` at span boundaries:

```
run R_k : D×2 of body[0 : b_k]      → -c (b_k + 1), --chunks 1      (k = 1..m; R_m is the whole-session run)
counts(window_k) = counts(R_k) − counts(R_{k−1})        with counts(R_0) := 0
```

`window_k = body[b_{k−1} : b_k)` and each window is attributed to the chunk type of the span(s) it contains. Rules:

- Cut points are placed at span boundaries only, and only at boundaries that fall on **pre-tokenization boundaries** (the special tokens 100/101/105/106, or a newline before a fence), so that `tokenize(body[0:b_k])` equals the first `b_k` tokens of `tokenize(body)` — asserted (§6.3), never assumed.
- Minimum window length **256 tokens**: consecutive short spans are merged into one window. A window is attributed to type `T` if ≥ 90 % of its tokens are `T` (by the span index); otherwise it is `mixed` and excluded from the affinity tables but reported (coverage must be ≥ 85 % of body tokens, §6.3).
- Consecutive spans of the same type share a window (no cut between them).
- The BOS position 0 belongs to window 1 (`control`).

Exactness conditions, each tested in V3: (i) determinism of identical runs (bitwise-equal counts), (ii) insensitivity of routing to the ubatch tiling (the last ubatch of `R_{k−1}` is shorter than the same tokens' ubatch in `R_k`; near-tie experts could flip). If (ii) shows flips, they are measured as a noise floor (expected ≪ 0.1 % of token-slots) and reported alongside every differential table; if they exceed 0.5 % the differential method is abandoned for Tier 2 (which needs none of this) and the corpus is reused unchanged.

Cost: `Σ_k (b_k + 1)` tokens ≈ `m · L / 2`. For `L ≈ 8 000` and `m ≈ 12–16` windows: 48–64 k tokens ≈ **40–55 min per session** at 20 tok/s plus `m` model loads (~25 s each from warm page cache). Eight sessions ≈ 6–7 h unattended. Knob: raising the minimum window to 512 tokens roughly halves `m`.

### 5.5 Context-size deviation and why it is safe

Differential runs use `-c` between 257 and 8 192, below the production envelope's 8 192–16 384. This is not a convenience: `-c` *is* the document length here, and a document is never longer than the session (≤ 8 191 body tokens). Safety: VRAM decreases monotonically with `-c` (only the 5 global layers' KV scales with it; SWA cache is fixed at 1 536 cells), so every run is at or below the validated `-ngl 9`, `-c 8192` point (≤ 1 633 MiB). `-fit off` is passed so the `llama_params_fit` step cannot alter `ngl`/`ctx` (the v1 logs show it trying and aborting). Runs never exceed 8 192 because sessions are capped at 8 191 body tokens; extending to 16 384 is possible (`-ngl 9` @ 16 k = 1 633 MiB validated) but doubles cost and is deferred until V4 says depth matters beyond 8 k.

---

## 6. Rigorous empirical validation plan

All checks are implemented in `scripts/validate_affinity_pipeline.py` (new) and produce a single `logs/affinity_validation_report.md` with PASS/FAIL per check. Every command below runs under the §7 wrapper. Thresholds are stated up-front so results cannot be rationalised afterwards.

### 6.1 V0 — Pilot (before any full generation; ≈ 2 h machine time)

Pilot corpus: **two** sessions only — `planner_pilot` (6 user turns) and `coder_pilot` (6 user turns), target `L ≈ 3 000–4 000` each.

| # | Check | Command / method | Pass criterion |
|---|---|---|---|
| P0 | `-c` honoured un-padded | `llama-imatrix.exe … -c 1001 --chunks 1 -f <any 2 002+-token file> --parse-special --no-ppl -fit off` | log line `computing over N chunks, n_ctx=1001` |
| P1 | imatrix's auto-BOS and no double BOS | run imatrix on `D×2` of the pilot body with `-c (2L+3)` (deliberately too large) → read `the data file you provided tokenizes to only N tokens` | `N == 2L + 2` exactly (auto-BOS present, none from text). If `N == 2L + 1`, auto-BOS is absent → §5.3 layout switches to writing `<bos>` at file start; recorded in the report |
| P2 | chunks are independent identical documents | `counts(D×2, --chunks 2)` vs `counts(D×2, --chunks 1)` (same as V3b, run early on the pilot body) | `counts(2 chunks) == 2 × counts(1 chunk)` bitwise (requires V3a). Note: chunk 1 starts with the literal separator `<bos>`=2, so the result holds whether or not imatrix overwrites the first token with BOS — v2's layout never has a chunk start mid-text, so the overwrite is a no-op by construction; the property v2 depends on and P2 verifies is that the two chunks are processed as independent, identical causal documents |
| P3 | thinking channel is real | inspect both pilot transcripts: every model turn (incl. turns ≥ 2, thoughts in history) starts with `<|channel>thought\n`, contains exactly one `<channel|>`, then answer text | 100 % of accepted turns; reasoning span ≥ 64 tokens in ≥ 80 % of planner turns |
| P4 | `-sp` rendering of `<turn|>` | check whether `content` of a `stop_type:"eos"` turn ends with `<turn|>` | documented either way; the renderer appends `<turn|>\n` only if absent |
| P5 | `return_tokens` support | response has non-empty `tokens` array | if absent: P6 is skipped and noted |
| P6 | re-tokenization fidelity | `return_tokens` ids vs `llama-tokenize --ids --no-bos --parse-special` of the same turn text | mismatch ≤ 1 % of tokens per turn (report the value) |
| P7 | span index integrity | `segment_chunk_types.py` on both pilots | spans cover `[0, L)` exactly; fence snap ≤ 1 token; every reasoning span starts at id 100 and ends at id 101 |
| P8 | `.counts` readable and per-expert | `python -c "from gguf import GGUFReader; r=GGUFReader('logs/…gguf'); [print(t.name,t.shape) for t in r.tensors if t.name.endswith('ffn_down_exps.weight.counts')]"` | 30 tensors, shape `(128,)`; `Σ_e counts[l,e] == 8·(L+1)` for all `l` |
| P9 | VRAM envelope during the largest pilot run (`-c ≈ 4 000`) | sum of the three `Vulkan0 … buffer size` lines in the imatrix log | ≤ 1 700 MiB |

Only when P0–P9 pass does full generation start. P0/P1/P2 each need a model load (~30 s warm) plus at most one chunk pass (≤ 4 min).

### 6.2 V1 — CoT elicitation (automated, per session)

- Per accepted model turn: exactly one `<|channel>` and one `<channel|>`, in that order, `<|channel>` at turn start.
- Reasoning span length ≥ 64 tokens for every turn labelled `reasoning`-expected; ≥ 16 tokens for all other model turns (an empty or 1-line channel is treated as "thinking off" and the turn is regenerated with `seed+1`, once).
- Session level: reasoning tokens / model-generated tokens ≥ 0.25 for `planner`, ≥ 0.10 for `coder`, ≥ 0.10 for `writer`.
- ≥ 90 % of turns end with `stop_type == "eos"`.
FAIL ⇒ the session is not used; the turn script is revised.

### 6.3 V2 — Chunk-alignment (automated, per imatrix input file, before the run)

For every file `imatrix_inputs/{session}_{k}.txt` with declared `-c = b_k + 1`:
1. `llama-tokenize.exe --no-bos --parse-special --show-count -f file` ⇒ count `== 2·b_k + 1` (auto-BOS excluded from this count; with it, `2·(b_k+1)` — the P1 result fixes which).
2. `tokenize(file)[b_k+1] == 2` (the separator `<bos>`) — i.e. chunk 1 starts with BOS.
3. `tokenize(prefix_text_k) == tokenize(body)[0 : b_k]` (prefix re-tokenization identical to the in-context tokenization).
4. Purity/coverage: `mixed` windows ≤ 15 % of body tokens per session; every non-mixed window ≥ 90 % one type.
5. After the run: log line `computing over 1 chunks, n_ctx=<b_k+1>` and `Σ_e counts[l,e] == 8·(b_k+1)`.
Any failure aborts the job list before the model is loaded.

### 6.4 V3 — Determinism, ubatch invariance, chunk independence, depth (≈ 45 min)

| # | Test | Pass |
|---|---|---|
| V3a | same `D×2` file, same flags, run twice | `counts` bitwise identical in all 30 × 128 cells |
| V3b | `D×2`, `--chunks 2` vs `--chunks 1` | `counts(2 chunks) == 2 · counts(1 chunk)` exactly (chunks independent + deterministic). A non-multiple result would mean state leaks across chunks ⇒ the whole Tier-1 approach is re-examined before anything else |
| V3c | same file, `-ub 512` vs `-ub 256` | flipped token-slots `Σ_l Σ_e |Δ| / 2` ≤ 0.1 % of `8·(L+1)·30`; the measured value is the differential noise floor; > 0.5 % ⇒ §5.4 abandoned in favour of Tier 2 |
| V4 | **depth sensitivity**: choose a `code` span `S` (≥ 768 tokens) that sits at body position ≥ 4 096 in a coder session; `p_deep` = its differential window; `p_shallow` = `D×2` run of `S` alone (with a minimal `<|turn>model\n` header). Also take a *different* shallow code span `S'` of similar length. | Report per-layer `JSD(p_deep, p_shallow)` vs `JSD(p_shallow(S), p_shallow(S'))`. **Depth matters** if `JSD_deep > JSD_between-spans` in ≥ 15/30 layers ⇒ long sessions stay mandatory. Otherwise ⇒ record "no first-order depth effect at 8 k", allow `L ≥ 2 048` sessions for cost |
| V4b | thought-retention fidelity (§3.4): regenerate `planner_02` with thoughts stripped from history (template-faithful), profile the same way | `JSD(p_T,retained, p_T,stripped)` ≤ split-half `JSD_within,T` (V6) in ≥ 25/30 layers for each `T` ⇒ retention validated; else switch to the stripped variant for the final run |

### 6.5 V5 — Type separation (signal vs noise)

For each pair of types `(T1, T2)` among the four, and each layer `l`:
```
R_l = JSD(p_T1[l], p_T2[l]) / max(JSD_within,T1[l], JSD_within,T2[l])
```
where `JSD_within,T[l]` is the split-half JSD from V6. PASS (locality signal present) if `R_l ≥ 3` in ≥ 20 of 30 layers for *every* pair; WEAK if 1.5 ≤ median `R_l` < 3 (double the corpus, re-test); **FAIL** if median `R_l` < 1.5 for any pair — then either the span index is wrong (re-inspect `exports/{type}.txt`) or this model does not route by chunk type, and thesis §6 must say so for this model. Also reported: `Jaccard(S_T1,l(0.8), S_T2,l(0.8))` per layer — expected well below the within-type value of V6.

### 6.6 V6 — Reproducibility (split-half)

Split each type's windows into halves **by session parity** (odd/even session numbers, so both halves have full-depth context and no within-session correlation). Per layer: Spearman `ρ` between `counts_T,half1[l, :]` and `counts_T,half2[l, :]`, and `Jaccard(S_half1,l(0.8), S_half2,l(0.8))`. PASS: `ρ ≥ 0.90` in ≥ 27/30 layers **and** median Jaccard ≥ 0.75, for every type. FAIL ⇒ that type's corpus is too small; generate more sessions of the family richest in it (§3.2) and re-run V6. `JSD_within,T[l]` between the halves is the noise floor used by V5.

### 6.7 V7 — Sample-size justification and stopping rule

Per layer and token, 8 of 128 slots are filled. Expert `e`'s count over `N` tokens is ≈ Binomial(`N`, `q_e`) with `q_e = Pr[e ∈ top-8]`; relative standard error ≈ `1/√(N·q_e)`.

| Requirement | Arithmetic | `N` (tokens per type) |
|---|---|---|
| ±20 % (2σ) on any expert with `q_e ≥ 1 %` (selected on 1 token in 100) | `N·q_e ≥ 100` | ≥ 10 000 |
| ±20 % (2σ) on any expert with `q_e ≥ 0.5 %` | same | **≥ 20 000** |
| "cold" claim: an expert with **zero** observations has `q_e < 0.015 %` at 95 % (rule of three, `3/N`) | `N ≥ 3 / 0.00015` | ≥ 20 000 |
| stable α = 0.8 set boundary: from v1 the cutoff falls among experts with `q ≈ 1–3 %`; at `N = 20 000` their counts are 200–600 ⇒ SE 4–7 %, so only experts within ≈ 10 % of the cutoff can flip | — | 20 000 suffices; 40 000 halves the flip band |

v1 had `N ≈ 2 048` per role ⇒ an expert at `q = 1 %` had an expected count of 20 (SE 22 %), and 14 % of cells were never observed at all — consistent with the table. **Rule: every chunk type must reach `N_T ≥ 20 000` tokens (target 40 000) and pass V6.** With the §3.2 mix (≈ 55 k body tokens) `creative` and possibly `factual` will fall short initially; the stopping rule generates additional `writer`/`planner` sessions until both conditions hold for all four types. Uniform routing would give `q = 6.25 %`; the observed v1 distribution (max 73 %, median 2 %) is far from uniform, which is itself the first evidence of structured routing.

### 6.8 V8 — Bookkeeping identities (exact)

- `Σ_k counts(window_k) == counts(R_m)` per session (telescoping — trivially true; a mismatch means a file/flag mix-up).
- `p_R` computed from §4.5 with measured `w_R,T` equals the normalised sum of `R`'s whole-session `counts(R_m)` (up to `mixed` windows).
- `Σ_e counts[l, e] == 8 · (tokens processed)` in every run and layer.

### 6.9 V9 — Comparison with v1 (report only)

`JSD(p_coder-sessions, p_v1-coder)` and `JSD(p_planner-sessions, p_v1-planner)` per layer, alongside V6's within-type values, to quantify how far v1 was from the corrected measurement. No pass criterion; v1 is known-invalid (§0).

---

## 7. Execution safety (mandatory for every process this design spawns)

This machine (i5-4460 4-core, GT 1030 2 GB) **froze completely** earlier this session when a model process was launched unpinned. Every `llama-server.exe`, `llama-imatrix.exe`, `llama-tokenize.exe` (it loads the 17 GB GGUF via mmap for the vocab) and Python aggregation process uses the idiom from `scripts/probe_role_models.ps1` lines 33–41 (already replicated in `build_routing_corpus.ps1` and `profile_experts_imatrix.ps1`):

```powershell
$proc = Start-Process -FilePath $Exe -ArgumentList $argList `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -PassThru -WindowStyle Hidden
Start-Sleep -Milliseconds 500
$proc.ProcessorAffinity = 12                                        # cores 2+3 only (mask 0xC)
$proc.PriorityClass     = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
```

Rules (none may be relaxed):

1. Always pass `-t 2 -tb 2` (server and imatrix). Affinity alone does not stop ggml from spawning 4 threads on 2 cores. The `--cpu-mask`/`--prio` flags of llama.cpp are **not** a substitute for the wrapper (they do not cover the Vulkan/driver threads).
2. **One model process at a time.** The server (generation) and imatrix (profiling) are never alive together — each takes ≈ 1.6 GB of a 2 GB card. The job runner waits for `HasExited` and then 5 s before the next launch.
3. VRAM guard: after load, parse the three `Vulkan0 … buffer size` lines; abort the job if their sum > 1 700 MiB. Operating point for all runs: `-ngl 9 --override-tensor "\.ffn_.*_exps\.=CPU" -b 512 -ub 512 -fit off`; `-c` ≤ 8 192 (§5.5).
4. Per-job wall-clock ceilings: server turn 600 s (`Invoke-RestMethod -TimeoutSec`), imatrix job `30 s + 0.12 s × c` (≈ 17 min at `c = 8 192`, 3× the expected pass time); on timeout `Stop-Process -Force` and mark the job failed — never retry automatically in the same run.
5. Warm the page cache once per session of work (`Get-Content <gguf> -ReadCount 0 | Out-Null` under the same wrapper) before timing anything; cold-mmap stalls were observed in the benchmark doc.
6. Speed numbers from imatrix/`cb_eval` runs are **not** inference benchmarks and are never written into `docs/gemma4_26b_a4b_vram_benchmark.md`.
7. All large outputs go to `logs/` (gitignored). Only the turn scripts, session transcripts, span indexes and the final markdown/JSON tables are committed.

---

## 8. Files, scripts and the fate of v1

### 8.1 Directory layout (new)

```
data/routing_corpus_v2/
  turns/{family}_{nn}.json          scripted user turns + labels (committed)
  sessions/{family}_{nn}.body.txt   raw transcript, no leading <bos> (committed)
  sessions/{family}_{nn}.turns.jsonl per-turn metadata from the server (committed)
  sessions/{family}_{nn}.spans.json  span index in imatrix token coordinates (committed)
  imatrix_inputs/{session}_{k}.txt   D×2 prefix files (generated, gitignored via logs-style rule)
  jobs.json                          [{file, ctx, out, session, k, window_type}] (generated)
  exports/{type}.txt                 human-readable span dumps — NEVER fed to imatrix
logs/imatrix_v2/{session}_{k}.gguf|.log|.log.err
logs/affinity_validation_report.md
logs/expert_affinity_{type|role}.json
```

### 8.2 Script changes

- **`scripts/build_routing_corpus.ps1`** (restructure, keep the wrapper and readiness loop):
  - parameters `-Family`, `-Session nn`, `-Seed`, `-TargetTokens 8000`, `-MaxTokens 8191`, `-TurnsFile`;
  - server args: add `-sp`, `-tb 2`, `-fit off`; keep `-ngl 9 -c 8192 -np 1`;
  - prompt rendering per §4.1 (`<|turn>`/`<turn|>`/`<|think|>`), **remove** `<bos>`/`<start_of_turn>`/`<end_of_turn>` entirely;
  - request body per §3.3 (`cache_prompt`, `seed`, model-default sampling, `return_tokens`);
  - per-turn acceptance/regeneration per §3.3 and §6.2; stop when the next turn would exceed `MaxTokens` (token counts from `tokens_evaluated + tokens_predicted`);
  - write `body.txt` + `turns.jsonl`; drop the `*_combined.txt` concatenation.
- **`scripts/segment_chunk_types.py`** (new): §4.3; calls `llama-tokenize.exe` under the wrapper; emits `spans.json` and `exports/`.
- **`scripts/build_imatrix_jobs.py`** (new): cut-point selection (§5.4 rules), `D×2` file generation, `jobs.json`, and the pre-run V2 assertions (§6.3).
- **`scripts/profile_experts_imatrix.ps1`** (parametrize): `-Jobs jobs.json` (default) or `-File/-Ctx` for ad-hoc runs; per job `-c <ctx> --chunks 1 --parse-special --no-ppl -fit off --output-format gguf -o <out>`; the wrapper, VRAM guard, and timeout per §7; log check `computing over 1 chunks, n_ctx=<ctx>`.
- **`scripts/aggregate_expert_affinity.py`** (new): differential subtraction, per-type/role accumulators, `S_T,l(α)`, JSD/Jaccard/Spearman, V5–V9; writes the JSON tables and the report.
- **`scripts/validate_affinity_pipeline.py`** (new): V0 pilot driver + V3/V4 experiments (calls the two runners).

### 8.3 v1 corpus and outputs — disposition

- `data/routing_corpus/{coder,planner,control}/*.txt` and `*_combined.txt`: **not folded** into v2 (wrong turn tokens, empty thought channels, `n_predict=200` truncations, no depth). Moved to `data/routing_corpus_v1_deprecated/` with a one-line `README` stating why, kept for provenance of this document's §1 claims.
- `logs/imatrix_{coder,planner}.gguf`: kept (gitignored) solely as the V9 comparison baseline and as the source of the §1.4 numbers.
- The `expert_routing_visibility_plan.md` §2.4/§2.5 corpus description and the 2.1 MiB expert size are superseded by this document; a pointer is added there, the rest of that plan (Tier 1/2/3 mechanics) stands.

---

## 9. Step-by-step plan for the implementer

| Step | Work | Machine time | Gate |
|---|---|---|---|
| 0 | Move v1 corpus; create `routing_corpus_v2/turns/*.json` for 2 pilot + 8 full sessions; implement script changes of §8.2 (no model runs) | — | code review of the prompt renderer against §1.1 token table |
| 1 | Pilot generation: `build_routing_corpus.ps1 -Family planner -Session pilot …` then `coder pilot` | ≈ 1 h | P3–P6 |
| 2 | Segment pilots; build jobs; run P0, P1, P2, P8, P9 and V3a/V3b/V3c on the pilot bodies | ≈ 1 h | all P-checks PASS; V3c noise floor recorded |
| 3 | Full generation: 3 coder, 3 planner, 2 writer sessions (sequential, one server at a time) | ≈ 4–5 h | V1 per session |
| 4 | Segment all sessions; build `jobs.json`; run V2 assertions | minutes | V2 PASS for every job |
| 5 | Profiling: all `D×2` differential jobs (≈ 8 × 12–16 jobs) | ≈ 6–7 h unattended | each job's log check |
| 6 | V4 depth test and V4b retention test (uses existing windows + 3 extra runs + 1 regenerated session) | ≈ 1.5 h | decision recorded in §3.1/§3.4 |
| 7 | Aggregate; run V5–V9; write `logs/affinity_validation_report.md` | minutes | V5 ≠ FAIL, V6 PASS, V7 `N_T ≥ 20 000` for all four types — else loop to step 3 with more sessions of the deficient family |
| 8 | Write `docs/expert_affinity_gemma4_26b.md` (per-type and per-role sets, pool budgets with `e_size = 4.14 MiB`, all validation numbers); update thesis §6.3 rows; hand the same sessions + span index to Tier 2 for the per-token locality statistic (`> 0.85` claim, thesis §2.1) | — | — |

Total: ≈ 14–16 h of machine time over 2–3 days, all under §7.

---

## 10. Known approximations and open items (to be closed by the pilot)

1. Re-tokenized text vs generated tokens (P6) — Tier-1-only approximation, measured.
2. Ubatch numeric non-determinism at near-tie experts (V3c) — measured noise floor for the differential method.
3. Thoughts retained in history (§3.4) — tested by V4b; may flip to the stripped variant.
4. 25/30 layers have a 1 024-token attention window — depth effects only via 5 global layers; V4 measures whether it matters at all.
5. `factual` vs `creative` are turn-labelled, not marker-detected — a corpus-side ground truth, unlike the runtime classifier; documented so downstream users do not assume the runtime detector was validated here.
6. Whether `return_tokens` exists in this server build (P5) — only affects P6.
7. Whether `-c` is honoured exactly (P0) and whether imatrix auto-adds BOS (P1) — both change only the file-layout arithmetic, both are decided by one log line each.
