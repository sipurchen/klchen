# Expert Routing Visibility Plan — Gemma4 26B-A4B (128 experts, top-8)

> Status: **planning document only** — nothing here has been executed. Branch `VRAM-Opt-StreamingExperts`, 2026-09-05.
> Goal: obtain per-layer, per-token selected-expert IDs from the real local MoE model so that thesis §6 ("block-boundary expert affinity") can be re-run on it and produce separate **coder** and **planner** affinity profiles.
> Path placeholders follow `bin/README.md`: `<PROJECT_ROOT>` = this repo, `<LLMS_DIR>` = the local GGUF store (the directory that holds `gemma4-26B-A4B-it-GGUF/`).

---

## 0. TL;DR

| | Verdict |
|---|---|
| Does the prebuilt b8679 zip already contain a ready-made expert-ID dumper? | **No** `llama-eval-callback.exe` — but **yes, indirectly**: `bin/llama-cpp/llama-imatrix.exe` is a shipped eval-callback consumer that already reads the router's selected-expert `ids` tensor and writes **per-layer, per-expert routing counts** into its GGUF output. That alone yields the per-role expert-frequency histograms (§6.1's "affinity experts" table) with **zero code and zero compiler**. |
| Can the eval callback be driven without a source rebuild? | **Yes.** `ggml_backend_sched_set_eval_callback` and every other symbol a harness needs are exported by the shipped `llama.dll` / `ggml-base.dll` / `ggml.dll`. No headers or `.lib` files ship, but Python `ctypes` needs neither. The cost is hand-transcribing two structs (`llama_model_params`, `llama_context_params`) from the b8679 headers. |
| Is `llama-cpp-python` useful here? | **Not recommended.** Not installed; its Windows PyPI wheels are CPU-only (no Vulkan), and its ctypes struct definitions are pinned to *its own* vendored llama.cpp, not b8679 — pointing it at our DLLs via `LLAMA_CPP_LIB_PATH` risks silent struct-layout corruption. |
| Is `gemma4` in mainline llama.cpp at b8679? | **Yes** — `llama.dll` embeds the strings `gemma4`, `gemma4_iswa`, `gemma4-iswa.cpp`; the official ggml-org release binary loads the model, so `src/models/gemma4-iswa.cpp` exists at tag `b8679`. The source-patch fallback is therefore *possible*, but it is not needed. |
| Overall feasibility of the primary route | **Easy** (Tier 1, imatrix, aggregate histograms) → **Moderate** (Tier 2, ctypes harness, per-token sequences). No compiler, no SDK, no rebuild for either. |
| Recommendation | Run **Tier 1 (imatrix) first**, then **Tier 2 (ctypes harness)** for the per-token locality statistic. Source patch (**Tier 3**) only on the explicit go/no-go criteria in §5. |

---

## 1. What was actually checked on this machine

All facts below were verified by directory listing / binary string search during this session; nothing was executed against the model.

### 1.1 `bin/llama-cpp/` contents (b8679, `llama-b8679-bin-win-vulkan-x64.zip`)

Executables (20): `llama-batched-bench.exe`, `llama-bench.exe`, `llama-cli.exe`, `llama-completion.exe`, `llama-fit-params.exe`, `llama-gemma3-cli.exe`, `llama-gguf-split.exe`, **`llama-imatrix.exe`**, `llama-llava-cli.exe`, `llama-minicpmv-cli.exe`, `llama-mtmd-cli.exe`, `llama-mtmd-debug.exe`, `llama-perplexity.exe`, `llama-quantize.exe`, `llama-qwen2vl-cli.exe`, `llama-results.exe`, `llama-server.exe`, `llama-template-analysis.exe`, `llama-tokenize.exe`, `llama-tts.exe`, `rpc-server.exe`.

DLLs: `llama.dll`, `mtmd.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-vulkan.dll` (56.8 MB), `ggml-rpc.dll`, 14× `ggml-cpu-<arch>.dll` variants (the loader picks the best one for the host CPU — on the i5-4460 that is `ggml-cpu-haswell.dll`), `libomp140.x86_64.dll`.

**Absent:** `llama-eval-callback.exe`, `llama-simple.exe`, any `*.h`, `*.hpp`, `*.lib`, `*.def`. (`find bin -iname '*.h' -o -iname '*.lib'` returns nothing.) The upstream release workflow builds with `LLAMA_BUILD_EXAMPLES=OFF`, so the `examples/eval-callback` target is not in the zip — as expected for a runtime-only release.

### 1.2 Symbols present in the shipped DLLs (string search, `grep -a`)

- `llama.dll`: `gemma4`, `gemma4_iswa`, `gemma4-iswa.cpp`; graph-node names `ffn_moe_logits`, `ffn_moe_logits_biased`, `ffn_moe_probs`, `ffn_moe_probs_biased`, `ffn_moe_probs_masked`, **`ffn_moe_topk`**, `ffn_moe_weights*`, `ffn_gate_inp`, `ffn_gate_inp_shexp`.
- Exported and needed by a harness (all found): `ggml_backend_sched_set_eval_callback`, `ggml_backend_tensor_get`, `ggml_backend_load_all`, `ggml_backend_load_all_from_path`, `ggml_backend_load`, `ggml_backend_load_best`, `ggml_backend_cpu_buffer_type`, `ggml_backend_dev_buffer_type`, `ggml_backend_dev_count`, `ggml_get_name`, `ggml_nbytes`, `ggml_nelements`, `ggml_n_dims`, `ggml_type_name`, `ggml_op_name`, `llama_backend_init`, `llama_model_default_params`, `llama_model_load_from_file`, `llama_model_free`, `llama_context_default_params`, `llama_init_from_model`, `llama_free`, `llama_decode`, `llama_batch_get_one`, `llama_batch_init`, `llama_batch_free`, `llama_tokenize`, `llama_token_to_piece`, `llama_model_get_vocab`, `llama_vocab_n_tokens`, `llama_vocab_bos`, `llama_vocab_is_eog`, `llama_get_logits_ith`, `llama_sampler_chain_init`, `llama_sampler_init_greedy`, `llama_sampler_sample`, `llama_get_memory`, `llama_memory_clear`, `llama_model_chat_template`, `llama_chat_apply_template`, `llama_log_set`, `llama_model_n_layer`.
- `ggml-base.dll` contains op names `TOP_K` and `ARGSORT` and the symbol `ggml_top_k` → at b8679 top-k is a **native ggml op**, so `ffn_moe_topk-N` is a real computed node (int32, shape `[8, n_tokens]`) that the eval callback can observe after it is evaluated. (In older builds it was `argsort` + `view`; not the case here.)
- `ggml.dll` contains `GGML_BACKEND_PATH` → the dynamic backend loader honours that environment variable; relevant because a Python host process is not in `bin/llama-cpp/`.
- `llama-imatrix.exe` contains `--output-format`, `--show-statistics`, `--chunk-size`, `--chunks`, `--from-chunk`, `--in-file`, `--output-file`, `--save-frequency`, `--no-ppl`, `--parse-special`, `--process-output`, `--override-tensor`, `--n-gpu-layers`, `--ctx-size`, `--batch-size`, `--ubatch-size`, `--threads`, and the GGUF keys `imatrix.chunk_count`, `imatrix.chunk_size`, `imatrix.datasets`, tensor suffixes `in_sum2` and `.counts`. This is the post-2025 GGUF-output imatrix (upstream PR "imatrix: use GGUF to store importance matrices"), which is the version that keeps **per-expert** counts for `MUL_MAT_ID` tensors.

### 1.3 Host toolchain

- Python 3.11.9 (Anaconda, `D:\ProgramData\anaconda`), `gguf` 0.18.0 installed (`gguf-py` — can read the imatrix GGUF output directly). `llama_cpp` (llama-cpp-python) **not** installed.
- `cmake` present (Anaconda). **No** `cl.exe`/MSVC (only `Installer` + `Shared` under `Program Files (x86)\Microsoft Visual Studio`), no `clang`, no `gcc`, no `ninja`, no Vulkan SDK (`VULKAN_SDK` unset, no `C:\VulkanSDK`). ⇒ a source rebuild would start from a bare machine (see §4).
- 34 GB RAM (`TotalPhysicalMemory` = 34 295 214 080). Model file `google_gemma-4-26B-A4B-it-Q4_K_M.gguf` = 17.0 GB; `IQ2_S` (10.2 GB) and `IQ2_M` (10.7 GB) also present in `<LLMS_DIR>\gemma4-26B-A4B-it-GGUF\`.
- Safe-launch idiom already in `scripts/probe_role_models.ps1` / `scripts/probe_expert_visibility2.ps1` (affinity mask 12, `BelowNormal`, `-t 2`) — reused verbatim in §6.

### 1.4 Where the expert IDs actually live in the b8679 graph

`llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp`, shared by all MoE archs including `gemma4-iswa.cpp`) builds, per layer `il`:

```
logits  = ggml_mul_mat(ffn_gate_inp, cur)           -> cb "ffn_moe_logits-{il}"   f32 [128, n_tokens]
probs   = softmax(logits)                            -> cb "ffn_moe_probs-{il}"    f32 [128, n_tokens]
selected= ggml_top_k(probs, n_expert_used=8)         -> cb "ffn_moe_topk-{il}"     i32 [8,   n_tokens]   <-- THE TARGET
weights = ggml_get_rows(probs, selected)             -> cb "ffn_moe_weights-{il}"
experts = ggml_mul_mat_id(ffn_gate_up_exps, cur, selected)   // selected == t->src[2] of the MUL_MAT_ID op
```

So there are **two** equivalent places to read the selection: the `ffn_moe_topk-N` node itself (cleanest, already int32 IDs), or `src[2]` of any `MUL_MAT_ID` node in layer N (what `llama-imatrix` reads). Hooking the pre-softmax `ffn_gate_inp` output and computing top-8 ourselves is also valid (softmax is monotonic, top-k order is identical) but unnecessary.

---

## 2. Primary approach: ggml eval callback (no source patch, no recompilation)

### 2.1 Tier 1 — `llama-imatrix.exe` as a zero-code expert-frequency profiler

**Why this works.** `tools/imatrix/imatrix.cpp` installs `collect_imatrix` as `cparams.cb_eval`. For every `GGML_OP_MUL_MAT_ID` node it copies `ids = t->src[2]` (`[n_expert_used, n_tokens]`, int32) to host with `ggml_backend_tensor_get`, then for each token and each of its 8 selected experts increments `e.counts[expert]` and accumulates `in_sum2[expert, :]`. In the GGUF output every expert weight gets two tensors:

- `blk.N.ffn_down_exps.weight.in_sum2` — f32 `[2816, 128]` (activation energy per expert, bonus signal)
- `blk.N.ffn_down_exps.weight.counts` — f32 `[1, 128]` (or `[128]`) — **number of tokens routed to each of the 128 experts in layer N**
- same pair for `blk.N.ffn_gate_up_exps.weight` (fused gate+up in this GGUF; identical ids, so use one of the two to avoid double counting)

Running imatrix once on a coder corpus and once on a planner corpus gives two `[30 layers × 128 experts]` histograms — exactly the aggregate thesis §6.1 tabulates ("chunk_type → affinity experts"), now for a real 128-expert model.

**What it does not give:** per-token sequences. The locality statistic `Pr[top_k(g(x_i)) ∩ top_k(g(x_{i+1})) ≠ ∅]` needs token order → Tier 2.

**Flags (verified against this build's actual `--help` output — corrected after an initial draft wrongly assumed a `--chunk-size` flag that does not exist in b8679):**

```
llama-imatrix.exe -m <LLMS_DIR>\gemma4-26B-A4B-it-GGUF\google_gemma-4-26B-A4B-it-Q4_K_M.gguf
  -ngl 9 --override-tensor "\.ffn_.*_exps\.=CPU"           # identical placement to the validated server config
  -c 8192 -b 512 -ub 512 -t 2 -tb 2
  -f <PROJECT_ROOT>\data\routing_corpus\coder.txt           # plain text; one role per run
  --parse-special --no-ppl
  --output-format gguf -o <PROJECT_ROOT>\logs\imatrix_coder.gguf
```

Notes / things the implementer must confirm before the first run:

1. b8679's actual imatrix flags (confirmed via `--help`, not assumed) are `-c`/`-b`/`-ub` for context/batch/ubatch sizing, `--chunks N` (max number of chunks to process, default -1 = all), and `--chunk`/`--from-chunk N` (start offset) — there is **no separate `--chunk-size` flag**; "chunk" here means a `-c`-sized (or shorter, for the tail) slice of the input file's tokenized text, so `-c 8192` alone controls the window each pass sees. No extra flag needed.
2. Whether `--output-format gguf` is already the default (it is in current upstream — confirmed via `--help`: "default: gguf" — harmless to pass explicitly).
3. That `counts` is per-expert (`shape [1,128]`) in the produced file — the single verification that decides whether Tier 1 stands alone. Check with:
   `python -c "from gguf import GGUFReader; r=GGUFReader(r'logs\imatrix_coder.gguf'); [print(t.name,t.shape) for t in r.tensors if t.name.endswith('.counts')][:5]"`
   If the shape is `[1]` (legacy scalar counts), Tier 1 collapses to Tier 2 — nothing else changes.
4. imatrix processes text as **prefill only** (no generation). Router decisions are causal and deterministic, so profiling the *text* of a prompt+answer pair gives exactly the experts the model would have used while generating that answer — provided the answer text is what the model itself produced. Hence the corpus-building step in §2.4 generates answers first with `llama-server`.
5. `--parse-special` is required so `<start_of_turn>` / `<end_of_turn>` in the corpus tokenize as the real Gemma control tokens instead of literal text; the corpus files must use the Gemma chat format that `scripts/probe_expert_visibility2.ps1` already uses.
6. With `cb_eval` set, the scheduler splits the graph at every observed node and disables some fusion; expect 10–40 % slower prefill than the benchmark's numbers. Speed numbers from imatrix runs must **not** be recorded as inference benchmarks.

**Effort:** ~1 hour of setup (corpus files + wrapper script) + unattended run time (§2.5). **Feasibility: Easy.** Blocking dependency: none.

### 2.2 Tier 2 — Python `ctypes` harness against the shipped DLLs (per-token IDs)

**Why this works without headers/`.lib`.** On Windows, `ctypes.CDLL` resolves exported symbols by name at load time — an import library is only needed for link-time binding from C/C++. Every function the harness needs is exported (§1.2). Two things still have to come from the b8679 *headers*: the byte layout of `struct llama_model_params` and `struct llama_context_params` (both are returned **by value** from `*_default_params()`, so ctypes must declare them exactly) and the field order of `struct llama_batch`. The `ggml_tensor` layout is *not* needed if the harness only uses the exported accessors (`ggml_get_name`, `ggml_nbytes`, `ggml_nelements`, `ggml_backend_tensor_get`).

**One-time inputs (no compiler):**

- `https://raw.githubusercontent.com/ggml-org/llama.cpp/b8679/include/llama.h` (~60 KB)
- `https://raw.githubusercontent.com/ggml-org/llama.cpp/b8679/ggml/include/ggml-backend.h` (for the callback typedef)
- reference for ctypes idioms only: `llama_cpp/llama_cpp.py` from the llama-cpp-python repo (copy the *pattern*, never its field list).

**Harness skeleton (design, not code):**

1. `os.add_dll_directory(bin/llama-cpp)`; `CDLL('ggml-base.dll')`, `CDLL('ggml.dll')`, `CDLL('llama.dll')`.
2. `ggml_backend_load_all_from_path(b"<PROJECT_ROOT>/bin/llama-cpp")` — mandatory: the default `ggml_backend_load_all()` searches the *executable's* directory (`python.exe`), not the DLL's, so Vulkan and the CPU variant would silently not load and the model would fail or run on a fallback. Alternatively set `GGML_BACKEND_PATH`. Verify with `ggml_backend_dev_count()` ≥ 2 and `ggml_backend_dev_name()` showing `Vulkan0` + `CPU`.
3. `mp = llama_model_default_params()`; set `n_gpu_layers = 9`; set `tensor_buft_overrides` to a NULL-terminated array of `{pattern: b"\\.ffn_.*_exps\\.", buft: ggml_backend_cpu_buffer_type()}` — this reproduces `--override-tensor "\.ffn_.*_exps\.=CPU"` exactly (that CLI flag is just sugar over this field).
4. `cp = llama_context_default_params()`; `n_ctx = 8192`, `n_batch = n_ubatch = 512`, `n_seq_max = 1`, `n_threads = n_threads_batch = 2`; **`cb_eval = <CFUNCTYPE(c_bool, c_void_p, c_bool, c_void_p)>`**, `cb_eval_user_data = <state ptr>`. Nothing else changes — same VRAM/ctx/ngl envelope as `docs/gemma4_26b_a4b_vram_benchmark.md`.
5. Callback logic: `ask=True` → return `ggml_get_name(t).startswith(b"ffn_moe_topk-")`; `ask=False` → `n = ggml_nelements(t)`; `n_tokens = n // 8`; `ggml_backend_tensor_get(t, buf, 0, ggml_nbytes(t))`; reshape `[n_tokens, 8]` int32; append `(prompt_id, layer=int(name.split(b"-")[1]), token_idx = n_past + j, ids)` for each row. The state object tracks `n_past` (incremented after every `llama_decode`).
6. Driver: for each corpus sample, `llama_memory_clear`, tokenize with the chat template, prefill in 512-token batches, then greedy-decode up to `n_predict` tokens (so generated tokens are profiled *live* — the Tier 1 two-stage trick is unnecessary here). Write JSONL as it goes.
7. Struct-layout sanity gate before any decode: print `cp.n_ctx == 512`, `cp.n_batch == 2048`, `cp.n_ubatch == 512`, `cp.n_seq_max == 1`, `cp.rope_freq_base == 0.0`, `cp.type_k == 1 (F16)`, `mp.n_gpu_layers == 999`, `mp.use_mmap == True`, `mp.vocab_only == False`. Any mismatch ⇒ the struct transcription is wrong; **stop** (a wrong layout can crash or silently mis-configure the context — never proceed to model load on a failed gate).

**Risks and mitigations**

| Risk | Impact | Mitigation |
|---|---|---|
| Struct layout transcription error (`llama_context_params` gains/loses fields between tags) | crash or wrong config | Transcribe from the *b8679* header only; §2.2 step 7 gate; also assert `ctypes.sizeof(llama_context_params)` against a value computed by hand from the header field list. |
| Backend DLLs not found from a Python host | model loads on no backend / CPU only | Step 2 explicit `load_all_from_path`; verify device list before loading the model. |
| `ffn_moe_topk-N` name differs for gemma4 (e.g. shared-expert path, or Gemma4 routing through a different helper) | callback never fires | First run in "discovery mode": `ask` returns True for every node whose name contains `moe`, log names once, then narrow. The `MUL_MAT_ID` `src[2]` route is the fallback within the same harness (needs the `ggml_tensor` struct's `src` field — layout has been stable: `type,buffer,ne[4],nb[4],op,op_params[16],flags,src[10],view_src,view_offs,data,name[64],extra,padding[8]`). |
| Callback overhead / graph splitting | slower, more compute buffer | Acceptable for offline profiling; keep `-ngl 9` (1633 MiB @16k / lower @8k) for headroom instead of the 10/11 operating points; abort if the log's `Vulkan0 compute buffer` line exceeds the benchmark's 517 MiB by more than ~100 MiB. |
| GIL / callback re-entrancy | ctypes callbacks are invoked from the ggml compute thread while `llama_decode` holds the GIL release | ctypes handles the GIL acquire automatically; keep the callback tiny (copy bytes, append), do JSON writing in the driver loop, not in the callback. |

**Effort:** ~half a day including header transcription and the discovery run. **Feasibility: Moderate.** Blocking dependency: two header files (~100 KB download); no compiler, no SDK, no rebuild.

### 2.3 `llama-cpp-python` assessment

- Not installed; PyPI Windows wheels are built CPU-only. Vulkan requires `CMAKE_ARGS=-DGGML_VULKAN=on pip install --no-binary` = a full source build needing MSVC + Vulkan SDK — i.e. the exact toolchain the fallback route needs, with none of its benefits.
- Its low-level module does declare `cb_eval` / `cb_eval_user_data` in `llama_context_params` and a `ggml_backend_sched_eval_callback` CFUNCTYPE, so it *demonstrates* the Tier 2 approach — but its struct definitions match whichever llama.cpp commit it vendors, not b8679. Redirecting it at our DLLs via `LLAMA_CPP_LIB_PATH` would mix a foreign struct layout with the b8679 ABI → undefined behaviour. **Do not do this.**
- CPU-only fallback speed would also be worse than the Vulkan DLLs we already have (attention would leave the GPU).
- Verdict: use it only as *reading material* for ctypes idioms.

### 2.4 Data-collection plan (shared by Tier 1 and Tier 2)

**Corpus.** `N = 12` prompts per role, ~150–400 tokens each, in Gemma chat format (`<bos><start_of_turn>user … <end_of_turn>\n<start_of_turn>model\n`), stored as `data/routing_corpus/{coder,planner}/NN.txt` (new dir; commit the prompts, gitignore nothing — they are small text).

- *coder*: implement a function (Python / PowerShell / JS), fix a given bug, refactor a snippet, write a unit test, explain a regex, translate code between languages, write a CLI parser, SQL query, shell one-liner, data-class design, error-handling rewrite, small algorithm (two-pointer, BFS).
- *planner*: multi-step project decomposition, dependency ordering, risk/trade-off analysis, capacity/time estimation, step-by-step arithmetic/logic word problems, "plan a benchmark", incident runbook outline, prioritisation with constraints, decision matrix, chain-of-thought math, meeting agenda from goals, architecture option comparison.
- Keep a third small `control` set (4 factual Q&A prompts) to sanity-check that the two role histograms differ from each other more than from noise.

**Stage A — generate answers (Tier 1 only).** Run `llama-server` with the safe wrapper (§6), `n_predict = 256`, `temperature = 0.2`, save `prompt + answer` back into the corpus file so imatrix profiles the model's own continuation.

**Stage B — collect.**
- Tier 1: `llama-imatrix.exe` once per role → `logs/imatrix_{coder,planner,control}.gguf`.
- Tier 2: harness once per role → `logs/routing_{role}.jsonl`, records `{"prompt_id", "role", "layer", "token_idx", "phase": "prefill|decode", "experts": [8 ints]}`.

**Stage C — aggregate (Python, `gguf-py` + numpy, no model needed).**
- `p_role[l, e]` = normalised count matrix `[30 × 128]` per role (Tier 1 from `.counts`; Tier 2 by counting JSONL rows).
- **Affinity set per layer:** smallest expert subset covering ≥ 80 % (and 90 %) of routing mass — the direct analogue of §6.1's per-type set; report `|S_l|` per layer, expected to be far below 128 if locality holds.
- **Role separation:** Jaccard(S_coder,l, S_planner,l), symmetric KL(p_coder,l ‖ p_planner,l), and the union size (`|S_coder ∪ S_planner|` = the RAM-pool budget §6.2 needs: `E_warm ≥ |S|` per layer with `e_size ≈ 272 MiB / 128 ≈ 2.1 MiB` per expert per layer for Q4_K_M).
- **Locality (Tier 2 only):** `Pr[E_i ∩ E_{i+1} ≠ ∅]` over consecutive tokens, per layer and per role, plus the mean `|E_i ∩ E_{i+1}|`; compare against the thesis's `> 0.85` claim. Also report the same statistic across a *role boundary* (last token of a coder answer vs first of a planner prompt) to quantify the "block-boundary" effect the pool relies on.
- Outputs: `logs/expert_affinity_{role}.json` (machine-readable sets) and a new `docs/expert_affinity_gemma4_26b.md` with the tables — mirroring the format of `docs/role_expert_screening.md`.

### 2.5 Run-time budget

Prefill on this box is CPU-bound on the experts (benchmark: 3.68 tok/s at a 34-token prompt with `-t 4`; larger batches amortise expert reads, so 5–15 tok/s prefill at `-t 2` is the realistic range, minus the callback overhead). Per role: ~12 samples × ~500 tokens ≈ 6 k tokens ⇒ **~10–25 min per role**, unattended. Tier 2 decode adds 12 × 256 tokens at ~3–4 tok/s ⇒ **+15–20 min per role**. Whole study ≈ 1.5–2.5 h of machine time, all under the safety wrapper.

RAM: 17 GB mmap'd model + KV/compute on host ≈ 20 GB peak, within 34 GB. Do **not** run imatrix and a server simultaneously (RAM fine, VRAM not: each would allocate its own ~1.6 GB on a 2 GB card).

---

## 3. Feasibility verdict for the primary approach

**Easy → Moderate. Go.**

- Tier 1 needs nothing that is not already on disk (`llama-imatrix.exe`, `gguf` 0.18.0, PowerShell wrapper idiom). The only unknown is whether the `.counts` tensor is per-expert in this exact build — checkable in the first five minutes of the first run's output.
- Tier 2 needs two header files fetched from the `b8679` tag for struct transcription. No compiler, no Vulkan SDK, no rebuild, no `.lib`. The inference configuration (`ngl`, `-ot`, ctx, threads) is reproduced field-for-field via the public C API, so the observation harness does not perturb the model.
- Neither tier touches `llama-server.exe` or the benchmark numbers already in `docs/gemma4_26b_a4b_vram_benchmark.md` / `docs/role_expert_screening.md`.

---

## 4. Fallback approach: source patch (only if §5 criteria trigger)

### 4.1 Is there upstream source to patch?

Yes. `general.architecture = gemma4` is supported at tag `b8679` (proof: the unmodified ggml-org release binary loads and runs the model; `llama.dll` embeds `gemma4-iswa.cpp`). Relevant files at that tag:

- `src/models/gemma4-iswa.cpp` — graph builder for this arch (`llm_build_gemma4_iswa`), which calls the shared MoE helper.
- `src/llama-graph.cpp` — `llm_graph_context::build_moe_ffn`: the `ggml_top_k` call and the `cb(selected_experts, "ffn_moe_topk", il)` line.
- `src/llama-arch.cpp` — `LLM_ARCH_GEMMA4` tensor-name table (`ffn_gate_inp`, `ffn_gate_up_exps`, `ffn_down_exps`, `ffn_gate_inp_shexp`).
- `examples/eval-callback/eval-callback.cpp` — upstream's reference `cb_eval` consumer (prints a *sample* of every node's values; would still need a name filter and a JSONL writer to be useful, so even had it shipped it would not have been a drop-in).

Risk flagged in the task ("gemma4 may not be mainline") is **resolved negative**: it is mainline. What remains to verify when actually opening the tag is only the file name (`gemma4-iswa.cpp` vs. a split into `gemma4.cpp` + `gemma4-iswa.cpp`).

### 4.2 What the minimal patch would be

`build_moe_ffn` builds a lazy graph, so an `fprintf` at the `ggml_top_k` line would print nothing useful (the tensor has no data at build time). The correct minimal patch is therefore not in the graph builder but in a *runner*: ~40 lines in `tools/server/server.cpp` (or `tools/completion`) that set `cparams.cb_eval` to a function filtering `ffn_moe_topk-` and appending to a JSONL file — i.e. **the same eval callback as Tier 2, just in C++ and in-process**. Alternatively enable `LLAMA_BUILD_EXAMPLES=ON` and add the filter to `eval-callback.cpp`. Either way the "patch" is an *observer*, identical in information content to Tier 2; the source route buys only freedom from ctypes struct transcription.

### 4.3 Toolchain needed on this Windows box (currently absent)

Per llama.cpp `docs/build.md` and the release workflow (`.github/workflows/release.yml`, job `windows-vulkan`):

1. **Visual Studio 2022 Build Tools** with "Desktop development with C++" (~7 GB) — none installed today.
2. **Vulkan SDK** (LunarG, ~1 GB) — needed for `glslc` + headers to compile `ggml-vulkan`'s shaders; not installed. *Avoidable:* build with `-DGGML_VULKAN=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DBUILD_SHARED_LIBS=ON` and drop the **prebuilt** `ggml-vulkan.dll` from the b8679 zip next to the self-built `ggml-base.dll` — the backend-registry ABI is stable within one tag, so a self-built host can load the upstream Vulkan backend. This is a legitimate shortcut but must be verified (device list at startup) rather than assumed.
3. CMake (present via Anaconda; confirm ≥ 3.14).
4. Build flags to stay comparable with the shipped binary: `-DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_EXAMPLES=ON` at tag `b8679`. The shader-generation step and full compile on 4 cores (pinned to 2, see §6) is **1–2 h**; with Vulkan disabled ~30–45 min.

Even so: **never replace `bin/llama-cpp/llama-server.exe` with a self-built one for benchmark numbers.** The existing tables were measured with the upstream binary; a locally built one (different compiler, possibly different CPU variant selection) is only for observation runs.

### 4.4 Effort / risk vs. primary

| | Tier 1 imatrix | Tier 2 ctypes | Tier 3 source patch |
|---|---|---|---|
| New installs | none | 2 header files | ~8 GB toolchain (+1 GB SDK unless the DLL-swap shortcut works) |
| Compiler | no | no | yes (MSVC) |
| Wall time to first data | < 1 h | ~0.5 day | 1–2 days incl. install, build, debugging |
| Per-token IDs | no | yes | yes |
| Perturbs benchmark binary | no | no | no if kept separate; yes if the self-built exe is ever reused for speed numbers |
| Main failure mode | `.counts` not per-expert in this build | struct-layout mismatch | build breakage on a bare Windows box; CPU freeze during compile if not pinned |

---

## 5. Decision recommendation and go/no-go

**Attempt order: Tier 1 → Tier 2 → (only if forced) Tier 3.**

1. **Start with Tier 1 (imatrix).** It is the only route that produces the §6.1 affinity table with nothing but files already on disk, and it doubles as the ground truth for validating Tier 2 (the per-layer histograms from the JSONL must match the imatrix `.counts` on the same corpus; a mismatch means the harness hooked the wrong tensor).
2. **Proceed to Tier 2 (ctypes) regardless of Tier 1's outcome**, because the locality statistic the thesis actually claims (`> 0.85` consecutive-token overlap) needs token order, which imatrix cannot provide.
3. **Fall back to Tier 3 only if all of the following hold:**
   - Tier 1: `.counts` is not per-expert **and** Tier 2 fails, or
   - Tier 2: after transcribing `llama_context_params` from the *b8679* header, the §2.2 step-7 sanity gate still fails, **and** the discovery run shows no node named `ffn_moe_*` **and** no `MUL_MAT_ID` node whose `src[2]` is readable (i.e. the callback fires for other nodes but the routing tensors are provably invisible — e.g. fused away by the Vulkan backend before the callback), or
   - the callback route is demonstrably too slow to finish a role in < 2 h (unlikely; the model is CPU-bound anyway).
   Any *one* of "header transcription is tedious", "first ctypes attempt crashed", or "names differ from expectation" is **not** a go signal for Tier 3 — those are expected iteration steps of Tier 2.

---

## 6. How to execute later — mandatory machine-safety and budget rules

These rules apply to **every** process this plan spawns: `llama-imatrix.exe`, `llama-server.exe` (corpus generation), `python.exe` running the ctypes harness, and — in the Tier 3 case — `cmake --build` / `cl.exe`. This machine (i5-4460, 4 cores, GT 1030 2 GB) froze completely earlier this session when a benchmark was launched unpinned.

**Launch idiom (copy from `scripts/probe_role_models.ps1`, lines 33–41):**

```powershell
$proc = Start-Process -FilePath $Exe -ArgumentList $argList `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -PassThru -WindowStyle Hidden
Start-Sleep -Milliseconds 500
$proc.ProcessorAffinity = 12                                        # cores 2+3 only (mask 0xC)
$proc.PriorityClass     = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
```

- Always also pass `-t 2 -tb 2` (imatrix/server) or `n_threads = n_threads_batch = 2` (harness); affinity alone does not stop ggml from spawning 4 threads on 2 cores.
- For a Tier 3 build: `cmake --build . --config Release -j 2` **and** wrap the `cmake` process with the same affinity/priority; MSVC's default `/MP` will otherwise use all 4 cores.
- One model process at a time. Never run imatrix and llama-server concurrently (VRAM).
- Poll the log for `Vulkan0 … buffer size` lines and abort if the three-buffer total exceeds **1.7 GB** (`docs/gemma4_26b_a4b_vram_benchmark.md` method). Suggested operating point for profiling: **`-ngl 9`, `-c 8192`, `-b 512 -ub 512`, `--override-tensor "\.ffn_.*_exps\.=CPU"`, `-np 1`** — one layer below the validated max to absorb any callback-induced compute-buffer growth. ctx stays within the 8192–16384 envelope; ngl stays within 7–11; the harness adds observation only and must not alter these.
- Warm the page cache once before timing anything (`Get-Content` the 17 GB GGUF to `$null`, pinned as above) — the IQ2_S stall in the benchmark doc was a cold-mmap artefact.
- All outputs go to `logs/` (already gitignored) except the small prompt corpus and the final markdown/JSON summaries, which are committed.

---

## 7. Deliverables checklist (for the implementation session)

- [ ] `data/routing_corpus/{coder,planner,control}/*.txt` — prompts (+ generated answers after Stage A)
- [ ] `scripts/profile_experts_imatrix.ps1` — safe-launch wrapper around `llama-imatrix.exe`, one role per invocation
- [ ] `scripts/expert_routing_harness.py` — ctypes harness (Tier 2), with the §2.2 step-7 sanity gate and a `--discover` mode
- [ ] `scripts/aggregate_expert_affinity.py` — reads `logs/imatrix_*.gguf` (via `gguf`) and/or `logs/routing_*.jsonl`, emits `logs/expert_affinity_{role}.json`
- [ ] `docs/expert_affinity_gemma4_26b.md` — results: per-layer affinity sets, coder-vs-planner separation, locality statistic vs. thesis §2.1/§6
- [ ] Update `docs/thesis_*.md` §6.3 with the real-model numbers alongside the Mixtral / DS-Coder-V2-Lite rows
