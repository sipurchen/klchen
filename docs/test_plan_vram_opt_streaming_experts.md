# Test Plan — VRAM-Opt-StreamingExperts Branch (Research Goals)

**Reviewer**: Independent test reviewer (Sonnet 4.6)  
**Date**: 2026-11-13  
**Branch**: `VRAM-Opt-StreamingExperts`  
**Scope**: Validate that the branch's scientific research targets have been achieved.  
**Prerequisite**: All slice tasks (S0–S7) accepted and merged. This plan executes AFTER slice work is complete.

---

## Fable5-CoT Analysis

### 1. Frame

The thesis §6 claim — "block-boundary expert affinity enables a pre-fetching RAM pool for Gemma4 26B-A4B on GT1030 2 GB VRAM" — rests on three empirical pillars that this branch must supply:

1. Different chunk types (reasoning / code / factual / creative) activate statistically distinguishable expert subsets in Gemma4's 30 MoE layers.
2. Those subsets are reproducible enough across sessions to plan a fixed pool budget.
3. Temporal locality holds at turn boundaries (Pr[E_i ∩ E_{i+1} ≠ ∅] > 0.85).

If the pipeline ships without producing real numbers that meet the stated thresholds, the branch is infrastructure only, not a completed research deliverable. This test plan exists to close that gap.

### 2. Facts (verified from branch documents)

| Item | Source | Status |
|------|--------|--------|
| Pilot data: reasoning=276 313 tokens, code=121 267, factual=19 020 | `logs/affinity_validation_report.md` | 2026-11-14 更新：全語料 10 session 已跑完（總 2 036 795 token；reasoning 1 248 355／code 464 177／factual 75 632／creative 162 170）；V7 四型 PASS、V8 PASS、V6 reasoning/code PASS、V5 reasoning-vs-code 與 code-vs-factual PASS（見 `tools/slice_runner/runs/S8/RESULT.md`）。上列 pilot 數字為歷史值 |
| V1–V9 spec with numeric thresholds | `docs/expert_affinity_corpus_design.md` §6 | Defined; not yet fully executed |
| Required 8 sessions (3 coder + 3 planner + 2 writer), ≥40 000 tokens/type | corpus design §3.2, §6.7 | 2026-11-14 更新：10 個 session 已齊（8 個新 session 由 gx10 生成，GGUF 與 GT1030 不同檔） |
| Pool budget formula: `|S_T,l(0.9)| × 4.14 MiB/expert-layer` | `docs/expert_routing_visibility_plan.md` §5 | 2026-11-14 更新：已由真實資料算出 |S| 大小（`runs/S7/pool_budget.txt`）：α=0.9 每層每型 ≈50–62 個 expert（4 型聯集 ≈83／128）；位元組換算仍用 4.14 MiB 假設值，未獨立驗證 |
| Locality statistic requires Tier 2 ctypes harness | `docs/expert_routing_visibility_plan.md` §3 | 2026-11-14 更新：Tier 2 已實作（`affinity_pipeline/tier2_ctypes.py`、`tier2_stats.py`、`scripts/tier2_locality.py`），在 gx10 受限條件（核心 0–4、nice 10、`-ngl 2 -ub 64`、VRAM 933 MiB）量到相鄰 token 專家交集機率 0.915（打亂對照 0.737）；T6.3 與 Tier 1 相關 0.9993/0.9995；尚未在 GT 1030 重跑 |
| Safety envelope: `ngl=9`, affinity mask 0xC, BelowNormal, VRAM ≤1 700 MiB | §6 safety rules + memory | Mandatory for every run |
| Thesis §6 的 Mixtral 表是估計值（非本分支量測） | `docs/thesis_zh.md` §6 | 2026-11-14 更新：查無 TODO 標記（此列原敘述過期）；§6.5／§10.5 已補入真實數字，§2.1 的 Pr>0.85 已改為待驗證假設 |

### 3. Forks

**F1: Run all tests end-to-end vs. accept existing pilot + only run incremental tests?**  
Pilot data is from 2 sessions; the split-half reproducibility check (V6) requires ≥4 sessions per type. Cannot skip full corpus generation. **Decision: generate full 8-session corpus from scratch** to reach the stopping rule (V7: all types ≥20 000 tokens, target 40 000).

**F2: Test on GT1030 (final target) or gx10 (portable structure check)?**  
Thesis targets GT1030 performance figures. gx10 serves only as a structure validator (portable scripts, no GPU offload). **Decision: pipeline portability tests run on gx10; affinity numbers and locality statistic collected only on GT1030.**

**F3: Tier 2 (ctypes locality) — required for branch acceptance?**  
Thesis §2.1 cites Pr > 0.85 as a core claim. Without it, the RAM-pool motivation is unverified. **Decision: Tier 2 is required for branch acceptance. If ctypes harness is missing, the branch is INCOMPLETE for this check.**

### 4. Fix Plan (Test Execution Order)

Execute sequentially. Each phase depends on the previous. All model processes use the §7 safety wrapper.

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
```

### 5. Final Proof

Each test section below states exact PASS/FAIL criteria drawn verbatim or derived from `docs/expert_affinity_corpus_design.md` §6 and `docs/expert_routing_visibility_plan.md`. The entire branch is ACCEPTED only when all mandatory checks reach PASS.

---

## Phase 0 — Safety & Environment Pre-flight

**Purpose**: Verify the hardware safety envelope before any model process.

### T0.1 — Affinity wrapper active
```powershell
# Verify scripts/profile_imatrix.py and profile_experts_imatrix.ps1
# contain the required wrapper idiom
Select-String -Path scripts\profile_imatrix.py -Pattern "affinity_mask|GuardedProcess"
```
PASS: `GuardedProcess` used; no raw `subprocess.run` with llama binaries.

### T0.2 — Single-process enforcement
Inspect `scripts/profile_imatrix.py` main(): each job waits for previous exit + 5 s settle before next launch.  
PASS: `--settle-s` default ≥ 5; no concurrent launch path exists.

### T0.3 — VRAM guard hard-coded
```powershell
Select-String -Path scripts\profile_imatrix.py -Pattern "1700|BudgetVerdict"
```
PASS: `check_budget(report, profile.vram) == BudgetVerdict.ABORT` triggers `stop()` when sum of three `Vulkan0 … buffer size` lines > 1 700 MiB.

### T0.4 — No hardcoded Windows paths
```powershell
Select-String -Recurse -Path scripts\, affinity_pipeline\ -Pattern "E:\\|C:\\" -Include "*.py"
```
PASS: zero matches (`.exe` checked separately: forbidden in `profile_imatrix.py` and `aggregate_expert_affinity.py`).

---

## Phase 1 — Pilot Pre-checks (P0–P9)

**Purpose**: Verify imatrix binary behaviour before full corpus run. Run on GT1030 only. ~2 h machine time.

Execute from `scripts/` with model at `profile.model_path` (GT1030 profile).

### T1.1 — P0: `-c` honoured un-padded
```powershell
# Use any body file ≥ 2 002 tokens; set ctx=1001
python scripts\profile_imatrix.py --profile gt1030_win --file data\routing_corpus_v2\sessions\coder_pilot.body.txt --ctx 1001 --out $env:TEMP\p0_test.gguf
```
PASS: log line `computing over 1 chunks, n_ctx=1001` present in `logs/imatrix_v2/coder_pilot_p0.log`.

### T1.2 — P1: Auto-BOS and no double BOS
Run imatrix on D×2 of pilot body with `-c (2L+3)` (too large). Read the token count from log.  
PASS: log `the data file you provided tokenizes to only N tokens` where `N == 2L + 2` (auto-BOS present, none from text). If `N == 2L + 1`, record "auto-BOS absent" and switch to writing `<bos>` at file start; must be noted in report.

### T1.3 — P2: Chunks independent and identical
Run `D×2` with `--chunks 2` and `--chunks 1`; compare `.counts` tensors.  
PASS: `counts(2 chunks) == 2 × counts(1 chunk)` bitwise in all 30 × 128 cells.

### T1.4 — P3: CoT channel present in pilot sessions
Inspect both pilot `.turns.jsonl` files: every model turn starts with `<|channel>thought\n`, contains exactly one `<channel|>`, then answer.  
PASS: 100% of accepted turns pass; reasoning span ≥ 64 tokens in ≥ 80% of planner turns.

### T1.5 — P7: Span index integrity
```powershell
python scripts\segment_chunk_types.py --session data\routing_corpus_v2\sessions\coder_pilot.body.txt --spans data\routing_corpus_v2\sessions\coder_pilot.spans.json
```
PASS: spans cover `[0, L)` exactly; every reasoning span starts at token id 100 (`<|channel>`), ends at id 101 (`<channel|>`).

### T1.6 — P8: `.counts` shape and sum identity
```python
from gguf import GGUFReader
# After any pilot run:
r = GGUFReader("logs/imatrix_v2/coder_pilot_1.gguf")
tensors = {t.name: t for t in r.tensors if t.name.endswith(".counts")}
assert len(tensors) == 30
for l, t in tensors.items():
    assert t.shape == (128,), f"{l}: {t.shape}"
    # Sum_e counts[l,e] == 8*(tokens processed)
```
PASS: 30 tensors, shape `(128,)`, sum identity holds for each layer.

### T1.7 — P9: VRAM envelope ≤ 1 700 MiB
Parse `logs/imatrix_v2/coder_pilot_1.log` for three `Vulkan0 … buffer size` lines; sum.  
PASS: total ≤ 1 700 MiB.

**Gate**: All P0–P9 checks PASS before proceeding to Phase 2.

---

## Phase 2 — Full Corpus Generation (8 Sessions, V1–V2 per session)

**Purpose**: Build the complete corpus satisfying V7 stopping rule. Run on GT1030. ~6–7 h unattended.

Target sessions: `coder_{01..03}`, `planner_{01..03}`, `writer_{01..02}`.  
Each session: 4 000–8 191 body tokens, `--chunks 1`, multi-turn with real Gemma4 template tokens.

### T2.1 — V1: CoT elicitation (automated, per session)

For each of 8 generated sessions, `scripts/validate_affinity_pipeline.py` must report V1 PASS:
- Every model turn: exactly one `<|channel>` and one `<channel|>`, in that order.
- Reasoning span ≥ 64 tokens for every turn labelled reasoning-expected; ≥ 16 tokens for all others.
- Reasoning tokens / model-generated tokens ≥ 0.25 for `planner`, ≥ 0.10 for `coder`, ≥ 0.10 for `writer`.
- ≥ 90% of turns end with `stop_type == "eos"`.

PASS: all 8 sessions output V1=PASS in `logs/affinity_validation_report.md`.  
FAIL action: regenerate the failing session with `seed+1`; if V1 fails twice, escalate to corpus design review.

### T2.2 — V2: Chunk-alignment (per imatrix input file, before each run)

For every `imatrix_inputs/{session}_{k}.txt` with declared `-c = b_k + 1`:
1. `llama-tokenize --no-bos --parse-special --show-count -f file` count `== 2*b_k + 1`.
2. `tokenize(file)[b_k+1] == 2` (the BOS separator at chunk boundary).
3. Purity: `mixed` windows ≤ 15% of body tokens per session; every non-mixed window ≥ 90% one type.
4. After run: log line `computing over 1 chunks, n_ctx=<b_k+1>` and `Sum_e counts[l,e] == 8*(b_k+1)`.

PASS: all 8 sessions × all windows PASS. Any failure aborts that session's job list before model loads.

### T2.3 — V7 Stopping Rule

After corpus generation, run aggregate script:
```powershell
python scripts\aggregate_expert_affinity.py --root <PROJECT_ROOT>
```
Read `logs/affinity_validation_report.md` V7 section.

| Chunk type | Required N_T | Target N_T |
|-----------|------------|-----------|
| reasoning | ≥ 20 000 | ≥ 40 000 |
| code | ≥ 20 000 | ≥ 40 000 |
| factual | ≥ 20 000 | ≥ 40 000 |
| creative | ≥ 20 000 | ≥ 40 000 |

PASS: all four types ≥ 20 000.  
PARTIAL: types below 40 000 noted; additional sessions generated per §3.2 mix until threshold met.  
**Note**: Current pilot has factual=19 020 (FAIL). Full 8-session corpus must fix this before proceeding.

---

## Phase 3 — Determinism & Depth Validation (V3–V4b)

**Purpose**: Verify that the differential method is scientifically sound. ~45 min.

### T3.1 — V3a: Determinism
Run same `D×2` file with same flags twice; compare `.counts` output bitwise.  
PASS: all 30 × 128 cells identical.

### T3.2 — V3b: Chunk independence
`D×2` with `--chunks 2` vs `--chunks 1`; verify `counts(2 chunks) == 2 * counts(1 chunk)` exactly.  
FAIL action: entire Tier-1 approach is re-examined; halt before any other analysis.

### T3.3 — V3c: Ubatch invariance
Same file, `-ub 512` vs `-ub 256`; compute flipped token-slots.  
PASS: `Sum_l Sum_e |Delta| / 2 <= 0.1%` of `8*(L+1)*30` cells.  
Report the measured value as the differential noise floor.  
FAIL (> 0.5%): §5.4 differential method abandoned; Tier 2 (ctypes) used instead; corpus reused unchanged.

### T3.4 — V4: Depth sensitivity
Choose a `code` span S (≥ 768 tokens) at body position ≥ 4 096 in a coder session; get `p_deep` (differential window). Take a shallow span S' of similar length (D×2 of S alone with minimal header); get `p_shallow`.

Report per-layer `JSD(p_deep, p_shallow)` vs `JSD(p_shallow(S), p_shallow(S'))`.

**Depth matters** if `JSD_deep > JSD_between-spans` in ≥ 15/30 layers → long sessions (≥4 000 tokens) mandatory.  
**No depth effect** → record "no first-order depth effect at 8 k"; sessions ≥ 2 048 tokens acceptable for cost.

### T3.5 — V4b: Thought-retention fidelity
Regenerate `planner_02` with thoughts stripped from history. Profile same way.

PASS: `JSD(p_T,retained, p_T,stripped) <= split-half JSD_within,T` (from V6) in ≥ 25/30 layers for each type.  
FAIL: switch to stripped variant for final run.

---

## Phase 4 — Type Separation & Reproducibility (V5–V6)

**Purpose**: The core empirical test of the thesis claim that chunk types activate different experts.

### T4.1 — V6: Reproducibility (split-half, run before V5)

Split each type's windows by **session parity** (odd/even session numbers).  
Per layer, compute:
- Spearman rho between `counts_T,half1[l, :]` and `counts_T,half2[l, :]`
- `Jaccard(S_half1,l(0.8), S_half2,l(0.8))`

PASS: `rho >= 0.90` in ≥ 27/30 layers **AND** median Jaccard ≥ 0.75, for every type.

FAIL: that type's corpus is too small; generate more sessions of the richest family and re-run V6.

Report `JSD_within,T[l]` between halves as noise floor for V5.

### T4.2 — V5: Type separation (the thesis claim)

For each pair of chunk types `(T1, T2)` and each layer `l`:
```
R_l = JSD(p_T1[l], p_T2[l]) / max(JSD_within,T1[l], JSD_within,T2[l])
```

| Result | Criterion | Interpretation |
|--------|-----------|---------------|
| PASS | `R_l >= 3` in >= 20/30 layers for every pair | Locality signal present; thesis §6 claim supported |
| WEAK | `1.5 <= median R_l < 3` for any pair | Double corpus, re-test |
| FAIL | `median R_l < 1.5` for any pair | Span index wrong OR this model does not route by chunk type; thesis §6 must state this |

Also report: `Jaccard(S_T1,l(0.8), S_T2,l(0.8))` per layer — expected well below within-type value.

**This is the primary research gate for the entire branch.**

---

## Phase 5 — Bookkeeping & V1 Comparison (V8–V9)

### T5.1 — V8: Bookkeeping identities (exact)

For each session, verify telescoping identity:
- `Sum_k counts(window_k) == counts(R_m)` per session.
- `p_R` from §4.5 (using measured `w_R,T`) equals normalised sum of whole-session `counts(R_m)` (up to `mixed` windows).
- `Sum_e counts[l, e] == 8 * (tokens processed)` in every run and layer.

PASS: all identities hold exactly (bitwise, not approximate).

### T5.2 — V9: Comparison with v1 (report only, no pass criterion)

Compute `JSD(p_coder-sessions, p_v1-coder)` and `JSD(p_planner-sessions, p_v1-planner)` per layer. Report alongside V6's within-type values.

Reviewer note: v1 had N≈2 048 per role; expected JSD values are non-trivial given 73% max expert concentration in v1 tables. Any `JSD > 2 × JSD_within,T` indicates v1 was significantly corrupted — report as "v1 invalid; v2 supersedes".

---

## Phase 6 — Locality Statistic (Tier 2 ctypes harness)

**Purpose**: Validate the thesis §2.1 claim — the primary motivation for a RAM pool.

### T6.1 — Tier 2 harness exists and runs
```python
from affinity_pipeline.tier2_ctypes import collect_per_token_experts
print("OK")
```
PASS: imports without error.  
FAIL: Tier 2 harness not implemented → branch is **INCOMPLETE** for this check; must be noted as open blocker.

### T6.2 — Locality statistic measured
Run Tier 2 harness on ≥ 2 sessions of each type (can reuse existing corpus sessions).  
For each layer `l`, compute:
```
Pr[top_k(g(x_i)) ∩ top_k(g(x_{i+1})) != ∅]
```
across consecutive token pairs within each session.

PASS (thesis §2.1 validated): result > 0.85 in ≥ 25/30 layers for ≥ 3 of 4 chunk types.  
FAIL: locality claim cannot be made; thesis §2.1 must be revised with actual number.

### T6.3 — Tier 2 vs Tier 1 consistency check
For the same sessions, compare top-8 expert activations from Tier 2 (per-token) against the aggregate Tier 1 distributions from Phase 4.

PASS: Pearson r ≥ 0.90 between `counts[l,:]` reconstructed from Tier 2 token sequences and the Tier 1 `.counts` tensors for the same session.

---

## Phase 7 — Pool Budget Computation

**Purpose**: Translate research results into the thesis §6 table.

### T7.1 — Affinity sets S_T,l(alpha) computed at alpha=0.80 and alpha=0.90

For each type T and layer l, compute `S_T,l(alpha)` = smallest set of experts covering ≥alpha of routing mass.

```python
import json
data = json.load(open("logs/expert_affinity_type.json"))
for t in ["reasoning", "code", "factual", "creative"]:
    for l in range(30):
        s80 = data[t][str(l)]["0.8"]
        s90 = data[t][str(l)]["0.9"]
        assert len(s80) <= len(s90)
        assert len(s90) <= 128
```
PASS: all 4 types × 30 layers × 2 alpha-levels populated; `|S_T,l(0.9)|` ≤ 128.

### T7.2 — Pool budget per type

For each type T, compute:
```
budget_T = (1/30) × Sum_l |S_T,l(0.9)| × 4.14 MiB
```
where 4.14 MiB = gate_up Q4_K (2.127 MiB) + down Q8_0 (2.009 MiB) per expert per layer.

Report `budget_T` for all 4 types. Report `max_budget = max(budget_T)`.

PASS (pool is feasible): `max_budget <= 2 048 MiB` (GT1030 host RAM budget for pool).  
If `max_budget > 2 048 MiB`: thesis §6 must report unfeasibility and propose a reduced-alpha fallback.

### T7.3 — Thesis §6 tables populated
Verify `docs/thesis_zh.md` §6 contains:
- Table: `|S_T,l(0.9)|` averaged over layers, for all 4 types.
- Table: Pool budget per type (MiB).
- Locality statistic values from Phase 6.
- V5 `R_l` values (median and min across pairs).
- Statement: whether the pool approach is feasible on GT1030 given the computed budget.

PASS: all five items present with actual numbers (not "TODO" or placeholder).

---

## Phase 8 — Cross-Hardware Consistency (GT1030 vs gx10)

**Purpose**: Verify pipeline portability. gx10 runs with `-ngl 0` (CPU-only); numbers will differ from GT1030.

### T8.1 — Pipeline runs on gx10 without crash
```bash
# On gx10 (Linux):
python tools/slice_runner/gx10_pytest.py --args "tests/affinity_pipeline/ -q"
```
PASS: test suite passes (or failures match the known 6 environment-specific blockers from S2 r2).

### T8.2 — Structure consistency (not numeric equality)
Run one coder session through the full pipeline on gx10 (CPU-only, ngl=0).  
Verify:
- Same 30 layers produce `.counts` tensors.
- `Sum_e counts[l,e]` identity holds.
- V1 (CoT elicitation) passes.

PASS: structural identities hold; numeric values allowed to differ (gx10 uses different compute path).  
Report: state clearly "gx10 numbers are CPU-baseline only; GT1030 numbers are the thesis target."

---

## Acceptance Criteria Summary

| Check | Status needed | Blocker? |
|-------|--------------|---------|
| P0–P9 (pilot pre-checks) | All PASS | Yes |
| V1 (CoT elicitation) | All 8 sessions PASS | Yes |
| V2 (chunk-alignment) | All sessions/windows PASS | Yes |
| V3a–V3c (determinism/ubatch) | PASS; noise floor reported | Yes (V3b) |
| V4 (depth sensitivity) | Reported (no pass criterion) | No |
| V5 (type separation R_l >= 3) | PASS for all pairs | **Primary gate** |
| V6 (reproducibility rho >= 0.90, Jaccard >= 0.75) | PASS for all types | Yes |
| V7 (N_T >= 20 000 all types) | PASS | Yes |
| V8 (bookkeeping identities) | PASS | Yes |
| V9 (v1 comparison) | Reported | No |
| Tier 2 locality statistic | PASS (> 0.85) | Yes |
| Pool budget computed | Feasibility stated | Yes |
| Thesis §6 tables populated | All 5 items with real numbers | Yes |
| Cross-hardware (gx10 structure) | PASS | No |
| Safety compliance | All processes wrapped | Yes |

**Branch ACCEPTED** when: all "Yes" blockers pass, V5 primary gate passes, and thesis §6 tables are populated.

---

## Execution Constraints (All Phases)

Every `llama-server.exe`, `llama-imatrix.exe`, `llama-tokenize.exe` and Python aggregation process must:
1. CPU affinity: cores 2+3 only (mask 0xC via `ProcessorAffinity = 12` or `GuardedProcess`).
2. Worker threads: `-t 2 -tb 2`.
3. Priority: BelowNormal.
4. VRAM: abort if sum of three `Vulkan0 … buffer size` lines > 1 700 MiB.
5. Single process at a time: 5 s settle after each exit.
6. Per-job wall-clock ceiling: `ceil(30 + 0.12 × c)` seconds; timeout → mark FAILED, do not auto-retry.
7. After every phase: stop all model processes; verify with `nvidia-smi` and `tasklist`.

Operating point for all imatrix runs:
```
-ngl 9 --override-tensor "\.ffn_.*_exps\.=CPU" -b 512 -ub 512 -fit off -c <session_length+1>
```

---

## Estimated Machine Time

| Phase | Time (GT1030) | Notes |
|-------|-------------|-------|
| Phase 0 (pre-flight) | 10 min | Static analysis only |
| Phase 1 (P0–P9) | ~2 h | Includes model loads |
| Phase 2 (8 sessions generation) | ~4–6 h | gen_session + imatrix per session |
| Phase 3 (V3–V4b) | ~45 min | Select runs |
| Phase 4 (V5–V6) | ~1 h | Aggregate + compute stats |
| Phase 5 (V8–V9) | 10 min | Python-only |
| Phase 6 (Tier 2 locality) | ~2–3 h | Per-token ctypes harness |
| Phase 7 (pool budget) | 15 min | Python-only |
| Phase 8 (gx10 consistency) | 1–2 h | On gx10 |
| **Total** | **~12–14 h** | Mostly unattended |
