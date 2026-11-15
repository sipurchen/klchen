# Provenance of the 8 non-pilot sessions in this directory (coder_01..03, planner_01..03, writer_01..02)

Generated 2026-11-13 on **gx10** (not on the GT1030) as the structural-validation run of the expert-affinity pipeline
(generation stage of the gx10 structural validation). **The text is the model's own continuation by the gx10 copy of Gemma-4-26B-A4B Q4_K_M, which is NOT the same file as the GT1030 one.**

| item | value |
|---|---|
| generator | `python -m affinity_pipeline.gen_session --profile gx10_sim ... --ctx 8192 --ngl 0 --extra-server-arg=-ub --extra-server-arg=64 --extra-server-arg=-lv --extra-server-arg=4` (via `tools/slice_runner/gx10_run.py`, cores 0-4, nice 10) |
| llama.cpp | commit `d327bf1` (CUDA, aarch64) |
| GGUF | `~/models/gemma4-26b/gemma4-26b-Q4_K_M.gguf`, 16,796,017,760 B, sha256 prefix `1decb6b92391959c` (GT1030 file: 17,035,038,112 B) |
| seeds | as in `data/routing_corpus_v2/turns/<family>_<NN>.json` (2001-2003, 4001-4003, 5001-5002); per-turn `seed_used` is in each `.turns.jsonl` |
| `coder_03` | first attempt (seed 2003) was rejected by `segment_chunk_types.py` (`<|channel>` preceded by 4 spaces, "not at turn start"); regenerated with `--seed 2103` (the rejected files are in `tools/slice_runner/runs/S5/coder_03_seed2003_rejected/`) |
| `coder_02` | generation stopped after turn 3 (turn 4 hit the token limit twice, session truncated by design) - a short session (5639 tokens) |
| line endings | LF, no BOM (generated that way; do not let autocrlf rewrite them) |
| intended use | structure validation and, per the slicing doc's Fork 3, as session text to be re-profiled on the GT1030 (S7). Any claim that these are "the model's own text" must carry the caveat above. |
