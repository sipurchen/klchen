# v1 corpus — deprecated, do not use

Kept only for provenance (source of the §1.4 statistics in `docs/expert_affinity_corpus_design.md` and the V9 comparison baseline). **Do not feed to imatrix again.**

Three confirmed defects (full detail in `docs/expert_affinity_corpus_design.md`):

1. Wrong chat-template tokens — used `<start_of_turn>`/`<end_of_turn>` which are **not in this model's vocabulary** (tokenized as literal text, not control tokens). The model's real tokens are `<|turn>`(105)/`<turn|>`(106), `<|think|>`(98), `<|channel>`(100)/`<channel|>`(101) — verified directly against the GGUF's `tokenizer.chat_template` and `tokenizer.ggml.tokens`.
2. No reasoning content — `<|think|>` was never sent, so every "thought channel" in these samples is empty. Zero CoT was ever profiled.
3. imatrix chunk-boundary corruption — `-c 1024` sliced the concatenated snippet files mid-sample at every window boundary, with no causal continuity across chunks (`llama_memory_clear` between chunks). Router counts for split samples reflect corrupted context, not real routing.

Related outputs kept (gitignored, in `logs/`): `imatrix_coder.gguf`, `imatrix_planner.gguf` — kept solely as the V9 known-invalid baseline.

See `docs/expert_affinity_corpus_design.md` for the replacement methodology (`data/routing_corpus_v2/`).
