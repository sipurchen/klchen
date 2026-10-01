"""
Segment a v2 session body into chunk-type spans (docs/expert_affinity_corpus_design.md §4.3).

Input:  data/routing_corpus_v2/sessions/{id}.body.txt
        data/routing_corpus_v2/sessions/{id}.turns.jsonl
Output: data/routing_corpus_v2/sessions/{id}.spans.json   (token-index spans covering [0, n_tokens) exactly)
        data/routing_corpus_v2/exports/{id}.txt            (human-readable dump, never fed to imatrix)

Run under the CPU-safe wrapper (tokenization is lightweight/no inference, but keep the pattern consistent
with every other process that touches this GGUF — see docs/expert_affinity_corpus_design.md §7).

Usage: python scripts/segment_chunk_types.py <family>_<session>   [--model PATH]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"E:\Gemma4_E4B_Project")
DEFAULT_MODEL = Path(r"E:\LLMmodel\gemma4-26B-A4B-it-GGUF\google_gemma-4-26B-A4B-it-Q4_K_M.gguf")
TOKENIZE_EXE = PROJECT_ROOT / "bin" / "llama-cpp" / "llama-tokenize.exe"
SESSIONS_DIR = PROJECT_ROOT / "data" / "routing_corpus_v2" / "sessions"
EXPORTS_DIR = PROJECT_ROOT / "data" / "routing_corpus_v2" / "exports"

USER_HEADER = "<|turn>user\n"
USER_FOOTER = "<turn|>\n"
CHANNEL_OPEN = "<|channel>"
CHANNEL_CLOSE = "<channel|>"
FENCE_RE = re.compile(r"```")


_VOCAB_CACHE = {}  # model_path -> list[str], index = token id


def _load_vocab(model_path: Path):
    key = str(model_path)
    if key in _VOCAB_CACHE:
        return _VOCAB_CACHE[key]
    import gguf
    r = gguf.GGUFReader(str(model_path), mode="r")
    vocab = None
    for f in r.fields.values():
        if f.name == "tokenizer.ggml.tokens":
            vocab = []
            for i in range(len(f.data)):
                idx = f.data[i]
                piece = f.parts[idx].tobytes().decode("utf-8", errors="replace")
                # SentencePiece word-boundary marker U+2581 ('▁') means a literal space here
                # (confirmed empirically: token 609's raw bytes are b'\xe2\x96\x81$' == '▁$').
                piece = piece.replace("▁", " ")
                vocab.append(piece)
            break
    if vocab is None:
        raise RuntimeError(f"{model_path}: no tokenizer.ggml.tokens field found")
    _VOCAB_CACHE[key] = vocab
    return vocab


def run_tokenize(model_path: Path, body_path: Path):
    """Return list of (token_id:int, piece:str) in order.

    Uses llama-tokenize.exe's `--ids` mode (a clean `[1, 2, 3, ...]` list — no text formatting
    ambiguity) for the authoritative token-id sequence, then looks up each id's piece text directly
    in the GGUF's own `tokenizer.ggml.tokens` vocabulary array. This deliberately avoids llama-tokenize's
    human-readable piece-mode text output: it was found empirically this session to corrupt any piece
    whose literal text starts with a backslash followed by a letter that looks like a C escape code
    (e.g. a piece that is literally the two characters '\\r' from LaTeX '\\rightarrow' in real generated
    text) — the tool's own console-formatting code appears to reinterpret it as an actual control
    character, silently dropping it from captured output. Piece-mode is only used elsewhere for human
    inspection (exports/), never for the char<->token offset map this function feeds.
    """
    # --no-escape is mandatory: by default llama-tokenize interprets backslash escapes (\n, \r, \t, \\, ...)
    # in file content read via -f, which silently corrupts any real text containing a literal backslash
    # (confirmed empirically this session: literal '\rightarrow' in generated text lost its backslash,
    # shifting every subsequent char offset). The actual body text must be tokenized completely literally.
    out = subprocess.run(
        [str(TOKENIZE_EXE), "-m", str(model_path), "--no-bos", "--no-escape", "--ids", "-f", str(body_path), "--log-disable"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    if out.returncode != 0:
        raise RuntimeError(f"llama-tokenize --ids failed: {out.stderr[-2000:]}")
    m = re.search(r"\[(.*)\]", out.stdout, re.DOTALL)
    if not m:
        raise RuntimeError(f"llama-tokenize --ids: could not find a '[...]' id list in output: {out.stdout[:500]!r}")
    ids = [int(x.strip()) for x in m.group(1).split(",") if x.strip() != ""]

    vocab = _load_vocab(model_path)
    tokens = []
    for tid in ids:
        if tid < 0 or tid >= len(vocab):
            raise RuntimeError(f"token id {tid} out of vocab range (size {len(vocab)})")
        tokens.append((tid, vocab[tid]))
    return tokens


def build_char_offsets(tokens, original_text: str):
    """Reconstruct char offsets per token by concatenation; assert exact match against original_text."""
    offsets = []
    pos = 0
    for tid, piece in tokens:
        start = pos
        end = pos + len(piece)
        offsets.append((start, end))
        pos = end
    reconstructed = "".join(p for _, p in tokens)
    if reconstructed != original_text:
        # Report the first divergence point precisely rather than a wall of text.
        n = min(len(reconstructed), len(original_text))
        i = 0
        while i < n and reconstructed[i] == original_text[i]:
            i += 1
        raise AssertionError(
            "Token-piece concatenation does not reconstruct the original body text exactly. "
            f"First divergence at char {i}: original={original_text[max(0,i-20):i+20]!r} "
            f"reconstructed={reconstructed[max(0,i-20):i+20]!r}. "
            "Segmentation cannot proceed on an unreliable char->token map."
        )
    return offsets


def char_to_token_index(offsets, char_pos, side="start"):
    """Find the token index whose span contains/starts-at char_pos. side='start' rounds to the token
    that STARTS at or after char_pos (for range starts); side='end' rounds to the token that ENDS at or
    before char_pos (for range ends, exclusive)."""
    lo, hi = 0, len(offsets) - 1
    if side == "start":
        for i, (s, e) in enumerate(offsets):
            if s >= char_pos:
                return i
        return len(offsets)
    else:  # 'end' -> exclusive token index just past the token containing char_pos-1
        for i, (s, e) in enumerate(offsets):
            if e >= char_pos:
                return i + 1
        return len(offsets)


def classify_model_turn(body: str, m_start: int, m_end: int, label: str, spans_out, tok_start_fn, tok_end_fn):
    """Split one model turn's char range [m_start, m_end) into reasoning / code / label-fallback spans,
    all in CHAR coordinates first (caller converts to token coords). Appends (char_s, char_e, type) tuples."""
    text = body[m_start:m_end]
    cursor = 0  # relative to m_start
    pieces = []  # list of (rel_start, rel_end, type)

    ch_open = text.find(CHANNEL_OPEN)
    if ch_open == 0:
        ch_close = text.find(CHANNEL_CLOSE)
        if ch_close == -1:
            raise AssertionError(f"<|channel> with no matching <channel|> in model turn at char {m_start}")
        ch_end = ch_close + len(CHANNEL_CLOSE)
        pieces.append((0, ch_end, "reasoning"))
        cursor = ch_end
    elif ch_open > 0:
        raise AssertionError(f"<|channel> not at turn start (offset {ch_open}) in model turn at char {m_start}")
    # else: no channel at all in this turn (thinking produced nothing) — whole turn falls through below.

    rest = text[cursor:]
    fence_positions = [mm.start() for mm in FENCE_RE.finditer(rest)]
    if len(fence_positions) % 2 != 0:
        # Odd number of ``` markers: an unterminated fence (model cut off mid-block). Treat the last
        # fence as running to the end of the turn rather than crash the whole session's segmentation.
        fence_positions.append(len(rest))
    fi = 0
    pos = 0
    while fi < len(fence_positions):
        f_start = fence_positions[fi]
        f_end_marker = fence_positions[fi + 1] + 3  # include the closing ```
        if f_start > pos:
            pieces.append((cursor + pos, cursor + f_start, label))
        pieces.append((cursor + f_start, cursor + f_end_marker, "code"))
        pos = f_end_marker
        fi += 2
    if pos < len(rest):
        pieces.append((cursor + pos, cursor + len(rest), label))

    for rel_s, rel_e, typ in pieces:
        if rel_e > rel_s:
            spans_out.append((m_start + rel_s, m_start + rel_e, typ))


def segment(session_id: str, model_path: Path):
    body_path = SESSIONS_DIR / f"{session_id}.body.txt"
    turns_path = SESSIONS_DIR / f"{session_id}.turns.jsonl"
    if not body_path.exists() or not turns_path.exists():
        raise FileNotFoundError(f"missing {body_path} or {turns_path}")

    body = body_path.read_text(encoding="utf-8")
    turns = [json.loads(line) for line in turns_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    tokens = run_tokenize(model_path, body_path)
    offsets = build_char_offsets(tokens, body)
    n_tokens = len(tokens)

    char_spans = []  # (start, end, type) in char coords, to be merged with "control" default

    for t in turns:
        us, ue = t["user_start"], t["user_end"]
        seg = body[us:ue]
        if not seg.startswith(USER_HEADER) or not seg.endswith(USER_FOOTER):
            raise AssertionError(
                f"turn {t['turn']} user span does not start/end with the expected literal markers "
                f"({USER_HEADER!r}/{USER_FOOTER!r}); got prefix={seg[:20]!r} suffix={seg[-20:]!r}"
            )
        u_text_start = us + len(USER_HEADER)
        u_text_end = ue - len(USER_FOOTER)
        char_spans.append((us, u_text_start, "control"))
        if u_text_end > u_text_start:
            char_spans.append((u_text_start, u_text_end, "user_prompt"))
        char_spans.append((u_text_end, ue, "control"))

        ms, me = t["model_start"], t["model_end"]
        classify_model_turn(body, ms, me, t["label"], char_spans, None, None)

    char_spans.sort(key=lambda x: x[0])

    # Fill gaps (turn headers/closers, system block, inter-turn newlines) with "control".
    filled = []
    cursor = 0
    for s, e, typ in char_spans:
        if s > cursor:
            filled.append((cursor, s, "control"))
        filled.append((s, e, typ))
        cursor = max(cursor, e)
    if cursor < len(body):
        filled.append((cursor, len(body), "control"))

    # Merge adjacent same-type spans.
    merged = []
    for s, e, typ in filled:
        if merged and merged[-1][2] == typ and merged[-1][1] == s:
            merged[-1] = (merged[-1][0], e, typ)
        else:
            merged.append((s, e, typ))

    # Convert char spans -> token-index spans.
    token_spans = []
    turn_map = []  # parallel: which turn each char-span belonged to, best-effort for the output
    for s, e, typ in merged:
        ts = char_to_token_index(offsets, s, side="start")
        te = char_to_token_index(offsets, e, side="end")
        if te > ts:
            token_spans.append({"start": ts, "end": te, "type": typ})

    # Merge adjacent same-type token spans (char->token rounding can create abutting duplicates).
    final_spans = []
    for sp in token_spans:
        if final_spans and final_spans[-1]["type"] == sp["type"] and final_spans[-1]["end"] == sp["start"]:
            final_spans[-1]["end"] = sp["end"]
        else:
            final_spans.append(sp)

    # Assertion: spans cover [0, n_tokens) exactly, no gaps/overlaps.
    cov = 0
    for sp in final_spans:
        if sp["start"] != cov:
            raise AssertionError(f"span coverage gap/overlap at token {cov}: next span starts at {sp['start']}")
        cov = sp["end"]
    if cov != n_tokens:
        raise AssertionError(f"span coverage ends at token {cov}, expected {n_tokens}")

    out_path = SESSIONS_DIR / f"{session_id}.spans.json"
    out_path.write_text(json.dumps({"session": session_id, "n_tokens": n_tokens, "spans": final_spans}, indent=2), encoding="utf-8")

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    export_path = EXPORTS_DIR / f"{session_id}.txt"
    with export_path.open("w", encoding="utf-8") as f:
        f.write(f"# session={session_id} n_tokens={n_tokens}\n")
        for sp in final_spans:
            piece_text = "".join(p for _, p in tokens[sp["start"]:sp["end"]])
            f.write(f"\n--- [{sp['type']}] tokens[{sp['start']}:{sp['end']}] ---\n{piece_text}\n")

    type_counts = {}
    for sp in final_spans:
        type_counts[sp["type"]] = type_counts.get(sp["type"], 0) + (sp["end"] - sp["start"])
    print(f"{session_id}: {n_tokens} tokens, spans written to {out_path}")
    for typ, cnt in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {typ:12s} {cnt:6d} tokens ({100*cnt/n_tokens:.1f}%)")

    return final_spans, type_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id", nargs="+", help="e.g. coder_pilot or coder_01 (one or more)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    model_path = Path(args.model)
    for sid in args.session_id:
        try:
            segment(sid, model_path)
        except Exception as e:
            print(f"FAILED {sid}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
