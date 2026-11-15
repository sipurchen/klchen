"""
Build imatrix jobs from a session's span index (docs/expert_affinity_corpus_design.md §5.3, §5.4).

For each session: group spans into windows (>=256 tokens, >=90% one type else "mixed"), compute
cumulative cut points b_1..b_m, write the D-times-2 prefix file for each cut point, and emit jobs.json
with the exact -c value and output path for each imatrix invocation. Each job's counts, once run, must
be prefix-differenced against the PREVIOUS job's counts by aggregate_expert_affinity.py to get that
window's isolated counts (imatrix itself only gives whole-prefix counts).

Run under the CPU-safe wrapper for the tokenizer subprocess (docs/expert_affinity_corpus_design.md §7).

Usage: python scripts/build_imatrix_jobs.py <family>_<session> [<family>_<session> ...] [--model PATH]
       python scripts/build_imatrix_jobs.py --all      (every *.spans.json under sessions/)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # For segment_chunk_types
from segment_chunk_types import run_tokenize, build_char_offsets  # noqa: E402
from affinity_pipeline.config import REPO_ROOT, load_profile, resolve_binary, MachineProfile

# Keep these constants unchanged
MIN_WINDOW_TOKENS = 256
PURITY_THRESHOLD = 0.90
MIN_COVERAGE = 0.85


def get_profile(name_or_path: str | None = None) -> MachineProfile:
    """Priority: parameter > env var AFFINITY_PROFILE > default "gt1030_win".
    Return load_profile(...) result."""
    import os
    if name_or_path is not None:
        return load_profile(name_or_path)
    profile_name = os.environ.get("AFFINITY_PROFILE", "gt1030_win")
    return load_profile(profile_name)


def resolve_job_path(p: str, root: Path) -> Path:
    """If p is absolute (Path(p).is_absolute()), return Path(p).
    Otherwise return root / p. Enables reading old absolute-path jobs.json files."""
    path = Path(p)
    if path.is_absolute():
        return path
    return root / p


def build_jobs_for_session(session_id: str, model_path: Path, *, root: Path = REPO_ROOT, profile: MachineProfile | None = None) -> dict:
    """Dirs derived from root: root / "data" / "routing_corpus_v2" / {sessions, imatrix_inputs, jobs.json},
    root / "logs" / "imatrix_v2".
    Calls run_tokenize (from segment_chunk_types) with profile passed down.
    CRITICAL: infile/outfile in returned jobs["jobs"] list must be **relative POSIX strings**:
      - infile: "data/routing_corpus_v2/imatrix_inputs/{session}_{k}.txt"
      - outfile: "logs/imatrix_v2/{session}_{k}.gguf"
    Use Path.relative_to(root).as_posix() to get these.
    Other fields (session, k, b_prev, b_k, window_type, window_dominant, window_purity, etc.) unchanged."""
    
    if profile is None:
        profile = get_profile()
        
    body_path = root / "data" / "routing_corpus_v2" / "sessions" / f"{session_id}.body.txt"
    spans_path = root / "data" / "routing_corpus_v2" / "sessions" / f"{session_id}.spans.json"
    if not body_path.exists() or not spans_path.exists():
        raise FileNotFoundError(f"missing {body_path} or {spans_path} — run segment_chunk_types.py first")

    body = body_path.read_text(encoding="utf-8")
    spans_doc = json.loads(spans_path.read_text(encoding="utf-8"))
    spans = spans_doc["spans"]
    n_tokens = spans_doc["n_tokens"]

    tokens = run_tokenize(model_path, body_path, profile=profile)
    offsets = build_char_offsets(tokens, body)
    assert len(offsets) == n_tokens, f"{session_id}: retokenization length {len(offsets)} != spans.json n_tokens {n_tokens}"

    windows = build_windows(spans)

    covered_pure = sum((w["end"] - w["start"]) for w in windows if w["type"] != "mixed")
    coverage = covered_pure / n_tokens if n_tokens else 0.0

    imatrix_inputs_dir = root / "data" / "routing_corpus_v2" / "imatrix_inputs"
    logs_dir = root / "logs" / "imatrix_v2"
    
    imatrix_inputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    b_prev = 0
    for k, w in enumerate(windows, start=1):
        b_k = w["end"]  # cumulative token count through this window
        char_end = offsets[b_k - 1][1] if b_k > 0 else 0
        prefix_text = body[:char_end]
        # Triplicate (not just duplicate) the prefix, separated by literal "<bos>". imatrix's hard
        # "needs >= 2*ctx tokens in the file" gate was found empirically to have near-zero margin with
        # only 2 copies: re-tokenizing the SAME text as part of a longer file can retokenize a few dozen-
        # to-hundred tokens differently near the "<bos>" boundary than in isolation (BPE merge-boundary
        # effect), which pushed the largest sessions' D×2 files just under the 2x threshold. Only chunk 0
        # (the first `ctx` tokens: [BOS]+prefix_text) is ever processed (--chunks 1), so the extra
        # redundant copies cost nothing but a little tokenization time and guarantee ample margin.
        d3_text = prefix_text + "<bos>" + prefix_text + "<bos>" + prefix_text
        infile = imatrix_inputs_dir / f"{session_id}_{k}.txt"
        infile.write_text(d3_text, encoding="utf-8")

        ctx = b_k + 1
        outfile = logs_dir / f"{session_id}_{k}.gguf"
        
        # Convert to relative POSIX paths
        rel_infile = infile.relative_to(root).as_posix()
        rel_outfile = outfile.relative_to(root).as_posix()
        
        jobs.append({
            "session": session_id, "k": k, "b_prev": b_prev, "b_k": b_k,
            "window_type": w["type"], "window_dominant": w["dominant"], "window_purity": w["purity"],
            "window_start": w["start"], "window_end": w["end"],
            "ctx": ctx, "infile": rel_infile, "outfile": rel_outfile,
            "is_whole_session": (k == len(windows)),
        })
        b_prev = b_k

    return {
        "session": session_id, "n_tokens": n_tokens, "n_windows": len(windows),
        "coverage_non_mixed": round(coverage, 4), "windows": windows, "jobs": jobs,
    }


def main(argv: list[str] | None = None) -> None:
    """argparse: session_id (nargs="*"), --all, -o/--out (default None → root/data/routing_corpus_v2/jobs.json),
    --profile (default None), --root (default REPO_ROOT), --model (default None).
    Load profile when needed. Write jobs.json (list of job dicts) with relative paths."""
    
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id", nargs="*", help="e.g. coder_pilot (omit with --all)")
    ap.add_argument("--all", action="store_true", help="process every *.spans.json under sessions/")
    ap.add_argument("--model", default=None)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--root", type=str, default=None, help="root directory (default: REPO_ROOT)")
    args = ap.parse_args(argv)

    root = Path(args.root) if args.root else REPO_ROOT

    if args.all:
        ids = sorted(p.stem.replace(".spans", "") for p in (root / "data" / "routing_corpus_v2" / "sessions").glob("*.spans.json"))
    else:
        ids = args.session_id
    if not ids:
        print("No session ids given (use --all or list session ids).", file=sys.stderr)
        sys.exit(1)

    profile = get_profile(args.profile)
    
    # If model path is not provided, use the one from profile
    if args.model is None:
        model_path = profile.model_path
    else:
        model_path = Path(args.model)
        
    all_jobs = []
    summary = []
    for sid in ids:
        try:
            result = build_jobs_for_session(sid, model_path, root=root, profile=profile)
        except Exception as e:
            print(f"FAILED {sid}: {e}", file=sys.stderr)
            sys.exit(1)
        all_jobs.extend(result["jobs"])
        summary.append({k: v for k, v in result.items() if k != "jobs"})
        print(f"{sid}: {result['n_tokens']} tokens, {result['n_windows']} windows, "
              f"coverage(non-mixed)={result['coverage_non_mixed']*100:.1f}% "
              f"({'OK' if result['coverage_non_mixed'] >= MIN_COVERAGE else 'BELOW 85% THRESHOLD'})")

    # Set default output path
    if args.out is None:
        out_path = root / "data" / "routing_corpus_v2" / "jobs.json"
    else:
        out_path = Path(args.out)
        
    out_path.write_text(json.dumps(all_jobs, indent=2), encoding="utf-8")
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {len(all_jobs)} jobs to {out_path}")
    print(f"Wrote per-session window summary to {summary_path}")

    below = [s for s in summary if s["coverage_non_mixed"] < MIN_COVERAGE]
    if below:
        print(f"\nWARNING: {len(below)} session(s) below {MIN_COVERAGE*100:.0f}% non-mixed coverage: "
              + ", ".join(s["session"] for s in below), file=sys.stderr)


# Keep this function unchanged
def build_windows(spans):
    """Greedily group ordered token-index spans into windows >=256 tokens; type = majority if
    >=90%, else 'mixed'. Returns list of {start,end,type,purity}."""
    windows = []
    buf = []  # list of span dicts
    buf_len = 0

    def close(buf):
        start = buf[0]["start"]
        end = buf[-1]["end"]
        total = end - start
        counts = {}
        for sp in buf:
            counts[sp["type"]] = counts.get(sp["type"], 0) + (sp["end"] - sp["start"])
        best_type, best_count = max(counts.items(), key=lambda kv: kv[1])
        purity = best_count / total
        wtype = best_type if purity >= PURITY_THRESHOLD else "mixed"
        return {"start": start, "end": end, "type": wtype, "purity": round(purity, 4), "dominant": best_type}

    for sp in spans:
        buf.append(sp)
        buf_len += sp["end"] - sp["start"]
        if buf_len >= MIN_WINDOW_TOKENS:
            windows.append(close(buf))
            buf = []
            buf_len = 0
    if buf:
        if windows:
            # merge trailing remainder into the previous window rather than leave a <256-token tail
            prev = windows.pop()
            merged_buf = [{"start": prev["start"], "end": prev["end"], "type": prev["dominant"]}] + buf
            # re-derive purity/type from the merged span using original per-span breakdown is lost for
            # prev (it's already collapsed); approximate by weighting prev's dominant type as one block.
            total = buf[-1]["end"] - prev["start"]
            counts = {prev["dominant"]: prev["end"] - prev["start"]}
            for sp in buf:
                counts[sp["type"]] = counts.get(sp["type"], 0) + (sp["end"] - sp["start"])
            best_type, best_count = max(counts.items(), key=lambda kv: kv[1])
            purity = best_count / total
            wtype = best_type if purity >= PURITY_THRESHOLD else "mixed"
            windows.append({"start": prev["start"], "end": buf[-1]["end"], "type": wtype,
                             "purity": round(purity, 4), "dominant": best_type})
        else:
            windows.append(close(buf))
    return windows


if __name__ == "__main__":
    main()
