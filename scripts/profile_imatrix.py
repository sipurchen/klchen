#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from pathlib import Path
from math import ceil
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from affinity_pipeline.config import REPO_ROOT, MachineProfile, resolve_binary
from affinity_pipeline.guarded_process import GuardedProcess
from affinity_pipeline.llama_logs import parse_buffer_lines, check_budget, BudgetVerdict
from build_imatrix_jobs import resolve_job_path, get_profile


def build_imatrix_argv(profile: MachineProfile, infile: Path, ctx: int, outfile: Path) -> List[str]:
    return [
        str(resolve_binary(profile, "llama-imatrix")),
        "-m", str(profile.model_path),
        "-ngl", "9",
        "--override-tensor", r"\.ffn_.*_exps\.=CPU",
        "-c", str(ctx),
        "-b", "512",
        "-ub", "512",
        "-t", str(profile.cpu.threads),
        "-tb", str(profile.cpu.threads),
        "-fit", "off",
        "-f", str(infile),
        "--chunks", "1",
        "--parse-special",
        "--no-escape",
        "--no-ppl",
        "--output-format", "gguf",
        "-o", str(outfile)
    ]


def job_ceiling_seconds(ctx: int) -> int:
    return ceil(30 + 0.12 * ctx)


def run_job(job: Dict[str, Any], profile: MachineProfile, root: Path, *, argv: Optional[List[str]] = None,
            poll_s: float = 2.0, extra_args: Optional[List[str]] = None,
            ceiling_scale: float = 1.0) -> str:
    infile = resolve_job_path(job["infile"], root)
    outfile = resolve_job_path(job["outfile"], root)
    ctx = job["ctx"]
    label = f"{job['session']}_{job['k']}"
    log = root / "logs" / "imatrix_v2" / f"{label}.log"
    
    # Create parent directories
    log.parent.mkdir(parents=True, exist_ok=True)
    
    # Delete old log if exists
    if log.exists():
        log.unlink()
        
    # Build argv if not provided
    if argv is None:
        argv = build_imatrix_argv(profile, infile, ctx, outfile) + list(extra_args or [])
    
    timeout_s = ceil(job_ceiling_seconds(ctx) * ceiling_scale)
    
    try:
        with GuardedProcess(argv, profile, cwd=root, log_path=log, timeout_s=timeout_s) as gp:
            # Poll process
            first_complete_check = True
            while True:
                # Read log file
                if log.exists():
                    text = log.read_text(encoding="utf-8", errors="replace")
                    parsed = parse_buffer_lines(text)
                    verdict = check_budget(parsed, profile.vram)
                    
                    # Check for VRAM_ABORT only once after first complete report
                    if first_complete_check and parsed.complete:
                        if verdict == BudgetVerdict.ABORT:
                            gp.stop()
                            return "VRAM_ABORT"
                        first_complete_check = False
                
                # Check if process exited
                exit_code = gp.poll()
                if exit_code is not None:
                    break
                    
                time.sleep(poll_s)
                
        # Process finished or timed out
        if gp.timed_out:
            return "TIMEOUT"
            
        # Check output file exists
        if not outfile.exists():
            # Print last 20 lines of log to stderr
            if log.exists():
                text = log.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                tail_lines = lines[-20:] if len(lines) >= 20 else lines
                print("\n".join(tail_lines), file=sys.stderr)
            return "NO_OUTPUT"
            
        # Read final log to verify completion line
        text = log.read_text(encoding="utf-8", errors="replace")
        
        expected_line = f"computing over 1 chunks, n_ctx={ctx}"
        if expected_line not in text:
            return "OK_UNVERIFIED"
            
        return "OK"
        
    except RuntimeError as e:
        # Re-raise RuntimeError to be caught by main()
        raise e


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--profile", type=str)
    parser.add_argument("--jobs", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--ctx", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--settle-s", type=float, default=5.0)
    # appended to every llama-imatrix command line, e.g. --extra-arg=-lv --extra-arg=4 (newer llama.cpp builds print
    # the device buffer sizes the VRAM guard needs only at a raised log verbosity)
    parser.add_argument("--extra-arg", action="append", default=[], dest="extra_arg")
    # the per-job wall-clock ceiling (30 s + 0.12 s/token, ~3x the expected time on the machine it was tuned on) was found
    # too tight on a busy GT1030 (S7: 10 of 22 jobs timed out, all killed at the ceiling): scale it instead of editing the formula
    parser.add_argument("--ceiling-scale", type=float, default=1.0)
    # skip jobs whose output file already exists (resume after a partial run)
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args(argv)

    # Set default jobs path based on root if not provided
    if args.jobs is None:
        args.jobs = args.root / "data" / "routing_corpus_v2" / "jobs.json"
    
    # Handle ad-hoc job
    if args.file is not None or args.ctx is not None or args.out is not None:
        if not all([args.file, args.ctx, args.out]):
            print("Error: --file, --ctx, and --out must be provided together", file=sys.stderr)
            return 1
            
        # Create a single job dict for ad-hoc execution
        job = {
            "session": "adhoc",
            "k": 0,
            "infile": str(args.file),
            "outfile": str(args.out),
            "ctx": args.ctx,
            "window_type": "adhoc"
        }
        
        jobs = [job]
    else:
        # Load jobs from file
        if not args.jobs.exists():
            print(f"Jobs file not found: {args.jobs} (run scripts/build_imatrix_jobs.py first)", file=sys.stderr)
            return 1
            
        with open(args.jobs, "r", encoding="utf-8") as f:
            jobs = json.load(f)

    # Get profile
    profile = get_profile(args.profile)
    
    results = []
    runtime_error_occurred = False

    for job in jobs:
        label = f"{job['session']}_{job['k']}"
        print(f"=== Job {label} (ctx={job['ctx']}, type={job['window_type']}) ===")

        try:
            if args.dry_run:
                argv = build_imatrix_argv(profile, resolve_job_path(job["infile"], args.root), job["ctx"],
                                          resolve_job_path(job["outfile"], args.root)) + list(args.extra_arg)
                print(" ".join(argv))
                status = "DRY_RUN"
            else:
                if args.skip_existing and resolve_job_path(job["outfile"], args.root).exists():
                    status = "SKIPPED_EXISTING"
                else:
                    extra = {"extra_args": args.extra_arg} if args.extra_arg else {}
                    if args.ceiling_scale != 1.0:
                        extra["ceiling_scale"] = args.ceiling_scale
                    status = run_job(job, profile, args.root, **extra)

            print(f"  {label} -> {status}")

            results.append({
                "session": job["session"],
                "k": job["k"],
                "status": status,
                "outfile": str(resolve_job_path(job["outfile"], args.root))
            })

            if not args.dry_run and args.settle_s > 0 and status != "SKIPPED_EXISTING":
                time.sleep(args.settle_s)

        except RuntimeError as e:
            print(f"Error running job {label}: {e}", file=sys.stderr)
            runtime_error_occurred = True
            break

    # Write results to file. Not in --dry-run and not in ad-hoc (--file) mode: both would otherwise overwrite the real
    # job_results.json of the last full run (the PowerShell original also returned before writing it for ad-hoc jobs).
    if not args.dry_run and args.file is None:
        logs_dir = args.root / "logs" / "imatrix_v2"
        logs_dir.mkdir(parents=True, exist_ok=True)

        with open(logs_dir / "job_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    status_counts = {}
    for r in results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\nSummary:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    return 2 if runtime_error_occurred else 0


if __name__ == "__main__":
    sys.exit(main())
