from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import sys
import time
import urllib.request
from typing import Sequence, Callable, Optional

from affinity_pipeline.config import MachineProfile, load_profile, resolve_binary, REPO_ROOT
from affinity_pipeline.guarded_process import GuardedProcess
from affinity_pipeline.llama_logs import parse_buffer_lines, check_budget, BudgetVerdict
from affinity_pipeline.render import Transcript, TurnMeta

@dataclass(frozen=True)
class TurnItem:
    label: str
    user: str

@dataclass(frozen=True)
class TurnSpec:
    family: str
    session: str
    seed: int
    system: str | None
    turns: tuple[TurnItem, ...]

@dataclass
class SessionResult:
    body: str
    turns: list[TurnMeta]
    accepted: bool

def load_turn_spec(path: str | Path) -> TurnSpec:
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    
    try:
        family = data["family"]
        session = data["session"]
        seed = data["seed"]
        system = data.get("system")
        turns_data = data["turns"]
    except KeyError as e:
        raise ValueError(f"Missing or invalid required field: {e.args[0]}")
    
    if not isinstance(turns_data, list) or len(turns_data) == 0:
        raise ValueError("turns must be a non-empty list")
    
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    
    turns = tuple(
        TurnItem(label=item["label"], user=item["user"])
        for item in turns_data
        if "label" in item and "user" in item
    )
    
    if len(turns) != len(turns_data):
        raise ValueError("Each turn must have 'label' and 'user'")
    
    return TurnSpec(
        family=family,
        session=session,
        seed=seed,
        system=system,
        turns=turns
    )

def build_server_argv(profile: MachineProfile, port: int, *, ctx: int = 8192, ngl: int = 0, extra_args: Sequence[str] = ()) -> list[str]:
    return [
        str(resolve_binary(profile, "llama-server")),
        "-m", str(profile.model_path),
        "-ngl", str(ngl),
        "--override-tensor", r"\.ffn_.*_exps\.=CPU",
        "-c", str(ctx),
        "-t", str(profile.cpu.threads),
        "-tb", str(profile.cpu.threads),
        "-np", "1",
        "-sp",
        "-fit", "off",
        "--swa-full",
        "--no-webui",
        "--no-warmup",
        "--port", str(port),
        *extra_args
    ]

def http_complete(base_url: str, *, timeout_s: float = 2400) -> Callable[[str, int], dict]:
    def complete(prompt: str, seed: int) -> dict:
        data = {
            "prompt": prompt,
            "n_predict": 3072,
            "cache_prompt": True,
            "seed": seed,
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "return_tokens": True
        }
        
        req = urllib.request.Request(
            url=f"{base_url}/completion",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    
    return complete

def generate_session(spec: TurnSpec, complete: Callable[[str, int], dict], *, seed: int | None = None, max_tokens: int = 8191, log: Callable[[str], None] = print) -> SessionResult:
    cur_seed = seed if seed is not None else spec.seed
    transcript = Transcript(spec.system)
    accepted = True

    for turn_idx, item in enumerate(spec.turns, start=1):
        user_start, user_end, model_start = transcript.open_user_turn(item.user)
        
        attempt = 1
        stop_type = "unknown"
        answer = ""
        tokens_predicted = 0
        gen_tokens = None
        
        while attempt <= 2:
            try:
                resp = complete(transcript.body, cur_seed)
                answer = resp.get("content", "")
                stop_type = resp.get("stop_type", "unknown")
                tokens_predicted = resp.get("tokens_predicted", 0)
                gen_tokens = resp.get("tokens")
            except Exception as e:
                log(f"[{spec.family}/{spec.session} turn {turn_idx}] request failed: {e}")
                stop_type = "error"
            
            if stop_type == "eos":
                break
            
            log(f"[{spec.family}/{spec.session} turn {turn_idx}] stop_type={stop_type}, retrying with seed+1")
            cur_seed += 1
            attempt += 1
        
        if stop_type != "eos":
            log(f"[{spec.family}/{spec.session} turn {turn_idx}] never got eos after 2 attempts — truncating session before this turn.")
            transcript.rollback(user_start)
            accepted = False
            break
        
        seed_used = cur_seed
        meta = transcript.close_model_turn(
            turn=turn_idx, label=item.label, user_start=user_start, user_end=user_end,
            model_start=model_start, answer=answer, stop_type=stop_type,
            tokens_predicted=tokens_predicted, seed_used=seed_used, gen_tokens=gen_tokens
        )
        
        if meta.model_end > max_tokens * 3:
            log(f"[{spec.family}/{spec.session}] model_end {meta.model_end} > {max_tokens * 3}, stopping.")
            break

    return SessionResult(body=transcript.body, turns=transcript.turns, accepted=accepted)

def write_outputs(result: SessionResult, out_dir: str | Path, family: str, session: str) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    body_path = out_dir / f"{family}_{session}.body.txt"
    turns_path = out_dir / f"{family}_{session}.turns.jsonl"
    
    with open(body_path, "wb") as f:
        f.write(result.body.encode("utf-8"))
    
    with open(turns_path, "w", encoding="utf-8", newline="\n") as f:
        for turn in result.turns:
            line = json.dumps(turn.to_dict(), separators=(",", ":"), ensure_ascii=False) + "\n"
            f.write(line)
    
    return (body_path, turns_path)

def start_server(profile: MachineProfile, argv: Sequence[str], log_path: Path, *, ready_timeout_s: float = 60.0, poll_s: float = 0.5, check_vram: bool = True) -> GuardedProcess:
    import time

    log_path.parent.mkdir(parents=True, exist_ok=True)

    gp = GuardedProcess(argv, profile, log_path=log_path)
    gp.start()

    start_time = time.time()
    while True:
        if gp.poll() is not None:
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                raise RuntimeError(f"Server process died early. Log:\n{content[-2000:]}")
            except FileNotFoundError:
                raise RuntimeError("Server process died early and log file not found")
        
        if time.time() - start_time > ready_timeout_s:
            gp.stop()
            raise RuntimeError("Server not ready within timeout.")
        
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            # b8679 prints "server is listening", the newer gx10 build (d327bf1) prints "listening on http://..."
            if any(marker in content for marker in ("server is listening", "listening on http")):
                break
        except FileNotFoundError:
            pass
        
        time.sleep(poll_s)

    if check_vram:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            log_text = f.read()
        verdict = check_budget(parse_buffer_lines(log_text), profile.vram)
        if verdict == BudgetVerdict.ABORT:
            gp.stop()
            raise RuntimeError("VRAM budget exceeded: server not suitable for this profile")

    return gp

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--turns-file")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--port", type=int, default=8260)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--ngl", type=int, default=0)
    parser.add_argument("--out-dir")
    parser.add_argument("--max-tokens", type=int, default=8191)
    parser.add_argument("--server-url")
    parser.add_argument("--extra-server-arg", action="append")
    parser.add_argument("--no-vram-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    try:
        profile = load_profile(args.profile)
    except Exception as e:
        print(f"Error loading profile: {e}", file=sys.stderr)
        return 2

    family = args.family
    session = args.session

    if args.turns_file is None:
        turns_file = REPO_ROOT / "data/routing_corpus_v2/turns" / f"{family}_{session}.json"
    else:
        turns_file = Path(args.turns_file)

    try:
        spec = load_turn_spec(turns_file)
    except Exception as e:
        print(f"Error loading turns spec: {e}", file=sys.stderr)
        return 2

    if args.out_dir is None:
        out_dir = REPO_ROOT / "data/routing_corpus_v2/sessions"
    else:
        out_dir = Path(args.out_dir)

    argv = build_server_argv(
        profile,
        port=args.port,
        ctx=args.ctx,
        ngl=args.ngl,
        extra_args=args.extra_server_arg or ()
    )

    if args.dry_run:
        body_path, turns_path = out_dir / f"{family}_{session}.body.txt", out_dir / f"{family}_{session}.turns.jsonl"
        print(json.dumps({
            "profile": args.profile,
            "turns_file": str(turns_file),
            "seed": spec.seed,
            "n_turns": len(spec.turns),
            "server_argv": argv,
            "outputs": [str(body_path), str(turns_path)]
        }))
        return 0

    server_process = None
    try:
        if args.server_url is not None:
            complete_func = http_complete(args.server_url)
        else:
            log_path = profile.work_dir / "logs" / f"gen_{family}_{session}.log"
            server_process = start_server(
                profile,
                argv,
                log_path,
                check_vram=not args.no_vram_check
            )
            complete_func = http_complete(f"http://localhost:{args.port}")
        
        result = generate_session(
            spec,
            complete_func,
            seed=args.seed,
            max_tokens=args.max_tokens
        )
        
        write_outputs(result, out_dir, family, session)
        
        if not result.accepted:
            return 3
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    finally:
        if server_process is not None:
            server_process.stop()

if __name__ == "__main__":
    sys.exit(main())
