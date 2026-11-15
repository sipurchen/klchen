"""Acceptor-written contract tests for S2 (render.py / gen_session.py).

Written by the acceptor after the Lead's own tests let 25 of 53 gen_session mutants survive (see
the acceptance record of that stage). Derived only from the S2 card contract and the committed golden files.
"""
import io
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from affinity_pipeline import gen_session as gs
from affinity_pipeline import render
from affinity_pipeline.config import REPO_ROOT, CpuLimit, MachineProfile, VramBudget, load_profile

SESS = REPO_ROOT / "data" / "routing_corpus_v2" / "sessions"
CRLF, LF = b"\r\n", b"\n"


def _golden():
    body = (SESS / "planner_pilot.body.txt").read_bytes().replace(CRLF, LF).decode("utf-8")
    raw = (SESS / "planner_pilot.turns.jsonl").read_bytes().replace(CRLF, LF)
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    return body, raw, rows


GOLDEN_BODY, GOLDEN_TURNS_BYTES, ROWS = _golden()
ANSWERS = [GOLDEN_BODY[r["model_start"]:r["model_end"]] for r in ROWS]
SPEC = gs.load_turn_spec(REPO_ROOT / "data" / "routing_corpus_v2" / "turns" / "planner_pilot.json")
QUIET = lambda s: None  # noqa: E731


def _prof(max_mb=1000):
    return MachineProfile(name="p", platform="windows" if sys.platform == "win32" else "linux",
                          llama_bin_dir=Path("x"), model_path=Path("m"), work_dir=Path("w"),
                          cpu=CpuLimit(mode="none", cores=(), priority="normal", threads=1),
                          vram=VramBudget(target_mb=max_mb // 2, max_mb=max_mb), max_load1=None)


def _alive(pid):
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_gone(pid, secs=5.0):
    end = time.time() + secs
    while time.time() < end:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


# ------------------------------------------------------------------ render
def test_render_header_variants():
    assert render.render_header(None) == "<|turn>system\n<|think|>\n<turn|>\n"
    assert render.render_header("") == "<|turn>system\n<|think|>\n<turn|>\n"
    assert render.render_header("S") == "<|turn>system\n<|think|>\nS<turn|>\n"


def test_render_user_turn_and_finalize():
    assert render.render_user_turn("hi") == "<|turn>user\nhi<turn|>\n"
    assert render.finalize_answer("abc  \n") == "abc<turn|>"
    assert render.finalize_answer("abc<turn|>") == "abc<turn|>"
    assert render.finalize_answer("abc<turn|>\n") == "abc<turn|>\n"


def test_transcript_offsets_are_characters_and_model_end_excludes_newline():
    t = render.Transcript(None)
    base = t.body
    us, ue, ms = t.open_user_turn("héllo 中文")
    assert us == len(base)
    assert ue == us + len("<|turn>user\nhéllo 中文<turn|>\n")
    assert ms == ue + len("<|turn>model\n")
    m = t.close_model_turn(turn=1, label="x", user_start=us, user_end=ue, model_start=ms, answer="ans",
                           stop_type="eos", tokens_predicted=3, seed_used=7, gen_tokens=[1, 2])
    assert t.body.endswith("ans<turn|>\n") and m.model_end == len(t.body) - 1
    assert t.turns == [m]
    assert list(m.to_dict()) == ["turn", "label", "user_start", "user_end", "model_start", "model_end",
                                 "stop_type", "tokens_predicted", "seed_used", "gen_tokens"]
    assert m.to_dict()["gen_tokens"] == [1, 2]
    t.rollback(us)
    assert t.body == base


# ------------------------------------------------------------------ generate_session
def _golden_complete(calls):
    def complete(prompt, seed):
        k = len(calls)
        calls.append((prompt, seed))
        r = ROWS[k]
        return {"content": ANSWERS[k], "stop_type": "eos", "tokens_predicted": r["tokens_predicted"], "tokens": r["gen_tokens"]}
    return complete


def test_generate_session_reproduces_golden_exactly():
    calls = []
    res = gs.generate_session(SPEC, _golden_complete(calls), log=QUIET)
    assert res.body == GOLDEN_BODY
    assert res.accepted is True and len(res.turns) == 4
    assert [t.to_dict() for t in res.turns] == ROWS
    assert [c[0] for c in calls] == [GOLDEN_BODY[:r["model_start"]] for r in ROWS]
    assert [c[1] for c in calls] == [3001] * 4


def test_missing_turn_close_is_re_added():
    calls = []
    inner = _golden_complete(calls)

    def complete(p, s):
        d = inner(p, s)
        d["content"] = d["content"].replace("<turn|>", "")
        return d
    assert gs.generate_session(SPEC, complete, log=QUIET).body == GOLDEN_BODY


def test_two_failures_reject_and_roll_back():
    seq = []

    def flaky(prompt, seed):
        seq.append(seed)
        if len(seq) == 1:
            return {"content": "partial", "stop_type": "length", "tokens_predicted": 5, "tokens": None}
        raise RuntimeError("boom")
    r = gs.generate_session(SPEC, flaky, log=QUIET)
    assert r.accepted is False and r.turns == []
    assert r.body == render.Transcript(SPEC.system).body          # rolled back to the header only
    assert seq == [3001, 3002]                                    # exactly 2 requests, seed + 1


def test_rejected_after_first_turn_keeps_only_accepted_turns():
    calls = []
    inner = _golden_complete(calls)
    n = {"c": 0}

    def complete(p, s):
        n["c"] += 1
        if n["c"] == 1:
            return inner(p, s)
        return {"content": "x", "stop_type": "length"}
    r = gs.generate_session(SPEC, complete, log=QUIET)
    assert r.accepted is False and len(r.turns) == 1
    assert r.body == GOLDEN_BODY[:ROWS[0]["model_end"] + 1]      # turn 1 + its trailing newline, turn 2 rolled back
    assert n["c"] == 3                                            # no request for turn 3


def test_seed_is_carried_across_turns_and_recorded():
    seq = []

    def flaky(prompt, seed):
        seq.append(seed)
        if len(seq) == 1:
            return {"content": "partial", "stop_type": "length"}
        return {"content": "ok", "stop_type": "eos", "tokens_predicted": 1, "tokens": [9]}
    r = gs.generate_session(SPEC, flaky, log=QUIET)
    n = len(SPEC.turns)
    assert seq == [3001] + [3002] * n
    assert [t.seed_used for t in r.turns] == [3002] * n


def test_seed_override_and_response_defaults():
    r = gs.generate_session(SPEC, lambda p, s: {"content": "x", "stop_type": "eos"}, seed=42, log=QUIET)
    t = r.turns[0]
    assert t.seed_used == 42 and t.gen_tokens is None and t.tokens_predicted == 0


def test_missing_stop_type_counts_as_failure_not_eos():
    r = gs.generate_session(SPEC, lambda p, s: {"content": "x"}, log=QUIET)
    assert r.accepted is False and r.turns == []


def test_max_tokens_valve_stops_after_first_turn_but_stays_accepted():
    r = gs.generate_session(SPEC, lambda p, s: {"content": "x" * 100, "stop_type": "eos"}, max_tokens=10, log=QUIET)
    assert len(r.turns) == 1 and r.accepted is True


def test_http_error_is_a_failed_attempt():
    def complete(p, s):
        raise OSError("HTTP Error 500")
    r = gs.generate_session(SPEC, complete, log=QUIET)
    assert r.accepted is False and r.turns == []


# ------------------------------------------------------------------ write_outputs
def test_write_outputs_bytes_equal_golden(tmp_path):
    res = gs.generate_session(SPEC, _golden_complete([]), log=QUIET)
    bp, tp = gs.write_outputs(res, tmp_path / "sub" / "dir", "planner", "pilot")   # missing dirs are created
    bb = bp.read_bytes()
    assert bb == GOLDEN_BODY.encode("utf-8") and b"\r" not in bb and not bb.startswith(b"\xef\xbb\xbf")
    tb = tp.read_bytes()
    assert tb == GOLDEN_TURNS_BYTES and b"\r" not in tb and not tb.startswith(b"\xef\xbb\xbf")
    assert bp.name == "planner_pilot.body.txt" and tp.name == "planner_pilot.turns.jsonl"


def test_write_outputs_non_ascii_is_not_escaped_and_empty_is_empty(tmp_path):
    t = render.TurnMeta(1, "中文", 0, 1, 2, 3, "eos", 1, 1, None)
    _, tp = gs.write_outputs(gs.SessionResult("b", [t], True), tmp_path, "f", "s")
    raw = tp.read_bytes()
    assert "中文".encode("utf-8") in raw and b"\\u" not in raw and raw.endswith(b"\n") and b"\r" not in raw
    assert b", " not in raw and b": " not in raw                    # compact separators
    _, tp2 = gs.write_outputs(gs.SessionResult("", [], True), tmp_path, "e", "s")
    assert tp2.read_bytes() == b""


# ------------------------------------------------------------------ build_server_argv
def test_server_argv_exact_windows_profile():
    gt = load_profile("gt1030_win")
    a = gs.build_server_argv(gt, 8260)
    assert a[0].lower().endswith("llama-server.exe")
    assert a[1:] == ["-m", str(gt.model_path), "-ngl", "0", "--override-tensor", r"\.ffn_.*_exps\.=CPU", "-c", "8192",
                     "-t", "2", "-tb", "2", "-np", "1", "-sp", "-fit", "off", "--swa-full", "--no-webui", "--no-warmup",
                     "--port", "8260"]


def test_server_argv_linux_profile_params_and_extra_last():
    gx = load_profile("gx10_sim")
    b = gs.build_server_argv(gx, 8261, ctx=4096, ngl=3, extra_args=["-lv", "4"])
    assert not b[0].endswith(".exe") and b[0].endswith("llama-server")
    assert (b[b.index("-t") + 1], b[b.index("-tb") + 1]) == ("5", "5")
    assert b[b.index("-ngl") + 1] == "3" and b[b.index("-c") + 1] == "4096"
    assert b[b.index("--port") + 1] == "8261" and b[-2:] == ["-lv", "4"]
    assert b.count(r"\.ffn_.*_exps\.=CPU") == 1


# ------------------------------------------------------------------ http_complete
class _Rec(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        _Rec.seen.append({"path": self.path, "ctype": self.headers.get("Content-Type"), "body": json.loads(self.rfile.read(n))})
        out = json.dumps({"content": "hi", "stop_type": "eos"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def mock_server():
    _Rec.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Rec)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(5)


def test_http_complete_request_is_exact(mock_server):
    out = gs.http_complete(mock_server)("PROMPT 中文", 77)
    assert out == {"content": "hi", "stop_type": "eos"}
    (req,) = _Rec.seen
    assert req["path"] == "/completion" and req["ctype"] == "application/json"
    assert req["body"] == {"prompt": "PROMPT 中文", "n_predict": 3072, "cache_prompt": True, "seed": 77,
                           "temperature": 1.0, "top_k": 64, "top_p": 0.95, "return_tokens": True}


# ------------------------------------------------------------------ load_turn_spec
def test_load_turn_spec_validation(tmp_path):
    ok = {"family": "f", "session": "s", "seed": 1, "turns": [{"label": "l", "user": "u"}]}

    def w(obj, bom=False):
        p = tmp_path / "s.json"
        p.write_bytes((b"\xef\xbb\xbf" if bom else b"") + json.dumps(obj).encode())
        return p
    assert gs.load_turn_spec(w(ok, bom=True)).system is None
    assert gs.load_turn_spec(w({**ok, "system": "S"})).system == "S"
    for key in ("family", "session", "seed", "turns"):
        o = dict(ok)
        o.pop(key)
        with pytest.raises(ValueError, match=key):
            gs.load_turn_spec(w(o))
    for bad in ({**ok, "turns": []}, {**ok, "turns": [{"label": "l"}]}, {**ok, "turns": [{"user": "u"}]},
                {**ok, "seed": "1"}, {**ok, "seed": True}):
        with pytest.raises(ValueError):
            gs.load_turn_spec(w(bad))


# ------------------------------------------------------------------ start_server
def test_start_server_ready_creates_log_dir_and_stops_cleanly(tmp_path):
    log = tmp_path / "nested" / "srv.log"
    gp = gs.start_server(_prof(), [sys.executable, "-c", "import time; print('server is listening', flush=True); time.sleep(60)"],
                         log, ready_timeout_s=20, poll_s=0.2)
    pid = gp.pid
    try:
        assert gp.poll() is None and _alive(pid) and log.exists()
    finally:
        gp.stop()
    assert _wait_gone(pid)


def test_start_server_early_exit_reports_log_tail(tmp_path):
    with pytest.raises(RuntimeError, match="dying words"):
        gs.start_server(_prof(), [sys.executable, "-c", "print('dying words')"], tmp_path / "a.log", ready_timeout_s=10, poll_s=0.2)


def test_start_server_timeout_kills_process(tmp_path):
    pidf = tmp_path / "pid.txt"
    code = f"import os,time; open({str(pidf)!r},'w').write(str(os.getpid())); time.sleep(60)"
    t0 = time.time()
    with pytest.raises(RuntimeError, match="(?i)not ready"):
        gs.start_server(_prof(), [sys.executable, "-c", code], tmp_path / "b.log", ready_timeout_s=1.0, poll_s=0.2)
    assert time.time() - t0 < 10
    assert pidf.exists() and _wait_gone(int(pidf.read_text()))


def _vram_server_code(tmp_path):
    fx = (REPO_ROOT / "tests" / "fixtures" / "affinity_pipeline" / "llama_b8679_vulkan_ngl9_ctx16384.log").read_text(
        encoding="utf-8", errors="replace")
    fpy = tmp_path / "fx.txt"
    fpy.write_text(fx + "\nserver is listening\n", encoding="utf-8")
    pidf = tmp_path / "pid2.txt"
    return (f"import os,time; open({str(pidf)!r},'w').write(str(os.getpid())); "
            f"print(open({str(fpy)!r},encoding='utf-8').read(), flush=True); time.sleep(60)"), pidf


def test_start_server_vram_over_budget_aborts_and_kills(tmp_path):
    code, pidf = _vram_server_code(tmp_path)
    with pytest.raises(RuntimeError, match="(?i)vram"):
        gs.start_server(_prof(max_mb=1000), [sys.executable, "-c", code], tmp_path / "c.log", ready_timeout_s=15, poll_s=0.2)
    assert pidf.exists() and _wait_gone(int(pidf.read_text()))


def test_start_server_check_vram_false_accepts_same_log(tmp_path):
    code, _ = _vram_server_code(tmp_path)
    gp = gs.start_server(_prof(max_mb=1000), [sys.executable, "-c", code], tmp_path / "d.log", ready_timeout_s=15,
                         poll_s=0.2, check_vram=False)
    try:
        assert gp.poll() is None
    finally:
        gp.stop()


# ------------------------------------------------------------------ main
def test_cli_dry_run_prints_one_json_and_writes_nothing(tmp_path):
    out = tmp_path / "out"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = gs.main(["--profile", "gt1030_win", "--family", "planner", "--session", "pilot", "--dry-run", "--out-dir", str(out)])
    j = json.loads(buf.getvalue())
    assert rc == 0 and set(j) >= {"profile", "turns_file", "seed", "n_turns", "server_argv", "outputs"}
    assert j["seed"] == 3001 and j["n_turns"] == len(SPEC.turns)
    assert j["server_argv"] == gs.build_server_argv(load_profile("gt1030_win"), 8260)   # default port 8260, ctx 8192, ngl 0
    assert len(j["outputs"]) == 2 and not out.exists()


def test_cli_missing_turns_file_returns_2(tmp_path):
    assert gs.main(["--profile", "gt1030_win", "--family", "x", "--session", "y",
                    "--turns-file", str(tmp_path / "nope.json"), "--out-dir", str(tmp_path / "o")]) == 2


def _small_spec(tmp_path):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"family": "t", "session": "s", "seed": 5, "system": None,
                             "turns": [{"label": "a", "user": "one"}, {"label": "b", "user": "two"}]}), encoding="utf-8")
    return p


def test_cli_server_url_runs_end_to_end_without_starting_a_server(tmp_path, monkeypatch, mock_server):
    def boom(*a, **k):
        raise AssertionError("start_server must not be called when --server-url is given")
    monkeypatch.setattr(gs, "start_server", boom)
    out = tmp_path / "out"
    rc = gs.main(["--profile", "gt1030_win", "--family", "t", "--session", "s", "--turns-file", str(_small_spec(tmp_path)),
                  "--out-dir", str(out), "--server-url", mock_server])
    assert rc == 0 and len(_Rec.seen) == 2
    assert (out / "t_s.body.txt").exists() and len((out / "t_s.turns.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_cli_rejected_session_returns_3_but_still_writes_files(tmp_path):
    class Fail(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            out = json.dumps({"content": "x", "stop_type": "length"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fail)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        out = tmp_path / "out"
        rc = gs.main(["--profile", "gt1030_win", "--family", "t", "--session", "s", "--turns-file", str(_small_spec(tmp_path)),
                      "--out-dir", str(out), "--server-url", f"http://127.0.0.1:{srv.server_address[1]}"])
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(5)
    assert rc == 3 and (out / "t_s.body.txt").exists() and (out / "t_s.turns.jsonl").read_bytes() == b""


def test_cli_starts_server_passes_defaults_and_always_stops_it(tmp_path, monkeypatch):
    events = {}

    class FakeGP:
        def stop(self):
            events["stopped"] = True

    def fake_start(profile, argv, log_path, **kw):
        events["argv"], events["log_path"], events["kw"] = list(argv), Path(log_path), kw
        return FakeGP()

    def fake_generate(spec, complete, **kw):
        events["gen_kw"] = kw
        raise RuntimeError("generation exploded")          # the server must still be stopped
    monkeypatch.setattr(gs, "start_server", fake_start)
    monkeypatch.setattr(gs, "generate_session", fake_generate)
    rc = gs.main(["--profile", "gt1030_win", "--family", "t", "--session", "s", "--turns-file", str(_small_spec(tmp_path)),
                  "--out-dir", str(tmp_path / "o")])
    assert rc == 2 and events["stopped"] is True
    assert events["gen_kw"]["max_tokens"] == 8191 and events["gen_kw"]["seed"] is None
    assert "8260" in events["argv"] and events["kw"]["check_vram"] is True
    assert events["log_path"].name == "gen_t_s.log"


def test_start_server_accepts_the_newer_llama_server_ready_line(tmp_path):
    """gx10's llama.cpp (d327bf1) says 'listening on http://127.0.0.1:PORT' instead of 'server is listening'
    (found in S5: every generation timed out with 'not ready' although the server was up after 9 s)."""
    code = "import time; print('srv server_on_li: listening on http://127.0.0.1:8260', flush=True); time.sleep(60)"
    gp = gs.start_server(_prof(), [sys.executable, "-c", code], tmp_path / "n.log", ready_timeout_s=20, poll_s=0.2)
    try:
        assert gp.poll() is None
    finally:
        gp.stop()
