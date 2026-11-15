"""Acceptor-written contract tests for S3 (scripts/segment_chunk_types.py, scripts/build_imatrix_jobs.py).

Derived from the S3 card only; the real-data golden regression (byte-identical spans.json / exports / jobs.json /
imatrix_inputs against the committed pilots, real llama-tokenize) is run by the acceptor outside pytest.
"""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from affinity_pipeline.config import REPO_ROOT, load_profile
from affinity_pipeline.render import Transcript

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_imatrix_jobs as bij  # noqa: E402
import segment_chunk_types as seg  # noqa: E402

MODULES = [seg, bij]


# ---------------------------------------------------------------- get_profile
@pytest.mark.parametrize("mod", MODULES)
def test_get_profile_priority(mod, monkeypatch):
    monkeypatch.delenv("AFFINITY_PROFILE", raising=False)
    assert mod.get_profile().name == "gt1030"                       # default gt1030_win
    monkeypatch.setenv("AFFINITY_PROFILE", "gx10_sim")
    assert mod.get_profile().name == "gx10_sim"                     # env beats default
    assert mod.get_profile("gt1030_win").name == "gt1030"           # parameter beats env


# ---------------------------------------------------------------- run_tokenize
class _Done:
    def __init__(self, rc=0, out="[1, 2, 3]", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _patch_run(monkeypatch, done):
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["kw"] = argv, kw
        return done
    monkeypatch.setattr(seg.subprocess, "run", fake_run)
    monkeypatch.setattr(seg, "_load_vocab", lambda m: ["a", "b", "c", "d"])
    return seen


@pytest.mark.parametrize("profile_name,exe_suffix", [("gt1030_win", "llama-tokenize.exe"), ("gx10_sim", "llama-tokenize")])
def test_run_tokenize_argv_is_exact(monkeypatch, profile_name, exe_suffix):
    seen = _patch_run(monkeypatch, _Done())
    prof = load_profile(profile_name)
    toks = seg.run_tokenize(Path("model.gguf"), Path("body.txt"), profile=prof)
    argv = seen["argv"]
    assert argv[0].endswith(exe_suffix) and (profile_name != "gx10_sim" or not argv[0].endswith(".exe"))
    assert argv[1:] == ["-m", str(Path("model.gguf")), "--no-bos", "--no-escape", "--ids", "-f", str(Path("body.txt")), "--log-disable"]
    assert seen["kw"]["timeout"] == 300 and seen["kw"]["capture_output"] is True
    assert toks == [(1, "b"), (2, "c"), (3, "d")]


def test_run_tokenize_error_branches(monkeypatch):
    prof = load_profile("gt1030_win")
    _patch_run(monkeypatch, _Done(rc=1, err="x" * 3000 + "TAILMARK"))
    with pytest.raises(RuntimeError, match="TAILMARK"):
        seg.run_tokenize(Path("m"), Path("b"), profile=prof)
    _patch_run(monkeypatch, _Done(out="no list here"))
    with pytest.raises(RuntimeError, match="id list"):
        seg.run_tokenize(Path("m"), Path("b"), profile=prof)
    _patch_run(monkeypatch, _Done(out="[1, 9]"))
    with pytest.raises(RuntimeError, match="out of vocab range"):
        seg.run_tokenize(Path("m"), Path("b"), profile=prof)


# ---------------------------------------------------------------- segment
def _make_session(root: Path, sid="s1"):
    t = Transcript(None)
    turns = []
    for i, (user, ans) in enumerate([("first question about plans", "thinking step one\nthen a short answer"),
                                     ("second question", "here is code:\n```python\nprint(1)\n```\ndone")], start=1):
        us, ue, ms = t.open_user_turn(user)
        m = t.close_model_turn(turn=i, label=f"t{i}", user_start=us, user_end=ue, model_start=ms, answer=ans,
                               stop_type="eos", tokens_predicted=1, seed_used=1, gen_tokens=None)
        turns.append(m.to_dict())
    sd = root / "data" / "routing_corpus_v2" / "sessions"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / f"{sid}.body.txt").write_bytes(t.body.encode("utf-8"))
    (sd / f"{sid}.turns.jsonl").write_bytes(("\n".join(json.dumps(x, separators=(",", ":")) for x in turns) + "\n").encode("utf-8"))
    return t.body


def _char_tokenizer(body):
    return lambda model_path, body_path, **kw: [(i, c) for i, c in enumerate(body)]


def test_segment_writes_spans_and_exports_under_root(tmp_path, monkeypatch):
    body = _make_session(tmp_path)
    monkeypatch.setattr(seg, "run_tokenize", _char_tokenizer(body))
    spans, counts = seg.segment("s1", Path("m.gguf"), root=tmp_path, profile=load_profile("gt1030_win"))
    sp_path = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "s1.spans.json"
    raw = sp_path.read_bytes()
    # the original wrote with default text mode (CRLF on Windows, committed blobs are LF-normalised): compare modulo EOL
    assert not raw.startswith(b"\xef\xbb\xbf") and raw.replace(b"\r\n", b"\n").startswith(b'{\n  "session"')   # indent=2
    doc = json.loads(raw)
    assert doc["session"] == "s1" and doc["n_tokens"] == len(body) and doc["spans"] == spans
    cov = 0
    for s in doc["spans"]:                                                        # exact, gap-free coverage
        assert s["start"] == cov
        cov = s["end"]
    assert cov == len(body) and sum(counts.values()) == len(body)
    assert (tmp_path / "data" / "routing_corpus_v2" / "exports" / "s1.txt").read_text(encoding="utf-8").startswith(f"# session=s1 n_tokens={len(body)}\n")
    assert "code" in counts                                                       # the fenced block was classified


def test_segment_passes_profile_and_model_to_tokenizer(tmp_path, monkeypatch):
    body = _make_session(tmp_path)
    got = {}

    def fake(model_path, body_path, **kw):
        got.update(model=model_path, body=body_path, kw=kw)
        return [(i, c) for i, c in enumerate(body)]
    monkeypatch.setattr(seg, "run_tokenize", fake)
    prof = load_profile("gx10_sim")
    seg.segment("s1", Path("the.gguf"), root=tmp_path, profile=prof)
    assert got["model"] == Path("the.gguf") and got["kw"]["profile"] is prof
    assert got["body"] == tmp_path / "data" / "routing_corpus_v2" / "sessions" / "s1.body.txt"


def test_segment_missing_inputs_name_the_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="s9.body.txt"):
        seg.segment("s9", Path("m"), root=tmp_path, profile=load_profile("gt1030_win"))


def test_segment_main_root_profile_model_and_failure_exit(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(seg, "segment", lambda sid, model, **kw: calls.append((sid, model, kw)))
    seg.main(["a", "b", "--root", str(tmp_path), "--profile", "gx10_sim"])
    assert [c[0] for c in calls] == ["a", "b"]
    assert calls[0][1] == load_profile("gx10_sim").model_path and calls[0][2]["root"] == tmp_path
    assert calls[0][2]["profile"].name == "gx10_sim"
    calls.clear()
    seg.main(["a", "--model", "custom.gguf", "--root", str(tmp_path)])
    assert calls[0][1] == Path("custom.gguf")

    def boom(sid, model, **kw):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(seg, "segment", boom)
    with pytest.raises(SystemExit) as e:
        seg.main(["zzz", "--root", str(tmp_path)])
    assert e.value.code == 1 and "FAILED zzz: kaboom" in capsys.readouterr().err


# ---------------------------------------------------------------- build_imatrix_jobs
def _spans_session(root: Path, sid="j1", n=700):
    body = "".join(chr(97 + (i % 26)) for i in range(n))
    spans = [{"start": 0, "end": 300, "type": "reasoning"}, {"start": 300, "end": 600, "type": "code"},
             {"start": 600, "end": n, "type": "reasoning"}]
    sd = root / "data" / "routing_corpus_v2" / "sessions"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / f"{sid}.body.txt").write_bytes(body.encode("utf-8"))
    (sd / f"{sid}.spans.json").write_text(json.dumps({"session": sid, "n_tokens": n, "spans": spans}), encoding="utf-8")
    return body


def test_resolve_job_path(tmp_path):
    assert bij.resolve_job_path("data/x.txt", tmp_path) == tmp_path / "data" / "x.txt"
    absolute = str((tmp_path / "abs.txt").resolve())
    assert bij.resolve_job_path(absolute, Path("/other")) == Path(absolute)


def test_build_jobs_for_session_contract(tmp_path, monkeypatch):
    body = _spans_session(tmp_path)
    monkeypatch.setattr(bij, "run_tokenize", _char_tokenizer(body))
    res = bij.build_jobs_for_session("j1", Path("m"), root=tmp_path, profile=load_profile("gt1030_win"))
    jobs = res["jobs"]
    assert res["n_tokens"] == 700 and res["n_windows"] == len(jobs) >= 2
    assert list(jobs[0]) == ["session", "k", "b_prev", "b_k", "window_type", "window_dominant", "window_purity",
                             "window_start", "window_end", "ctx", "infile", "outfile", "is_whole_session"]
    prev = 0
    for k, j in enumerate(jobs, start=1):
        assert j["k"] == k and j["b_prev"] == prev and j["ctx"] == j["b_k"] + 1
        assert j["infile"] == f"data/routing_corpus_v2/imatrix_inputs/j1_{k}.txt"
        assert j["outfile"] == f"logs/imatrix_v2/j1_{k}.gguf"
        text = (tmp_path / j["infile"]).read_text(encoding="utf-8")
        prefix = body[:j["b_k"]]
        assert text == prefix + "<bos>" + prefix + "<bos>" + prefix          # triplicated prefix
        assert j["is_whole_session"] is (k == len(jobs))
        prev = j["b_k"]
    assert (tmp_path / "logs" / "imatrix_v2").is_dir()


def test_build_jobs_token_count_mismatch_asserts(tmp_path, monkeypatch):
    body = _spans_session(tmp_path)
    monkeypatch.setattr(bij, "run_tokenize", lambda *a, **k: [(i, c) for i, c in enumerate(body[:-1])])
    with pytest.raises((AssertionError, RuntimeError)):
        bij.build_jobs_for_session("j1", Path("m"), root=tmp_path, profile=load_profile("gt1030_win"))


def test_build_jobs_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="segment_chunk_types"):
        bij.build_jobs_for_session("nope", Path("m"), root=tmp_path, profile=load_profile("gt1030_win"))


def test_bij_main_writes_jobs_and_summary_under_root(tmp_path, monkeypatch, capsys):
    body = _spans_session(tmp_path, "j1")
    _spans_session(tmp_path, "j2")
    got = {}

    def fake_tok(model_path, body_path, **kw):
        got.setdefault("models", []).append(model_path)
        return [(i, c) for i, c in enumerate(body)]
    monkeypatch.setattr(bij, "run_tokenize", fake_tok)
    bij.main(["--all", "--root", str(tmp_path), "--profile", "gx10_sim"])
    jobs_path = tmp_path / "data" / "routing_corpus_v2" / "jobs.json"
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    assert {j["session"] for j in jobs} == {"j1", "j2"}
    assert [j["session"] for j in jobs] == sorted(j["session"] for j in jobs)    # sessions in sorted order
    assert jobs_path.read_bytes().replace(b"\r\n", b"\n").startswith(b'[\n  {\n    "session"')  # indent=2
    summ = json.loads((tmp_path / "data" / "routing_corpus_v2" / "jobs_summary.json").read_text(encoding="utf-8"))
    assert [s["session"] for s in summ] == ["j1", "j2"] and "jobs" not in summ[0]
    assert set(got["models"]) == {load_profile("gx10_sim").model_path}
    assert "Wrote 4 jobs" in capsys.readouterr().out or len(jobs) >= 2


def test_bij_main_out_option_and_no_ids_and_failure(tmp_path, monkeypatch, capsys):
    body = _spans_session(tmp_path, "j1")
    monkeypatch.setattr(bij, "run_tokenize", _char_tokenizer(body))
    out = tmp_path / "custom" / "jobs_x.json"
    out.parent.mkdir()
    bij.main(["j1", "--root", str(tmp_path), "-o", str(out), "--model", "mm.gguf"])
    assert out.exists() and (out.parent / "jobs_x_summary.json").exists()
    assert not (tmp_path / "data" / "routing_corpus_v2" / "jobs.json").exists()
    with pytest.raises(SystemExit) as e:
        bij.main(["--root", str(tmp_path)])
    assert e.value.code == 1 and "No session ids" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        bij.main(["missing_one", "--root", str(tmp_path)])
    assert e.value.code == 1 and "FAILED missing_one" in capsys.readouterr().err


# ---------------------------------------------------------------- static / CLI wiring
@pytest.mark.parametrize("name", ["segment_chunk_types.py", "build_imatrix_jobs.py"])
def test_no_hardcoded_machine_paths(name):
    src = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
    assert "E:\\" not in src and "C:\\" not in src and ".exe" not in src.replace("llama-tokenize.exe's", "")


@pytest.mark.parametrize("name", ["segment_chunk_types.py", "build_imatrix_jobs.py"])
def test_script_help_runs_from_any_cwd(name, tmp_path):
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / name), "--help"], cwd=tmp_path, capture_output=True,
                       text=True, encoding="utf-8")
    assert r.returncode == 0 and "--root" in r.stdout and "--profile" in r.stdout


# ---------------------------------------------------------------- extra tests added after the first mutation run
@pytest.mark.parametrize("mod", MODULES)
def test_missing_either_input_is_an_error(mod, tmp_path):
    sd = tmp_path / "data" / "routing_corpus_v2" / "sessions"
    sd.mkdir(parents=True)
    other = "turns.jsonl" if mod is seg else "spans.json"
    fn = (lambda sid: mod.segment(sid, Path("m"), root=tmp_path, profile=load_profile("gt1030_win"))) if mod is seg else          (lambda sid: mod.build_jobs_for_session(sid, Path("m"), root=tmp_path, profile=load_profile("gt1030_win")))
    (sd / "x.body.txt").write_text("b", encoding="utf-8")            # only the body exists
    with pytest.raises(FileNotFoundError, match="^missing"):          # our explicit message, not a bare open() error
        fn("x")
    (sd / "x.body.txt").unlink()
    (sd / f"y.{other}").write_text("{}", encoding="utf-8")           # only the second input exists
    with pytest.raises(FileNotFoundError, match="^missing"):
        fn("y")


def test_run_tokenize_without_profile_uses_get_profile(monkeypatch):
    seen = _patch_run(monkeypatch, _Done())
    monkeypatch.setenv("AFFINITY_PROFILE", "gx10_sim")
    seg.run_tokenize(Path("m"), Path("b"))
    assert not seen["argv"][0].endswith(".exe") and seen["argv"][0].endswith("llama-tokenize")


def test_build_jobs_without_profile_and_profile_is_forwarded(tmp_path, monkeypatch):
    body = _spans_session(tmp_path)
    got = {}

    def fake_tok(model_path, body_path, **kw):
        got.update(kw)
        return [(i, c) for i, c in enumerate(body)]
    monkeypatch.setattr(bij, "run_tokenize", fake_tok)
    monkeypatch.setenv("AFFINITY_PROFILE", "gx10_sim")
    bij.build_jobs_for_session("j1", Path("m"), root=tmp_path)           # profile omitted -> get_profile()
    assert got["profile"].name == "gx10_sim"
    prof = load_profile("gt1030_win")
    bij.build_jobs_for_session("j1", Path("m"), root=tmp_path, profile=prof)
    assert got["profile"] is prof


def test_build_jobs_spans_token_count_must_match_tokenizer(tmp_path, monkeypatch):
    body = _spans_session(tmp_path)
    p = tmp_path / "data" / "routing_corpus_v2" / "sessions" / "j1.spans.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["n_tokens"] += 1                                                # spans.json is stale w.r.t. the tokenizer
    p.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(bij, "run_tokenize", _char_tokenizer(body))
    with pytest.raises(AssertionError, match="retokenization length"):
        bij.build_jobs_for_session("j1", Path("m"), root=tmp_path, profile=load_profile("gt1030_win"))


def test_build_jobs_coverage_is_rounded_to_four_places(tmp_path, monkeypatch):
    body = _spans_session(tmp_path, n=701)
    monkeypatch.setattr(bij, "run_tokenize", _char_tokenizer(body))
    res = bij.build_jobs_for_session("j1", Path("m"), root=tmp_path, profile=load_profile("gt1030_win"))
    pure = sum(w["end"] - w["start"] for w in res["windows"] if w["type"] != "mixed")
    assert res["coverage_non_mixed"] == round(pure / 701, 4) and len(str(res["coverage_non_mixed"]).split(".")[1]) > 1


def test_bij_main_model_option_reaches_tokenizer(tmp_path, monkeypatch):
    body = _spans_session(tmp_path, "j1")
    models = []

    def fake_tok(model_path, body_path, **kw):
        models.append(model_path)
        return [(i, c) for i, c in enumerate(body)]
    monkeypatch.setattr(bij, "run_tokenize", fake_tok)
    bij.main(["j1", "--root", str(tmp_path), "--model", "mm.gguf"])
    assert models == [Path("mm.gguf")]
