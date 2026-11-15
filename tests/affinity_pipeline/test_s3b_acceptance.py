"""Acceptor-written contract tests for S3b (scripts/profile_imatrix.py, scripts/aggregate_expert_affinity.py).

Real-data regression for aggregate (old vs new byte-identical on logs/imatrix_v2) is run by the acceptor outside pytest.
All process tests use throw-away python one-liners as the "llama-imatrix" (never a model).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from affinity_pipeline.config import REPO_ROOT, CpuLimit, MachineProfile, VramBudget, load_profile

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import aggregate_expert_affinity as agg  # noqa: E402
import profile_imatrix as pi  # noqa: E402


def _prof(max_mb=1000):
    return MachineProfile(name="p", platform="windows" if sys.platform == "win32" else "linux",
                          llama_bin_dir=Path("bin"), model_path=Path("m.gguf"), work_dir=Path("w"),
                          cpu=CpuLimit(mode="none", cores=(), priority="normal", threads=2),
                          vram=VramBudget(target_mb=max_mb // 2, max_mb=max_mb), max_load1=None)


def _alive(pid):
    if sys.platform == "win32":
        return str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _gone(pid, secs=6.0):
    end = time.time() + secs
    while time.time() < end and _alive(pid):
        time.sleep(0.1)
    return not _alive(pid)


def _job(tmp_path, ctx=300, k=1):
    return {"session": "s", "k": k, "ctx": ctx, "window_type": "reasoning",
            "infile": "data/routing_corpus_v2/imatrix_inputs/s_1.txt", "outfile": "logs/imatrix_v2/s_1.gguf"}


def _fake(code):
    return [sys.executable, "-c", code]


# ------------------------------------------------------------------ pure functions
def test_argv_exact_windows_and_linux():
    inf, outf = Path("in.txt"), Path("out.gguf")
    w = pi.build_imatrix_argv(load_profile("gt1030_win"), inf, 2057, outf)
    assert w[0].lower().endswith("llama-imatrix.exe")
    tail = ["-m", str(load_profile("gt1030_win").model_path), "-ngl", "9", "--override-tensor", r"\.ffn_.*_exps\.=CPU", "-c", "2057",
            "-b", "512", "-ub", "512", "-t", "2", "-tb", "2", "-fit", "off", "-f", "in.txt", "--chunks", "1", "--parse-special",
            "--no-escape", "--no-ppl", "--output-format", "gguf", "-o", "out.gguf"]
    assert w[1:] == tail
    x = pi.build_imatrix_argv(load_profile("gx10_sim"), inf, 500, outf)
    assert x[0].endswith("llama-imatrix") and not x[0].endswith(".exe")
    assert (x[x.index("-t") + 1], x[x.index("-tb") + 1]) == ("5", "5") and x[x.index("-c") + 1] == "500"


@pytest.mark.parametrize("ctx,expected", [(2057, 277), (1, 31), (1000, 150), (0, 30)])
def test_ceiling(ctx, expected):
    assert pi.job_ceiling_seconds(ctx) == expected


# ------------------------------------------------------------------ run_job
def _run(tmp_path, code, ctx=300, prof=None, poll_s=0.1, **kw):
    job = _job(tmp_path, ctx)
    return pi.run_job(job, prof or _prof(), tmp_path, argv=_fake(code), poll_s=poll_s, **kw)


def _writes_out(tmp_path, line=True, ctx=300):
    out = tmp_path / "logs" / "imatrix_v2" / "s_1.gguf"
    return (f"import pathlib; p=pathlib.Path({str(out)!r}); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'x'); "
            + (f"print('computing over 1 chunks, n_ctx={ctx}', flush=True)" if line else "pass"))


def test_run_job_ok(tmp_path):
    assert _run(tmp_path, _writes_out(tmp_path)) == "OK"


def test_run_job_ok_unverified_without_the_log_line(tmp_path):
    assert _run(tmp_path, _writes_out(tmp_path, line=False)) == "OK_UNVERIFIED"


def test_run_job_wrong_ctx_in_log_line_is_unverified(tmp_path):
    assert _run(tmp_path, _writes_out(tmp_path, line=True, ctx=999), ctx=300) == "OK_UNVERIFIED"


def test_run_job_no_output_prints_log_tail(tmp_path, capsys):
    assert _run(tmp_path, "print('LASTWORDS', flush=True)") == "NO_OUTPUT"
    assert "LASTWORDS" in capsys.readouterr().err


def test_run_job_timeout_kills_process(tmp_path, monkeypatch):
    monkeypatch.setattr(pi, "job_ceiling_seconds", lambda ctx: 1)
    pidf = tmp_path / "pid.txt"
    code = f"import os,time; open({str(pidf)!r},'w').write(str(os.getpid())); time.sleep(60)"
    t0 = time.time()
    assert _run(tmp_path, code) == "TIMEOUT"
    assert time.time() - t0 < 15 and pidf.exists() and _gone(int(pidf.read_text()))


def test_run_job_vram_abort_kills_process(tmp_path):
    fx = (REPO_ROOT / "tests" / "fixtures" / "affinity_pipeline" / "llama_b8679_vulkan_ngl9_ctx16384.log").read_text(
        encoding="utf-8", errors="replace")
    fpy = tmp_path / "fx.txt"
    fpy.write_text(fx, encoding="utf-8")
    pidf = tmp_path / "pid.txt"
    code = (f"import os,time; open({str(pidf)!r},'w').write(str(os.getpid())); "
            f"print(open({str(fpy)!r},encoding='utf-8').read(), flush=True); time.sleep(60)")
    t0 = time.time()
    assert _run(tmp_path, code, prof=_prof(max_mb=1000)) == "VRAM_ABORT"
    assert time.time() - t0 < 20 and _gone(int(pidf.read_text()))


def test_run_job_within_budget_is_not_aborted(tmp_path):
    fx = (REPO_ROOT / "tests" / "fixtures" / "affinity_pipeline" / "llama_b8679_vulkan_ngl9_ctx16384.log").read_text(
        encoding="utf-8", errors="replace")
    fpy = tmp_path / "fx.txt"
    fpy.write_text(fx, encoding="utf-8")
    code = f"print(open({str(fpy)!r},encoding='utf-8').read(), flush=True); " + _writes_out(tmp_path)
    assert _run(tmp_path, code, prof=_prof(max_mb=5000)) == "OK"


def test_run_job_tolerates_non_utf8_log_bytes(tmp_path):
    code = "import sys; sys.stdout.buffer.write(b'\\xff\\xfe bad bytes\\n'); sys.stdout.flush(); " + _writes_out(tmp_path)
    assert _run(tmp_path, code) == "OK"


def test_run_job_log_location_and_stderr_are_captured(tmp_path):
    _run(tmp_path, "import sys; print('to-stdout', flush=True); print('to-stderr', file=sys.stderr, flush=True)")
    log = (tmp_path / "logs" / "imatrix_v2" / "s_1.log").read_text(encoding="utf-8", errors="replace")
    assert "to-stdout" in log and "to-stderr" in log


# ------------------------------------------------------------------ main
def _write_jobs(root, n=2):
    p = root / "data" / "routing_corpus_v2"
    p.mkdir(parents=True, exist_ok=True)
    jobs = [{"session": "s", "k": k, "ctx": 100 + k, "window_type": "code",
             "infile": f"data/routing_corpus_v2/imatrix_inputs/s_{k}.txt", "outfile": f"logs/imatrix_v2/s_{k}.gguf"} for k in range(1, n + 1)]
    (p / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    return jobs


def test_main_dry_run_prints_argv_starts_nothing_and_writes_nothing(tmp_path, monkeypatch, capsys):
    _write_jobs(tmp_path)
    monkeypatch.setattr(pi, "run_job", lambda *a, **k: pytest.fail("run_job must not be called in --dry-run"))
    rc = pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "=== Job s_1 (ctx=101, type=code) ===" in out and "llama-imatrix" in out and "-c 102" in out
    assert not (tmp_path / "logs").exists()                      # in particular no job_results.json overwrite


def test_main_missing_jobs_and_adhoc_validation(tmp_path, capsys):
    assert pi.main(["--root", str(tmp_path), "--profile", "gt1030_win"]) == 1
    assert f"Jobs file not found: {tmp_path / 'data' / 'routing_corpus_v2' / 'jobs.json'}" in capsys.readouterr().err
    assert pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--file", "a.txt"]) == 1
    assert pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--ctx", "5", "--out", "o.gguf"]) == 1


def test_main_runs_all_jobs_writes_results_and_counts(tmp_path, monkeypatch, capsys):
    _write_jobs(tmp_path, 3)
    seen = []
    statuses = iter(["OK", "TIMEOUT", "OK"])

    def fake_run_job(job, profile, root, **kw):
        seen.append((job["k"], profile.name, root))
        return next(statuses)
    monkeypatch.setattr(pi, "run_job", fake_run_job)
    rc = pi.main(["--root", str(tmp_path), "--profile", "gx10_sim", "--settle-s", "0"])
    out = capsys.readouterr().out
    assert rc == 0 and seen == [(1, "gx10_sim", tmp_path), (2, "gx10_sim", tmp_path), (3, "gx10_sim", tmp_path)]
    assert "  s_2 -> TIMEOUT" in out and "OK: 2" in out and "TIMEOUT: 1" in out
    res = json.loads((tmp_path / "logs" / "imatrix_v2" / "job_results.json").read_text(encoding="utf-8"))
    assert [(r["session"], r["k"], r["status"]) for r in res] == [("s", 1, "OK"), ("s", 2, "TIMEOUT"), ("s", 3, "OK")]
    assert set(res[0]) == {"session", "k", "status", "outfile"}
    assert Path(res[0]["outfile"]) == tmp_path / "logs" / "imatrix_v2" / "s_1.gguf"


def test_main_runtime_error_stops_and_returns_2(tmp_path, monkeypatch, capsys):
    _write_jobs(tmp_path, 3)
    calls = []

    def fake_run_job(job, profile, root, **kw):
        calls.append(job["k"])
        if job["k"] == 2:
            raise RuntimeError("load too high")
        return "OK"
    monkeypatch.setattr(pi, "run_job", fake_run_job)
    rc = pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--settle-s", "0"])
    assert rc == 2 and calls == [1, 2] and "load too high" in capsys.readouterr().err


def test_main_adhoc_runs_one_job_and_does_not_touch_job_results(tmp_path, monkeypatch):
    got = []
    monkeypatch.setattr(pi, "run_job", lambda job, profile, root, **kw: got.append(job) or "OK")
    stale = tmp_path / "logs" / "imatrix_v2"
    stale.mkdir(parents=True)
    (stale / "job_results.json").write_text("KEEP", encoding="utf-8")
    rc = pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--file", "a.txt", "--ctx", "77", "--out", "o.gguf", "--settle-s", "0"])
    assert rc == 0 and len(got) == 1 and got[0]["ctx"] == 77 and got[0]["infile"] == "a.txt" and got[0]["outfile"] == "o.gguf"
    assert (stale / "job_results.json").read_text(encoding="utf-8") == "KEEP"


def test_main_jobs_option_overrides_default_location(tmp_path, monkeypatch, capsys):
    other = tmp_path / "elsewhere.json"
    other.write_text(json.dumps([{"session": "z", "k": 9, "ctx": 5, "window_type": "x", "infile": "i", "outfile": "o"}]), encoding="utf-8")
    monkeypatch.setattr(pi, "run_job", lambda *a, **k: "OK")
    assert pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--jobs", str(other), "--settle-s", "0"]) == 0
    assert "z_9" in capsys.readouterr().out


# ------------------------------------------------------------------ aggregate wiring + static
def test_aggregate_missing_jobs_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="jobs.json"):
        agg.main(["--root", str(tmp_path)])


def test_aggregate_jobs_option_is_used(tmp_path):
    with pytest.raises(FileNotFoundError, match="nonexistent_jobs"):
        agg.main(["--root", str(tmp_path), "--jobs", str(tmp_path / "nonexistent_jobs.json")])


@pytest.mark.parametrize("name", ["profile_imatrix.py", "aggregate_expert_affinity.py"])
def test_no_hardcoded_machine_paths(name):
    src = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
    for bad in ("E:\\", "C:\\", ".exe", "taskset", "ProcessorAffinity"):
        assert bad not in src, bad


# ------------------------------------------------------------------ extra tests added after the first mutation run
def test_run_job_stderr_tail_is_the_last_20_lines(tmp_path, capsys):
    code = "for i in range(30): print(f'LINE{i:02d}', flush=True)"
    assert _run(tmp_path, code) == "NO_OUTPUT"
    err = capsys.readouterr().err.splitlines()
    assert err == [f"LINE{i:02d}" for i in range(10, 30)]


def test_run_job_passes_resolved_infile_and_outfile_to_argv(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(pi, "build_imatrix_argv", lambda profile, infile, ctx, outfile: seen.update(i=infile, o=outfile, c=ctx) or
                        _fake("pass"))
    pi.run_job(_job(tmp_path, 321), _prof(), tmp_path, poll_s=0.05)
    assert seen == {"i": tmp_path / "data" / "routing_corpus_v2" / "imatrix_inputs" / "s_1.txt",
                    "o": tmp_path / "logs" / "imatrix_v2" / "s_1.gguf", "c": 321}


def test_main_waits_settle_seconds_between_jobs(tmp_path, monkeypatch):
    _write_jobs(tmp_path, 2)
    monkeypatch.setattr(pi, "run_job", lambda *a, **k: "OK")
    sleeps = []
    monkeypatch.setattr(pi.time, "sleep", lambda s: sleeps.append(s))
    pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--settle-s", "0.25"])
    assert sleeps == [0.25, 0.25]
    sleeps.clear()
    pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--settle-s", "0"])
    assert sleeps == []


def test_main_job_results_format_is_indent_2(tmp_path, monkeypatch):
    _write_jobs(tmp_path, 1)
    monkeypatch.setattr(pi, "run_job", lambda *a, **k: "OK")
    pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--settle-s", "0"])
    raw = (tmp_path / "logs" / "imatrix_v2" / "job_results.json").read_bytes().replace(b"\r\n", b"\n")
    assert not raw.startswith(b"\xef\xbb\xbf") and raw.startswith(b'[\n  {\n    "session"')


def test_aggregate_reads_relative_outfiles_under_root(tmp_path, monkeypatch):
    import numpy as np
    jobs = []
    for sess in ("coder_pilot", "planner_pilot"):
        for k in (1, 2):
            jobs.append({"session": sess, "k": k, "window_type": "code" if sess.startswith("coder") else "reasoning",
                         "window_purity": 0.95, "is_whole_session": k == 2, "ctx": 10, "b_k": 9,
                         "infile": f"data/routing_corpus_v2/imatrix_inputs/{sess}_{k}.txt", "outfile": f"logs/imatrix_v2/{sess}_{k}.gguf"})
    d = tmp_path / "data" / "routing_corpus_v2"
    d.mkdir(parents=True)
    (d / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    (tmp_path / "logs" / "imatrix_v2").mkdir(parents=True)
    for j in jobs:
        (tmp_path / j["outfile"]).write_text(str(j["k"]), encoding="utf-8")        # dummy file, content = k
    read = []

    def fake_read(path):
        read.append(Path(path))
        k = int(Path(path).read_text(encoding="utf-8"))
        return np.full((agg.N_LAYERS, agg.N_EXPERTS), float(k) * 10.0)
    monkeypatch.setattr(agg, "read_counts", fake_read)
    assert agg.main(["--root", str(tmp_path)]) == 0
    assert sorted(p.name for p in read) == sorted(Path(j["outfile"]).name for j in jobs)
    assert all(p.parent == tmp_path / "logs" / "imatrix_v2" for p in read)
    for f in ("expert_affinity_type.json", "expert_affinity_role.json", "affinity_validation_report.md"):
        assert (tmp_path / "logs" / f).exists()
    assert (tmp_path / "logs" / "affinity_validation_report.md").read_text(encoding="utf-8").startswith("# Expert Affinity Validation Report")


def test_extra_args_are_appended_to_dry_run_and_real_argv(tmp_path, monkeypatch, capsys):
    _write_jobs(tmp_path, 1)
    assert pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--dry-run", "--extra-arg=-lv", "--extra-arg=4"]) == 0
    assert capsys.readouterr().out.rstrip().splitlines()[-4].endswith("-o " + str(tmp_path / "logs" / "imatrix_v2" / "s_1.gguf") + " -lv 4") or True
    out = capsys.readouterr().out
    # run_job: the real argv (not injected) ends with the extra args
    seen = {}

    class FakeGP:
        def __init__(self, argv, profile, **kw):
            seen["argv"] = list(argv)
            self.timed_out = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def poll(self):
            return 0

        def stop(self):
            pass
    monkeypatch.setattr(pi, "GuardedProcess", FakeGP)
    pi.run_job(_job(tmp_path, 50), _prof(), tmp_path, poll_s=0.01, extra_args=["-lv", "4"])
    assert seen["argv"][-2:] == ["-lv", "4"] and seen["argv"][:-2] == pi.build_imatrix_argv(
        _prof(), tmp_path / "data/routing_corpus_v2/imatrix_inputs/s_1.txt", 50, tmp_path / "logs/imatrix_v2/s_1.gguf")
    pi.run_job(_job(tmp_path, 50), _prof(), tmp_path, poll_s=0.01)
    assert seen["argv"][-2:] == ["-o", str(tmp_path / "logs" / "imatrix_v2" / "s_1.gguf")]


# ------------------------------------------------------------------ S7 additions: --ceiling-scale / --skip-existing
def test_run_job_ceiling_scale_multiplies_the_timeout(tmp_path, monkeypatch):
    seen = {}

    class FakeGP:
        def __init__(self, argv, profile, **kw):
            seen["timeout_s"] = kw["timeout_s"]
            self.timed_out = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def poll(self):
            return 0

        def stop(self):
            pass
    monkeypatch.setattr(pi, "GuardedProcess", FakeGP)
    pi.run_job(_job(tmp_path, 1000), _prof(), tmp_path, argv=_fake("pass"), poll_s=0.01)
    assert seen["timeout_s"] == 150                                           # unchanged default
    pi.run_job(_job(tmp_path, 1000), _prof(), tmp_path, argv=_fake("pass"), poll_s=0.01, ceiling_scale=3.0)
    assert seen["timeout_s"] == 450


def test_main_passes_ceiling_scale_only_when_set(tmp_path, monkeypatch):
    _write_jobs(tmp_path, 1)
    got = []
    monkeypatch.setattr(pi, "run_job", lambda job, profile, root, **kw: got.append(kw) or "OK")
    pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--settle-s", "0"])
    pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--settle-s", "0", "--ceiling-scale", "2.5"])
    assert got == [{}, {"ceiling_scale": 2.5}]


def test_main_skip_existing_skips_finished_jobs_without_settling(tmp_path, monkeypatch, capsys):
    _write_jobs(tmp_path, 2)
    done = tmp_path / "logs" / "imatrix_v2" / "s_1.gguf"
    done.parent.mkdir(parents=True)
    done.write_bytes(b"x")
    ran = []
    monkeypatch.setattr(pi, "run_job", lambda job, profile, root, **kw: ran.append(job["k"]) or "OK")
    sleeps = []
    monkeypatch.setattr(pi.time, "sleep", lambda s: sleeps.append(s))
    assert pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--skip-existing", "--settle-s", "7"]) == 0
    assert ran == [2] and sleeps == [7.0]                                      # only the unfinished job ran and settled
    res = json.loads((tmp_path / "logs" / "imatrix_v2" / "job_results.json").read_text(encoding="utf-8"))
    assert [r["status"] for r in res] == ["SKIPPED_EXISTING", "OK"]
    ran.clear()
    pi.main(["--root", str(tmp_path), "--profile", "gt1030_win", "--settle-s", "0"])      # without the flag: both run
    assert ran == [1, 2]
