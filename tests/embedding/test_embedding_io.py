"""embedding_io: truncation, vector-count check and retry logic without any model file."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affinity_pipeline import embedding_io as eio  # noqa: E402


def test_truncate_budget_and_separator_removal():
    long = "word " * 2000
    cut = eio.truncate(long, 100)
    assert eio.estimated_tokens(cut) <= 100 and len(cut) > 50
    assert eio.SEP not in eio.truncate(f"a{eio.SEP}b", 100)
    assert eio.truncate("", 10) == " "
    zh = "中" * 1000
    assert eio.estimated_tokens(eio.truncate(zh, 100)) <= 100


class FakeRun:
    """pretends to be llama-embedding: one vector per separator-delimited piece; aborts on pieces longer than `limit` chars"""

    def __init__(self, limit=10**9, drop=False):
        self.limit, self.drop, self.calls = limit, drop, 0

    def __call__(self, cmd, **kw):
        self.calls += 1
        text = Path(cmd[cmd.index("-f") + 1]).read_text(encoding="utf-8")
        pieces = text.split(eio.SEP)
        if any(len(p) > self.limit for p in pieces):
            return subprocess.CompletedProcess(cmd, 134, "", "abort")
        if self.drop:
            pieces = pieces[:-1]
        rows = [{"index": i, "embedding": [float(len(p)), 1.0, 0.0]} for i, p in enumerate(pieces)]
        return subprocess.CompletedProcess(cmd, 0, "log line\n" + json.dumps({"object": "list", "data": rows}), "")


def test_multi_line_text_gives_exactly_one_vector(monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    out = eio.embed_texts(Path("bin"), Path("m.gguf"), ["line one\nline two\nline three", "second"])
    assert out.shape == (2, 3)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert fake.calls == 1                                       # one process for the whole batch


def test_batch_retries_in_halves_then_shortens(monkeypatch):
    fake = FakeRun(limit=60)
    monkeypatch.setattr(subprocess, "run", fake)
    texts = ["x" * 30, "y" * 200, "z" * 20]
    out = eio.embed_texts(Path("bin"), Path("m.gguf"), texts, max_tokens=480)
    assert out.shape == (3, 3)                                    # the 200-char text was shortened until accepted


def test_vector_count_mismatch_is_an_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run", FakeRun(drop=True))
    with pytest.raises(RuntimeError):
        eio._run(Path("bin"), Path("m.gguf"), ["a", "b"], 2, 10)
