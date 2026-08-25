import json
from pathlib import Path

import pytest

from mdlscore import cli
from mdlscore.scorer import ScoreResult, Scorer


class _FakeScorer(Scorer):
    """Deterministic stand-in for a real model, so the CLI can be tested
    without downloading or running one."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def score(self, target_text: str, context_text: str = "") -> ScoreResult:
        # 1 nat per character of target text, regardless of context
        return ScoreResult(num_tokens=len(target_text), total_nats=float(len(target_text)))


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setattr(cli, "get_backend", lambda name: _FakeScorer)


def test_missing_file_returns_error(capsys):
    code = cli.main(["/no/such/file.py"])

    assert code == 1
    assert "no such file" in capsys.readouterr().err


def test_human_readable_output(tmp_path: Path, capsys):
    f = tmp_path / "a.py"
    f.write_text("abcd")

    code = cli.main([str(f)])

    out = capsys.readouterr().out
    assert code == 0
    assert "tokens: 4" in out
    assert "bits:" in out


def test_json_output(tmp_path: Path, capsys):
    f = tmp_path / "a.py"
    f.write_text("abcd")

    code = cli.main([str(f), "--json", "--unit", "nats"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["tokens"] == 4
    assert payload["total"] == pytest.approx(4.0)
    assert payload["unit"] == "nats"


def test_context_flag_is_passed_through(tmp_path: Path, capsys, monkeypatch):
    f = tmp_path / "a.py"
    f.write_text("ab")
    ctx = tmp_path / "ctx.py"
    ctx.write_text("some context")

    seen = {}
    original_gather = cli.gather_context

    def spy(paths):
        seen["paths"] = list(paths)
        return original_gather(paths)

    monkeypatch.setattr(cli, "gather_context", spy)

    cli.main([str(f), "--context", str(ctx)])

    assert seen["paths"] == [str(ctx)]


def test_unknown_backend_returns_error(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setattr(
        cli,
        "get_backend",
        lambda name: (_ for _ in ()).throw(ValueError(f"unknown backend {name!r}")),
    )
    f = tmp_path / "a.py"
    f.write_text("ab")

    code = cli.main([str(f), "--backend", "nope"])

    assert code == 1
    assert "unknown backend" in capsys.readouterr().err
