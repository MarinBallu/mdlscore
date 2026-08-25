import json
from pathlib import Path

import pytest

from mdlscore import cli
from mdlscore.scorer import ScoreResult


@pytest.fixture
def fake_score_file(monkeypatch):
    """Stands in for mdlscore.score_file, recording how the CLI called it."""
    calls = []

    def fake(target_path, context_paths=(), backend="local-lm", **backend_kwargs):
        calls.append({
            "target_path": target_path,
            "context_paths": list(context_paths),
            "backend": backend,
            **backend_kwargs,
        })
        text = Path(target_path).read_text()
        return ScoreResult(num_tokens=len(text), total_nats=float(len(text)))

    monkeypatch.setattr(cli, "score_file", fake)
    return calls


def test_missing_file_returns_error(capsys):
    code = cli.main(["/no/such/file.py"])

    assert code == 1
    assert "no such file" in capsys.readouterr().err


def test_human_readable_output(tmp_path: Path, capsys, fake_score_file):
    f = tmp_path / "a.py"
    f.write_text("abcd")

    code = cli.main([str(f)])

    out = capsys.readouterr().out
    assert code == 0
    assert "tokens: 4" in out
    assert "bits:" in out


def test_json_output(tmp_path: Path, capsys, fake_score_file):
    f = tmp_path / "a.py"
    f.write_text("abcd")

    code = cli.main([str(f), "--json", "--unit", "nats"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["tokens"] == 4
    assert payload["total"] == pytest.approx(4.0)
    assert payload["unit"] == "nats"


def test_context_flag_is_passed_through(tmp_path: Path, fake_score_file):
    f = tmp_path / "a.py"
    f.write_text("ab")
    ctx = tmp_path / "ctx.py"
    ctx.write_text("some context")

    cli.main([str(f), "--context", str(ctx)])

    assert fake_score_file[0]["context_paths"] == [str(ctx)]


def test_model_flag_is_passed_through(tmp_path: Path, fake_score_file):
    f = tmp_path / "a.py"
    f.write_text("ab")

    cli.main([str(f), "--model", "some/model"])

    assert fake_score_file[0]["model_name"] == "some/model"


def test_unknown_backend_returns_error(tmp_path: Path, capsys, monkeypatch):
    def fake(*args, **kwargs):
        raise ValueError("unknown backend 'nope'; available: local-lm")

    monkeypatch.setattr(cli, "score_file", fake)
    f = tmp_path / "a.py"
    f.write_text("ab")

    code = cli.main([str(f), "--backend", "nope"])

    assert code == 1
    assert "unknown backend" in capsys.readouterr().err
