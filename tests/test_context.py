from pathlib import Path

from mdlscore.context import gather_context


def test_single_file(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("print('hi')\n")

    ctx = gather_context([f], root=tmp_path)

    assert "a.py" in ctx
    assert "print('hi')" in ctx


def test_directory_recurses_and_skips_noise_dirs(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("x = 1\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("should not appear\n")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "mod.cpython-311.pyc").write_text("junk\n")

    ctx = gather_context([tmp_path], root=tmp_path)

    assert "x = 1" in ctx
    assert "should not appear" not in ctx
    assert "junk" not in ctx


def test_binary_file_is_skipped(tmp_path: Path):
    binary = tmp_path / "data.bin"
    binary.write_bytes(bytes(range(256)))
    text_file = tmp_path / "readme.txt"
    text_file.write_text("hello\n")

    ctx = gather_context([tmp_path], root=tmp_path)

    assert "hello" in ctx


def test_overlapping_paths_are_deduplicated(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")

    ctx = gather_context([tmp_path, f], root=tmp_path)

    assert ctx.count("x = 1") == 1


def test_no_paths_yields_empty_string():
    assert gather_context([]) == ""
