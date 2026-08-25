"""Gather repo context text from arbitrary files/directories."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
    ".idea", ".vscode", ".egg-info",
}


def _iter_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for child in sorted(path.iterdir()):
        if child.is_dir():
            if child.name in _SKIP_DIRS or child.name.startswith("."):
                continue
            yield from _iter_files(child)
        elif child.is_file():
            yield child


def gather_context(paths: Iterable[str | Path], root: Path | None = None) -> str:
    """Concatenate every file under `paths` (files or directories) into one context string.

    Each file is preceded by a path header so the model sees filenames as part of
    the signal. Files that aren't valid UTF-8 (binaries, etc.) are skipped, since
    they carry no text signal a code LM can condition on. Paths are deduplicated
    if they overlap.
    """
    root = root or Path.cwd()
    parts = []
    seen = set()
    for raw in paths:
        for f in _iter_files(Path(raw)):
            resolved = f.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            try:
                label = resolved.relative_to(root.resolve())
            except ValueError:
                label = f
            parts.append(f"# --- {label} ---\n{text}")
    return "\n\n".join(parts)
