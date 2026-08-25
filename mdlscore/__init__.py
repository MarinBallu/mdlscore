"""Score a file's description length under a code LM, conditioned on context.

    Σ −log p(token | context, preceding tokens)

A proxy for Kolmogorov complexity: lower is more compressible/predictable
*given what already exists* in the surrounding repo, which is what an agent
minimizing complexity actually wants — not raw line or character count.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .backends import get_backend
from .context import gather_context
from .scorer import ScoreResult, Scorer

__all__ = ["Scorer", "ScoreResult", "gather_context", "get_backend", "score_file"]


def score_file(
    target_path: str | Path,
    context_paths: Iterable[str | Path] = (),
    backend: str = "local-lm",
    **backend_kwargs,
) -> ScoreResult:
    """Score the file at `target_path`, conditioned on the text under `context_paths`."""
    target_text = Path(target_path).read_text(encoding="utf-8")
    context_text = gather_context(context_paths) if context_paths else ""
    scorer = get_backend(backend)(**backend_kwargs)
    return scorer.score(target_text, context_text)
