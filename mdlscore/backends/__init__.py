"""Backend registry. `local-lm` is the only backend today; the `Scorer` contract
in `mdlscore.scorer` is what any additional backend (e.g. a compression-based
proxy, or an API-based scorer) would need to implement."""
from __future__ import annotations

from typing import Callable

from ..scorer import Scorer
from .local_lm import LocalLMScorer

_BACKENDS: dict[str, Callable[..., Scorer]] = {
    "local-lm": LocalLMScorer,
}


def get_backend(name: str) -> Callable[..., Scorer]:
    try:
        return _BACKENDS[name]
    except KeyError:
        available = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"unknown backend {name!r}; available: {available}") from None
