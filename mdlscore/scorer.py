"""The scorer contract: assign a description length to text, given context."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

_LN2 = math.log(2)


@dataclass(frozen=True)
class ScoreResult:
    """Description length of a target text, in nats, conditioned on some context."""

    num_tokens: int
    total_nats: float

    @property
    def total_bits(self) -> float:
        return self.total_nats / _LN2

    @property
    def nats_per_token(self) -> float:
        return self.total_nats / self.num_tokens if self.num_tokens else 0.0

    @property
    def bits_per_token(self) -> float:
        return self.nats_per_token / _LN2


class Scorer(ABC):
    """Assigns -log p(token | context, preceding tokens), summed over `target_text`."""

    @abstractmethod
    def score(self, target_text: str, context_text: str = "") -> ScoreResult:
        """Score `target_text` under this backend's model, conditioned on `context_text`."""
