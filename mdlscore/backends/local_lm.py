"""Score text with a local, open-weight causal LM (teacher-forced cross-entropy).

This is the baseline backend: no API key, no network at score time beyond the
one-off model download. Any causal LM on the Hugging Face Hub works via
`model_name`; the default is a small open code model.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..scorer import ScoreResult, Scorer

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-0.5B"


def cross_entropy_sum(
    logits: torch.Tensor, input_ids: torch.Tensor, target_start: int
) -> tuple[float, int]:
    """Sum of -log p(token) in nats for input_ids[target_start:].

    `logits[i]` are the model's predictions for `input_ids[i + 1]`, so the very
    first token of the whole sequence is never scored (nothing precedes it to
    condition on) — the standard convention for causal-LM perplexity.
    """
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    next_ids = input_ids[1:]
    token_logp = log_probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
    start = max(target_start - 1, 0)
    target_logp = token_logp[start:]
    return float(-target_logp.sum().item()), int(target_logp.shape[0])


class LocalLMScorer(Scorer):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        max_context_tokens: int | None = None,
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.window = max_context_tokens or getattr(
            self.model.config, "max_position_embeddings", 2048
        )

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text) if text else []

    def _logits(self, input_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self.model(ids).logits[0]

    def _tail(self, ids: list[int]) -> list[int]:
        """The most recent half-window of `ids`, carried forward as left-context."""
        keep = min(len(ids), self.window // 2)
        return ids[len(ids) - keep :] if keep else []

    def score(self, target_text: str, context_text: str = "") -> ScoreResult:
        context_ids = self._encode(context_text)
        target_ids = self._encode(target_text)
        if not target_ids:
            return ScoreResult(num_tokens=0, total_nats=0.0)

        # Left-context carried into each window: the tail of `context_ids`
        # initially, then the tail of the previous window once we're deep
        # enough into `target_ids` that a single window can't hold it all.
        buffer = self._tail(context_ids)

        total_nats = 0.0
        total_tokens = 0
        pos = 0
        while pos < len(target_ids):
            space = max(self.window - len(buffer), 1)
            chunk = target_ids[pos : pos + space]
            input_ids = buffer + chunk
            logits = self._logits(input_ids)
            nats, n = cross_entropy_sum(
                logits, torch.tensor(input_ids), target_start=len(buffer)
            )
            total_nats += nats
            total_tokens += n
            pos += len(chunk)
            buffer = self._tail(input_ids)

        return ScoreResult(num_tokens=total_tokens, total_nats=total_nats)
