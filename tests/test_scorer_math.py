import math

import pytest
import torch

from mdlscore.backends.local_lm import LocalLMScorer, cross_entropy_sum
from mdlscore.scorer import ScoreResult


class TestCrossEntropySum:
    """Uniform logits (all zeros) make every prediction -log(vocab_size), so the
    expected sum is just (num scored tokens) * log(vocab_size) — easy to check
    by hand without needing a trained model."""

    VOCAB = 4

    def _uniform_logits(self, seq_len: int) -> torch.Tensor:
        return torch.zeros(seq_len, self.VOCAB)

    def test_whole_sequence_is_target(self):
        input_ids = torch.tensor([0, 1, 2, 3, 1])
        nats, n = cross_entropy_sum(self._uniform_logits(5), input_ids, target_start=0)

        # first token is never scored (nothing precedes it to condition on)
        assert n == 4
        assert nats == pytest.approx(4 * math.log(self.VOCAB))

    def test_leading_context_is_excluded(self):
        input_ids = torch.tensor([0, 1, 2, 3, 1])
        nats, n = cross_entropy_sum(self._uniform_logits(5), input_ids, target_start=2)

        assert n == 3
        assert nats == pytest.approx(3 * math.log(self.VOCAB))

    def test_single_unconditioned_token_scores_nothing(self):
        input_ids = torch.tensor([0])
        nats, n = cross_entropy_sum(self._uniform_logits(1), input_ids, target_start=0)

        assert n == 0
        assert nats == 0.0


class _FakeLocalLM(LocalLMScorer):
    """A LocalLMScorer that skips loading real weights, so the sliding-window
    chunking logic in `score()` can be tested without network access. Each
    token's logits put all mass on the *next* token id, making the sum-of-nats
    trivially predictable: 0 nats per correctly-scored token."""

    VOCAB = 50

    def __init__(self, window: int):
        self.device = "cpu"
        self.window = window

    def _encode(self, text):
        # one token per character, offset so ids stay in range
        return [ord(c) % self.VOCAB for c in text]

    def _logits(self, input_ids):
        seq_len = len(input_ids)
        logits = torch.full((seq_len, self.VOCAB), -1e4)
        for i in range(seq_len - 1):
            logits[i, input_ids[i + 1]] = 1e4  # near-certain correct prediction
        return logits


class TestLocalLMScorerWindowing:
    def test_fits_in_one_window(self):
        scorer = _FakeLocalLM(window=64)
        result = scorer.score(target_text="abcdef", context_text="context")

        assert isinstance(result, ScoreResult)
        assert result.num_tokens == 6
        assert result.total_nats == pytest.approx(0.0, abs=1e-2)

    def test_empty_target_scores_nothing(self):
        scorer = _FakeLocalLM(window=64)
        result = scorer.score(target_text="", context_text="context")

        assert result.num_tokens == 0
        assert result.total_nats == 0.0

    def test_target_larger_than_window_slides(self):
        # window=4 forces multiple chunks for an 11-token target
        scorer = _FakeLocalLM(window=4)
        target = "abcdefghijk"

        result = scorer.score(target_text=target, context_text="")

        assert result.num_tokens == len(target) - 1  # first token of all is unconditioned
        assert result.total_nats == pytest.approx(0.0, abs=1e-2)

    def test_matches_single_window_result_when_context_is_irrelevant(self):
        # with near-certain logits, splitting into windows shouldn't change the
        # total score as long as each chunk's own predictions are still exact
        small_window = _FakeLocalLM(window=3).score("abcdefgh", "")
        big_window = _FakeLocalLM(window=64).score("abcdefgh", "")

        assert small_window.num_tokens == big_window.num_tokens
        assert small_window.total_nats == pytest.approx(big_window.total_nats, abs=1e-2)
