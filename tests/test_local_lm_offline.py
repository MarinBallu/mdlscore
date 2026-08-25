"""Integration test against a real, tiny, randomly-initialized GPT-2 built
entirely in memory, so it needs no network access. Unlike
test_local_lm_integration.py (which downloads a real checkpoint and skips
when offline), this exercises the real transformers/torch plumbing --
tokenization, model forward, logits shape -- on every run, everywhere.

Only the network boundary (`from_pretrained`) is faked; the tokenizer and
model handed back are genuine `transformers`/`torch` objects, so this catches
what the hand-rolled fakes in test_scorer_math.py can't: API drift between
mdlscore and the real HF classes.
"""
from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from mdlscore.backends import local_lm
from mdlscore.backends.local_lm import LocalLMScorer

CORPUS = [
    "def add(a, b):",
    "    return a + b",
    "def subtract(a, b):",
    "    return a - b",
    "class Calculator:",
    "    def __init__(self):",
    "        self.value = 0",
]


def _build_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=256, special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(CORPUS, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")


def _build_model(vocab_size: int) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=None,
        eos_token_id=None,
    )
    return GPT2LMHeadModel(config)


@pytest.fixture
def scorer(monkeypatch):
    tokenizer = _build_tokenizer()
    model = _build_model(len(tokenizer))

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            return tokenizer

    class _StubAutoModel:
        @staticmethod
        def from_pretrained(name):
            return model

    monkeypatch.setattr(local_lm, "AutoTokenizer", _StubAutoTokenizer)
    monkeypatch.setattr(local_lm, "AutoModelForCausalLM", _StubAutoModel)

    return LocalLMScorer(model_name="offline-tiny-gpt2", max_context_tokens=32)


def test_init_loads_a_real_model_and_tokenizer(scorer):
    assert isinstance(scorer.model, GPT2LMHeadModel)
    assert isinstance(scorer.tokenizer, PreTrainedTokenizerFast)


def test_scores_real_code_through_the_real_forward_pass(scorer):
    result = scorer.score(
        target_text="def multiply(a, b):\n    return a * b",
        context_text="def add(a, b):\n    return a + b",
    )

    assert result.num_tokens > 0
    assert result.total_nats > 0
    assert result.total_bits == pytest.approx(result.total_nats / 0.6931471805599453)


def test_sliding_window_engages_for_a_target_longer_than_the_window(scorer):
    long_target = "\n".join(CORPUS * 5)  # forces multiple windows at max_context_tokens=32

    result = scorer.score(target_text=long_target, context_text="")

    assert result.num_tokens > 0
    assert result.total_nats > 0
