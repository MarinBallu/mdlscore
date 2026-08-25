"""End-to-end check against a real (tiny) model. Skips automatically if the
model can't be downloaded — e.g. no network access — rather than failing CI
in offline environments."""
import pytest

from mdlscore.backends.local_lm import LocalLMScorer

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def scorer():
    try:
        return LocalLMScorer(model_name=TINY_MODEL, max_context_tokens=64)
    except Exception as exc:  # network/hub errors vary by transformers version
        pytest.skip(f"could not load {TINY_MODEL}: {exc}")


def test_scoring_a_real_file_returns_a_positive_score(scorer):
    result = scorer.score(
        target_text="def add(a, b):\n    return a + b\n",
        context_text="# a tiny arithmetic library\n",
    )

    assert result.num_tokens > 0
    assert result.total_nats > 0
    assert result.total_bits == pytest.approx(result.total_nats / 0.6931471805599453)


def test_more_context_changes_the_score(scorer):
    target = "def add(a, b):\n    return a + b\n"

    bare = scorer.score(target_text=target, context_text="")
    with_context = scorer.score(
        target_text=target, context_text="def add(a, b):\n    return a + b\n" * 5
    )

    assert bare.total_nats != with_context.total_nats
