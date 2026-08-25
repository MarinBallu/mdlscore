# mdlscore

Score a file's description length under a code LM, conditioned on context:

    Σ −log p(token | context, preceding tokens)

This is a proxy for Kolmogorov complexity: how many bits it takes a model to
describe a file, *given what already exists* around it. Lower means more
predictable/compressible relative to the context — the file reuses patterns
already established in the repo rather than introducing new ones. It's meant
as a signal an agent can check before/after an edit: does this change reduce
the file's description length given the rest of the codebase, or does it add
entropy?

It is not line count, character count, or a cyclomatic-complexity score —
those don't know what's already been said elsewhere in the repo.

## Install

```
pip install -e .
```

Requires `torch` and `transformers` (the `local-lm` backend runs a real,
open-weight causal LM locally — no API key, but the first run downloads
model weights).

## Usage

```
mdlscore path/to/file.py --context src/ shared/utils.py
```

```
file: path/to/file.py
tokens: 214
bits: 1305.42
bits/token: 6.10
```

- `--context PATH [PATH ...]` — files/directories to condition on (walked
  recursively; binaries and noise directories like `.git`, `node_modules`,
  `__pycache__` are skipped). Not scored themselves, only used as left
  context for the model.
- `--model NAME` — override the default model for the `local-lm` backend
  (any causal LM on the Hugging Face Hub).
- `--unit {bits,nats}` — default `bits`.
- `--json` — machine-readable output, for driving an agent loop.

## As a library

```python
from mdlscore import score_file

result = score_file("path/to/file.py", context_paths=["src/"])
result.total_bits
result.bits_per_token
```

## Backends

`local-lm` (`mdlscore/backends/local_lm.py`) is the only backend today: it
computes real teacher-forced cross-entropy under a local Hugging Face causal
LM, with sliding-window context truncation for files/context that don't fit
the model's window in one pass.

Scoring is defined by the `Scorer` interface in `mdlscore/scorer.py` —
`score(target_text, context_text) -> ScoreResult` — so an alternative backend
(a compression-based proxy, an API-based scorer) is a drop-in.

## Development

```
pip install -e ".[dev]"
pytest
```

`tests/test_local_lm_offline.py` runs the real `transformers`/`torch`
plumbing against a tiny, randomly-initialized GPT-2 built entirely in
memory (only the network boundary, `from_pretrained`, is faked), so it
exercises the real API on every run without needing network access.

`tests/test_local_lm_integration.py` additionally downloads a real
checkpoint from the Hub and skips automatically if there's no network
access — CI (`.github/workflows/tests.yml`) has internet, so it runs there
on every push/PR.
