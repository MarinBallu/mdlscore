from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import score_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdlscore",
        description=(
            "Score a file's description length under a code LM, conditioned on "
            "context: Sum of -log p(token | context). A Kolmogorov-complexity "
            "proxy for how much a file adds, given what already exists."
        ),
    )
    parser.add_argument("file", help="path to the file to score")
    parser.add_argument(
        "--context",
        nargs="*",
        default=(),
        metavar="PATH",
        help="files/directories to condition on (not scored themselves)",
    )
    parser.add_argument(
        "--backend", default="local-lm", help="scoring backend (default: local-lm)"
    )
    parser.add_argument(
        "--model", default=None, help="model name for the local-lm backend"
    )
    parser.add_argument(
        "--unit",
        choices=("bits", "nats"),
        default="bits",
        help="unit for the total/per-token score (default: bits)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not Path(args.file).is_file():
        print(f"mdlscore: {args.file}: no such file", file=sys.stderr)
        return 1

    backend_kwargs = {"model_name": args.model} if args.model else {}
    try:
        result = score_file(
            args.file, args.context, backend=args.backend, **backend_kwargs
        )
    except ValueError as exc:
        print(f"mdlscore: {exc}", file=sys.stderr)
        return 1

    total = result.total_bits if args.unit == "bits" else result.total_nats
    per_token = result.bits_per_token if args.unit == "bits" else result.nats_per_token

    if args.json:
        print(json.dumps({
            "file": args.file,
            "backend": args.backend,
            "unit": args.unit,
            "tokens": result.num_tokens,
            "total": total,
            "per_token": per_token,
        }))
    else:
        print(f"file: {args.file}")
        print(f"tokens: {result.num_tokens}")
        print(f"{args.unit}: {total:.2f}")
        print(f"{args.unit}/token: {per_token:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
