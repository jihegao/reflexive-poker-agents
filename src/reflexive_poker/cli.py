from __future__ import annotations

import argparse
from pathlib import Path

from .simulation import CONFIRMATORY_CONDITIONS, IMAGE_SHAPING_CONDITIONS, run_study, summarize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reflexive poker agent simulations.")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run a small confirmatory-style demo.")
    demo.add_argument("--output", default="results/demo")
    demo.add_argument("--hands", type=int, default=80)
    demo.add_argument("--seeds", type=int, default=2)

    shaping = sub.add_parser("image-shaping", help="Run the strategic public-image shaping study.")
    shaping.add_argument("--output", default="results/image_shaping/static")
    shaping.add_argument("--hands", type=int, default=320)
    shaping.add_argument("--seeds", type=int, default=20)
    shaping.add_argument("--hidden-shift", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        seeds = range(11, 11 + args.seeds)
        data = run_study(
            CONFIRMATORY_CONDITIONS,
            seeds=seeds,
            hands=args.hands,
            hidden_shift=True,
            output=args.output,
        )
    elif args.command == "image-shaping":
        seeds = range(1000, 1000 + args.seeds)
        data = run_study(
            IMAGE_SHAPING_CONDITIONS,
            seeds=seeds,
            hands=args.hands,
            hidden_shift=args.hidden_shift,
            output=args.output,
        )
    else:  # pragma: no cover
        raise SystemExit(f"unknown command: {args.command}")

    summary = summarize(data)
    print(summary.to_string(index=False))
    print(f"\nWrote results to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
