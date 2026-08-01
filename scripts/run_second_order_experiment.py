from __future__ import annotations

import argparse
from pathlib import Path

from reflexive_poker.second_order_experiment import SecondOrderConfig, run_second_order_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired six-max second-order LLM ablations")
    parser.add_argument("--seed-start", type=int, default=9200)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--hands", type=int, default=6)
    parser.add_argument("--equity-samples", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["opencode-go:deepseek-v4-flash", "codex:gpt-5.6-luna"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/second_order/low_cost_pilot"),
    )
    args = parser.parse_args()
    model_specs = tuple(tuple(value.split(":", 1)) for value in args.models)
    result = run_second_order_experiment(
        SecondOrderConfig(
            model_specs=model_specs,
            seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
            hands=args.hands,
            equity_samples=args.equity_samples,
            workers=args.workers,
            output_dir=args.output,
        )
    )
    print(result["paired_summary"].to_string(index=False))
    print(f"\nSaved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
