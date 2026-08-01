from __future__ import annotations

import argparse
from pathlib import Path

from reflexive_poker.six_max_experiment import SixMaxConfig, run_six_max_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a six-player no-limit reflexive LLM pilot")
    parser.add_argument("--provider", choices=("mock", "opencode-go", "codex"), default="mock")
    parser.add_argument("--model", default="current")
    parser.add_argument("--hands", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=9100)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--equity-samples", type=int, default=64)
    parser.add_argument(
        "--condition", choices=("reflexive_on", "reflexive_off"), default="reflexive_on"
    )
    parser.add_argument("--output", type=Path, default=Path("results/six_max/mock_pilot"))
    args = parser.parse_args()
    result = run_six_max_experiment(
        SixMaxConfig(
            provider=args.provider,
            model=args.model,
            seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
            hands=args.hands,
            equity_samples=args.equity_samples,
            condition=args.condition,
            reflexive_enabled=args.condition == "reflexive_on",
            output_dir=args.output,
        )
    )
    print(result["summary"].to_string(index=False))
    print(f"\nSaved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
