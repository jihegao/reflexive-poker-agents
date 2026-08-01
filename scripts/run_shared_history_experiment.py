from __future__ import annotations

import argparse
from pathlib import Path

from reflexive_poker.shared_history_experiment import (
    SharedHistoryConfig,
    run_shared_history_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a shared-formation-history six-max reflexive experiment"
    )
    parser.add_argument("--provider", default="opencode-go")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--seed", type=int, default=9300)
    parser.add_argument("--formation-hands", type=int, default=30)
    parser.add_argument("--exploitation-hands", type=int, default=30)
    parser.add_argument("--equity-samples", type=int, default=64)
    parser.add_argument("--memory-hands", type=int, default=30)
    parser.add_argument("--branch-workers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/shared_history/deepseek_30x30_seed9300"),
    )
    args = parser.parse_args()
    result = run_shared_history_experiment(
        SharedHistoryConfig(
            provider=args.provider,
            model=args.model,
            seed=args.seed,
            formation_hands=args.formation_hands,
            exploitation_hands=args.exploitation_hands,
            equity_samples=args.equity_samples,
            memory_hands=args.memory_hands,
            branch_workers=args.branch_workers,
            output_dir=args.output,
        )
    )
    print(result["summary"].to_string(index=False))
    print("\nCall accounting")
    print(result["calls"].to_string(index=False))
    print(f"\nFork: {result['fork']}")
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
