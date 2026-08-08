from __future__ import annotations

import argparse
from pathlib import Path

from reflexive_poker.three_round_experiment import ThreeRoundConfig, run_three_round_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run/resume the three-round DeepSeek-vs-Luna tournament"
    )
    parser.add_argument("--seed-start", type=int, default=9950)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--hands", type=int, default=1)
    parser.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--round3-lineup-count", type=int, default=1)
    parser.add_argument("--gto-iterations", type=int, default=2000)
    parser.add_argument("--equity-samples", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("results/three_round/pilot"))
    args = parser.parse_args()
    result = run_three_round_experiment(
        ThreeRoundConfig(
            seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
            hands=args.hands,
            rounds=tuple(args.rounds),
            round3_lineup_count=args.round3_lineup_count,
            gto_iterations=args.gto_iterations,
            equity_samples=args.equity_samples,
            output_dir=args.output,
        )
    )
    print(result["match_summary"].to_string(index=False))
    print(f"\nProvider gate valid: {result['provider_gate']['valid']}")
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
