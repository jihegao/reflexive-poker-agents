from __future__ import annotations

import argparse
from pathlib import Path

from reflexive_poker.type_matchup_experiment import TypeMatchupConfig, run_type_matchups


def main() -> None:
    parser = argparse.ArgumentParser(description="Run player-type matchup and ecology experiments")
    parser.add_argument("--output", type=Path, default=Path("results/type_matchups"))
    parser.add_argument("--pairwise-hands", type=int, default=200)
    parser.add_argument("--pairwise-seeds", type=int, default=24)
    parser.add_argument("--ecology-hands", type=int, default=300)
    parser.add_argument("--ecology-seeds", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--equity-samples", type=int, default=2)
    args = parser.parse_args()
    config = TypeMatchupConfig(
        pairwise_hands=args.pairwise_hands,
        pairwise_seeds=tuple(range(5000, 5000 + args.pairwise_seeds)),
        ecology_hands=args.ecology_hands,
        ecology_seeds=tuple(range(7000, 7000 + args.ecology_seeds)),
        equity_samples=args.equity_samples,
        workers=args.workers,
        output_dir=args.output,
    )
    frames = run_type_matchups(config)
    print(f"pairwise rows: {len(frames['pairwise'])}")
    print(f"ecology rows: {len(frames['ecology'])}")
    print(f"saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
