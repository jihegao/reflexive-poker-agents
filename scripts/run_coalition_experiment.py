from __future__ import annotations

import argparse
from pathlib import Path

from reflexive_poker.coalition_experiment import CoalitionConfig, run_coalition_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paired coalition mechanism pilot")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--hands", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("results/coalition/mock_smoke"))
    args = parser.parse_args()
    seeds = tuple(args.seeds or (9400,))
    result = run_coalition_experiment(
        CoalitionConfig(seeds=seeds, hands=args.hands, output_dir=args.output)
    )
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
