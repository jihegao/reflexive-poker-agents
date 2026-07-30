from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reflexive_poker.type_matchup_experiment import _run_ecology_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--hands", type=int, default=300)
    parser.add_argument("--equity-samples", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/type_matchups/ecology_chunks"))
    args = parser.parse_args()
    rows = []
    for offset in range(args.chunk, args.seeds, args.chunks):
        rows.extend(_run_ecology_seed(7000 + offset, args.hands, args.equity_samples))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"ecology_chunk_{args.chunk}.csv"
    pd.DataFrame(rows).sort_values(["seed", "player_type"]).to_csv(path, index=False)
    print(path, flush=True)


if __name__ == "__main__":
    main()
