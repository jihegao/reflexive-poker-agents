from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexive_poker.regime_adaptation import (
    RegimeExperimentConfig,
    paired_regime_effects,
    run_regime_switch_experiment,
    summarize_paired_regime_effects,
    summarize_regime_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the calibrated full-hand regime-switch adaptation experiment"
    )
    parser.add_argument("--seed-start", type=int, default=9300)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--hands", type=int, default=320)
    parser.add_argument("--switch-hand", type=int, default=160)
    parser.add_argument("--equity-samples", type=int, default=4)
    parser.add_argument("--recovery-window", type=int, default=32)
    parser.add_argument("--simulation-rollout-hands", type=int, default=36)
    parser.add_argument("--simulation-equity-samples", type=int, default=1)
    parser.add_argument("--formation-observations", type=int, default=48)
    parser.add_argument("--calibration-observations", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/regime_adaptation/smoke"),
    )
    args = parser.parse_args()
    rows = run_regime_switch_experiment(
        RegimeExperimentConfig(
            seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
            hands=args.hands,
            switch_hand=args.switch_hand,
            equity_samples=args.equity_samples,
            recovery_window=args.recovery_window,
            simulation_rollout_hands=args.simulation_rollout_hands,
            simulation_equity_samples=args.simulation_equity_samples,
            formation_observations=args.formation_observations,
            calibration_observations=args.calibration_observations,
            output_dir=args.output,
        )
    )
    payload = {
        "conditions": summarize_regime_experiment(rows),
        "paired_effect": summarize_paired_regime_effects(
            paired_regime_effects(rows)
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
