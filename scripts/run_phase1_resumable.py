from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reflexive_poker.phase1_models import ReasoningTreatment
from reflexive_poker.phase1_resumable import (
    FullSimulationRunConfig,
    LLMConfirmationRunConfig,
    run_full_simulation_matrix,
    run_llm_confirmation_resumable,
)


def _simulation(args: argparse.Namespace) -> None:
    result = run_full_simulation_matrix(
        FullSimulationRunConfig(
            output_dir=args.output,
            seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
            horizon=args.hands,
            formation_hands=args.formation_hands,
            equity_samples=args.equity_samples,
            mccfr_iterations=args.mccfr_iterations,
            max_cells=args.max_cells,
            max_seed_blocks=args.max_seed_blocks,
            allow_dirty_worktree=args.allow_dirty_worktree,
        )
    )
    print(result.to_string(index=False))
    print(f"\nResumable simulation state: {args.output.resolve()}")


def _llm(args: argparse.Namespace) -> None:
    models = tuple(tuple(value.split(":", 1)) for value in args.models)
    result = run_llm_confirmation_resumable(
        LLMConfirmationRunConfig(
            output_dir=args.output,
            selected_depth=ReasoningTreatment(args.selected_depth),
            models=models,
            seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
            horizon=args.hands,
            formation_hands=args.formation_hands,
            equity_samples=args.equity_samples,
            minimum_primary_calls_to_start_block=args.minimum_calls,
            max_blocks=args.max_blocks,
            allow_dirty_worktree=args.allow_dirty_worktree,
        )
    )
    print(result.to_string(index=False))
    print(f"\nResumable LLM state: {args.output.resolve()}")


def _status(args: argparse.Namespace) -> None:
    candidates = (
        args.output / "FULL_SIMULATION_STATUS.csv",
        args.output / "LLM_CONFIRMATION_STATUS.csv",
    )
    found = False
    for path in candidates:
        if path.exists():
            found = True
            print(f"# {path.name}\n")
            print(pd.read_csv(path).to_string(index=False))
            print()
    live_json = [args.output / "RUN_STATUS.json"]
    live_json.extend(args.output.glob("models/*/MODEL_BUDGET_STATUS.json"))
    for path in live_json:
        if path.exists():
            found = True
            print(f"# {path.relative_to(args.output)}\n")
            print(path.read_text(encoding="utf-8"))
            print()
    if not found:
        raise SystemExit(f"No resumable status artifacts found under {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or inspect resumable Phase 1 experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulation = subparsers.add_parser("simulation", help="Run/resume the 40-cell rule matrix")
    simulation.add_argument("--output", type=Path, default=Path("results/phase1/full_simulation"))
    simulation.add_argument("--seed-start", type=int, default=9400)
    simulation.add_argument("--seed-count", type=int, default=60)
    simulation.add_argument("--hands", type=int, default=400)
    simulation.add_argument("--formation-hands", type=int, default=100)
    simulation.add_argument("--equity-samples", type=int, default=8)
    simulation.add_argument("--mccfr-iterations", type=int, default=20_000)
    simulation.add_argument("--max-cells", type=int)
    simulation.add_argument("--max-seed-blocks", type=int)
    simulation.add_argument("--allow-dirty-worktree", action="store_true")
    simulation.set_defaults(handler=_simulation)

    llm = subparsers.add_parser("llm", help="Run/resume the dual-model confirmation plan")
    llm.add_argument("--output", type=Path, default=Path("results/phase1/llm_confirmation"))
    llm.add_argument(
        "--models",
        nargs="+",
        default=["opencode-go:deepseek-v4-flash", "codex:gpt-5.6-luna"],
    )
    llm.add_argument(
        "--selected-depth",
        choices=("action_prediction", "recursive_d2", "recursive_d3"),
        default="recursive_d2",
    )
    llm.add_argument("--seed-start", type=int, default=9700)
    llm.add_argument("--seed-count", type=int, default=200)
    llm.add_argument("--hands", type=int, default=20)
    llm.add_argument("--formation-hands", type=int, default=5)
    llm.add_argument("--equity-samples", type=int, default=8)
    llm.add_argument("--minimum-calls", type=int, default=20)
    llm.add_argument("--max-blocks", type=int)
    llm.add_argument("--allow-dirty-worktree", action="store_true")
    llm.set_defaults(handler=_llm)

    status = subparsers.add_parser("status", help="Read current progress without running work")
    status.add_argument("--output", type=Path, required=True)
    status.set_defaults(handler=_status)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
