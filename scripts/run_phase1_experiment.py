from __future__ import annotations

import argparse
from pathlib import Path

from reflexive_poker.phase1_experiment import (
    DEFAULT_TREATMENTS,
    DEPTH_TREATMENTS,
    Phase1ExperimentConfig,
    run_phase1_experiment,
    run_phase1_matrix_smoke,
    write_llm_confirmation_plan,
)
from reflexive_poker.phase1_models import (
    Arena,
    OpponentComposition,
    ProviderBudget,
    ReasoningTreatment,
    Stability,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Phase 1 opponent-modeling experiment"
    )
    parser.add_argument("--arena", choices=[item.value for item in Arena], default="heads_up")
    parser.add_argument(
        "--treatments",
        nargs="+",
        choices=[item.value for item in ReasoningTreatment],
    )
    parser.add_argument("--opponent", default="tag")
    parser.add_argument(
        "--composition",
        choices=[item.value for item in OpponentComposition],
        default=OpponentComposition.HETEROGENEOUS_CLASSIC.value,
    )
    parser.add_argument(
        "--stability", choices=[item.value for item in Stability], default="fixed"
    )
    parser.add_argument("--epsilon", type=float, choices=(0.05, 0.35), default=0.05)
    parser.add_argument("--seed-start", type=int, default=9400)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--hands", type=int, default=40)
    parser.add_argument("--formation-hands", type=int, default=10)
    parser.add_argument("--equity-samples", type=int, default=8)
    parser.add_argument(
        "--provider", choices=("rule", "mock", "opencode-go", "codex"), default="rule"
    )
    parser.add_argument("--model", default="none")
    parser.add_argument("--max-calls", type=int, default=10_000)
    parser.add_argument("--max-retries", type=int, default=400)
    parser.add_argument("--mccfr-iterations", type=int, default=20_000)
    parser.add_argument("--preregistered", action="store_true")
    parser.add_argument("--matrix-smoke", action="store_true")
    parser.add_argument("--write-confirmation-plan", action="store_true")
    parser.add_argument(
        "--selected-depth",
        choices=("action_prediction", "recursive_d2", "recursive_d3"),
        default="recursive_d2",
    )
    parser.add_argument("--output", type=Path, default=Path("results/phase1/smoke"))
    args = parser.parse_args()

    if args.matrix_smoke:
        result = run_phase1_matrix_smoke(args.output, seed=args.seed_start)
        print(result.to_string(index=False))
        print(f"\nSaved matrix smoke to {args.output.resolve()}")
        return
    if args.write_confirmation_plan:
        payload = write_llm_confirmation_plan(
            args.output, ReasoningTreatment(args.selected_depth)
        )
        print(f"Confirmation plan hash: {payload['plan_hash']}")
        print(f"Saved confirmation plan to {args.output.resolve()}")
        return

    arena = Arena(args.arena)
    default_treatments = DEPTH_TREATMENTS if arena is Arena.SIX_MAX else DEFAULT_TREATMENTS
    treatments = (
        tuple(ReasoningTreatment(value) for value in args.treatments)
        if args.treatments
        else default_treatments
    )
    result = run_phase1_experiment(
        Phase1ExperimentConfig(
            arena=arena,
            treatments=treatments,
            opponent_type=args.opponent,
            opponent_composition=OpponentComposition(args.composition),
            stability=Stability(args.stability),
            epsilon=args.epsilon,
            seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
            horizon=args.hands,
            formation_hands=args.formation_hands,
            equity_samples=args.equity_samples,
            provider=args.provider,
            model=args.model,
            provider_budget=ProviderBudget(
                max_calls=args.max_calls,
                max_retries=args.max_retries,
            ),
            mccfr_iterations=args.mccfr_iterations,
            output_dir=args.output,
            preregistered=args.preregistered,
        )
    )
    print(result["inference"].to_string(index=False))
    print(f"\nProvider gate: {result['provider_gate']['valid']}")
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
