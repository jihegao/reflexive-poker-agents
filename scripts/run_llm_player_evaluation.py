from __future__ import annotations

import argparse
from pathlib import Path

from reflexive_poker.llm_evaluation import LLMEvaluationConfig, run_llm_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLMPlayer and save decision/reflection traces")
    parser.add_argument("--provider", choices=("mock", "openai", "opencode-go", "codex"), default="mock")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--hands", type=int, default=24)
    parser.add_argument("--seed-start", type=int, default=8100)
    parser.add_argument("--seed-count", type=int, default=6)
    parser.add_argument("--equity-samples", type=int, default=2)
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["tag", "calling_station", "closed_loop_shaper"],
    )
    parser.add_argument("--output", type=Path, default=Path("results/llm_player/mock_evaluation"))
    args = parser.parse_args()
    result = run_llm_evaluation(
        LLMEvaluationConfig(
            provider=args.provider,
            model=args.model,
            opponents=tuple(args.opponents),
            seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
            hands_per_mirror=args.hands,
            equity_samples=args.equity_samples,
            output_dir=args.output,
        )
    )
    print(result["summary"].to_string(index=False))
    print(f"\nSaved traces to {args.output.resolve()}")


if __name__ == "__main__":
    main()
