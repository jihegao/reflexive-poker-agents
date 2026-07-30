from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("results/llm_player/mock_evaluation")
OUTPUT_DIR = Path("results/llm_player/analysis")


def read_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(INPUT_DIR / "matches.csv")
    decisions = read_jsonl(INPUT_DIR / "decision_traces.jsonl.gz")
    reflections = read_jsonl(INPUT_DIR / "reflection_traces.jsonl.gz")

    paired = (
        matches.groupby(["opponent_type", "seed"], as_index=False)
        .agg(llm_chips_per_100=("llm_chips_per_100", "mean"))
    )
    rows = []
    for opponent, group in paired.groupby("opponent_type"):
        d = [item for item in decisions if item["opponent_type"] == opponent]
        r = [item for item in reflections if item["opponent_type"] == opponent]
        legal = [
            item["final_action"] in (item.get("state") or {}).get("legal_actions", [])
            for item in d
        ]
        rows.append(
            {
                "opponent_type": opponent,
                "paired_seeds": len(group),
                "mean_chips_per_100": group["llm_chips_per_100"].mean(),
                "positive_seed_rate": (group["llm_chips_per_100"] > 0).mean(),
                "decisions": len(d),
                "reflections": len(r),
                "legal_action_rate": sum(legal) / max(len(legal), 1),
                "fallback_rate": sum(bool(item.get("fallback")) for item in d) / max(len(d), 1),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    report = [
        "# LLMPlayer integration report",
        "",
        "This frozen evaluation uses the deterministic mock provider, not a live LLM.",
        "It validates the provider contract, legal-action checks, trace logging, and post-hand reflection pipeline.",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Decision traces: {len(decisions):,}",
        f"Reflection traces: {len(reflections):,}",
    ]
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
