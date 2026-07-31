from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .agents import AgentStyle
from .environment import EnvironmentConfig, HoldemEnvironment
from .llm_player import (
    DeterministicNarrativeProvider,
    LLMPlayer,
    OpenAIResponsesProvider,
)
from .tournament_agents import make_tournament_agent


@dataclass(frozen=True)
class LLMEvaluationConfig:
    provider: str = "mock"
    model: str = "gpt-5-mini"
    opponents: tuple[str, ...] = ("tag", "calling_station", "closed_loop_shaper")
    seeds: tuple[int, ...] = tuple(range(8100, 8106))
    hands_per_mirror: int = 24
    equity_samples: int = 2
    output_dir: Path = Path("results/llm_player/mock_evaluation")


def _provider(kind: str, model: str, seed: int):
    if kind == "mock":
        return DeterministicNarrativeProvider(seed=seed)
    if kind == "openai":
        return OpenAIResponsesProvider(model=model)
    raise ValueError(f"Unknown provider: {kind}")


def _run_mirror(
    *,
    opponent_type: str,
    seed: int,
    swap: bool,
    config: LLMEvaluationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    names = ("seat_0", "seat_1")
    llm_seat = 1 if swap else 0
    opponent_seat = 1 - llm_seat
    provider = _provider(config.provider, config.model, seed * 2 + int(swap))
    llm = LLMPlayer(
        names[llm_seat],
        seed * 1009 + 17,
        provider,
        AgentStyle(
            aggression=0.47,
            risk_margin=0.055,
            belief_sensitivity=0.24,
            social_learning_rate=0.20,
            equity_samples=config.equity_samples,
        ),
        memory_hands=5,
        opponents=(names[opponent_seat],),
    )
    opponent = make_tournament_agent(
        opponent_type,
        names[opponent_seat],
        (names[llm_seat],),
        seed * 1009 + 31,
        equity_samples=config.equity_samples,
    )
    agents = [None, None]
    agents[llm_seat] = llm
    agents[opponent_seat] = opponent
    started = time.perf_counter()
    records = HoldemEnvironment(
        agents,  # type: ignore[arg-type]
        seed=seed,
        config=EnvironmentConfig(regime_switch_hand=config.hands_per_mirror + 1),
    ).play(config.hands_per_mirror)
    elapsed = time.perf_counter() - started
    llm_reward = sum(record.rewards[llm.name] for record in records)
    opponent_reward = sum(record.rewards[opponent.name] for record in records)
    decisions = [
        {
            "opponent_type": opponent_type,
            "seed": seed,
            "swap": int(swap),
            **item,
        }
        for item in llm.llm_decision_log
    ]
    reflections = [
        {
            "opponent_type": opponent_type,
            "seed": seed,
            "swap": int(swap),
            **item,
        }
        for item in llm.llm_reflection_log
    ]
    latency_values = [
        item.get("provider", {}).get("latency_ms")
        for item in llm.llm_decision_log
        if isinstance(item.get("provider"), dict)
        and item.get("provider", {}).get("latency_ms") is not None
    ]
    total_tokens = [
        item.get("provider", {}).get("total_tokens")
        for item in llm.llm_decision_log + llm.llm_reflection_log
        if isinstance(item.get("provider"), dict)
        and item.get("provider", {}).get("total_tokens") is not None
    ]
    row = {
        "provider": config.provider,
        "model": getattr(provider, "model", config.model),
        "opponent_type": opponent_type,
        "seed": seed,
        "swap": int(swap),
        "hands": config.hands_per_mirror,
        "llm_chips_per_100": 100.0 * llm_reward / config.hands_per_mirror,
        "opponent_chips_per_100": 100.0 * opponent_reward / config.hands_per_mirror,
        "decision_count": len(decisions),
        "reflection_count": len(reflections),
        "fallback_count": sum(bool(item.get("fallback")) for item in decisions),
        "provider_failures": llm.provider_failures,
        "invalid_actions": llm.invalid_actions,
        "mean_decision_latency_ms": (
            sum(latency_values) / len(latency_values) if latency_values else float("nan")
        ),
        "total_tokens": sum(int(value) for value in total_tokens) if total_tokens else 0,
        "elapsed_seconds": elapsed,
        "showdown_rate": sum(record.showdown for record in records) / len(records),
    }
    return row, decisions, reflections


def _paired_summary(matches: pd.DataFrame) -> pd.DataFrame:
    mirror = matches.groupby(["opponent_type", "seed"], as_index=False).agg(
        llm_chips_per_100=("llm_chips_per_100", "mean"),
        fallback_count=("fallback_count", "sum"),
        decision_count=("decision_count", "sum"),
        reflection_count=("reflection_count", "sum"),
        invalid_actions=("invalid_actions", "sum"),
        provider_failures=("provider_failures", "sum"),
        mean_decision_latency_ms=("mean_decision_latency_ms", "mean"),
        total_tokens=("total_tokens", "sum"),
    )
    rows = []
    for opponent, group in mirror.groupby("opponent_type"):
        rows.append(
            {
                "opponent_type": opponent,
                "seeds": group["seed"].nunique(),
                "mean_llm_chips_per_100": group["llm_chips_per_100"].mean(),
                "median_llm_chips_per_100": group["llm_chips_per_100"].median(),
                "positive_seed_rate": (group["llm_chips_per_100"] > 0).mean(),
                "decision_count": group["decision_count"].sum(),
                "reflection_count": group["reflection_count"].sum(),
                "fallback_rate": (
                    group["fallback_count"].sum() / max(1, group["decision_count"].sum())
                ),
                "invalid_action_rate": (
                    group["invalid_actions"].sum() / max(1, group["decision_count"].sum())
                ),
                "provider_failure_count": group["provider_failures"].sum(),
                "mean_decision_latency_ms": group["mean_decision_latency_ms"].mean(),
                "total_tokens": group["total_tokens"].sum(),
            }
        )
    return pd.DataFrame(rows)


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    opener = gzip.open if path.suffix == ".gz" else path.open
    with (
        opener(path, "wt", encoding="utf-8")
        if path.suffix == ".gz"
        else opener("w", encoding="utf-8") as handle
    ):
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _example_markdown(decisions: list[dict[str, Any]], reflections: list[dict[str, Any]]) -> str:
    lines = ["# LLMPlayer trace examples", ""]
    for item in decisions[:8]:
        output = item.get("output") or {}
        state = item.get("state") or {}
        lines.extend(
            [
                f"## Hand {item['hand_index']} · {item['street']} · vs {item['opponent_type']}",
                "",
                f"- Cards: `{state.get('hole_cards')}`; board: `{state.get('board')}`",
                f"- Pot / to call: {state.get('pot')} / {state.get('to_call')}",
                f"- Equity / pot odds: {state.get('equity_estimate')} / {state.get('pot_odds')}",
                f"- Action: **{item.get('final_action')}**; confidence: {output.get('confidence')}",
                f"- Situation: {output.get('situation_summary')}",
                f"- Rationale: {output.get('rationale')}",
                f"- Self model: {output.get('self_model')}",
                f"- Opponent model: {output.get('opponent_model')}",
                f"- Risk flags: {output.get('risk_flags')}",
                "",
            ]
        )
    lines.extend(["# Post-hand reflection examples", ""])
    for item in reflections[:5]:
        output = item.get("output") or {}
        lines.extend(
            [
                f"## Hand {item['hand_index']} · vs {item['opponent_type']}",
                "",
                f"- Outcome: {output.get('outcome_summary')}",
                f"- Review: {output.get('decision_review')}",
                f"- Worked: {output.get('what_worked')}",
                f"- Failed: {output.get('what_failed')}",
                f"- Belief updates: {output.get('belief_updates')}",
                f"- Adjustment: {output.get('strategy_adjustment')}",
                f"- Calibration: {output.get('calibration_note')}",
                "",
            ]
        )
    return "\n".join(lines)


def run_llm_evaluation(config: LLMEvaluationConfig) -> dict[str, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    match_rows: list[dict[str, Any]] = []
    decision_traces: list[dict[str, Any]] = []
    reflection_traces: list[dict[str, Any]] = []
    for opponent_type in config.opponents:
        for seed in config.seeds:
            for swap in (False, True):
                row, decisions, reflections = _run_mirror(
                    opponent_type=opponent_type,
                    seed=seed,
                    swap=swap,
                    config=config,
                )
                match_rows.append(row)
                decision_traces.extend(decisions)
                reflection_traces.extend(reflections)

    matches = pd.DataFrame(match_rows).sort_values(["opponent_type", "seed", "swap"])
    summary = _paired_summary(matches)
    matches.to_csv(config.output_dir / "matches.csv", index=False)
    summary.to_csv(config.output_dir / "summary.csv", index=False)
    _write_jsonl(config.output_dir / "decision_traces.jsonl.gz", decision_traces)
    _write_jsonl(config.output_dir / "reflection_traces.jsonl.gz", reflection_traces)
    (config.output_dir / "trace_examples.md").write_text(
        _example_markdown(decision_traces, reflection_traces), encoding="utf-8"
    )
    return {"matches": matches, "summary": summary}
