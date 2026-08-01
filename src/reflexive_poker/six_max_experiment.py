from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .agents import AgentStyle
from .environment import EnvironmentConfig, HoldemEnvironment
from .llm_player import CodexProvider, DeterministicNarrativeProvider, LLMPlayer, OpenCodeGoProvider
from .tournament_agents import make_tournament_agent

SIX_MAX_LINEUP = ("llm", "tag", "lag", "rock", "calling_station", "myopic")


@dataclass(frozen=True)
class SixMaxConfig:
    provider: str = "mock"
    model: str = "current"
    seeds: tuple[int, ...] = (9100,)
    hands: int = 6
    equity_samples: int = 64
    condition: str = "reflexive_on"
    reflexive_enabled: bool = True
    output_dir: Path = Path("results/six_max/mock_pilot")


def _provider(kind: str, model: str, seed: int):
    if kind == "mock":
        return DeterministicNarrativeProvider(seed=seed)
    if kind == "opencode-go":
        return OpenCodeGoProvider(model=model)
    if kind == "codex":
        return CodexProvider(model=model)
    raise ValueError(f"Unknown provider: {kind}")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _design_markdown(config: SixMaxConfig) -> str:
    return "\n".join(
        [
            "# Six-max no-limit LLM experiment",
            "",
            f"- Provider/model: `{config.provider}` / `{config.model}`",
            f"- Paired seeds: `{', '.join(str(seed) for seed in config.seeds)}`",
            f"- Hands per seed: `{config.hands}`",
            f"- Condition: `{config.condition}`",
            f"- Reflexive tools enabled: `{str(config.reflexive_enabled).lower()}`",
            "- Lineup: LLM, TAG, LAG, Rock, Calling Station, myopic control.",
            "- Button rotates every hand; six hands give every seat exactly one button.",
            "- Betting has no artificial raise-count cap; an agent can raise until its stack is exhausted.",
            "- The action interface is bounded: `raise_scale=1.25` represents an all-in raise.",
            "- Main and side pots are awarded from total contributions; stacks reset each hand, so this is a cash-game hand-sampling study, not a tournament-elimination simulation.",
            "- Both LLM conditions receive cards, board, legal actions, multiway equity and pot odds. Only reflexive-on receives own public image, opponent aggression/fold summaries, predicted collective fold probability, and prior reflections.",
            "",
            "## Interpretation boundary",
            "",
            "Condition effects require paired seeds and must be analyzed within each model. A smoke run validates wiring only; it does not establish a profitability ranking.",
        ]
    )


def run_six_max_experiment(config: SixMaxConfig) -> dict[str, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    reflections: list[dict[str, Any]] = []
    for seed in config.seeds:
        names = tuple(
            f"seat_{index}_{player_type}" for index, player_type in enumerate(SIX_MAX_LINEUP)
        )
        provider = _provider(config.provider, config.model, seed)
        llm = LLMPlayer(
            names[0],
            seed * 1009 + 1,
            provider,
            AgentStyle(
                aggression=0.47,
                risk_margin=0.055,
                belief_sensitivity=0.24,
                social_learning_rate=0.20,
                equity_samples=config.equity_samples,
            ),
            opponents=names[1:],
            memory_hands=6,
            reflexive_enabled=config.reflexive_enabled,
        )
        agents = [llm]
        for index, player_type in enumerate(SIX_MAX_LINEUP[1:], start=1):
            agents.append(
                make_tournament_agent(
                    player_type,
                    names[index],
                    tuple(name for name in names if name != names[index]),
                    seed * 1009 + index + 1,
                    equity_samples=config.equity_samples,
                )
            )
        records = HoldemEnvironment(
            agents,
            seed=seed,
            config=EnvironmentConfig(
                max_raises_per_street=None, regime_switch_hand=config.hands + 1
            ),
        ).play(config.hands)
        rewards = {
            agent.name: sum(record.rewards[agent.name] for record in records) for agent in agents
        }
        action_counts = {
            agent.name: sum(
                event.actor == agent.name for record in records for event in record.actions
            )
            for agent in agents
        }
        raise_counts = {
            agent.name: sum(
                event.actor == agent.name and event.action.value == "raise"
                for record in records
                for event in record.actions
            )
            for agent in agents
        }
        provider_traces = llm.llm_decision_log + llm.llm_reflection_log
        provider_latencies = [
            float(trace["latency_ms"])
            for trace in provider_traces
            if trace.get("latency_ms") is not None
        ]
        input_token_values = [
            int(trace["input_tokens"])
            for trace in provider_traces
            if trace.get("input_tokens") is not None
        ]
        output_token_values = [
            int(trace["output_tokens"])
            for trace in provider_traces
            if trace.get("output_tokens") is not None
        ]
        total_token_values = [
            int(trace["total_tokens"])
            for trace in provider_traces
            if trace.get("total_tokens") is not None
        ]
        cost_values = [
            float(trace["cost_usd"])
            for trace in provider_traces
            if trace.get("cost_usd") is not None
        ]
        fallback_count = sum(
            bool(trace.get("final_decision", {}).get("fallback_used"))
            for trace in llm.llm_decision_log
        )
        for index, player_type in enumerate(SIX_MAX_LINEUP):
            rows.append(
                {
                    "seed": seed,
                    "condition": config.condition,
                    "reflexive_enabled": config.reflexive_enabled,
                    "player_type": player_type,
                    "seat": index,
                    "hands": config.hands,
                    "chips_per_100": 100.0 * rewards[names[index]] / config.hands,
                    "decision_count": action_counts[names[index]],
                    "raise_rate": raise_counts[names[index]] / max(1, action_counts[names[index]]),
                    "button_hands": sum(record.button == index for record in records),
                    "showdown_rate": sum(record.showdown for record in records) / len(records),
                    "provider": config.provider if player_type == "llm" else "rule_based",
                    "model": getattr(provider, "model", None) if player_type == "llm" else None,
                    "provider_call_count": len(provider_traces) if player_type == "llm" else 0,
                    "provider_failure_count": llm.provider_failures if player_type == "llm" else 0,
                    "fallback_count": fallback_count if player_type == "llm" else 0,
                    "invalid_action_count": llm.invalid_actions if player_type == "llm" else 0,
                    "input_tokens": (
                        sum(input_token_values)
                        if player_type == "llm" and input_token_values
                        else float("nan")
                    ),
                    "output_tokens": (
                        sum(output_token_values)
                        if player_type == "llm" and output_token_values
                        else float("nan")
                    ),
                    "total_tokens": (
                        sum(total_token_values)
                        if player_type == "llm" and total_token_values
                        else float("nan")
                    ),
                    "token_observed_call_count": (
                        len(total_token_values) if player_type == "llm" else 0
                    ),
                    "reported_cost_usd": (
                        sum(cost_values) if player_type == "llm" and cost_values else float("nan")
                    ),
                    "cost_observed_call_count": len(cost_values) if player_type == "llm" else 0,
                    "latency_observed_call_count": (
                        len(provider_latencies) if player_type == "llm" else 0
                    ),
                    "mean_provider_latency_ms": (
                        sum(provider_latencies) / len(provider_latencies)
                        if player_type == "llm" and provider_latencies
                        else float("nan")
                    ),
                }
            )
        decisions.extend({"seed": seed, **trace} for trace in llm.llm_decision_log)
        reflections.extend({"seed": seed, **trace} for trace in llm.llm_reflection_log)

    per_seed = pd.DataFrame(rows).sort_values(["condition", "seed", "seat"])
    summary = (
        per_seed.groupby(["condition", "player_type"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_chips_per_100=("chips_per_100", "mean"),
            mean_raise_rate=("raise_rate", "mean"),
            decision_count=("decision_count", "sum"),
            mean_showdown_rate=("showdown_rate", "mean"),
            provider_failure_count=("provider_failure_count", "sum"),
            fallback_count=("fallback_count", "sum"),
            total_tokens=("total_tokens", "sum"),
            reported_cost_usd=("reported_cost_usd", "sum"),
        )
        .sort_values("mean_chips_per_100", ascending=False)
    )
    per_seed.to_csv(config.output_dir / "per_seed.csv", index=False)
    summary.to_csv(config.output_dir / "summary.csv", index=False)
    _write_jsonl(config.output_dir / "llm_decision_traces.jsonl.gz", decisions)
    _write_jsonl(config.output_dir / "llm_reflection_traces.jsonl.gz", reflections)
    (config.output_dir / "design.md").write_text(_design_markdown(config), encoding="utf-8")
    return {"per_seed": per_seed, "summary": summary}
