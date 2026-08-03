from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agents import AgentStyle, PokerAgent
from .environment import EnvironmentConfig, HoldemEnvironment
from .regime_agents import (
    ReflectionTrackerAgent,
    RegimeSwitchingOpponent,
    SimulationEnhancedReflectionAgent,
)
from .regime_detection import SurpriseDetector
from .regime_simulation import WorldSimulator


@dataclass(frozen=True)
class RegimeExperimentConfig:
    seeds: tuple[int, ...] = tuple(range(9300, 9310))
    hands: int = 320
    switch_hand: int = 160
    equity_samples: int = 6
    recovery_window: int = 32
    output_dir: Path | None = None


@dataclass(frozen=True)
class RegimeExperimentRow:
    condition: str
    seed: int
    mirror: int
    total_reward_bb: float
    pre_switch_reward_bb: float
    post_switch_reward_bb: float
    post_switch_bb100: float
    recovery_hands: int | None
    detected_change_hand: int | None
    detection_delay_hands: int | None
    hypothesis_calls: int
    simulation_calls: int


def _recovery_hands(rewards: Sequence[float], switch_hand: int, window: int) -> int | None:
    post = rewards[switch_hand:]
    if len(post) < window:
        return None
    consecutive = 0
    for end in range(window, len(post) + 1):
        mean = sum(post[end - window : end]) / window
        consecutive = consecutive + 1 if mean > 0.0 else 0
        if consecutive >= 3:
            return end
    return None


def _make_hero(condition: str, seed: int, equity_samples: int) -> PokerAgent:
    style = AgentStyle(
        aggression=0.40,
        risk_margin=-0.045,
        belief_sensitivity=0.22,
        social_learning_rate=0.20,
        equity_samples=equity_samples,
    )
    if condition == "baseline":
        agent = PokerAgent("hero", seed, style)
        agent.condition = condition
        return agent
    if condition == "reflection":
        return ReflectionTrackerAgent("hero", seed, style)
    if condition == "reflection_simulation":
        return SimulationEnhancedReflectionAgent(
            "hero",
            seed,
            style,
            opponent_name="opponent",
            detector=SurpriseDetector(
                window_size=20,
                min_observations=10,
                threshold=0.040,
                cooldown_observations=20,
            ),
            simulator=WorldSimulator(rollouts=1200, seed=seed + 41),
            formation_observations=24,
        )
    raise ValueError(f"Unknown condition: {condition}")


def run_regime_switch_experiment(config: RegimeExperimentConfig) -> list[RegimeExperimentRow]:
    if config.switch_hand <= 0 or config.switch_hand >= config.hands:
        raise ValueError("switch_hand must fall inside the experiment horizon")
    rows: list[RegimeExperimentRow] = []
    for condition in ("baseline", "reflection", "reflection_simulation"):
        for seed in config.seeds:
            for mirror in (0, 1):
                hero = _make_hero(condition, seed * 17 + 1, config.equity_samples)
                opponent = RegimeSwitchingOpponent(
                    "opponent",
                    seed * 17 + 2,
                    config.switch_hand,
                    AgentStyle(equity_samples=config.equity_samples),
                )
                agents = [hero, opponent] if mirror == 0 else [opponent, hero]
                environment = HoldemEnvironment(
                    agents,
                    seed=seed,
                    config=EnvironmentConfig(
                        starting_stack=100.0,
                        small_blind=0.5,
                        big_blind=1.0,
                        max_raises_per_street=2,
                        regime_switch_hand=config.switch_hand,
                    ),
                )
                rewards = [
                    record.rewards["hero"] for record in environment.play(config.hands)
                ]
                pre = sum(rewards[: config.switch_hand])
                post = sum(rewards[config.switch_hand :])
                detected_change_hand = None
                hypothesis_calls = simulation_calls = 0
                if isinstance(hero, SimulationEnhancedReflectionAgent):
                    detected_change_hand = next(
                        (
                            hand
                            for hand in hero.state.detected_changes
                            if hand >= config.switch_hand
                        ),
                        None,
                    )
                    hypothesis_calls = hero.generator.calls
                    simulation_calls = hero.simulator.calls
                rows.append(
                    RegimeExperimentRow(
                        condition,
                        seed,
                        mirror,
                        sum(rewards),
                        pre,
                        post,
                        100.0 * post / (config.hands - config.switch_hand),
                        _recovery_hands(
                            rewards,
                            config.switch_hand,
                            config.recovery_window,
                        ),
                        detected_change_hand,
                        None
                        if detected_change_hand is None
                        else detected_change_hand - config.switch_hand,
                        hypothesis_calls,
                        simulation_calls,
                    )
                )
    if config.output_dir is not None:
        write_regime_experiment(rows, config.output_dir)
    return rows


def summarize_regime_experiment(rows: Sequence[RegimeExperimentRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RegimeExperimentRow]] = {}
    for row in rows:
        grouped.setdefault(row.condition, []).append(row)
    summary: list[dict[str, Any]] = []
    for condition, values in grouped.items():
        recovery = [row.recovery_hands for row in values if row.recovery_hands is not None]
        delays = [
            row.detection_delay_hands
            for row in values
            if row.detection_delay_hands is not None
        ]
        summary.append(
            {
                "condition": condition,
                "matches": len(values),
                "mean_total_reward_bb": sum(row.total_reward_bb for row in values) / len(values),
                "mean_post_switch_bb100": (
                    sum(row.post_switch_bb100 for row in values) / len(values)
                ),
                "mean_recovery_hands": sum(recovery) / len(recovery) if recovery else None,
                "recovery_rate": len(recovery) / len(values),
                "mean_detection_delay_hands": sum(delays) / len(delays) if delays else None,
                "mean_hypothesis_calls": sum(row.hypothesis_calls for row in values) / len(values),
                "mean_simulation_calls": sum(row.simulation_calls for row in values) / len(values),
            }
        )
    return sorted(summary, key=lambda item: item["condition"])


def write_regime_experiment(rows: Sequence[RegimeExperimentRow], output_dir: Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    with (output_dir / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
        writer.writeheader()
        writer.writerows(payload)
    (output_dir / "summary.json").write_text(
        json.dumps(summarize_regime_experiment(rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
