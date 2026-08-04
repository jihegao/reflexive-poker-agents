from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t

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
    equity_samples: int = 4
    recovery_window: int = 32
    simulation_rollout_hands: int = 36
    simulation_equity_samples: int = 1
    formation_observations: int = 48
    calibration_observations: int = 32
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
    simulated_hands: int
    final_response_policy: str | None
    surprise_threshold: float | None
    calibration_complete: bool | None


@dataclass(frozen=True)
class RegimePairedEffect:
    seed: int
    mirror: int
    treatment: str
    control: str
    total_reward_delta_bb: float
    post_switch_bb100_delta: float
    recovery_hands_delta: float | None


def _recovery_hands(
    rewards: Sequence[float],
    switch_hand: int,
    window: int,
) -> int | None:
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


def _make_hero(
    condition: str,
    seed: int,
    config: RegimeExperimentConfig,
) -> PokerAgent:
    style = AgentStyle(
        aggression=0.40,
        risk_margin=-0.045,
        belief_sensitivity=0.22,
        social_learning_rate=0.20,
        equity_samples=config.equity_samples,
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
                threshold=0.07,
                cooldown_observations=20,
            ),
            simulator=WorldSimulator(
                rollouts=config.simulation_rollout_hands,
                seed=seed + 41,
                equity_samples=config.simulation_equity_samples,
            ),
            observation_window=max(config.formation_observations * 2, 96),
            formation_observations=config.formation_observations,
            calibration_observations=config.calibration_observations,
        )
    raise ValueError(f"Unknown condition: {condition}")


def run_regime_switch_experiment(
    config: RegimeExperimentConfig,
) -> list[RegimeExperimentRow]:
    if config.switch_hand <= 0 or config.switch_hand >= config.hands:
        raise ValueError("switch_hand must fall inside the experiment horizon")
    if config.simulation_rollout_hands < 2:
        raise ValueError("simulation_rollout_hands must be at least two")
    rows: list[RegimeExperimentRow] = []
    for condition in ("baseline", "reflection", "reflection_simulation"):
        for seed in config.seeds:
            for mirror in (0, 1):
                hero = _make_hero(condition, seed * 17 + 1, config)
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
                hypothesis_calls = 0
                simulation_calls = 0
                simulated_hands = 0
                final_response_policy = None
                surprise_threshold = None
                calibration_complete = None
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
                    simulated_hands = hero.simulator.simulated_hands
                    final_response_policy = hero.state.response_policy
                    surprise_threshold = hero.detector.threshold
                    calibration_complete = hero.calibration_complete
                rows.append(
                    RegimeExperimentRow(
                        condition=condition,
                        seed=seed,
                        mirror=mirror,
                        total_reward_bb=sum(rewards),
                        pre_switch_reward_bb=pre,
                        post_switch_reward_bb=post,
                        post_switch_bb100=(
                            100.0 * post / (config.hands - config.switch_hand)
                        ),
                        recovery_hands=_recovery_hands(
                            rewards,
                            config.switch_hand,
                            config.recovery_window,
                        ),
                        detected_change_hand=detected_change_hand,
                        detection_delay_hands=(
                            None
                            if detected_change_hand is None
                            else detected_change_hand - config.switch_hand
                        ),
                        hypothesis_calls=hypothesis_calls,
                        simulation_calls=simulation_calls,
                        simulated_hands=simulated_hands,
                        final_response_policy=final_response_policy,
                        surprise_threshold=surprise_threshold,
                        calibration_complete=calibration_complete,
                    )
                )
    if config.output_dir is not None:
        write_regime_experiment(rows, config.output_dir)
    return rows


def summarize_regime_experiment(
    rows: Sequence[RegimeExperimentRow],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[RegimeExperimentRow]] = {}
    for row in rows:
        grouped.setdefault(row.condition, []).append(row)
    summary: list[dict[str, Any]] = []
    for condition, values in grouped.items():
        recovery = [
            row.recovery_hands for row in values if row.recovery_hands is not None
        ]
        delays = [
            row.detection_delay_hands
            for row in values
            if row.detection_delay_hands is not None
        ]
        policies = Counter(
            row.final_response_policy
            for row in values
            if row.final_response_policy is not None
        )
        summary.append(
            {
                "condition": condition,
                "matches": len(values),
                "mean_total_reward_bb": (
                    sum(row.total_reward_bb for row in values) / len(values)
                ),
                "mean_post_switch_bb100": (
                    sum(row.post_switch_bb100 for row in values) / len(values)
                ),
                "mean_recovery_hands": (
                    sum(recovery) / len(recovery) if recovery else None
                ),
                "recovery_rate": len(recovery) / len(values),
                "detection_rate": len(delays) / len(values),
                "mean_detection_delay_hands": (
                    sum(delays) / len(delays) if delays else None
                ),
                "mean_hypothesis_calls": (
                    sum(row.hypothesis_calls for row in values) / len(values)
                ),
                "mean_simulation_calls": (
                    sum(row.simulation_calls for row in values) / len(values)
                ),
                "mean_simulated_hands": (
                    sum(row.simulated_hands for row in values) / len(values)
                ),
                "final_response_policy_counts": dict(sorted(policies.items())),
            }
        )
    return sorted(summary, key=lambda item: item["condition"])


def paired_regime_effects(
    rows: Sequence[RegimeExperimentRow],
    *,
    treatment: str = "reflection_simulation",
    control: str = "reflection",
) -> list[RegimePairedEffect]:
    indexed: dict[tuple[str, int, int], RegimeExperimentRow] = {}
    for row in rows:
        key = (row.condition, row.seed, row.mirror)
        if key in indexed:
            raise ValueError(f"Duplicate experiment row: {key}")
        indexed[key] = row
    pairs: list[RegimePairedEffect] = []
    pair_keys = sorted(
        {
            (row.seed, row.mirror)
            for row in rows
            if row.condition in {treatment, control}
        }
    )
    for seed, mirror in pair_keys:
        treatment_row = indexed.get((treatment, seed, mirror))
        control_row = indexed.get((control, seed, mirror))
        if treatment_row is None or control_row is None:
            continue
        recovery_delta = None
        if (
            treatment_row.recovery_hands is not None
            and control_row.recovery_hands is not None
        ):
            recovery_delta = float(
                treatment_row.recovery_hands - control_row.recovery_hands
            )
        pairs.append(
            RegimePairedEffect(
                seed=seed,
                mirror=mirror,
                treatment=treatment,
                control=control,
                total_reward_delta_bb=(
                    treatment_row.total_reward_bb - control_row.total_reward_bb
                ),
                post_switch_bb100_delta=(
                    treatment_row.post_switch_bb100 - control_row.post_switch_bb100
                ),
                recovery_hands_delta=recovery_delta,
            )
        )
    return pairs


def _mean_ci95(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "ci95_low": mean, "ci95_high": mean}
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    half_width = critical * standard_error
    return {
        "n": len(values),
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def summarize_paired_regime_effects(
    effects: Sequence[RegimePairedEffect],
) -> dict[str, Any]:
    return {
        "treatment": effects[0].treatment if effects else None,
        "control": effects[0].control if effects else None,
        "total_reward_delta_bb": _mean_ci95(
            [effect.total_reward_delta_bb for effect in effects]
        ),
        "post_switch_bb100_delta": _mean_ci95(
            [effect.post_switch_bb100_delta for effect in effects]
        ),
        "recovery_hands_delta": _mean_ci95(
            [
                effect.recovery_hands_delta
                for effect in effects
                if effect.recovery_hands_delta is not None
            ]
        ),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_regime_experiment(
    rows: Sequence[RegimeExperimentRow],
    output_dir: Path,
) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    match_payload = [asdict(row) for row in rows]
    _write_csv(output_dir / "matches.csv", match_payload)
    (output_dir / "summary.json").write_text(
        json.dumps(summarize_regime_experiment(rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    effects = paired_regime_effects(rows)
    effect_payload = [asdict(effect) for effect in effects]
    _write_csv(output_dir / "paired_effects.csv", effect_payload)
    (output_dir / "paired_summary.json").write_text(
        json.dumps(summarize_paired_regime_effects(effects), indent=2, sort_keys=True),
        encoding="utf-8",
    )
