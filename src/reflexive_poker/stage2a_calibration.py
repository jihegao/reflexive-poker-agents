from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from scipy.stats import t as student_t

from .agents import AgentStyle, PokerAgent
from .environment import EnvironmentConfig, HoldemEnvironment
from .models import ActionEvent, ActionType, Decision, DecisionContext
from .regime_agents import ReflectionTrackerAgent
from .regime_detection import (
    ConditionalDistributionDetector,
    OpponentObservation,
    OpponentWorld,
    empirical_world,
)
from .regime_simulation import WorldSimulator, response_policy_decision

STAGE2A_PROTOCOL = "regime-stage2a-calibration-v1"
MIRRORS = (0, 1)
SWITCH_CONDITIONS = ("baseline", "reflection", "simulation")


@dataclass(frozen=True)
class Stage2AConfig:
    seed_start: int
    seed_count: int
    hands: int
    switch_hand: int
    equity_samples: int
    simulation_rollouts: int
    simulation_equity_samples: int
    policy_holdout_hands: int
    reference_size: int
    recent_size: int
    min_recent_observations: int
    likelihood_ratio_threshold: float
    min_probability_delta: float
    required_streak: int
    evaluation_stride: int
    detector_calibration_observations: int
    detector_probe_direction_delta: float
    earliest_detection_hand: int
    selector_confidence_z: float
    selector_minimum_improvement: float
    selector_probe_open_raise_minimum: float
    selector_probe_fold_minimum: float
    selector_probe_reraise_maximum: float
    detection_rate_gate: float
    median_delay_gate: int
    false_positive_rate_gate: float
    pressure_selection_rate_gate: float
    reflection_divergence_gate: float
    pre_world: OpponentWorld
    switch_world: OpponentWorld
    noise_levels: tuple[float, ...]
    holdout_worlds: tuple[OpponentWorld, ...]
    proxy_worlds: tuple[OpponentWorld, ...]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_start, self.seed_start + self.seed_count))


def _world_from_mapping(name: str, value: Mapping[str, Any]) -> OpponentWorld:
    return OpponentWorld(
        name=name,
        open_raise_probability=float(value["open_raise_probability"]),
        fold_vs_bet_probability=float(value["fold_vs_bet_probability"]),
        reraise_probability=float(value["reraise_probability"]),
        prior=float(value.get("prior", 1.0)),
        rationale=str(value.get("rationale", "")),
    )


def load_stage2a_config(path: Path, *, seed_count: int | None = None) -> Stage2AConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = payload["stage2a_calibration"]
    if section.get("protocol") != STAGE2A_PROTOCOL:
        raise ValueError(f"protocol must be {STAGE2A_PROTOCOL}")
    detector = section["detector"]
    selector = section["selector"]
    gates = section["gates"]
    worlds = section["worlds"]
    configured_seed_count = int(section["seed_count"])
    return Stage2AConfig(
        seed_start=int(section["seed_start"]),
        seed_count=configured_seed_count if seed_count is None else seed_count,
        hands=int(section["hands"]),
        switch_hand=int(section["switch_hand"]),
        equity_samples=int(section["equity_samples"]),
        simulation_rollouts=int(section["simulation_rollouts"]),
        simulation_equity_samples=int(section["simulation_equity_samples"]),
        policy_holdout_hands=int(section["policy_holdout_hands"]),
        reference_size=int(detector["reference_size"]),
        recent_size=int(detector["recent_size"]),
        min_recent_observations=int(detector["min_recent_observations"]),
        likelihood_ratio_threshold=float(detector["likelihood_ratio_threshold"]),
        min_probability_delta=float(detector["min_probability_delta"]),
        required_streak=int(detector["required_streak"]),
        evaluation_stride=int(detector["evaluation_stride"]),
        detector_calibration_observations=int(detector["calibration_observations"]),
        detector_probe_direction_delta=float(detector["probe_direction_delta"]),
        earliest_detection_hand=int(detector["earliest_detection_hand"]),
        selector_confidence_z=float(selector["confidence_z"]),
        selector_minimum_improvement=float(selector["minimum_improvement_bb_per_hand"]),
        selector_probe_open_raise_minimum=float(selector["probe_open_raise_minimum"]),
        selector_probe_fold_minimum=float(selector["probe_fold_minimum"]),
        selector_probe_reraise_maximum=float(selector["probe_reraise_maximum"]),
        detection_rate_gate=float(gates["detection_rate"]),
        median_delay_gate=int(gates["median_delay_hands"]),
        false_positive_rate_gate=float(gates["no_switch_false_positive_rate"]),
        pressure_selection_rate_gate=float(gates["pressure_selection_rate"]),
        reflection_divergence_gate=float(gates["reflection_action_divergence"]),
        pre_world=_world_from_mapping("pre_tag", worlds["pre"]),
        switch_world=_world_from_mapping("confirm_probe_fold", worlds["switch"]),
        noise_levels=tuple(float(value) for value in section["noise_levels"]),
        holdout_worlds=tuple(
            _world_from_mapping(str(name), value)
            for name, value in worlds["policy_holdouts"].items()
        ),
        proxy_worlds=tuple(
            _world_from_mapping(str(name), value)
            for name, value in worlds["proxy_hypotheses"].items()
        ),
    )


class ConditionalRegimeOpponent(PokerAgent):
    """Independent execution model used only as the Stage 2A actual opponent."""

    def __init__(
        self,
        name: str,
        seed: int,
        switch_hand: int,
        pre_world: OpponentWorld,
        post_world: OpponentWorld,
        execution_noise: float,
    ) -> None:
        super().__init__(name, seed, AgentStyle(equity_samples=1))
        if not 0.0 <= execution_noise <= 1.0:
            raise ValueError("execution_noise must be in [0, 1]")
        self.switch_hand = switch_hand
        self.pre_world = pre_world
        self.post_world = post_world
        self.execution_noise = execution_noise

    def act(self, context: DecisionContext) -> Decision:
        world = self.post_world if context.hand_index >= self.switch_hand else self.pre_world
        legal = tuple(context.legal_actions)
        facing_bet = context.to_call > 1e-9
        if self.rng.random() < self.execution_noise:
            action = legal[self.rng.randrange(len(legal))]
        elif not facing_bet:
            action = (
                ActionType.RAISE
                if ActionType.RAISE in legal
                and self.rng.random() < world.open_raise_probability
                else ActionType.CHECK_CALL
            )
        elif ActionType.FOLD in legal and self.rng.random() < world.fold_vs_bet_probability:
            action = ActionType.FOLD
        elif ActionType.RAISE in legal and self.rng.random() < world.reraise_probability:
            action = ActionType.RAISE
        else:
            action = ActionType.CHECK_CALL
        decision = Decision(
            action=action,
            raise_scale=0.55,
            equity=0.5,
            reasoning_depth=0,
            reasoning_ops=1,
            metadata={"scenario_world": world.name},
        )
        self.decision_log.append(
            {
                "hand_index": context.hand_index,
                "street": context.street.value,
                "action": action.value,
                "scenario_world": world.name,
            }
        )
        return decision


class FrozenProxyGenerator:
    """Frozen structured-world proxy; it is not a live LLM call."""

    def __init__(self, worlds: Sequence[OpponentWorld]) -> None:
        self.worlds = tuple(worlds)
        self.calls = 0

    def generate(
        self,
        observations: Sequence[OpponentObservation],
        current_worlds: Sequence[OpponentWorld],
    ) -> list[OpponentWorld]:
        del current_worlds
        self.calls += 1
        return [
            empirical_world(observations, name="recent_empirical", prior=1.25),
            *self.worlds,
        ]


@dataclass(frozen=True)
class PolicyCalibration:
    pressure_allowed: bool
    conservative_lower_bound_bb100: float


class Stage2ASimulationAgent(PokerAgent):
    """One-shot detector + frozen hypotheses + robust rollout selector."""

    condition = "stage2a_simulation"

    def __init__(
        self,
        name: str,
        seed: int,
        config: Stage2AConfig,
        policy_calibration: PolicyCalibration,
    ) -> None:
        super().__init__(name, seed, hero_style(config.equity_samples))
        self.detector = ConditionalDistributionDetector(
            reference_size=config.reference_size,
            recent_size=config.recent_size,
            min_recent_observations=config.min_recent_observations,
            likelihood_ratio_threshold=config.likelihood_ratio_threshold,
            min_probability_delta=config.min_probability_delta,
            required_streak=config.required_streak,
            evaluation_stride=config.evaluation_stride,
            calibration_observations=config.detector_calibration_observations,
        )
        self.generator = FrozenProxyGenerator(config.proxy_worlds)
        self.simulator = WorldSimulator(
            rollouts=config.simulation_rollouts,
            seed=seed + 41,
            equity_samples=config.simulation_equity_samples,
        )
        self.formation: deque[OpponentObservation] = deque(maxlen=config.reference_size)
        self.detected_change_hand: int | None = None
        self.response_policy: str | None = None
        self.raw_simulator_policy: str | None = None
        self.expected_value = 0.0
        self.selector_lower_bound = 0.0
        self.last_likelihood_ratio = 0.0
        self.last_probability_delta = 0.0
        self.last_probability_deltas: dict[str, float] = {}
        self.selector_confidence_z = config.selector_confidence_z
        self.selector_minimum_improvement = config.selector_minimum_improvement
        self.detector_probe_direction_delta = config.detector_probe_direction_delta
        self.earliest_detection_hand = config.earliest_detection_hand
        self.policy_calibration = policy_calibration
        self.selector_probe_open_raise_minimum = config.selector_probe_open_raise_minimum
        self.selector_probe_fold_minimum = config.selector_probe_fold_minimum
        self.selector_probe_reraise_maximum = config.selector_probe_reraise_maximum

    def observe_action(self, event: ActionEvent) -> None:
        super().observe_action(event)
        if event.actor == self.name:
            return
        observation = OpponentObservation.from_event(event)
        if not self.detector.ready:
            self.formation.append(observation)
            if len(self.formation) == self.detector.reference_size:
                self.detector.fit_reference(tuple(self.formation))
            return
        if self.detected_change_hand is not None:
            return
        update = self.detector.update(observation)
        self.last_likelihood_ratio = update.likelihood_ratio
        self.last_probability_delta = update.max_probability_delta
        self.last_probability_deltas = update.probability_deltas
        if not update.change_detected:
            return
        if event.hand_index < self.earliest_detection_hand:
            return
        probe_signature = (
            update.probability_deltas.get("unopened:raise", 0.0)
            >= self.detector_probe_direction_delta
            and update.probability_deltas.get("facing_bet:fold", 0.0)
            >= self.detector_probe_direction_delta
        )
        if not probe_signature:
            return
        self.detected_change_hand = event.hand_index
        recent = tuple(self.detector.recent)
        worlds = self.generator.generate(recent, ())
        results = self.simulator.evaluate(worlds, recent)
        (
            self.raw_simulator_policy,
            self.expected_value,
            self.selector_lower_bound,
        ) = self.simulator.choose_response_robust(
            results,
            confidence_z=self.selector_confidence_z,
            minimum_improvement=self.selector_minimum_improvement,
        )
        recent_world = empirical_world(recent, name="selector_recent", prior=1.0)
        calibrated_probe_fold = (
            self.policy_calibration.pressure_allowed
            and recent_world.open_raise_probability
            >= self.selector_probe_open_raise_minimum
            and recent_world.fold_vs_bet_probability >= self.selector_probe_fold_minimum
            and recent_world.reraise_probability <= self.selector_probe_reraise_maximum
        )
        if calibrated_probe_fold:
            self.response_policy = "pressure"
            self.selector_lower_bound = (
                self.policy_calibration.conservative_lower_bound_bb100 / 100.0
            )
        else:
            self.response_policy = self.raw_simulator_policy

    def act(self, context: DecisionContext) -> Decision:
        metadata = {
            "adaptation_condition": self.condition,
            "detected_change_hand": self.detected_change_hand,
            "response_policy": self.response_policy,
            "selector_lower_bound": self.selector_lower_bound,
            "likelihood_ratio": self.last_likelihood_ratio,
            "max_probability_delta": self.last_probability_delta,
        }
        if self.response_policy is None:
            return self._policy(context, reasoning_depth=1, metadata=metadata)
        decision = response_policy_decision(
            self,
            context,
            self.response_policy,
            metadata=metadata,
        )
        self.decision_log.append(
            {
                "hand_index": context.hand_index,
                "street": context.street.value,
                "action": decision.action.value,
                "equity": decision.equity,
                **metadata,
            }
        )
        return decision


class FixedResponseAgent(PokerAgent):
    def __init__(self, name: str, seed: int, policy: str, equity_samples: int) -> None:
        super().__init__(name, seed, hero_style(equity_samples))
        self.policy = policy

    def act(self, context: DecisionContext) -> Decision:
        return response_policy_decision(self, context, self.policy)


def hero_style(equity_samples: int) -> AgentStyle:
    return AgentStyle(
        aggression=0.40,
        risk_margin=-0.045,
        belief_sensitivity=0.22,
        social_learning_rate=0.20,
        equity_samples=equity_samples,
    )


def _make_switch_hero(
    condition: str,
    seed: int,
    config: Stage2AConfig,
    policy_calibration: PolicyCalibration,
) -> PokerAgent:
    if condition == "baseline":
        agent = PokerAgent("hero", seed, hero_style(config.equity_samples))
        agent.condition = condition
        return agent
    if condition == "reflection":
        return ReflectionTrackerAgent("hero", seed, hero_style(config.equity_samples))
    if condition == "simulation":
        return Stage2ASimulationAgent("hero", seed, config, policy_calibration)
    raise ValueError(f"Unknown condition: {condition}")


def _action_signature(agent: PokerAgent) -> str:
    return "|".join(str(item["action"]) for item in agent.decision_log)


def run_switch_match(
    config: Stage2AConfig,
    *,
    scenario_id: str,
    post_world: OpponentWorld,
    noise: float,
    condition: str,
    seed: int,
    mirror: int,
    policy_calibration: PolicyCalibration,
) -> dict[str, Any]:
    hero = _make_switch_hero(condition, seed * 17 + 1, config, policy_calibration)
    opponent = ConditionalRegimeOpponent(
        "opponent",
        seed * 17 + 2,
        config.switch_hand,
        config.pre_world,
        post_world,
        noise,
    )
    agents = [hero, opponent] if mirror == 0 else [opponent, hero]
    records = HoldemEnvironment(
        agents,
        seed=seed,
        config=EnvironmentConfig(
            starting_stack=100.0,
            small_blind=0.5,
            big_blind=1.0,
            max_raises_per_street=2,
            regime_switch_hand=config.switch_hand,
        ),
    ).play(config.hands)
    rewards = [record.rewards["hero"] for record in records]
    detected_hand = None
    response_policy = None
    raw_simulator_policy = None
    selector_lower_bound = None
    simulation_calls = 0
    simulated_hands = 0
    likelihood_ratio = None
    probability_delta = None
    probability_deltas = None
    if isinstance(hero, Stage2ASimulationAgent):
        detected_hand = hero.detected_change_hand
        response_policy = hero.response_policy
        raw_simulator_policy = hero.raw_simulator_policy
        selector_lower_bound = hero.selector_lower_bound
        simulation_calls = hero.simulator.calls
        simulated_hands = hero.simulator.simulated_hands
        likelihood_ratio = hero.last_likelihood_ratio
        probability_delta = hero.last_probability_delta
        probability_deltas = json.dumps(hero.last_probability_deltas, sort_keys=True)
    return {
        "scenario_id": scenario_id,
        "noise": noise,
        "condition": condition,
        "seed": seed,
        "mirror": mirror,
        "total_reward_bb": sum(rewards),
        "post_switch_bb100": 100.0
        * sum(rewards[config.switch_hand :])
        / (config.hands - config.switch_hand),
        "detected_change_hand": detected_hand,
        "detection_delay_hands": (
            None if detected_hand is None else detected_hand - config.switch_hand
        ),
        "response_policy": response_policy,
        "raw_simulator_policy": raw_simulator_policy,
        "selector_lower_bound": selector_lower_bound,
        "simulation_calls": simulation_calls,
        "simulated_hands": simulated_hands,
        "last_likelihood_ratio": likelihood_ratio,
        "last_probability_delta": probability_delta,
        "last_probability_deltas": probability_deltas,
        "action_signature": _action_signature(hero),
    }


def run_policy_holdout(
    config: Stage2AConfig,
    *,
    world: OpponentWorld,
    policy: str,
    seed: int,
    mirror: int,
) -> dict[str, Any]:
    hero = FixedResponseAgent("hero", seed * 19 + 1, policy, config.equity_samples)
    opponent = ConditionalRegimeOpponent(
        "opponent",
        seed * 19 + 2,
        0,
        world,
        world,
        0.10,
    )
    agents = [hero, opponent] if mirror == 0 else [opponent, hero]
    records = HoldemEnvironment(
        agents,
        seed=seed + 200_000,
        config=EnvironmentConfig(
            starting_stack=100.0,
            small_blind=0.5,
            big_blind=1.0,
            max_raises_per_street=2,
            regime_switch_hand=config.policy_holdout_hands + 1,
        ),
    ).play(config.policy_holdout_hands)
    reward = sum(record.rewards["hero"] for record in records)
    return {
        "world": world.name,
        "policy": policy,
        "seed": seed,
        "mirror": mirror,
        "reward_bb": reward,
        "bb100": 100.0 * reward / config.policy_holdout_hands,
    }


def _interval(values: Sequence[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "ci95_low": mean, "ci95_high": mean}
    half_width = (
        student_t.ppf(0.975, len(values) - 1)
        * statistics.stdev(values)
        / math.sqrt(len(values))
    )
    return {
        "n": len(values),
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _signature_divergence(left: str, right: str) -> float:
    left_actions = left.split("|") if left else []
    right_actions = right.split("|") if right else []
    denominator = max(len(left_actions), len(right_actions), 1)
    shared = min(len(left_actions), len(right_actions))
    mismatches = sum(left_actions[index] != right_actions[index] for index in range(shared))
    mismatches += abs(len(left_actions) - len(right_actions))
    return mismatches / denominator


def build_policy_calibration(
    config: Stage2AConfig,
    policy_rows: Sequence[dict[str, Any]],
) -> PolicyCalibration:
    by_key = {
        (row["world"], row["policy"], row["seed"], row["mirror"]): row
        for row in policy_rows
    }
    lower_bounds: list[float] = []
    mirror_directions: list[float] = []
    for world in config.holdout_worlds:
        effects: list[float] = []
        per_mirror = {0: [], 1: []}
        for seed in config.seeds:
            paired: list[float] = []
            for mirror in MIRRORS:
                delta = (
                    by_key[(world.name, "pressure", seed, mirror)]["bb100"]
                    - by_key[(world.name, "balanced", seed, mirror)]["bb100"]
                )
                paired.append(delta)
                per_mirror[mirror].append(delta)
            effects.append(statistics.fmean(paired))
        interval = _interval(effects)
        lower_bounds.append(float(interval["ci95_low"]))
        mirror_directions.extend(statistics.fmean(per_mirror[value]) for value in MIRRORS)
    conservative_lower_bound = min(lower_bounds)
    return PolicyCalibration(
        pressure_allowed=bool(
            config.seed_count >= 2
            and conservative_lower_bound > 0.0
            and all(value > 0.0 for value in mirror_directions)
        ),
        conservative_lower_bound_bb100=conservative_lower_bound,
    )


def summarize_stage2a(
    config: Stage2AConfig,
    switch_rows: Sequence[dict[str, Any]],
    policy_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    switch_cells: list[dict[str, Any]] = []
    by_switch_key = {
        (row["scenario_id"], row["condition"], row["seed"], row["mirror"]): row
        for row in switch_rows
    }
    for noise in config.noise_levels:
        scenario_id = f"probe_fold_e{int(noise * 100):02d}"
        simulation_rows = [
            row
            for row in switch_rows
            if row["scenario_id"] == scenario_id and row["condition"] == "simulation"
        ]
        valid_detections = [
            row
            for row in simulation_rows
            if row["detected_change_hand"] is not None
            and row["detected_change_hand"] >= config.switch_hand
        ]
        delays = [float(row["detection_delay_hands"]) for row in valid_detections]
        divergences = [
            _signature_divergence(
                by_switch_key[(scenario_id, "baseline", seed, mirror)]["action_signature"],
                by_switch_key[(scenario_id, "reflection", seed, mirror)]["action_signature"],
            )
            for seed in config.seeds
            for mirror in MIRRORS
        ]
        pressure_selections = sum(
            row["response_policy"] == "pressure" for row in valid_detections
        )
        cell = {
            "scenario_id": scenario_id,
            "noise": noise,
            "matches": len(simulation_rows),
            "detection_rate": len(valid_detections) / len(simulation_rows),
            "median_detection_delay_hands": statistics.median(delays) if delays else None,
            "pre_switch_false_detection_rate": sum(
                row["detected_change_hand"] is not None
                and row["detected_change_hand"] < config.switch_hand
                for row in simulation_rows
            )
            / len(simulation_rows),
            "pressure_selection_rate": pressure_selections / len(simulation_rows),
            "pressure_given_detection_rate": (
                pressure_selections / len(valid_detections) if valid_detections else 0.0
            ),
            "response_policy_counts": dict(
                sorted(Counter(row["response_policy"] for row in valid_detections).items())
            ),
            "mean_reflection_action_divergence": statistics.fmean(divergences),
        }
        cell["gate_pass"] = bool(
            cell["detection_rate"] >= config.detection_rate_gate
            and cell["median_detection_delay_hands"] is not None
            and cell["median_detection_delay_hands"] <= config.median_delay_gate
            and cell["pressure_selection_rate"] >= config.pressure_selection_rate_gate
            and cell["mean_reflection_action_divergence"]
            >= config.reflection_divergence_gate
        )
        switch_cells.append(cell)

    no_switch_rows = [
        row
        for row in switch_rows
        if row["scenario_id"].startswith("no_switch_")
        and row["condition"] == "simulation"
    ]
    no_switch_false_positive_rate = sum(
        row["detected_change_hand"] is not None for row in no_switch_rows
    ) / len(no_switch_rows)

    by_policy_key = {
        (row["world"], row["policy"], row["seed"], row["mirror"]): row
        for row in policy_rows
    }
    policy_cells: list[dict[str, Any]] = []
    for world in config.holdout_worlds:
        seed_effects: list[float] = []
        mirror_effects = {0: [], 1: []}
        for seed in config.seeds:
            pair: list[float] = []
            for mirror in MIRRORS:
                pressure = by_policy_key[(world.name, "pressure", seed, mirror)]["bb100"]
                balanced = by_policy_key[(world.name, "balanced", seed, mirror)]["bb100"]
                delta = pressure - balanced
                pair.append(delta)
                mirror_effects[mirror].append(delta)
            seed_effects.append(statistics.fmean(pair))
        interval = _interval(seed_effects)
        cell = {
            "world": world.name,
            "pressure_minus_balanced_seed_mirror_mean": interval,
            "mirror_0_mean": statistics.fmean(mirror_effects[0]),
            "mirror_1_mean": statistics.fmean(mirror_effects[1]),
        }
        cell["gate_pass"] = bool(
            interval["ci95_low"] > 0.0
            and cell["mirror_0_mean"] > 0.0
            and cell["mirror_1_mean"] > 0.0
        )
        policy_cells.append(cell)

    false_positive_gate_pass = (
        no_switch_false_positive_rate <= config.false_positive_rate_gate
    )
    return {
        "protocol": STAGE2A_PROTOCOL,
        "claim_boundary": (
            "calibration_and_proxy_simulation_only_not_live_llm_or_formal_profitability_evidence"
        ),
        "switch_cells": switch_cells,
        "no_switch": {
            "matches": len(no_switch_rows),
            "false_positive_rate": no_switch_false_positive_rate,
            "gate_pass": false_positive_gate_pass,
        },
        "policy_holdout_cells": policy_cells,
        "stage2b_ready": bool(
            all(cell["gate_pass"] for cell in switch_cells)
            and false_positive_gate_pass
            and all(cell["gate_pass"] for cell in policy_cells)
        ),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_provenance(config_path: Path) -> dict[str, Any]:
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("regime_detection.py"),
        Path(__file__).with_name("regime_agents.py"),
        Path(__file__).with_name("regime_simulation.py"),
        config_path,
    )
    source_hasher = hashlib.sha256()
    for source_path in source_paths:
        source_hasher.update(str(source_path.resolve()).encode())
        source_hasher.update(source_path.read_bytes())
    return {
        "source_tree_sha256": source_hasher.hexdigest(),
        "source_files": [str(path.resolve()) for path in source_paths],
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }


def run_stage2a(config: Stage2AConfig, output_dir: Path, config_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "protocol": STAGE2A_PROTOCOL,
        "claim_boundary": "bounded_calibration_no_live_llm",
        "config": asdict(config),
        "statistical_unit": "seed-level mean across two seat mirrors",
        "frozen_before_stochastic_run": True,
        **_source_provenance(config_path),
    }
    (output_dir / "PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2, default=str),
        encoding="utf-8",
    )
    policy_rows: list[dict[str, Any]] = []
    for world in config.holdout_worlds:
        for seed in config.seeds:
            for mirror in MIRRORS:
                for policy in ("pressure", "balanced", "bluff_catch"):
                    policy_rows.append(
                        run_policy_holdout(
                            config,
                            world=world,
                            policy=policy,
                            seed=seed,
                            mirror=mirror,
                        )
                    )
            print(f"policy_holdout world={world.name} seed={seed} complete", flush=True)

    policy_calibration = build_policy_calibration(config, policy_rows)
    (output_dir / "policy_calibration.json").write_text(
        json.dumps(asdict(policy_calibration), indent=2),
        encoding="utf-8",
    )

    switch_rows: list[dict[str, Any]] = []
    for noise in config.noise_levels:
        scenario_id = f"probe_fold_e{int(noise * 100):02d}"
        for seed in config.seeds:
            for mirror in MIRRORS:
                for condition in SWITCH_CONDITIONS:
                    switch_rows.append(
                        run_switch_match(
                            config,
                            scenario_id=scenario_id,
                            post_world=config.switch_world,
                            noise=noise,
                            condition=condition,
                            seed=seed,
                            mirror=mirror,
                            policy_calibration=policy_calibration,
                        )
                    )
            print(f"switch scenario={scenario_id} seed={seed} complete", flush=True)

        no_switch_id = f"no_switch_e{int(noise * 100):02d}"
        for seed in config.seeds:
            for mirror in MIRRORS:
                switch_rows.append(
                    run_switch_match(
                        config,
                        scenario_id=no_switch_id,
                        post_world=config.pre_world,
                        noise=noise,
                        condition="simulation",
                        seed=seed,
                        mirror=mirror,
                        policy_calibration=policy_calibration,
                    )
                )
            print(f"control scenario={no_switch_id} seed={seed} complete", flush=True)

    _write_csv(output_dir / "policy_holdout_rows.csv", policy_rows)
    _write_csv(output_dir / "switch_rows.csv", switch_rows)
    summary = summarize_stage2a(config, switch_rows, policy_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Stage 2A regime calibration")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-count", type=int)
    args = parser.parse_args()
    config = load_stage2a_config(args.config, seed_count=args.seed_count)
    run_stage2a(config, args.output, args.config)


if __name__ == "__main__":
    main()
