from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .agents import PokerAgent
from .environment import EnvironmentConfig, HoldemEnvironment
from .models import ActionEvent, ActionType, Decision, DecisionContext
from .regime_agents import ReflectionTrackerAgent
from .regime_detection import (
    OpponentObservation,
    OpponentWorld,
    SignedConditionalEProcessDetector,
)
from .regime_simulation import WorldSimulator, response_policy_decision
from .stage2a_calibration import (
    ConditionalRegimeOpponent,
    FrozenProxyGenerator,
    _interval,
    _signature_divergence,
    hero_style,
)

STAGE2A1_PROTOCOL = "regime-stage2a1-eprocess-v1"
MIRRORS = (0, 1)
SWITCH_CONDITIONS = ("baseline", "reflection", "simulation")


@dataclass(frozen=True)
class Stage2A1Config:
    tuning_seed_start: int
    tuning_seed_count: int
    validation_seed_start: int
    validation_seed_count: int
    hands: int
    switch_hand: int
    equity_samples: int
    simulation_rollouts: int
    simulation_equity_samples: int
    noise_levels: tuple[float, ...]
    reference_size: int
    block_size: int
    threshold_candidates: tuple[float, ...]
    alternative_delta: float
    alternative_concentration: float
    minimum_direction_delta: float
    maximum_blocks: int
    earliest_detection_hand: int
    detection_rate_gate: float
    median_delay_gate: int
    pre_switch_false_rate_gate: float
    no_switch_false_rate_gate: float
    pressure_selection_rate_gate: float
    reflection_divergence_gate: float
    policy_calibration_path: Path
    policy_calibration_sha256: str
    pre_world: OpponentWorld
    switch_world: OpponentWorld
    proxy_worlds: tuple[OpponentWorld, ...]

    @property
    def tuning_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.tuning_seed_start, self.tuning_seed_start + self.tuning_seed_count))

    @property
    def validation_seeds(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.validation_seed_start,
                self.validation_seed_start + self.validation_seed_count,
            )
        )


def _world_from_mapping(name: str, value: Mapping[str, Any]) -> OpponentWorld:
    return OpponentWorld(
        name=name,
        open_raise_probability=float(value["open_raise_probability"]),
        fold_vs_bet_probability=float(value["fold_vs_bet_probability"]),
        reraise_probability=float(value["reraise_probability"]),
        prior=float(value.get("prior", 1.0)),
    )


def load_stage2a1_config(
    path: Path,
    *,
    tuning_seed_count: int | None = None,
    validation_seed_count: int | None = None,
) -> Stage2A1Config:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = payload["stage2a1_calibration"]
    if section.get("protocol") != STAGE2A1_PROTOCOL:
        raise ValueError(f"protocol must be {STAGE2A1_PROTOCOL}")
    detector = section["detector"]
    gates = section["gates"]
    calibration = section["policy_calibration"]
    worlds = section["worlds"]
    return Stage2A1Config(
        tuning_seed_start=int(section["tuning_seed_start"]),
        tuning_seed_count=(
            int(section["tuning_seed_count"])
            if tuning_seed_count is None
            else tuning_seed_count
        ),
        validation_seed_start=int(section["validation_seed_start"]),
        validation_seed_count=(
            int(section["validation_seed_count"])
            if validation_seed_count is None
            else validation_seed_count
        ),
        hands=int(section["hands"]),
        switch_hand=int(section["switch_hand"]),
        equity_samples=int(section["equity_samples"]),
        simulation_rollouts=int(section["simulation_rollouts"]),
        simulation_equity_samples=int(section["simulation_equity_samples"]),
        noise_levels=tuple(float(value) for value in section["noise_levels"]),
        reference_size=int(detector["reference_size"]),
        block_size=int(detector["block_size"]),
        threshold_candidates=tuple(
            float(value) for value in detector["threshold_candidates"]
        ),
        alternative_delta=float(detector["alternative_delta"]),
        alternative_concentration=float(detector["alternative_concentration"]),
        minimum_direction_delta=float(detector["minimum_direction_delta"]),
        maximum_blocks=int(detector["maximum_blocks"]),
        earliest_detection_hand=int(detector["earliest_detection_hand"]),
        detection_rate_gate=float(gates["detection_rate"]),
        median_delay_gate=int(gates["median_delay_hands"]),
        pre_switch_false_rate_gate=float(gates["pre_switch_false_detection_rate"]),
        no_switch_false_rate_gate=float(gates["no_switch_false_positive_rate"]),
        pressure_selection_rate_gate=float(gates["pressure_selection_rate"]),
        reflection_divergence_gate=float(gates["reflection_action_divergence"]),
        policy_calibration_path=Path(calibration["path"]),
        policy_calibration_sha256=str(calibration["sha256"]),
        pre_world=_world_from_mapping("pre_tag", worlds["pre"]),
        switch_world=_world_from_mapping("confirm_probe_fold", worlds["switch"]),
        proxy_worlds=tuple(
            _world_from_mapping(str(name), value)
            for name, value in worlds["proxy_hypotheses"].items()
        ),
    )


def _detector(config: Stage2A1Config, threshold: float) -> SignedConditionalEProcessDetector:
    return SignedConditionalEProcessDetector(
        reference_size=config.reference_size,
        block_size=config.block_size,
        e_value_threshold=threshold,
        alternative_delta=config.alternative_delta,
        alternative_concentration=config.alternative_concentration,
        minimum_direction_delta=config.minimum_direction_delta,
        maximum_blocks=config.maximum_blocks,
    )


class TraceCollectingAgent(PokerAgent):
    condition = "stage2a1_trace_collector"

    def __init__(self, name: str, seed: int, equity_samples: int) -> None:
        super().__init__(name, seed, hero_style(equity_samples))
        self.opponent_observations: list[tuple[int, OpponentObservation]] = []

    def observe_action(self, event: ActionEvent) -> None:
        super().observe_action(event)
        if event.actor != self.name:
            self.opponent_observations.append(
                (event.hand_index, OpponentObservation.from_event(event))
            )


class Stage2A1SimulationAgent(PokerAgent):
    condition = "stage2a1_simulation"

    def __init__(
        self,
        name: str,
        seed: int,
        config: Stage2A1Config,
        threshold: float,
        policy_lower_bound_bb100: float,
    ) -> None:
        super().__init__(name, seed, hero_style(config.equity_samples))
        self.detector = _detector(config, threshold)
        self.formation: deque[OpponentObservation] = deque(maxlen=config.reference_size)
        self.recent: deque[OpponentObservation] = deque(maxlen=48)
        self.generator = FrozenProxyGenerator(config.proxy_worlds)
        self.simulator = WorldSimulator(
            rollouts=config.simulation_rollouts,
            seed=seed + 41,
            equity_samples=config.simulation_equity_samples,
        )
        self.earliest_detection_hand = config.earliest_detection_hand
        self.policy_lower_bound_bb100 = policy_lower_bound_bb100
        self.detected_change_hand: int | None = None
        self.response_policy: str | None = None
        self.raw_simulator_policy: str | None = None
        self.raw_simulator_value = 0.0
        self.detector_trace: list[dict[str, Any]] = []

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
        self.recent.append(observation)
        update = self.detector.update(observation)
        if update.block_complete:
            self.detector_trace.append(
                {
                    "hand_index": event.hand_index,
                    **asdict(update),
                }
            )
        if (
            not update.change_detected
            or update.direction != "up"
            or event.hand_index < self.earliest_detection_hand
        ):
            return
        self.detected_change_hand = event.hand_index
        worlds = self.generator.generate(tuple(self.recent), ())
        results = self.simulator.evaluate(worlds, tuple(self.recent))
        self.raw_simulator_policy, self.raw_simulator_value = self.simulator.choose_response(
            results
        )
        self.response_policy = "pressure"

    def act(self, context: DecisionContext) -> Decision:
        metadata = {
            "adaptation_condition": self.condition,
            "detected_change_hand": self.detected_change_hand,
            "response_policy": self.response_policy,
            "raw_simulator_policy": self.raw_simulator_policy,
            "calibrated_lower_bound_bb100": self.policy_lower_bound_bb100,
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


def _environment(
    agents: Sequence[PokerAgent],
    seed: int,
    config: Stage2A1Config,
) -> HoldemEnvironment:
    return HoldemEnvironment(
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


def collect_tuning_trace(
    config: Stage2A1Config,
    *,
    scenario_id: str,
    post_world: OpponentWorld,
    noise: float,
    seed: int,
    mirror: int,
) -> dict[str, Any]:
    hero = TraceCollectingAgent("hero", seed * 23 + 1, config.equity_samples)
    opponent = ConditionalRegimeOpponent(
        "opponent",
        seed * 23 + 2,
        config.switch_hand,
        config.pre_world,
        post_world,
        noise,
    )
    agents = [hero, opponent] if mirror == 0 else [opponent, hero]
    _environment(agents, seed, config).play(config.hands)
    return {
        "scenario_id": scenario_id,
        "noise": noise,
        "seed": seed,
        "mirror": mirror,
        "observations": [
            {
                "hand_index": hand_index,
                "action": observation.action.value,
                "facing_bet": observation.facing_bet,
            }
            for hand_index, observation in hero.opponent_observations
        ],
    }


def evaluate_detector_trace(
    config: Stage2A1Config,
    trace: Mapping[str, Any],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detector = _detector(config, threshold)
    formation: list[OpponentObservation] = []
    detected_hand = None
    block_rows: list[dict[str, Any]] = []
    for item in trace["observations"]:
        observation = OpponentObservation(
            action=ActionType(item["action"]),
            facing_bet=bool(item["facing_bet"]),
        )
        if not detector.ready:
            formation.append(observation)
            if len(formation) == detector.reference_size:
                detector.fit_reference(formation)
            continue
        update = detector.update(observation)
        if update.block_complete:
            block_rows.append(
                {
                    "scenario_id": trace["scenario_id"],
                    "noise": trace["noise"],
                    "seed": trace["seed"],
                    "mirror": trace["mirror"],
                    "threshold": threshold,
                    "hand_index": item["hand_index"],
                    **asdict(update),
                }
            )
        if (
            detected_hand is None
            and update.change_detected
            and update.direction == "up"
            and int(item["hand_index"]) >= config.earliest_detection_hand
        ):
            detected_hand = int(item["hand_index"])
    return (
        {
            "scenario_id": trace["scenario_id"],
            "noise": trace["noise"],
            "seed": trace["seed"],
            "mirror": trace["mirror"],
            "threshold": threshold,
            "detected_change_hand": detected_hand,
            "detection_delay_hands": (
                None if detected_hand is None else detected_hand - config.switch_hand
            ),
        },
        block_rows,
    )


def summarize_threshold(
    config: Stage2A1Config,
    rows: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    selected = [row for row in rows if row["threshold"] == threshold]
    switch_cells: list[dict[str, Any]] = []
    for noise in config.noise_levels:
        scenario_id = f"probe_fold_e{int(noise * 100):02d}"
        values = [row for row in selected if row["scenario_id"] == scenario_id]
        valid = [
            row
            for row in values
            if row["detected_change_hand"] is not None
            and row["detected_change_hand"] >= config.switch_hand
        ]
        delays = [float(row["detection_delay_hands"]) for row in valid]
        switch_cells.append(
            {
                "scenario_id": scenario_id,
                "detection_rate": len(valid) / len(values),
                "median_delay_hands": statistics.median(delays) if delays else None,
                "pre_switch_false_rate": sum(
                    row["detected_change_hand"] is not None
                    and row["detected_change_hand"] < config.switch_hand
                    for row in values
                )
                / len(values),
            }
        )
    no_switch = [row for row in selected if row["scenario_id"].startswith("no_switch_")]
    false_rate = sum(row["detected_change_hand"] is not None for row in no_switch) / len(
        no_switch
    )
    gate_pass = bool(
        false_rate <= config.no_switch_false_rate_gate
        and all(
            cell["detection_rate"] >= config.detection_rate_gate
            and cell["median_delay_hands"] is not None
            and cell["median_delay_hands"] <= config.median_delay_gate
            and cell["pre_switch_false_rate"] <= config.pre_switch_false_rate_gate
            for cell in switch_cells
        )
    )
    detection_deficit = sum(
        max(config.detection_rate_gate - cell["detection_rate"], 0.0)
        for cell in switch_cells
    )
    delay_excess = sum(
        max(float(cell["median_delay_hands"] or config.hands) - config.median_delay_gate, 0.0)
        / config.hands
        for cell in switch_cells
    )
    false_excess = max(false_rate - config.no_switch_false_rate_gate, 0.0)
    pre_false_excess = sum(
        max(cell["pre_switch_false_rate"] - config.pre_switch_false_rate_gate, 0.0)
        for cell in switch_cells
    )
    return {
        "threshold": threshold,
        "switch_cells": switch_cells,
        "no_switch_false_positive_rate": false_rate,
        "gate_pass": gate_pass,
        "selection_penalty": (
            4.0 * false_excess
            + 4.0 * pre_false_excess
            + detection_deficit
            + delay_excess
        ),
    }


def select_threshold(threshold_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    passing = [summary for summary in threshold_summaries if summary["gate_pass"]]
    if passing:
        return min(passing, key=lambda value: float(value["threshold"]))
    return min(
        threshold_summaries,
        key=lambda value: (float(value["selection_penalty"]), float(value["threshold"])),
    )


def _make_validation_hero(
    condition: str,
    seed: int,
    config: Stage2A1Config,
    threshold: float,
    policy_lower_bound_bb100: float,
) -> PokerAgent:
    if condition == "baseline":
        agent = PokerAgent("hero", seed, hero_style(config.equity_samples))
        agent.condition = condition
        return agent
    if condition == "reflection":
        return ReflectionTrackerAgent("hero", seed, hero_style(config.equity_samples))
    if condition == "simulation":
        return Stage2A1SimulationAgent(
            "hero",
            seed,
            config,
            threshold,
            policy_lower_bound_bb100,
        )
    raise ValueError(f"Unknown condition: {condition}")


def run_validation_match(
    config: Stage2A1Config,
    *,
    scenario_id: str,
    post_world: OpponentWorld,
    noise: float,
    condition: str,
    seed: int,
    mirror: int,
    threshold: float,
    policy_lower_bound_bb100: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hero = _make_validation_hero(
        condition,
        seed * 29 + 1,
        config,
        threshold,
        policy_lower_bound_bb100,
    )
    opponent = ConditionalRegimeOpponent(
        "opponent",
        seed * 29 + 2,
        config.switch_hand,
        config.pre_world,
        post_world,
        noise,
    )
    agents = [hero, opponent] if mirror == 0 else [opponent, hero]
    records = _environment(agents, seed, config).play(config.hands)
    rewards = [record.rewards["hero"] for record in records]
    detected_hand = None
    response_policy = None
    raw_policy = None
    simulation_calls = 0
    trace_rows: list[dict[str, Any]] = []
    if isinstance(hero, Stage2A1SimulationAgent):
        detected_hand = hero.detected_change_hand
        response_policy = hero.response_policy
        raw_policy = hero.raw_simulator_policy
        simulation_calls = hero.simulator.calls
        trace_rows = [
            {
                "scenario_id": scenario_id,
                "noise": noise,
                "seed": seed,
                "mirror": mirror,
                **row,
            }
            for row in hero.detector_trace
        ]
    action_signature = "|".join(str(item["action"]) for item in hero.decision_log)
    return (
        {
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
            "raw_simulator_policy": raw_policy,
            "simulation_calls": simulation_calls,
            "action_signature": action_signature,
        },
        trace_rows,
    )


def summarize_validation(
    config: Stage2A1Config,
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
    policy_calibration_valid: bool,
) -> dict[str, Any]:
    by_key = {
        (row["scenario_id"], row["condition"], row["seed"], row["mirror"]): row
        for row in rows
    }
    switch_cells: list[dict[str, Any]] = []
    for noise in config.noise_levels:
        scenario_id = f"probe_fold_e{int(noise * 100):02d}"
        simulations = [
            row
            for row in rows
            if row["scenario_id"] == scenario_id and row["condition"] == "simulation"
        ]
        valid = [
            row
            for row in simulations
            if row["detected_change_hand"] is not None
            and row["detected_change_hand"] >= config.switch_hand
        ]
        delays = [float(row["detection_delay_hands"]) for row in valid]
        divergences = [
            _signature_divergence(
                by_key[(scenario_id, "baseline", seed, mirror)]["action_signature"],
                by_key[(scenario_id, "reflection", seed, mirror)]["action_signature"],
            )
            for seed in config.validation_seeds
            for mirror in MIRRORS
        ]
        payoff_effects: list[float] = []
        for seed in config.validation_seeds:
            paired = []
            for mirror in MIRRORS:
                simulation = float(
                    by_key[(scenario_id, "simulation", seed, mirror)]["post_switch_bb100"]
                )
                reflection = float(
                    by_key[(scenario_id, "reflection", seed, mirror)]["post_switch_bb100"]
                )
                paired.append(simulation - reflection)
            payoff_effects.append(statistics.fmean(paired))
        pressure_count = sum(row["response_policy"] == "pressure" for row in valid)
        cell = {
            "scenario_id": scenario_id,
            "matches": len(simulations),
            "detection_rate": len(valid) / len(simulations),
            "median_delay_hands": statistics.median(delays) if delays else None,
            "pre_switch_false_rate": sum(
                row["detected_change_hand"] is not None
                and row["detected_change_hand"] < config.switch_hand
                for row in simulations
            )
            / len(simulations),
            "pressure_selection_rate": pressure_count / len(simulations),
            "pressure_given_detection_rate": (
                pressure_count / len(valid) if valid else 0.0
            ),
            "raw_policy_counts": dict(
                sorted(Counter(row["raw_simulator_policy"] for row in valid).items())
            ),
            "mean_reflection_action_divergence": statistics.fmean(divergences),
            "simulation_minus_reflection_post_bb100": _interval(payoff_effects),
        }
        cell["gate_pass"] = bool(
            cell["detection_rate"] >= config.detection_rate_gate
            and cell["median_delay_hands"] is not None
            and cell["median_delay_hands"] <= config.median_delay_gate
            and cell["pre_switch_false_rate"] <= config.pre_switch_false_rate_gate
            and cell["pressure_selection_rate"] >= config.pressure_selection_rate_gate
            and cell["mean_reflection_action_divergence"]
            >= config.reflection_divergence_gate
        )
        switch_cells.append(cell)
    no_switch = [row for row in rows if row["scenario_id"].startswith("no_switch_")]
    false_rate = sum(row["detected_change_hand"] is not None for row in no_switch) / len(
        no_switch
    )
    no_switch_gate = false_rate <= config.no_switch_false_rate_gate
    return {
        "protocol": STAGE2A1_PROTOCOL,
        "claim_boundary": (
            "fresh_detector_validation_and_proxy_simulation_only_not_live_llm_or_formal_profit"
        ),
        "selected_threshold": threshold,
        "policy_calibration_valid": policy_calibration_valid,
        "switch_cells": switch_cells,
        "no_switch": {
            "matches": len(no_switch),
            "false_positive_rate": false_rate,
            "gate_pass": no_switch_gate,
        },
        "stage2b_ready": bool(
            policy_calibration_valid
            and no_switch_gate
            and all(cell["gate_pass"] for cell in switch_cells)
        ),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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
        Path(__file__).with_name("stage2a_calibration.py"),
        config_path,
    )
    hasher = hashlib.sha256()
    for source_path in source_paths:
        hasher.update(str(source_path.resolve()).encode())
        hasher.update(source_path.read_bytes())
    return {
        "source_tree_sha256": hasher.hexdigest(),
        "source_files": [str(path.resolve()) for path in source_paths],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }


def _load_policy_calibration(config: Stage2A1Config) -> dict[str, Any]:
    path = config.policy_calibration_path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != config.policy_calibration_sha256:
        raise ValueError("policy calibration hash does not match frozen config")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("pressure_allowed"):
        raise ValueError("frozen policy calibration does not allow pressure")
    return payload


def run_stage2a1(
    config: Stage2A1Config,
    output_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    policy_calibration = _load_policy_calibration(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "protocol": STAGE2A1_PROTOCOL,
        "claim_boundary": "tuning_then_fresh_validation_no_live_llm",
        "config": asdict(config),
        "tuning_and_validation_seed_sets_disjoint": not bool(
            set(config.tuning_seeds) & set(config.validation_seeds)
        ),
        "threshold_selection_rule": (
            "smallest passing threshold; if none pass, minimum preregistered weighted penalty"
        ),
        "statistical_unit": "seed-level mean across two seat mirrors",
        "frozen_before_stochastic_run": True,
        **_source_provenance(config_path),
    }
    (output_dir / "PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2, default=str),
        encoding="utf-8",
    )

    tuning_traces: list[dict[str, Any]] = []
    for noise in config.noise_levels:
        for scenario_id, post_world in (
            (f"probe_fold_e{int(noise * 100):02d}", config.switch_world),
            (f"no_switch_e{int(noise * 100):02d}", config.pre_world),
        ):
            for seed in config.tuning_seeds:
                for mirror in MIRRORS:
                    tuning_traces.append(
                        collect_tuning_trace(
                            config,
                            scenario_id=scenario_id,
                            post_world=post_world,
                            noise=noise,
                            seed=seed,
                            mirror=mirror,
                        )
                    )
                print(f"tuning trace scenario={scenario_id} seed={seed} complete", flush=True)
    _write_jsonl(output_dir / "tuning_observations.jsonl", tuning_traces)

    tuning_rows: list[dict[str, Any]] = []
    tuning_block_rows: list[dict[str, Any]] = []
    threshold_summaries: list[dict[str, Any]] = []
    for threshold in config.threshold_candidates:
        for trace in tuning_traces:
            row, blocks = evaluate_detector_trace(config, trace, threshold)
            tuning_rows.append(row)
            tuning_block_rows.extend(blocks)
        threshold_summaries.append(summarize_threshold(config, tuning_rows, threshold))
    selected = select_threshold(threshold_summaries)
    selected_threshold = float(selected["threshold"])
    tuning_summary = {
        "threshold_candidates": threshold_summaries,
        "selected": selected,
        "tuning_gate_pass": bool(selected["gate_pass"]),
    }
    (output_dir / "tuning_summary.json").write_text(
        json.dumps(tuning_summary, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "tuning_rows.csv", tuning_rows)
    _write_jsonl(output_dir / "tuning_detector_trace.jsonl", tuning_block_rows)
    print(
        f"selected detector threshold={selected_threshold} "
        f"tuning_gate_pass={selected['gate_pass']}",
        flush=True,
    )

    validation_rows: list[dict[str, Any]] = []
    validation_trace: list[dict[str, Any]] = []
    lower_bound = float(policy_calibration["conservative_lower_bound_bb100"])
    for noise in config.noise_levels:
        scenario_id = f"probe_fold_e{int(noise * 100):02d}"
        for seed in config.validation_seeds:
            for mirror in MIRRORS:
                for condition in SWITCH_CONDITIONS:
                    row, traces = run_validation_match(
                        config,
                        scenario_id=scenario_id,
                        post_world=config.switch_world,
                        noise=noise,
                        condition=condition,
                        seed=seed,
                        mirror=mirror,
                        threshold=selected_threshold,
                        policy_lower_bound_bb100=lower_bound,
                    )
                    validation_rows.append(row)
                    validation_trace.extend(traces)
            print(f"validation switch scenario={scenario_id} seed={seed} complete", flush=True)
        control_id = f"no_switch_e{int(noise * 100):02d}"
        for seed in config.validation_seeds:
            for mirror in MIRRORS:
                row, traces = run_validation_match(
                    config,
                    scenario_id=control_id,
                    post_world=config.pre_world,
                    noise=noise,
                    condition="simulation",
                    seed=seed,
                    mirror=mirror,
                    threshold=selected_threshold,
                    policy_lower_bound_bb100=lower_bound,
                )
                validation_rows.append(row)
                validation_trace.extend(traces)
            print(f"validation control scenario={control_id} seed={seed} complete", flush=True)
    _write_csv(output_dir / "validation_rows.csv", validation_rows)
    _write_jsonl(output_dir / "validation_detector_trace.jsonl", validation_trace)
    summary = summarize_validation(
        config,
        validation_rows,
        threshold=selected_threshold,
        policy_calibration_valid=True,
    )
    summary["tuning_gate_pass"] = bool(selected["gate_pass"])
    summary["stage2b_ready"] = bool(summary["stage2b_ready"] and selected["gate_pass"])
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2A.1 e-process calibration")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning-seed-count", type=int)
    parser.add_argument("--validation-seed-count", type=int)
    args = parser.parse_args()
    config = load_stage2a1_config(
        args.config,
        tuning_seed_count=args.tuning_seed_count,
        validation_seed_count=args.validation_seed_count,
    )
    run_stage2a1(config, args.output, args.config)


if __name__ == "__main__":
    main()
