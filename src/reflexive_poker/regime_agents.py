from __future__ import annotations

import math
import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from .agents import AgentStyle, PokerAgent
from .equity import estimate_equity
from .models import ActionEvent, ActionType, Decision, DecisionContext
from .regime_detection import (
    DEFAULT_WORLD,
    HeuristicHypothesisGenerator,
    HypothesisGenerator,
    OpponentObservation,
    OpponentWorld,
    SurpriseDetector,
    empirical_world,
)
from .regime_simulation import WorldSimulator, response_policy_decision

_ACTIONS = (ActionType.FOLD, ActionType.CHECK_CALL, ActionType.RAISE)


@dataclass
class AdaptationState:
    worlds: list[OpponentWorld]
    posterior: dict[str, float]
    response_policy: str = "balanced"
    expected_bb_per_decision: float = 0.0
    detected_changes: list[int] = field(default_factory=list)
    last_surprise: float = 0.0
    mean_surprise: float = 0.0


class ReflectionTrackerAgent(PokerAgent):
    """Reflection-only control with recency-weighted unconditional frequencies."""

    condition = "reflection_tracker"

    def __init__(self, name: str, seed: int, style: AgentStyle | None = None) -> None:
        super().__init__(name, seed, style)
        self.action_counts: Counter[ActionType] = Counter(
            {action: 1.0 for action in _ACTIONS}
        )

    def observe_action(self, event: ActionEvent) -> None:
        super().observe_action(event)
        if event.actor == self.name:
            return
        for action in _ACTIONS:
            self.action_counts[action] *= 0.985
        self.action_counts[event.action] += 1.0

    @property
    def estimated_raise_probability(self) -> float:
        return self.action_counts[ActionType.RAISE] / sum(self.action_counts.values())

    def act(self, context: DecisionContext) -> Decision:
        raise_probability = self.estimated_raise_probability
        shift = 0.10 if raise_probability < 0.13 else (-0.03 if raise_probability > 0.34 else 0.0)
        return self._policy(
            context,
            aggression_shift=shift,
            reasoning_depth=1,
            metadata={
                "adaptation_condition": self.condition,
                "estimated_raise_probability": raise_probability,
                "response_policy": "frequency_adjustment",
            },
        )


class SimulationEnhancedReflectionAgent(PokerAgent):
    """Calibrated change detection plus candidate-world full-hand rollout."""

    condition = "reflection_simulation"

    def __init__(
        self,
        name: str,
        seed: int,
        style: AgentStyle | None = None,
        *,
        opponent_name: str | None = None,
        detector: SurpriseDetector | None = None,
        generator: HypothesisGenerator | None = None,
        simulator: WorldSimulator | None = None,
        observation_window: int = 96,
        formation_observations: int = 48,
        calibration_observations: int = 32,
        calibration_sigma: float = 1.5,
        calibration_margin: float = 0.005,
    ) -> None:
        super().__init__(name, seed, style)
        if observation_window < formation_observations:
            raise ValueError("observation_window must cover formation_observations")
        if formation_observations < 3:
            raise ValueError("formation_observations must be at least three")
        if calibration_observations < 0:
            raise ValueError("calibration_observations must be non-negative")
        self.opponent_name = opponent_name
        self.detector = detector or SurpriseDetector()
        self.generator = generator or HeuristicHypothesisGenerator()
        self.simulator = simulator or WorldSimulator(seed=seed + 7_103)
        self.formation_observations = formation_observations
        self.formation_complete = False
        self.observations: deque[OpponentObservation] = deque(maxlen=observation_window)
        self.state = AdaptationState([DEFAULT_WORLD], {DEFAULT_WORLD.name: 1.0})
        self.calibration_observations = calibration_observations
        self.calibration_sigma = calibration_sigma
        self.calibration_margin = calibration_margin
        self.calibration_scores: list[float] = []
        self.calibration_complete = calibration_observations == 0

    def _mixture_probability(self, observation: OpponentObservation) -> float:
        return sum(
            self.state.posterior.get(world.name, 0.0) * world.probability(observation)
            for world in self.state.worlds
        )

    def _expected_surprise(self, facing_bet: bool) -> float:
        denominator = -math.log(self.detector.probability_floor)
        probabilities = {
            action: sum(
                self.state.posterior.get(world.name, 0.0)
                * world.action_probabilities(facing_bet)[action]
                for world in self.state.worlds
            )
            for action in _ACTIONS
        }
        return sum(
            probability
            * -math.log(max(probability, self.detector.probability_floor))
            / denominator
            for probability in probabilities.values()
            if probability > 0.0
        )

    def _finish_formation(self) -> None:
        world = empirical_world(
            tuple(self.observations),
            name="formation_empirical",
            prior=1.0,
        )
        self.state.worlds = [world]
        self.state.posterior = {world.name: 1.0}
        self.formation_complete = True
        self.detector.reset()

    def _calibrate(self, observation: OpponentObservation) -> None:
        score = self.detector.score(
            self._mixture_probability(observation),
            self._expected_surprise(observation.facing_bet),
        )
        self.calibration_scores.append(score)
        self.state.last_surprise = score
        self.state.mean_surprise = statistics.fmean(self.calibration_scores)
        if len(self.calibration_scores) < self.calibration_observations:
            return
        spread = (
            statistics.pstdev(self.calibration_scores)
            if len(self.calibration_scores) > 1
            else 0.0
        )
        self.detector.threshold = max(
            self.detector.threshold,
            statistics.fmean(self.calibration_scores)
            + self.calibration_sigma * spread
            + self.calibration_margin,
        )
        self.calibration_complete = True
        self.detector.reset()

    def _refresh_worlds(self, hand_index: int) -> None:
        worlds = self.generator.generate(tuple(self.observations), tuple(self.state.worlds))
        results = self.simulator.evaluate(worlds, tuple(self.observations))
        response, expected_value = self.simulator.choose_response(results)
        self.state.worlds = worlds
        self.state.posterior = {
            result.world_name: result.posterior for result in results
        }
        self.state.response_policy = response
        self.state.expected_bb_per_decision = expected_value
        self.state.detected_changes.append(hand_index)
        self.detector.reset()

    def observe_action(self, event: ActionEvent) -> None:
        super().observe_action(event)
        if event.actor == self.name:
            return
        if self.opponent_name is not None and event.actor != self.opponent_name:
            return
        observation = OpponentObservation.from_event(event)
        self.observations.append(observation)
        if not self.formation_complete:
            if len(self.observations) >= self.formation_observations:
                self._finish_formation()
            return
        if not self.calibration_complete:
            self._calibrate(observation)
            return
        update = self.detector.update(
            self._mixture_probability(observation),
            self._expected_surprise(observation.facing_bet),
        )
        self.state.last_surprise = update.surprise_score
        self.state.mean_surprise = update.mean_surprise
        if update.change_detected:
            self._refresh_worlds(event.hand_index)

    def _metadata(self) -> dict[str, Any]:
        return {
            "adaptation_condition": self.condition,
            "response_policy": self.state.response_policy,
            "expected_bb_per_decision": self.state.expected_bb_per_decision,
            "surprise_score": self.state.last_surprise,
            "mean_surprise": self.state.mean_surprise,
            "surprise_threshold": self.detector.threshold,
            "calibration_complete": self.calibration_complete,
            "detected_changes": tuple(self.state.detected_changes),
            "world_posterior": dict(self.state.posterior),
            "hypothesis_calls": self.generator.calls,
            "simulation_calls": self.simulator.calls,
            "simulated_hands": self.simulator.simulated_hands,
            "simulation_unit": "full_hand",
        }

    def act(self, context: DecisionContext) -> Decision:
        if not self.state.detected_changes:
            return self._policy(context, reasoning_depth=1, metadata=self._metadata())
        metadata = self._metadata()
        decision = response_policy_decision(
            self,
            context,
            self.state.response_policy,
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

    def snapshot(self) -> dict[str, Any]:
        return {**super().snapshot(), **self._metadata()}


class RegimeSwitchingOpponent(PokerAgent):
    condition = "regime_switching_opponent"

    def __init__(
        self,
        name: str,
        seed: int,
        switch_hand: int,
        style: AgentStyle | None = None,
    ) -> None:
        super().__init__(name, seed, style)
        self.switch_hand = switch_hand

    def act(self, context: DecisionContext) -> Decision:
        equity = estimate_equity(
            context.hole_cards,
            context.board,
            max(1, context.active_players - 1),
            self.rng,
            self.style.equity_samples,
        )
        pot_odds = context.to_call / max(context.pot + context.to_call, 1e-9)
        can_raise = ActionType.RAISE in context.legal_actions
        random_value = self.rng.random()
        shifted = context.hand_index >= self.switch_hand
        if shifted:
            if context.to_call > 0:
                if can_raise and (equity > 0.42 or random_value < 0.30):
                    action = ActionType.RAISE
                elif equity > pot_odds - 0.14 or random_value < 0.30:
                    action = ActionType.CHECK_CALL
                else:
                    action = ActionType.FOLD
            else:
                action = (
                    ActionType.RAISE
                    if can_raise and (equity > 0.32 or random_value < 0.36)
                    else ActionType.CHECK_CALL
                )
            raise_scale, regime = 0.54, "lag"
        else:
            if context.to_call > 0:
                if can_raise and equity > 0.90:
                    action = ActionType.RAISE
                elif equity > pot_odds + 0.12:
                    action = ActionType.CHECK_CALL
                else:
                    action = ActionType.FOLD
            else:
                action = (
                    ActionType.RAISE
                    if can_raise and (equity > 0.82 or random_value < 0.01)
                    else ActionType.CHECK_CALL
                )
            raise_scale, regime = 0.64, "tag"
        decision = Decision(
            action=action,
            raise_scale=raise_scale,
            equity=equity,
            reasoning_depth=1,
            reasoning_ops=3,
            metadata={"opponent_regime": regime},
        )
        self.decision_log.append(
            {
                "hand_index": context.hand_index,
                "street": context.street.value,
                "action": action.value,
                "equity": equity,
                "opponent_regime": regime,
            }
        )
        return decision
