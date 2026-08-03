from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .agents import AgentStyle, PokerAgent
from .equity import estimate_equity
from .environment import EnvironmentConfig, HoldemEnvironment
from .models import ActionEvent, ActionType, Decision, DecisionContext

_ACTIONS = (ActionType.FOLD, ActionType.CHECK_CALL, ActionType.RAISE)


@dataclass(frozen=True)
class OpponentWorld:
    """A compact action-distribution hypothesis used by the adaptation layer.

    This is intentionally a decision-layer model, not a full poker strategy solver. The
    experiment asks whether hypothesis generation plus counterfactual simulation can
    recover faster after an opponent policy switch.
    """

    name: str
    fold_probability: float
    call_probability: float
    raise_probability: float
    prior: float = 1.0
    rationale: str = ""

    def __post_init__(self) -> None:
        values = (
            self.fold_probability,
            self.call_probability,
            self.raise_probability,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("World probabilities must be non-negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("World action probabilities must sum to one")
        if self.prior <= 0.0:
            raise ValueError("World prior must be positive")

    def probability(self, action: ActionType, floor: float = 1e-4) -> float:
        mapping = {
            ActionType.FOLD: self.fold_probability,
            ActionType.CHECK_CALL: self.call_probability,
            ActionType.RAISE: self.raise_probability,
        }
        return max(mapping[action], floor)


@dataclass(frozen=True)
class SurpriseUpdate:
    surprise_score: float
    mean_surprise: float
    change_detected: bool
    observations: int


class SurpriseDetector:
    """Sliding-window normalized negative-log-likelihood change detector."""

    def __init__(
        self,
        window_size: int = 24,
        min_observations: int = 12,
        threshold: float = 0.54,
        probability_floor: float = 1e-4,
        cooldown_observations: int = 12,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least two")
        if not 1 <= min_observations <= window_size:
            raise ValueError("min_observations must be within the window")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.window_size = window_size
        self.min_observations = min_observations
        self.threshold = threshold
        self.probability_floor = probability_floor
        self.cooldown_observations = cooldown_observations
        self._scores: deque[float] = deque(maxlen=window_size)
        self._updates = 0
        self._last_detection = -10**9

    def reset(self) -> None:
        self._scores.clear()
        self._last_detection = -10**9

    def update(
        self,
        predicted_probability: float,
        expected_surprise: float = 0.0,
    ) -> SurpriseUpdate:
        probability = min(max(predicted_probability, self.probability_floor), 1.0)
        max_nll = -math.log(self.probability_floor)
        raw_score = min(-math.log(probability) / max_nll, 1.0)
        score = max(0.0, raw_score - max(expected_surprise, 0.0))
        self._scores.append(score)
        self._updates += 1
        mean_score = sum(self._scores) / len(self._scores)
        cooled_down = self._updates - self._last_detection >= self.cooldown_observations
        detected = (
            len(self._scores) >= self.min_observations
            and mean_score >= self.threshold
            and cooled_down
        )
        if detected:
            self._last_detection = self._updates
        return SurpriseUpdate(score, mean_score, detected, len(self._scores))


class HypothesisGenerator(Protocol):
    calls: int

    def generate(
        self,
        observations: Sequence[ActionType],
        current_worlds: Sequence[OpponentWorld],
    ) -> list[OpponentWorld]: ...


class HeuristicHypothesisGenerator:
    """Deterministic offline hypothesis generator for CI and ablations."""

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        self.calls = 0

    @staticmethod
    def _normalized(
        name: str,
        fold: float,
        call: float,
        raise_: float,
        rationale: str,
    ) -> OpponentWorld:
        total = fold + call + raise_
        return OpponentWorld(
            name=name,
            fold_probability=fold / total,
            call_probability=call / total,
            raise_probability=raise_ / total,
            rationale=rationale,
        )

    def generate(
        self,
        observations: Sequence[ActionType],
        current_worlds: Sequence[OpponentWorld],
    ) -> list[OpponentWorld]:
        del current_worlds
        self.calls += 1
        counts = Counter(observations)
        denominator = len(observations) + self.smoothing * len(_ACTIONS)
        empirical = {
            action: (counts[action] + self.smoothing) / denominator for action in _ACTIONS
        }
        worlds = [
            OpponentWorld(
                name="empirical_shift",
                fold_probability=empirical[ActionType.FOLD],
                call_probability=empirical[ActionType.CHECK_CALL],
                raise_probability=empirical[ActionType.RAISE],
                prior=1.5,
                rationale="Smoothed action frequencies in the surprise window.",
            ),
            self._normalized(
                "aggressive_switch",
                0.13,
                0.42,
                0.45,
                "A persistent increase in raises is consistent with a LAG or tilt regime.",
            ),
            self._normalized(
                "passive_switch",
                0.24,
                0.67,
                0.09,
                "A call-heavy, low-raise regime is consistent with passive play.",
            ),
            self._normalized(
                "tight_switch",
                0.48,
                0.39,
                0.13,
                "A fold-heavy regime is consistent with a tighter range.",
            ),
        ]
        unique: dict[str, OpponentWorld] = {}
        for world in worlds:
            unique[world.name] = world
        return list(unique.values())


HYPOTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["worlds"],
    "properties": {
        "worlds": {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "fold_probability",
                    "call_probability",
                    "raise_probability",
                    "prior",
                    "rationale",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "fold_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "call_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "raise_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "prior": {"type": "number", "exclusiveMinimum": 0.0},
                    "rationale": {"type": "string"},
                },
            },
        }
    },
}


class ProviderHypothesisGenerator:
    """Bounded provider-backed world generator using the repository LLMProvider contract."""

    def __init__(self, provider: Any) -> None:
        if not hasattr(provider, "structured"):
            raise TypeError("provider must implement structured(...)")
        self.provider = provider
        self.calls = 0
        self.last_response: Any | None = None

    def generate(
        self,
        observations: Sequence[ActionType],
        current_worlds: Sequence[OpponentWorld],
    ) -> list[OpponentWorld]:
        self.calls += 1
        counts = Counter(action.value for action in observations)
        state = {
            "recent_action_counts": dict(counts),
            "recent_actions": [action.value for action in observations],
            "current_worlds": [asdict(world) for world in current_worlds],
            "constraints": {
                "probabilities_must_sum_to_one": True,
                "maximum_worlds": 4,
                "purpose": "generate falsifiable opponent-regime hypotheses",
            },
        }
        response = self.provider.structured(
            instructions=(
                "Generate two to four concise, falsifiable opponent-policy hypotheses for a "
                "poker action stream that has become surprising. Use only fold, check_call, and "
                "raise frequencies. Probabilities for each world must sum to one. Do not choose "
                "a poker action and do not provide hidden chain-of-thought."
            ),
            state=state,
            schema_name="opponent_regime_hypotheses",
            schema=HYPOTHESIS_SCHEMA,
        )
        self.last_response = response
        worlds: list[OpponentWorld] = []
        for item in response.payload["worlds"]:
            total = (
                float(item["fold_probability"])
                + float(item["call_probability"])
                + float(item["raise_probability"])
            )
            if total <= 0.0:
                continue
            worlds.append(
                OpponentWorld(
                    name=str(item["name"]),
                    fold_probability=float(item["fold_probability"]) / total,
                    call_probability=float(item["call_probability"]) / total,
                    raise_probability=float(item["raise_probability"]) / total,
                    prior=float(item["prior"]),
                    rationale=str(item["rationale"]),
                )
            )
        if len(worlds) < 2:
            raise ValueError("Provider returned fewer than two valid hypotheses")
        return worlds


@dataclass(frozen=True)
class SimulationResult:
    world_name: str
    posterior: float
    best_response: str
    expected_bb_per_decision: float
    response_values: dict[str, float]


class WorldSimulator:
    """Monte Carlo counterfactual evaluator over compact opponent worlds."""

    PAYOFFS: dict[str, dict[ActionType, float]] = {
        "pressure": {
            ActionType.FOLD: 0.42,
            ActionType.CHECK_CALL: 0.08,
            ActionType.RAISE: -0.34,
        },
        "balanced": {
            ActionType.FOLD: 0.18,
            ActionType.CHECK_CALL: 0.12,
            ActionType.RAISE: -0.08,
        },
        "bluff_catch": {
            ActionType.FOLD: 0.05,
            ActionType.CHECK_CALL: 0.06,
            ActionType.RAISE: 0.20,
        },
    }

    def __init__(self, rollouts: int = 2000, seed: int = 0) -> None:
        if rollouts < 1:
            raise ValueError("rollouts must be positive")
        self.rollouts = rollouts
        self.rng = random.Random(seed)
        self.calls = 0

    @staticmethod
    def _sample_action(world: OpponentWorld, value: float) -> ActionType:
        if value < world.fold_probability:
            return ActionType.FOLD
        if value < world.fold_probability + world.call_probability:
            return ActionType.CHECK_CALL
        return ActionType.RAISE

    def evaluate(
        self,
        worlds: Sequence[OpponentWorld],
        observations: Sequence[ActionType],
    ) -> list[SimulationResult]:
        if not worlds:
            raise ValueError("At least one world is required")
        self.calls += 1
        log_weights: list[float] = []
        for world in worlds:
            value = math.log(world.prior)
            for action in observations:
                value += math.log(world.probability(action))
            log_weights.append(value)
        anchor = max(log_weights)
        weights = [math.exp(value - anchor) for value in log_weights]
        total_weight = sum(weights)
        posteriors = [value / total_weight for value in weights]

        results: list[SimulationResult] = []
        for world, posterior in zip(worlds, posteriors, strict=True):
            totals = {name: 0.0 for name in self.PAYOFFS}
            for _ in range(self.rollouts):
                action = self._sample_action(world, self.rng.random())
                for response, payoff in self.PAYOFFS.items():
                    totals[response] += payoff[action]
            values = {name: total / self.rollouts for name, total in totals.items()}
            best = max(values, key=values.get)
            results.append(
                SimulationResult(
                    world_name=world.name,
                    posterior=posterior,
                    best_response=best,
                    expected_bb_per_decision=values[best],
                    response_values=values,
                )
            )
        return sorted(results, key=lambda item: item.posterior, reverse=True)

    @staticmethod
    def choose_response(results: Sequence[SimulationResult]) -> tuple[str, float]:
        if not results:
            return "balanced", 0.0
        aggregate: dict[str, float] = Counter()
        for result in results:
            for response, value in result.response_values.items():
                aggregate[response] += result.posterior * value
        best = max(aggregate, key=aggregate.get)
        return best, aggregate[best]


@dataclass
class AdaptationState:
    worlds: list[OpponentWorld]
    posterior: dict[str, float]
    response_policy: str = "balanced"
    expected_bb_per_decision: float = 0.0
    detected_changes: list[int] = field(default_factory=list)
    last_surprise: float = 0.0
    mean_surprise: float = 0.0


DEFAULT_WORLD = OpponentWorld(
    name="stable_tag",
    fold_probability=0.31,
    call_probability=0.56,
    raise_probability=0.13,
    rationale="Initial balanced prior before enough opponent actions are observed.",
)


class ReflectionTrackerAgent(PokerAgent):
    """Reflection-only control: smooth action frequencies without change-point resets."""

    condition = "reflection_tracker"

    def __init__(self, name: str, seed: int, style: AgentStyle | None = None) -> None:
        super().__init__(name, seed, style)
        self.action_counts: Counter[ActionType] = Counter({action: 1.0 for action in _ACTIONS})

    def observe_action(self, event: ActionEvent) -> None:
        super().observe_action(event)
        if event.actor != self.name:
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
    """Surprise-triggered hypothesis generation and simulated response selection."""

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
        observation_window: int = 32,
        formation_observations: int = 24,
    ) -> None:
        super().__init__(name, seed, style)
        self.opponent_name = opponent_name
        self.detector = detector or SurpriseDetector()
        self.generator = generator or HeuristicHypothesisGenerator()
        self.simulator = simulator or WorldSimulator(seed=seed + 7103)
        if formation_observations < 3 or formation_observations > observation_window:
            raise ValueError("formation_observations must be between 3 and observation_window")
        self.formation_observations = formation_observations
        self.formation_complete = False
        self.observations: deque[ActionType] = deque(maxlen=observation_window)
        self.state = AdaptationState(
            worlds=[DEFAULT_WORLD],
            posterior={DEFAULT_WORLD.name: 1.0},
        )

    def _mixture_probability(self, action: ActionType) -> float:
        return sum(
            self.state.posterior.get(world.name, 0.0) * world.probability(action)
            for world in self.state.worlds
        )

    def _expected_surprise(self) -> float:
        max_nll = -math.log(self.detector.probability_floor)
        return sum(
            self._mixture_probability(action)
            * (-math.log(max(self._mixture_probability(action), self.detector.probability_floor)))
            / max_nll
            for action in _ACTIONS
        )

    def _finish_formation(self) -> None:
        counts = Counter(self.observations)
        denominator = len(self.observations) + len(_ACTIONS)
        world = OpponentWorld(
            name="formation_empirical",
            fold_probability=(counts[ActionType.FOLD] + 1.0) / denominator,
            call_probability=(counts[ActionType.CHECK_CALL] + 1.0) / denominator,
            raise_probability=(counts[ActionType.RAISE] + 1.0) / denominator,
            rationale="Smoothed opponent action frequencies from the formation period.",
        )
        self.state.worlds = [world]
        self.state.posterior = {world.name: 1.0}
        self.formation_complete = True
        self.detector.reset()

    def _refresh_worlds(self, hand_index: int) -> None:
        worlds = self.generator.generate(tuple(self.observations), tuple(self.state.worlds))
        results = self.simulator.evaluate(worlds, tuple(self.observations))
        response, expected_value = self.simulator.choose_response(results)
        self.state.worlds = list(worlds)
        self.state.posterior = {result.world_name: result.posterior for result in results}
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
        self.observations.append(event.action)
        if not self.formation_complete:
            if len(self.observations) >= self.formation_observations:
                self._finish_formation()
            return
        predicted = self._mixture_probability(event.action)
        update = self.detector.update(predicted, self._expected_surprise())
        self.state.last_surprise = update.surprise_score
        self.state.mean_surprise = update.mean_surprise
        if update.change_detected:
            self._refresh_worlds(event.hand_index)

    def _equity_adjustment(self) -> float:
        if self.state.response_policy == "bluff_catch":
            return 0.125
        if self.state.response_policy == "pressure":
            return -0.015
        return 0.0

    def act(self, context: DecisionContext) -> Decision:
        if not self.state.detected_changes:
            return self._policy(
                context,
                reasoning_depth=1,
                metadata={
                    "adaptation_condition": self.condition,
                    "response_policy": "balanced",
                    "expected_bb_per_decision": 0.0,
                    "surprise_score": self.state.last_surprise,
                    "mean_surprise": self.state.mean_surprise,
                    "detected_changes": (),
                    "world_posterior": dict(self.state.posterior),
                    "hypothesis_calls": self.generator.calls,
                    "simulation_calls": self.simulator.calls,
                },
            )
        equity = estimate_equity(
            context.hole_cards,
            context.board,
            max(1, context.active_players - 1),
            self.rng,
            self.style.equity_samples,
        )
        adjusted_equity = min(max(equity + self._equity_adjustment(), 0.0), 1.0)
        pot_odds = context.to_call / max(context.pot + context.to_call, 1e-9)
        can_raise = ActionType.RAISE in context.legal_actions
        if self.state.response_policy == "pressure":
            aggression = min(self.style.aggression + 0.18, 0.95)
        elif self.state.response_policy == "bluff_catch":
            aggression = max(self.style.aggression - 0.08, 0.05)
        else:
            aggression = self.style.aggression

        if context.to_call > 0:
            if can_raise and adjusted_equity > 0.79 - 0.10 * aggression:
                action = ActionType.RAISE
            elif adjusted_equity + 0.02 >= pot_odds - self.style.risk_margin:
                action = ActionType.CHECK_CALL
            else:
                action = ActionType.FOLD
        else:
            action = (
                ActionType.RAISE
                if can_raise and adjusted_equity > 0.66 - 0.18 * aggression
                else ActionType.CHECK_CALL
            )
        metadata = {
            "adaptation_condition": self.condition,
            "response_policy": self.state.response_policy,
            "expected_bb_per_decision": self.state.expected_bb_per_decision,
            "surprise_score": self.state.last_surprise,
            "mean_surprise": self.state.mean_surprise,
            "detected_changes": tuple(self.state.detected_changes),
            "world_posterior": dict(self.state.posterior),
            "hypothesis_calls": self.generator.calls,
            "simulation_calls": self.simulator.calls,
        }
        decision = Decision(
            action=action,
            raise_scale=0.62 if self.state.response_policy == "pressure" else 0.52,
            equity=equity,
            reasoning_depth=2 if self.state.detected_changes else 1,
            reasoning_ops=7 if self.state.detected_changes else 3,
            metadata=metadata,
        )
        self.decision_log.append(
            {
                "hand_index": context.hand_index,
                "street": context.street.value,
                "action": action.value,
                "equity": equity,
                **metadata,
            }
        )
        return decision

    def snapshot(self) -> dict[str, Any]:
        return {
            **super().snapshot(),
            "response_policy": self.state.response_policy,
            "world_posterior": dict(self.state.posterior),
            "detected_changes": tuple(self.state.detected_changes),
            "hypothesis_calls": self.generator.calls,
            "simulation_calls": self.simulator.calls,
        }


class RegimeSwitchingOpponent(PokerAgent):
    """Opponent with a frozen TAG-to-LAG switch at a known experimental hand."""

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
        shifted = context.hand_index >= self.switch_hand
        random_value = self.rng.random()
        if not shifted:
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
            raise_scale = 0.64
            regime = "tag"
        else:
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
            raise_scale = 0.54
            regime = "lag"
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


def _make_hero(condition: str, seed: int, opponent_name: str, equity_samples: int) -> PokerAgent:
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
            opponent_name=opponent_name,
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
    conditions = ("baseline", "reflection", "reflection_simulation")
    for condition in conditions:
        for seed in config.seeds:
            for mirror in (0, 1):
                hero_seed = seed * 17 + 1
                opponent_seed = seed * 17 + 2
                hero = _make_hero(condition, hero_seed, "opponent", config.equity_samples)
                opponent = RegimeSwitchingOpponent(
                    "opponent",
                    opponent_seed,
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
                records = environment.play(config.hands)
                rewards = [record.rewards["hero"] for record in records]
                pre = sum(rewards[: config.switch_hand])
                post = sum(rewards[config.switch_hand :])
                post_hands = config.hands - config.switch_hand
                detected_change_hand: int | None = None
                hypothesis_calls = 0
                simulation_calls = 0
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
                        condition=condition,
                        seed=seed,
                        mirror=mirror,
                        total_reward_bb=sum(rewards),
                        pre_switch_reward_bb=pre,
                        post_switch_reward_bb=post,
                        post_switch_bb100=100.0 * post / post_hands,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    row_payload = [asdict(row) for row in rows]
    with (output_dir / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_payload[0]))
        writer.writeheader()
        writer.writerows(row_payload)
    summary = summarize_regime_experiment(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
