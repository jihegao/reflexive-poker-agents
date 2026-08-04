from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .models import ActionEvent, ActionType


@dataclass(frozen=True)
class OpponentObservation:
    """One opponent action with the legal-action context needed for interpretation."""

    action: ActionType
    facing_bet: bool

    @classmethod
    def from_event(cls, event: ActionEvent) -> OpponentObservation:
        return cls(action=event.action, facing_bet=event.to_call > 1e-9)


@dataclass(frozen=True)
class OpponentWorld:
    """Compact conditional opponent policy used for detection and rollout.

    The three parameters are identifiable from public action histories without
    conflating checks with calls or treating an illegal fold as evidence.
    """

    name: str
    open_raise_probability: float
    fold_vs_bet_probability: float
    reraise_probability: float
    prior: float = 1.0
    rationale: str = ""

    def __post_init__(self) -> None:
        values = (
            self.open_raise_probability,
            self.fold_vs_bet_probability,
            self.reraise_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("World conditional probabilities must be in [0, 1]")
        if self.prior <= 0.0:
            raise ValueError("World prior must be positive")

    def action_probabilities(self, facing_bet: bool) -> dict[ActionType, float]:
        if not facing_bet:
            return {
                ActionType.FOLD: 0.0,
                ActionType.CHECK_CALL: 1.0 - self.open_raise_probability,
                ActionType.RAISE: self.open_raise_probability,
            }
        continue_probability = 1.0 - self.fold_vs_bet_probability
        return {
            ActionType.FOLD: self.fold_vs_bet_probability,
            ActionType.CHECK_CALL: continue_probability * (1.0 - self.reraise_probability),
            ActionType.RAISE: continue_probability * self.reraise_probability,
        }

    def probability(
        self,
        observation: OpponentObservation,
        floor: float = 1e-4,
    ) -> float:
        return max(
            self.action_probabilities(observation.facing_bet)[observation.action],
            floor,
        )


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
        threshold: float = 0.07,
        probability_floor: float = 1e-4,
        cooldown_observations: int = 16,
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

    def score(
        self,
        predicted_probability: float,
        expected_surprise: float = 0.0,
    ) -> float:
        probability = min(max(predicted_probability, self.probability_floor), 1.0)
        normalized = -math.log(probability) / -math.log(self.probability_floor)
        return max(0.0, min(normalized, 1.0) - max(expected_surprise, 0.0))

    def update(
        self,
        predicted_probability: float,
        expected_surprise: float = 0.0,
    ) -> SurpriseUpdate:
        score = self.score(predicted_probability, expected_surprise)
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
        observations: Sequence[OpponentObservation],
        current_worlds: Sequence[OpponentWorld],
    ) -> list[OpponentWorld]: ...


def empirical_world(
    observations: Sequence[OpponentObservation],
    *,
    name: str = "empirical_shift",
    prior: float = 1.5,
    smoothing: float = 1.0,
) -> OpponentWorld:
    unopened = [observation for observation in observations if not observation.facing_bet]
    faced = [observation for observation in observations if observation.facing_bet]
    open_raises = sum(observation.action is ActionType.RAISE for observation in unopened)
    open_raise_probability = (open_raises + smoothing) / (
        len(unopened) + 2.0 * smoothing
    )
    folds = sum(observation.action is ActionType.FOLD for observation in faced)
    fold_vs_bet_probability = (folds + smoothing) / (len(faced) + 2.0 * smoothing)
    continued = [
        observation for observation in faced if observation.action is not ActionType.FOLD
    ]
    reraises = sum(observation.action is ActionType.RAISE for observation in continued)
    reraise_probability = (reraises + smoothing) / (
        len(continued) + 2.0 * smoothing
    )
    return OpponentWorld(
        name=name,
        open_raise_probability=open_raise_probability,
        fold_vs_bet_probability=fold_vs_bet_probability,
        reraise_probability=reraise_probability,
        prior=prior,
        rationale="Smoothed conditional frequencies by betting context.",
    )


class HeuristicHypothesisGenerator:
    """Deterministic offline generator used for CI and mechanism ablations."""

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        self.calls = 0

    def generate(
        self,
        observations: Sequence[OpponentObservation],
        current_worlds: Sequence[OpponentWorld],
    ) -> list[OpponentWorld]:
        del current_worlds
        self.calls += 1
        return [
            empirical_world(observations, smoothing=self.smoothing),
            OpponentWorld(
                name="aggressive_switch",
                open_raise_probability=0.48,
                fold_vs_bet_probability=0.18,
                reraise_probability=0.30,
                rationale="Frequent opening raises, low folding, and more reraising.",
            ),
            OpponentWorld(
                name="passive_switch",
                open_raise_probability=0.08,
                fold_vs_bet_probability=0.18,
                reraise_probability=0.05,
                rationale="Low initiative and call-heavy continuing ranges.",
            ),
            OpponentWorld(
                name="tight_switch",
                open_raise_probability=0.14,
                fold_vs_bet_probability=0.56,
                reraise_probability=0.08,
                rationale="Low opening pressure and high fold-to-bet frequency.",
            ),
        ]


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
                    "open_raise_probability",
                    "fold_vs_bet_probability",
                    "reraise_probability",
                    "prior",
                    "rationale",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "open_raise_probability": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "fold_vs_bet_probability": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "reraise_probability": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "prior": {"type": "number", "exclusiveMinimum": 0.0},
                    "rationale": {"type": "string"},
                },
            },
        }
    },
}


class ProviderHypothesisGenerator:
    """Provider-backed conditional world generator using Structured Outputs."""

    def __init__(self, provider: Any) -> None:
        if not hasattr(provider, "structured"):
            raise TypeError("provider must implement structured(...)")
        self.provider = provider
        self.calls = 0
        self.last_response: Any | None = None

    def generate(
        self,
        observations: Sequence[OpponentObservation],
        current_worlds: Sequence[OpponentWorld],
    ) -> list[OpponentWorld]:
        self.calls += 1
        conditional_counts = Counter(
            f"{'facing_bet' if observation.facing_bet else 'unopened'}:"
            f"{observation.action.value}"
            for observation in observations
        )
        response = self.provider.structured(
            instructions=(
                "Generate two to four concise, falsifiable opponent-policy hypotheses. "
                "Model opening raise, fold versus bet, and reraise probabilities. "
                "Do not choose a hero action or provide hidden chain-of-thought."
            ),
            state={
                "recent_observations": [asdict(observation) for observation in observations],
                "conditional_counts": dict(conditional_counts),
                "current_worlds": [asdict(world) for world in current_worlds],
            },
            schema_name="opponent_regime_hypotheses",
            schema=HYPOTHESIS_SCHEMA,
        )
        self.last_response = response
        worlds = [
            OpponentWorld(
                name=str(item["name"]),
                open_raise_probability=float(item["open_raise_probability"]),
                fold_vs_bet_probability=float(item["fold_vs_bet_probability"]),
                reraise_probability=float(item["reraise_probability"]),
                prior=float(item["prior"]),
                rationale=str(item["rationale"]),
            )
            for item in response.payload["worlds"]
        ]
        if len(worlds) < 2:
            raise ValueError("Provider returned fewer than two valid hypotheses")
        return worlds


DEFAULT_WORLD = OpponentWorld(
    name="stable_tag",
    open_raise_probability=0.18,
    fold_vs_bet_probability=0.42,
    reraise_probability=0.10,
    rationale="Initial TAG-like conditional prior before formation.",
)
