from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .models import ActionType

_ACTIONS = (ActionType.FOLD, ActionType.CHECK_CALL, ActionType.RAISE)


@dataclass(frozen=True)
class OpponentWorld:
    name: str
    fold_probability: float
    call_probability: float
    raise_probability: float
    prior: float = 1.0
    rationale: str = ""

    def __post_init__(self) -> None:
        values = (self.fold_probability, self.call_probability, self.raise_probability)
        if any(value < 0.0 for value in values):
            raise ValueError("World probabilities must be non-negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("World action probabilities must sum to one")
        if self.prior <= 0.0:
            raise ValueError("World prior must be positive")

    def probability(self, action: ActionType, floor: float = 1e-4) -> float:
        values = {
            ActionType.FOLD: self.fold_probability,
            ActionType.CHECK_CALL: self.call_probability,
            ActionType.RAISE: self.raise_probability,
        }
        return max(values[action], floor)


@dataclass(frozen=True)
class SurpriseUpdate:
    surprise_score: float
    mean_surprise: float
    change_detected: bool
    observations: int


class SurpriseDetector:
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
        raw_score = -math.log(probability) / -math.log(self.probability_floor)
        score = max(0.0, min(raw_score, 1.0) - max(expected_surprise, 0.0))
        self._scores.append(score)
        self._updates += 1
        mean_score = sum(self._scores) / len(self._scores)
        detected = (
            len(self._scores) >= self.min_observations
            and mean_score >= self.threshold
            and self._updates - self._last_detection >= self.cooldown_observations
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
    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        self.calls = 0

    @staticmethod
    def _world(
        name: str,
        fold: float,
        call: float,
        raise_: float,
        rationale: str,
        prior: float = 1.0,
    ) -> OpponentWorld:
        total = fold + call + raise_
        return OpponentWorld(
            name,
            fold / total,
            call / total,
            raise_ / total,
            prior,
            rationale,
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
        return [
            OpponentWorld(
                "empirical_shift",
                empirical[ActionType.FOLD],
                empirical[ActionType.CHECK_CALL],
                empirical[ActionType.RAISE],
                1.5,
                "Smoothed action frequencies in the surprise window.",
            ),
            self._world(
                "aggressive_switch",
                0.13,
                0.42,
                0.45,
                "A persistent increase in raises is consistent with a LAG or tilt regime.",
            ),
            self._world(
                "passive_switch",
                0.24,
                0.67,
                0.09,
                "A call-heavy, low-raise regime is consistent with passive play.",
            ),
            self._world(
                "tight_switch",
                0.48,
                0.39,
                0.13,
                "A fold-heavy regime is consistent with a tighter range.",
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
        response = self.provider.structured(
            instructions=(
                "Generate two to four concise, falsifiable opponent-policy hypotheses for a "
                "surprising poker action stream. Use only fold, check_call, and raise "
                "frequencies. Each probability vector must sum to one. Do not choose an action."
            ),
            state={
                "recent_actions": [action.value for action in observations],
                "recent_action_counts": dict(Counter(action.value for action in observations)),
                "current_worlds": [asdict(world) for world in current_worlds],
            },
            schema_name="opponent_regime_hypotheses",
            schema=HYPOTHESIS_SCHEMA,
        )
        self.last_response = response
        worlds: list[OpponentWorld] = []
        for item in response.payload["worlds"]:
            total = sum(
                float(item[key])
                for key in ("fold_probability", "call_probability", "raise_probability")
            )
            if total <= 0.0:
                continue
            worlds.append(
                OpponentWorld(
                    str(item["name"]),
                    float(item["fold_probability"]) / total,
                    float(item["call_probability"]) / total,
                    float(item["raise_probability"]) / total,
                    float(item["prior"]),
                    str(item["rationale"]),
                )
            )
        if len(worlds) < 2:
            raise ValueError("Provider returned fewer than two valid hypotheses")
        return worlds


DEFAULT_WORLD = OpponentWorld(
    "stable_tag",
    0.31,
    0.56,
    0.13,
    rationale="Initial balanced prior before enough opponent actions are observed.",
)
