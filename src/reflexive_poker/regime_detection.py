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


@dataclass(frozen=True)
class ConditionalDriftUpdate:
    """Result from a two-sample conditional action-distribution comparison."""

    likelihood_ratio: float
    max_probability_delta: float
    change_detected: bool
    observations: int
    context_likelihood_ratios: dict[str, float]
    probability_deltas: dict[str, float]


@dataclass(frozen=True)
class SignedEProcessUpdate:
    """One update from the blockwise signed conditional Bayes-factor process."""

    block_complete: bool
    block_index: int
    e_value_up: float
    e_value_down: float
    log_bayes_factor_up: float
    log_bayes_factor_down: float
    direction: str | None
    change_detected: bool
    observations: int
    probability_deltas: dict[str, float]
    metric_counts: dict[str, dict[str, int]]


def _log_add_exp(left: float, right: float) -> float:
    anchor = max(left, right)
    if math.isinf(anchor):
        return anchor
    return anchor + math.log(math.exp(left - anchor) + math.exp(right - anchor))


class SignedConditionalEProcessDetector:
    """Blockwise signed Bayes-factor process for conditional Bernoulli shifts.

    Separate upward and downward processes track opening raises and folds while
    facing a bet. A uniform mixture over possible block-aligned change starts
    prevents repeated peeking from being treated as independent tests. The
    detector is calibrated empirically on held-out no-switch traces before it
    is used for validation; it is not claimed as a theorem-level universal
    anytime-valid test when the reference distribution is estimated.
    """

    _METRICS = ("open_raise", "fold_vs_bet")

    def __init__(
        self,
        *,
        reference_size: int = 96,
        block_size: int = 16,
        e_value_threshold: float = 10.0,
        alternative_delta: float = 0.18,
        alternative_concentration: float = 32.0,
        minimum_direction_delta: float = 0.06,
        maximum_blocks: int = 64,
        smoothing: float = 1.0,
    ) -> None:
        if reference_size < 16 or block_size < 4:
            raise ValueError("reference_size and block_size are too small")
        if e_value_threshold <= 1.0:
            raise ValueError("e_value_threshold must exceed one")
        if not 0.0 < alternative_delta < 1.0:
            raise ValueError("alternative_delta must be in (0, 1)")
        if alternative_concentration <= 0.0 or smoothing <= 0.0:
            raise ValueError("concentration and smoothing must be positive")
        if not 0.0 <= minimum_direction_delta < 1.0:
            raise ValueError("minimum_direction_delta must be in [0, 1)")
        if maximum_blocks < 2:
            raise ValueError("maximum_blocks must be at least two")
        self.reference_size = reference_size
        self.block_size = block_size
        self.e_value_threshold = e_value_threshold
        self.alternative_delta = alternative_delta
        self.alternative_concentration = alternative_concentration
        self.minimum_direction_delta = minimum_direction_delta
        self.maximum_blocks = maximum_blocks
        self.smoothing = smoothing
        self.reference: tuple[OpponentObservation, ...] = ()
        self.block: list[OpponentObservation] = []
        self.block_index = 0
        self._log_restart_up = -math.inf
        self._log_restart_down = -math.inf
        self.e_value_up = 1.0
        self.e_value_down = 1.0

    @property
    def ready(self) -> bool:
        return len(self.reference) >= self.reference_size

    def fit_reference(self, observations: Sequence[OpponentObservation]) -> None:
        if len(observations) < self.reference_size:
            raise ValueError("Not enough observations to fit e-process reference")
        self.reference = tuple(observations[-self.reference_size :])
        self.block.clear()
        self.block_index = 0
        self._log_restart_up = -math.inf
        self._log_restart_down = -math.inf
        self.e_value_up = 1.0
        self.e_value_down = 1.0

    @staticmethod
    def _metric_outcomes(
        observations: Sequence[OpponentObservation],
        metric: str,
    ) -> tuple[int, int]:
        if metric == "open_raise":
            eligible = [item for item in observations if not item.facing_bet]
            successes = sum(item.action is ActionType.RAISE for item in eligible)
        elif metric == "fold_vs_bet":
            eligible = [item for item in observations if item.facing_bet]
            successes = sum(item.action is ActionType.FOLD for item in eligible)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return successes, len(eligible) - successes

    @staticmethod
    def _log_beta(alpha: float, beta: float) -> float:
        return math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)

    def _metric_log_bayes_factor(
        self,
        metric: str,
        successes: int,
        failures: int,
        direction: int,
    ) -> float:
        reference_successes, reference_failures = self._metric_outcomes(
            self.reference,
            metric,
        )
        null_alpha = reference_successes + self.smoothing
        null_beta = reference_failures + self.smoothing
        reference_mean = null_alpha / (null_alpha + null_beta)
        alternative_mean = min(
            max(reference_mean + direction * self.alternative_delta, 0.01),
            0.99,
        )
        alternative_alpha = alternative_mean * self.alternative_concentration
        alternative_beta = (1.0 - alternative_mean) * self.alternative_concentration
        log_null = self._log_beta(
            null_alpha + successes,
            null_beta + failures,
        ) - self._log_beta(null_alpha, null_beta)
        log_alternative = self._log_beta(
            alternative_alpha + successes,
            alternative_beta + failures,
        ) - self._log_beta(alternative_alpha, alternative_beta)
        return log_alternative - log_null

    def _mixture_e_value(self, log_restart: float) -> float:
        remaining = max(self.maximum_blocks - self.block_index, 0)
        log_remaining = math.log(remaining) if remaining else -math.inf
        log_mixture = _log_add_exp(log_remaining, log_restart) - math.log(
            self.maximum_blocks
        )
        return math.exp(min(log_mixture, 700.0))

    def update(self, observation: OpponentObservation) -> SignedEProcessUpdate:
        if not self.ready:
            raise RuntimeError("fit_reference must be called before update")
        self.block.append(observation)
        if len(self.block) < self.block_size:
            return SignedEProcessUpdate(
                block_complete=False,
                block_index=self.block_index,
                e_value_up=self.e_value_up,
                e_value_down=self.e_value_down,
                log_bayes_factor_up=0.0,
                log_bayes_factor_down=0.0,
                direction=None,
                change_detected=False,
                observations=len(self.block),
                probability_deltas={},
                metric_counts={},
            )

        self.block_index += 1
        metric_counts: dict[str, dict[str, int]] = {}
        probability_deltas: dict[str, float] = {}
        log_bayes_factors = {1: 0.0, -1: 0.0}
        for metric in self._METRICS:
            successes, failures = self._metric_outcomes(self.block, metric)
            reference_successes, reference_failures = self._metric_outcomes(
                self.reference,
                metric,
            )
            recent_probability = (successes + self.smoothing) / (
                successes + failures + 2.0 * self.smoothing
            )
            reference_probability = (reference_successes + self.smoothing) / (
                reference_successes + reference_failures + 2.0 * self.smoothing
            )
            probability_deltas[metric] = recent_probability - reference_probability
            metric_counts[metric] = {
                "successes": successes,
                "failures": failures,
            }
            if successes + failures:
                for direction in (1, -1):
                    log_bayes_factors[direction] += self._metric_log_bayes_factor(
                        metric,
                        successes,
                        failures,
                        direction,
                    )

        self._log_restart_up = log_bayes_factors[1] + _log_add_exp(
            0.0,
            self._log_restart_up,
        )
        self._log_restart_down = log_bayes_factors[-1] + _log_add_exp(
            0.0,
            self._log_restart_down,
        )
        self.e_value_up = self._mixture_e_value(self._log_restart_up)
        self.e_value_down = self._mixture_e_value(self._log_restart_down)
        upward_signature = all(
            probability_deltas.get(metric, 0.0) >= self.minimum_direction_delta
            for metric in self._METRICS
        )
        downward_signature = all(
            probability_deltas.get(metric, 0.0) <= -self.minimum_direction_delta
            for metric in self._METRICS
        )
        direction = None
        if self.e_value_up >= self.e_value_threshold and upward_signature:
            direction = "up"
        elif self.e_value_down >= self.e_value_threshold and downward_signature:
            direction = "down"
        self.block.clear()
        return SignedEProcessUpdate(
            block_complete=True,
            block_index=self.block_index,
            e_value_up=self.e_value_up,
            e_value_down=self.e_value_down,
            log_bayes_factor_up=log_bayes_factors[1],
            log_bayes_factor_down=log_bayes_factors[-1],
            direction=direction,
            change_detected=direction is not None,
            observations=self.block_size,
            probability_deltas=probability_deltas,
            metric_counts=metric_counts,
        )


class ConditionalDistributionDetector:
    """Detect two-sided changes in context-conditional opponent actions.

    A frozen formation sample is compared with a rolling recent sample using a
    smoothed conditional multinomial likelihood ratio. Unlike surprise-only
    detection, this test responds when an already-likely action becomes still
    more frequent, as happens when fold-versus-bet increases.
    """

    def __init__(
        self,
        *,
        reference_size: int = 64,
        recent_size: int = 40,
        min_recent_observations: int = 28,
        likelihood_ratio_threshold: float = 6.0,
        min_probability_delta: float = 0.12,
        required_streak: int = 2,
        evaluation_stride: int = 4,
        calibration_observations: int = 0,
        smoothing: float = 1.0,
    ) -> None:
        if reference_size < 8:
            raise ValueError("reference_size must be at least eight")
        if recent_size < 4:
            raise ValueError("recent_size must be at least four")
        if not 1 <= min_recent_observations <= recent_size:
            raise ValueError("min_recent_observations must be within recent_size")
        if likelihood_ratio_threshold <= 0.0:
            raise ValueError("likelihood_ratio_threshold must be positive")
        if not 0.0 < min_probability_delta <= 1.0:
            raise ValueError("min_probability_delta must be in (0, 1]")
        if calibration_observations < 0:
            raise ValueError("calibration_observations must be non-negative")
        if required_streak < 1 or evaluation_stride < 1 or smoothing <= 0.0:
            raise ValueError("streak, stride, and smoothing must be positive")
        self.reference_size = reference_size
        self.recent_size = recent_size
        self.min_recent_observations = min_recent_observations
        self.likelihood_ratio_threshold = likelihood_ratio_threshold
        self.min_probability_delta = min_probability_delta
        self.required_streak = required_streak
        self.evaluation_stride = evaluation_stride
        self.calibration_observations = calibration_observations
        self.smoothing = smoothing
        self.reference: deque[OpponentObservation] = deque(maxlen=reference_size)
        self.recent: deque[OpponentObservation] = deque(maxlen=recent_size)
        self._updates = 0
        self._streak = 0

    @property
    def ready(self) -> bool:
        return len(self.reference) >= self.reference_size

    def fit_reference(self, observations: Sequence[OpponentObservation]) -> None:
        if len(observations) < self.reference_size:
            raise ValueError("Not enough observations to fit detector reference")
        self.reference = deque(observations[-self.reference_size :], maxlen=self.reference_size)
        self.recent.clear()
        self._updates = 0
        self._streak = 0

    @staticmethod
    def _context_actions(facing_bet: bool) -> tuple[ActionType, ...]:
        return (
            (ActionType.FOLD, ActionType.CHECK_CALL, ActionType.RAISE)
            if facing_bet
            else (ActionType.CHECK_CALL, ActionType.RAISE)
        )

    def _compare_context(
        self,
        facing_bet: bool,
    ) -> tuple[float, dict[str, float]]:
        reference = [item for item in self.reference if item.facing_bet is facing_bet]
        recent = [item for item in self.recent if item.facing_bet is facing_bet]
        actions = self._context_actions(facing_bet)
        if len(reference) < len(actions) * 2 or len(recent) < len(actions) * 2:
            return 0.0, {}
        reference_total = len(reference) + self.smoothing * len(actions)
        recent_total = len(recent) + self.smoothing * len(actions)
        pooled_total = reference_total + recent_total
        likelihood_ratio = 0.0
        deltas: dict[str, float] = {}
        for action in actions:
            reference_count = (
                sum(item.action is action for item in reference) + self.smoothing
            )
            recent_count = sum(item.action is action for item in recent) + self.smoothing
            reference_probability = reference_count / reference_total
            recent_probability = recent_count / recent_total
            pooled_probability = (reference_count + recent_count) / pooled_total
            likelihood_ratio += 2.0 * (
                reference_count * math.log(reference_probability / pooled_probability)
                + recent_count * math.log(recent_probability / pooled_probability)
            )
            deltas[action.value] = recent_probability - reference_probability
        return max(likelihood_ratio, 0.0), deltas

    def update(self, observation: OpponentObservation) -> ConditionalDriftUpdate:
        if not self.ready:
            raise RuntimeError("fit_reference must be called before update")
        if len(self.recent) == self.recent_size:
            self.reference.append(self.recent.popleft())
        self.recent.append(observation)
        self._updates += 1
        context_scores: dict[str, float] = {}
        probability_deltas: dict[str, float] = {}
        for facing_bet, context_name in ((False, "unopened"), (True, "facing_bet")):
            score, deltas = self._compare_context(facing_bet)
            context_scores[context_name] = score
            probability_deltas.update(
                {f"{context_name}:{action}": delta for action, delta in deltas.items()}
            )
        likelihood_ratio = sum(context_scores.values())
        max_delta = max((abs(value) for value in probability_deltas.values()), default=0.0)
        evaluated = (
            len(self.recent) >= self.min_recent_observations
            and self._updates > self.calibration_observations
            and self._updates % self.evaluation_stride == 0
        )
        exceeded = (
            likelihood_ratio >= self.likelihood_ratio_threshold
            and max_delta >= self.min_probability_delta
        )
        if evaluated:
            self._streak = self._streak + 1 if exceeded else 0
        detected = evaluated and self._streak >= self.required_streak
        return ConditionalDriftUpdate(
            likelihood_ratio=likelihood_ratio,
            max_probability_delta=max_delta,
            change_detected=detected,
            observations=len(self.recent),
            context_likelihood_ratios=context_scores,
            probability_deltas=probability_deltas,
        )


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
                name="probe_fold_switch",
                open_raise_probability=0.42,
                fold_vs_bet_probability=0.66,
                reraise_probability=0.08,
                rationale="Frequent probes followed by excessive folding to pressure.",
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
