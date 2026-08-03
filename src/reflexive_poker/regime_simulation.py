from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from .models import ActionType
from .regime_detection import OpponentWorld


@dataclass(frozen=True)
class SimulationResult:
    world_name: str
    posterior: float
    best_response: str
    expected_bb_per_decision: float
    response_values: dict[str, float]


class WorldSimulator:
    PAYOFFS: ClassVar[dict[str, dict[ActionType, float]]] = {
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
        log_weights = [
            math.log(world.prior)
            + sum(math.log(world.probability(action)) for action in observations)
            for world in worlds
        ]
        anchor = max(log_weights)
        weights = [math.exp(value - anchor) for value in log_weights]
        posteriors = [value / sum(weights) for value in weights]
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
                    world.name,
                    posterior,
                    best,
                    values[best],
                    values,
                )
            )
        return sorted(results, key=lambda result: result.posterior, reverse=True)

    @staticmethod
    def choose_response(results: Sequence[SimulationResult]) -> tuple[str, float]:
        if not results:
            return "balanced", 0.0
        aggregate: Counter[str] = Counter()
        for result in results:
            for response, value in result.response_values.items():
                aggregate[response] += result.posterior * value
        best = max(aggregate, key=aggregate.get)
        return best, aggregate[best]
