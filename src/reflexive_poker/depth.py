from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class DepthPrediction:
    depth: int
    per_opponent: dict[str, float]

    @property
    def all_fold_probability(self) -> float:
        return float(prod(self.per_opponent.values())) if self.per_opponent else 0.0


class AdaptiveDepthController:
    OPS = (1, 3, 7, 15)

    def __init__(self, opponents: tuple[str, ...] = (), max_depth: int = 3) -> None:
        self.max_depth = max_depth
        self.opponent_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for opponent in opponents:
            self.opponent_counts[opponent]

    def choose_depth(self) -> int:
        return 1 if self.max_depth else 0

    def predict(
        self, depth: int, opponents: tuple[str, ...], images: dict[object, object]
    ) -> DepthPrediction:
        del images
        values = {}
        for opponent in opponents:
            counts = self.opponent_counts[opponent]
            total = sum(counts.values())
            values[opponent] = (counts["fold"] + 1) / (total + 2) if total else 0.5
        return DepthPrediction(depth, values)
