from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    has_self_model: bool
    has_feedback: bool
    fixed_depth: int | None = None
    shaping: str = "none"


CONFIRMATORY_CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec("no_reflection", False, False, 0),
    ConditionSpec("local_reflection", False, False, 1),
    ConditionSpec("situated_open_loop", True, False, None),
    ConditionSpec("situated_reflection", True, True, None),
    ConditionSpec("situated_fixed_depth3", True, True, 3),
)

IMAGE_SHAPING_CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec("myopic_control", False, False, 0, "none"),
    ConditionSpec("passive_image_tracking", True, False, None, "passive"),
    ConditionSpec("open_loop_shaping", True, False, None, "open_loop"),
    ConditionSpec("closed_loop_shaping", True, True, None, "closed_loop"),
)


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _sample_hand_strength(rng: random.Random) -> float:
    # Lightweight stand-in for the full simulator's Monte Carlo equity estimate.
    return min(0.99, max(0.01, rng.betavariate(2.0, 2.0)))


def run_condition(
    spec: ConditionSpec,
    *,
    seed: int,
    hands: int = 320,
    hidden_shift: bool = False,
    switch_hand: int | None = None,
) -> pd.DataFrame:
    rng = random.Random(seed)
    switch_hand = hands // 2 if switch_hand is None else switch_hand
    rows: list[dict[str, float | int | str]] = []

    public_aggression = 0.50
    self_estimate = 0.50
    confidence = 0.0
    signal_done = False
    signal_stop_hand = hands
    strategy_shift_hand = hands
    cumulative_profit = 0.0

    for hand in range(hands):
        regime = "post_shift" if hidden_shift and hand >= switch_hand else "static"
        strength = _sample_hand_strength(rng)
        target_tight = spec.shaping in {"open_loop", "closed_loop"}
        early_signal = target_tight and not signal_done

        if spec.shaping == "open_loop" and hand >= 30:
            early_signal = False
            signal_done = True
            signal_stop_hand = min(signal_stop_hand, hand)
        if spec.shaping == "closed_loop" and hand >= 8 and public_aggression < 0.47 and confidence > 0.36:
            early_signal = False
            signal_done = True
            signal_stop_hand = min(signal_stop_hand, hand)
        if spec.shaping == "closed_loop" and hand >= 85:
            early_signal = False
            signal_done = True
            signal_stop_hand = min(signal_stop_hand, hand)

        base_raise = 0.21 + 0.34 * (strength - 0.50)
        if spec.has_self_model:
            base_raise += 0.04 * (0.50 - self_estimate)
        if early_signal:
            base_raise -= 0.055
        elif target_tight and hand >= 30:
            base_raise += 0.040
        if regime == "post_shift" and spec.has_self_model:
            # The v0.3.0 failure mode: stale self-models over-trust the old belief channel.
            base_raise += 0.020
        raise_prob = min(0.92, max(0.03, base_raise))
        raised = rng.random() < raise_prob

        # Opponents update a noisy public-image belief from observed focal actions.
        update_rate = 0.035 if regime == "static" else 0.020
        observed_signal = 1.0 if raised else 0.0
        public_aggression = (1 - update_rate) * public_aggression + update_rate * observed_signal
        public_aggression += rng.gauss(0.0, 0.006)
        public_aggression = min(0.95, max(0.05, public_aggression))

        if spec.has_self_model:
            if spec.has_feedback:
                self_estimate = 0.86 * self_estimate + 0.14 * public_aggression
            else:
                self_estimate = 0.97 * self_estimate + 0.03 * (0.42 if early_signal else 0.54)
        image_mae = abs(self_estimate - public_aggression) if spec.has_self_model else abs(0.50 - public_aggression)
        confidence = 0.92 * confidence + 0.08 * (1.0 - min(1.0, image_mae * 4.0))

        if spec.fixed_depth is not None:
            depth = spec.fixed_depth
        elif not spec.has_self_model:
            depth = 0
        else:
            depth = 1 + int(image_mae > 0.05) + int(confidence < 0.45)
        reasoning_ops = 0 if depth == 0 else sum(range(1, depth + 2))

        fold_rate = min(0.90, max(0.05, 0.40 - 0.30 * public_aggression + 0.18 * raised))
        if regime == "post_shift":
            fold_rate = 0.28 + 0.15 * raised
        opponent_fold = rng.random() < fold_rate

        immediate = (1.2 if opponent_fold and raised else 0.0)
        showdown = (strength - 0.50) * 2.0 + rng.gauss(0.0, 0.70)
        cost = 0.55 if raised else 0.12
        reward = immediate + showdown - cost
        if early_signal:
            reward -= 0.08
        cumulative_profit += reward

        if spec.has_self_model and strategy_shift_hand == hands and not early_signal and hand >= 5:
            strategy_shift_hand = hand

        rows.append(
            {
                "condition": spec.name,
                "seed": seed,
                "hand": hand,
                "phase": "signal" if hand < 30 else ("exploit" if hand < 120 else "late"),
                "regime": regime,
                "strength": strength,
                "raised": int(raised),
                "public_aggression": public_aggression,
                "self_estimate": self_estimate,
                "image_mae": image_mae,
                "depth": depth,
                "reasoning_ops": reasoning_ops,
                "opponent_fold": int(opponent_fold),
                "reward": reward,
                "cumulative_profit": cumulative_profit,
                "signal_stop_hand": signal_stop_hand,
                "strategy_shift_hand": strategy_shift_hand,
            }
        )
    return pd.DataFrame(rows)


def run_study(
    conditions: Iterable[ConditionSpec],
    *,
    seeds: Iterable[int],
    hands: int,
    hidden_shift: bool = False,
    output: str | Path | None = None,
) -> pd.DataFrame:
    frames = [
        run_condition(spec, seed=seed, hands=hands, hidden_shift=hidden_shift)
        for seed in seeds
        for spec in conditions
    ]
    data = pd.concat(frames, ignore_index=True)
    if output is not None:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        data.to_csv(out / "hands.csv", index=False)
        summarize(data).to_csv(out / "summary.csv", index=False)
    return data


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    per_run = (
        data.groupby(["condition", "seed"], as_index=False)
        .agg(
            chips_per_100=("reward", lambda x: float(x.sum()) / len(x) * 100.0),
            image_mae=("image_mae", "mean"),
            avg_reasoning_ops=("reasoning_ops", "mean"),
            raise_rate=("raised", "mean"),
            signal_stop_hand=("signal_stop_hand", "min"),
        )
    )
    return (
        per_run.groupby("condition", as_index=False)
        .agg(
            chips_per_100=("chips_per_100", "mean"),
            image_mae=("image_mae", "mean"),
            avg_reasoning_ops=("avg_reasoning_ops", "mean"),
            raise_rate=("raise_rate", "mean"),
            signal_stop_hand=("signal_stop_hand", "mean"),
        )
    )
