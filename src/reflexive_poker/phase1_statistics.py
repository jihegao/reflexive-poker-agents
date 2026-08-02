from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairedInference:
    contrast: str
    pairs: int
    mean_delta: float
    median_delta: float
    ci95_low: float
    ci95_high: float
    permutation_p: float
    holm_p: float
    positive_seed_rate: float
    worst_quartile_mean: float
    leave_largest_out_mean: float


def classify_core_hypothesis(
    simulation_inference: pd.DataFrame,
    simulation_paired: pd.DataFrame,
    llm_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Apply the frozen strong/limited/boundary decision rule without pooling models."""
    required = {"recursive_d2-state_only", "recursive_d2-action_prediction"}
    simulation_core = simulation_inference[
        simulation_inference["contrast"].isin(required)
    ]
    simulation_supported = (
        set(simulation_core["contrast"]) == required
        and bool((simulation_core["ci95_low"] > 0).all())
    )
    trimmed = simulation_paired[simulation_paired["contrast"].isin(required)]
    pot_robust = (
        set(trimmed["contrast"]) == required
        and bool((trimmed.groupby("contrast")["trimmed_delta"].mean() > 0).all())
    )
    model_rows: dict[str, dict[str, object]] = {}
    for model, payload in llm_results.items():
        inference = payload.get("inference")
        gate = payload.get("provider_gate")
        cost = payload.get("cost_metrics")
        if not isinstance(inference, pd.DataFrame) or not isinstance(cost, pd.DataFrame):
            model_rows[model] = {"valid": False, "reason": "missing_frames"}
            continue
        row = inference[inference["contrast"] == "recursive_d2-action_prediction"]
        cost_row = cost[cost["treatment"] == "recursive_d2"]
        gate_valid = bool(gate.get("valid")) if isinstance(gate, dict) else False
        direction_positive = not row.empty and float(row.iloc[0]["mean_delta"]) > 0
        interval_positive = not row.empty and float(row.iloc[0]["ci95_low"]) > 0
        cost_positive = (
            not cost_row.empty and float(cost_row.iloc[0]["chips_per_1000_tokens"]) > 0
        )
        model_rows[model] = {
            "valid": gate_valid,
            "direction_positive": direction_positive,
            "interval_positive": interval_positive,
            "cost_adjusted_positive": cost_positive,
        }
    both_models_present = len(model_rows) == 2
    both_valid = both_models_present and all(bool(row.get("valid")) for row in model_rows.values())
    both_direction = both_models_present and all(
        bool(row.get("direction_positive")) for row in model_rows.values()
    )
    both_intervals = both_models_present and all(
        bool(row.get("interval_positive")) for row in model_rows.values()
    )
    both_cost = both_models_present and all(
        bool(row.get("cost_adjusted_positive")) for row in model_rows.values()
    )
    if simulation_supported and pot_robust and both_valid and both_intervals and both_cost:
        classification = "strong_support"
    elif simulation_supported and pot_robust and both_valid and both_direction:
        classification = "limited_support"
    else:
        classification = "boundary_or_not_supported"
    return {
        "classification": classification,
        "simulation_supported": simulation_supported,
        "large_pot_robust": pot_robust,
        "models": model_rows,
        "models_analyzed_separately": True,
    }


def paired_bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int = 5_000,
    seed: int = 20260802,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_sign_permutation_p(
    values: np.ndarray,
    *,
    samples: int = 20_000,
    seed: int = 20260802,
) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    if len(values) <= 18:
        masks = np.arange(1 << len(values), dtype=np.uint64)[:, None]
        bits = (masks >> np.arange(len(values), dtype=np.uint64)) & 1
        signs = np.where(bits == 1, 1.0, -1.0)
        null_means = np.abs((signs * values).mean(axis=1))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(samples, len(values)))
        null_means = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null_means >= observed) + 1) / (len(null_means) + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    finite = [(index, value) for index, value in enumerate(p_values) if math.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return adjusted


def large_pot_sensitivity(per_hand: pd.DataFrame) -> dict[str, float]:
    if per_hand.empty:
        return {
            "total_reward": 0.0,
            "largest_abs_reward": 0.0,
            "top_1pct_abs_share": float("nan"),
            "trimmed_1pct_reward": 0.0,
        }
    rewards = per_hand["reward"].to_numpy(dtype=float)
    absolute = np.abs(rewards)
    total_absolute = float(absolute.sum())
    remove_count = max(1, math.ceil(len(rewards) * 0.01))
    removal = np.argsort(absolute)[-remove_count:]
    keep = np.ones(len(rewards), dtype=bool)
    keep[removal] = False
    return {
        "total_reward": float(rewards.sum()),
        "largest_abs_reward": float(absolute.max()),
        "top_1pct_abs_share": (
            float(absolute[removal].sum() / total_absolute) if total_absolute else 0.0
        ),
        "trimmed_1pct_reward": float(rewards[keep].sum()),
    }


def rolling_direction_consistency(per_hand_deltas: pd.DataFrame, window: int = 25) -> float:
    if per_hand_deltas.empty:
        return float("nan")
    ordered = per_hand_deltas.sort_values(["seed", "hand_index"])
    rolling = ordered.groupby("seed")["reward_delta"].rolling(window, min_periods=window).sum()
    values = rolling.dropna().to_numpy(dtype=float)
    return float((values > 0).mean()) if len(values) else float("nan")


def inference_table(
    paired: pd.DataFrame,
    *,
    metric: str = "chips_per_100_delta",
    bootstrap_samples: int = 5_000,
    permutation_samples: int = 20_000,
) -> pd.DataFrame:
    """Calculate seed-paired inference for one declared outcome metric."""
    if metric not in paired.columns:
        raise ValueError(f"paired frame does not contain metric: {metric}")
    rows: list[dict[str, float | int | str]] = []
    for contrast, group in paired.groupby("contrast", sort=False):
        values = group[metric].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        low, high = paired_bootstrap_interval(values, samples=bootstrap_samples)
        ordered = np.sort(values)
        worst_count = max(1, math.ceil(len(values) * 0.25))
        leave_largest = np.delete(values, np.argmax(np.abs(values))) if len(values) > 1 else values
        rows.append(
            {
                "metric": metric,
                "contrast": contrast,
                "pairs": len(values),
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "ci95_low": low,
                "ci95_high": high,
                "permutation_p": paired_sign_permutation_p(
                    values, samples=permutation_samples
                ),
                "positive_seed_rate": float((values > 0).mean()),
                "worst_quartile_mean": float(ordered[:worst_count].mean()),
                "leave_largest_out_mean": float(leave_largest.mean()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=[
            "metric", "contrast", "pairs", "mean_delta", "median_delta",
            "ci95_low", "ci95_high", "permutation_p", "positive_seed_rate",
            "worst_quartile_mean", "leave_largest_out_mean", "holm_p",
        ])
    frame["holm_p"] = holm_adjust(frame["permutation_p"].tolist())
    return frame


def prediction_scores(traces: pd.DataFrame) -> pd.DataFrame:
    required = {"predicted_probability", "observed_action", "treatment"}
    if traces.empty or not required.issubset(traces.columns):
        return pd.DataFrame(columns=["treatment", "n", "log_loss", "brier"])
    clipped = traces.copy()
    clipped["predicted_probability"] = clipped["predicted_probability"].clip(1e-9, 1 - 1e-9)
    clipped["target"] = (clipped["observed_action"] == clipped["predicted_action"]).astype(float)
    clipped["log_loss_row"] = -(
        clipped["target"] * np.log(clipped["predicted_probability"])
        + (1 - clipped["target"]) * np.log(1 - clipped["predicted_probability"])
    )
    clipped["brier_row"] = (
        clipped["predicted_probability"] - clipped["target"]
    ) ** 2
    return clipped.groupby("treatment", as_index=False).agg(
        n=("target", "size"),
        log_loss=("log_loss_row", "mean"),
        brier=("brier_row", "mean"),
    )
