"""Frozen, trajectory-level analysis helpers for the Phase 1 offline benchmark.

The offline benchmark contains several observations from the same generated
trajectory.  These helpers deliberately make a trajectory (rather than a row)
the resampling and paired-inference unit.  They are kept separate from the
benchmark runner so analysing a completed raw score file cannot change how it
was generated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusterBootstrap:
    """A mean estimated by resampling independent trajectories."""

    estimate: float
    ci95_low: float
    ci95_high: float
    clusters: int
    observations: int


@dataclass(frozen=True)
class PairedPermutation:
    """A trajectory-paired contrast and its two-sided randomisation p-value."""

    estimate: float
    p_value: float
    trajectories: int
    paired_observations: int
    exact: bool


@dataclass(frozen=True)
class BrierDecomposition:
    """Binary Brier-score decomposition (Murphy-style grouped forecasts)."""

    brier_score: float
    reliability: float
    resolution: float
    uncertainty: float
    within_bin_variance: float
    within_bin_forecast_outcome_covariance: float
    reconstructed_brier: float
    observations: int


def _finite_array(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_probability_inputs(
    observed: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    targets = _finite_array(observed, name="observed")
    forecasts = _finite_array(probabilities, name="probabilities")
    if len(targets) != len(forecasts):
        raise ValueError("observed and probabilities must have equal length")
    if not np.isin(targets, (0.0, 1.0)).all():
        raise ValueError("observed must contain binary 0/1 outcomes")
    if ((forecasts < 0.0) | (forecasts > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")
    return targets, forecasts


def trajectory_cluster_bootstrap(
    scores: pd.DataFrame,
    *,
    value_col: str,
    trajectory_col: str = "trajectory_id",
    samples: int = 5_000,
    seed: int = 20260802,
) -> ClusterBootstrap:
    """Bootstrap a row-level mean by resampling whole trajectories.

    Sampling a trajectory carries all its rows with it, preserving within-
    trajectory correlation and allowing unequal trajectory lengths.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    missing = {value_col, trajectory_col}.difference(scores.columns)
    if missing:
        raise ValueError(f"scores missing required columns: {sorted(missing)}")
    clean = scores[[trajectory_col, value_col]].dropna().copy()
    if clean.empty:
        raise ValueError("no finite observations available for bootstrap")
    clean[value_col] = pd.to_numeric(clean[value_col], errors="raise")
    if not np.isfinite(clean[value_col].to_numpy(dtype=float)).all():
        raise ValueError("value column must contain only finite values")
    # Keep both total and count: bootstrap samples retain the original statistic
    # even when generated trajectories contain a different number of cases.
    clusters = clean.groupby(trajectory_col, sort=False)[value_col].agg(["sum", "count"])
    cluster_sums = clusters["sum"].to_numpy(dtype=float)
    cluster_counts = clusters["count"].to_numpy(dtype=float)
    estimate = float(cluster_sums.sum() / cluster_counts.sum())
    if len(clusters) == 1:
        return ClusterBootstrap(estimate, float("nan"), float("nan"), 1, len(clean))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clusters), size=(samples, len(clusters)))
    means = cluster_sums[indices].sum(axis=1) / cluster_counts[indices].sum(axis=1)
    return ClusterBootstrap(
        estimate=estimate,
        ci95_low=float(np.quantile(means, 0.025)),
        ci95_high=float(np.quantile(means, 0.975)),
        clusters=len(clusters),
        observations=len(clean),
    )


def paired_trajectory_deltas(
    scores: pd.DataFrame,
    *,
    metric: str,
    treatment_a: str,
    treatment_b: str,
    trajectory_col: str = "trajectory_id",
    treatment_col: str = "treatment",
    pair_cols: Sequence[str] = ("case_id",),
) -> pd.DataFrame:
    """Return one mean ``A - B`` delta per complete trajectory.

    A row only enters after it has an exact counterpart in the other treatment.
    This fail-closed pairing prevents a partly failed arm from silently becoming
    evidence.  Duplicate treatment rows at the same pairing key are ambiguous
    and rejected rather than averaged.
    """
    required = {metric, trajectory_col, treatment_col, *pair_cols}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"scores missing required columns: {sorted(missing)}")
    if treatment_a == treatment_b:
        raise ValueError("treatment_a and treatment_b must differ")
    work = scores.loc[
        scores[treatment_col].isin((treatment_a, treatment_b)),
        [trajectory_col, *pair_cols, treatment_col, metric],
    ].dropna()
    if work.empty:
        return pd.DataFrame(columns=[trajectory_col, "delta", "paired_observations"])
    work = work.copy()
    work[metric] = pd.to_numeric(work[metric], errors="raise")
    if not np.isfinite(work[metric].to_numpy(dtype=float)).all():
        raise ValueError("metric must contain only finite values")
    keys = [trajectory_col, *pair_cols]
    if work.duplicated([*keys, treatment_col]).any():
        raise ValueError("duplicate treatment rows at a trajectory pairing key")
    wide = work.pivot(index=keys, columns=treatment_col, values=metric)
    if treatment_a not in wide or treatment_b not in wide:
        return pd.DataFrame(columns=[trajectory_col, "delta", "paired_observations"])
    paired = wide.dropna(subset=[treatment_a, treatment_b]).copy()
    if paired.empty:
        return pd.DataFrame(columns=[trajectory_col, "delta", "paired_observations"])
    paired["delta"] = paired[treatment_a] - paired[treatment_b]
    result = (
        paired.reset_index()
        .groupby(trajectory_col, as_index=False, sort=False)
        .agg(delta=("delta", "mean"), paired_observations=("delta", "size"))
    )
    return result


def within_trajectory_paired_permutation(
    deltas: pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    delta_col: str = "delta",
    paired_observations_col: str = "paired_observations",
    samples: int = 20_000,
    seed: int = 20260802,
) -> PairedPermutation:
    """Two-sided sign-flip test over paired trajectory means.

    Exact enumeration is used through 18 trajectories; otherwise reproducible
    Monte Carlo signs are used.  The +1 correction keeps sampled p-values from
    ever being reported as zero; enumerated exact distributions use their
    conventional exact tail mass.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    paired_observations = 0
    if isinstance(deltas, pd.DataFrame):
        if delta_col not in deltas:
            raise ValueError(f"deltas missing required column: {delta_col}")
        values = deltas[delta_col].dropna().to_numpy(dtype=float)
        if paired_observations_col in deltas:
            paired_observations = int(deltas[paired_observations_col].sum())
    else:
        values = np.asarray(deltas, dtype=float)
    values = _finite_array(values, name="deltas")
    observed = abs(float(values.mean()))
    exact = len(values) <= 18
    if exact:
        masks = np.arange(1 << len(values), dtype=np.uint64)[:, None]
        bits = (masks >> np.arange(len(values), dtype=np.uint64)) & 1
        signs = np.where(bits == 1, 1.0, -1.0)
        null_statistics = np.abs((signs * values).mean(axis=1))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(samples, len(values)))
        null_statistics = np.abs((signs * values).mean(axis=1))
    p_value = (
        float(np.count_nonzero(null_statistics >= observed) / len(null_statistics))
        if exact
        else float(
            (np.count_nonzero(null_statistics >= observed) + 1)
            / (len(null_statistics) + 1)
        )
    )
    return PairedPermutation(
        estimate=float(values.mean()),
        p_value=p_value,
        trajectories=len(values),
        paired_observations=paired_observations,
        exact=exact,
    )


def holm_adjust(p_values: Mapping[str, float] | Sequence[float]) -> dict[str, float] | list[float]:
    """Holm step-down adjustment while preserving names/order and NaN values."""
    is_mapping = isinstance(p_values, Mapping)
    items = list(p_values.items()) if is_mapping else list(enumerate(p_values))
    adjusted: dict[Any, float] = {key: float("nan") for key, _ in items}
    finite = [(key, float(value)) for key, value in items if math.isfinite(float(value))]
    if any(value < 0.0 or value > 1.0 for _, value in finite):
        raise ValueError("p-values must be in [0, 1]")
    ordered = sorted(finite, key=lambda item: item[1])
    running = 0.0
    count = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[key] = running
    if is_mapping:
        return {str(key): adjusted[key] for key, _ in items}
    return [adjusted[index] for index, _ in items]


def _bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    estimate = float(left.mean() - right.mean())
    if len(left) < 2 or len(right) < 2:
        return estimate, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    left_indices = rng.integers(0, len(left), size=(samples, len(left)))
    right_indices = rng.integers(0, len(right), size=(samples, len(right)))
    draws = left[left_indices].mean(axis=1) - right[right_indices].mean(axis=1)
    return estimate, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _two_sample_permutation_p(
    left: np.ndarray, right: np.ndarray, *, samples: int, seed: int
) -> tuple[float, bool]:
    """Random-label test for whether two trajectory-delta distributions differ."""
    observed = abs(float(left.mean() - right.mean()))
    combined = np.concatenate((left, right))
    # Exact combinations are tractable for small smoke fixtures.
    total_combinations = math.comb(len(combined), len(left))
    if total_combinations <= 50_000:
        from itertools import combinations

        null = np.fromiter(
            (
                abs(combined[list(selected)].mean() - combined[np.setdiff1d(np.arange(len(combined)), selected)].mean())
                for selected in combinations(range(len(combined)), len(left))
            ),
            dtype=float,
            count=total_combinations,
        )
        return float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1)), True
    rng = np.random.default_rng(seed)
    null = np.empty(samples, dtype=float)
    for index in range(samples):
        shuffled = rng.permutation(combined)
        null[index] = abs(shuffled[: len(left)].mean() - shuffled[len(left) :].mean())
    return float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1)), False


def d2_d1bm_post_switch_contrasts(
    scores: pd.DataFrame,
    *,
    metric: str,
    d2_treatment: str = "recursive_d2",
    d1bm_treatment: str = "d1_budget_matched",
    dynamic_regime: str = "adaptive_shift",
    fixed_regime: str = "fixed",
    regime_col: str = "regime",
    post_switch_col: str = "post_switch",
    bootstrap_samples: int = 5_000,
    permutation_samples: int = 20_000,
    seed: int = 20260802,
) -> pd.DataFrame:
    """Pre-registered post-switch D2-D1BM contrasts and adaptive-v-fixed interaction.

    The first two rows are within-trajectory paired sign-flip tests.  The final
    interaction row tests whether those trajectory contrasts differ between the
    dynamic and fixed regimes by randomly relabelling whole trajectories.
    """
    required = {regime_col, post_switch_col}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"scores missing required columns: {sorted(missing)}")
    if bootstrap_samples < 1 or permutation_samples < 1:
        raise ValueError("bootstrap_samples and permutation_samples must be positive")
    post_switch = scores.loc[scores[post_switch_col].astype(bool)].copy()
    by_regime: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for regime in (dynamic_regime, fixed_regime):
        deltas = paired_trajectory_deltas(
            post_switch.loc[post_switch[regime_col] == regime],
            metric=metric,
            treatment_a=d2_treatment,
            treatment_b=d1bm_treatment,
        )
        if deltas.empty:
            raise ValueError(f"no complete paired post-switch trajectories for regime: {regime}")
        by_regime[regime] = deltas
        boot = trajectory_cluster_bootstrap(deltas, value_col="delta", samples=bootstrap_samples, seed=seed)
        permutation = within_trajectory_paired_permutation(
            deltas, samples=permutation_samples, seed=seed
        )
        rows.append(
            {
                "contrast": f"D2-D1BM_post_switch_{regime}",
                "regime": regime,
                "estimate": boot.estimate,
                "ci95_low": boot.ci95_low,
                "ci95_high": boot.ci95_high,
                "permutation_p": permutation.p_value,
                "permutation_exact": permutation.exact,
                "trajectories": boot.clusters,
                "paired_observations": permutation.paired_observations,
                "inference_unit": "trajectory",
            }
        )
    dynamic = by_regime[dynamic_regime]["delta"].to_numpy(dtype=float)
    fixed = by_regime[fixed_regime]["delta"].to_numpy(dtype=float)
    estimate, low, high = _bootstrap_difference(
        dynamic, fixed, samples=bootstrap_samples, seed=seed + 1
    )
    interaction_p, exact = _two_sample_permutation_p(
        dynamic, fixed, samples=permutation_samples, seed=seed + 1
    )
    rows.append(
        {
            "contrast": "D2-D1BM_post_switch_adaptive_minus_fixed",
            "regime": "adaptive_minus_fixed",
            "estimate": estimate,
            "ci95_low": low,
            "ci95_high": high,
            "permutation_p": interaction_p,
            "permutation_exact": exact,
            "trajectories": len(dynamic) + len(fixed),
            "paired_observations": int(
                by_regime[dynamic_regime]["paired_observations"].sum()
                + by_regime[fixed_regime]["paired_observations"].sum()
            ),
            "inference_unit": "trajectory",
        }
    )
    result = pd.DataFrame(rows)
    result["holm_p"] = holm_adjust(result["permutation_p"].tolist())
    return result


def reliability_curve(
    observed: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Return non-empty fixed-width calibration bins for binary probabilities."""
    if bins < 1:
        raise ValueError("bins must be positive")
    targets, forecasts = _validate_probability_inputs(observed, probabilities)
    assignments = np.minimum((forecasts * bins).astype(int), bins - 1)
    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        mask = assignments == index
        if not mask.any():
            continue
        prediction = float(forecasts[mask].mean())
        frequency = float(targets[mask].mean())
        rows.append(
            {
                "bin": index,
                "bin_lower": index / bins,
                "bin_upper": (index + 1) / bins,
                "n": int(mask.sum()),
                "mean_prediction": prediction,
                "observed_frequency": frequency,
                "absolute_gap": abs(prediction - frequency),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    observed: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Fixed-bin expected calibration error for a binary forecast."""
    curve = reliability_curve(observed, probabilities, bins=bins)
    if curve.empty:
        return float("nan")
    return float((curve["n"] * curve["absolute_gap"]).sum() / curve["n"].sum())


def brier_decomposition(
    observed: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    bins: int | None = None,
) -> BrierDecomposition:
    """Decompose binary Brier score exactly, including optional bin coarsening.

    With ``bins=None`` each distinct forecast value is its own group and the
    usual reliability - resolution + uncertainty identity applies.  Fixed bins
    are convenient for visual reporting; their within-bin forecast variance is
    made explicit so ``reconstructed_brier`` still equals the raw Brier score.
    """
    targets, forecasts = _validate_probability_inputs(observed, probabilities)
    if bins is not None and bins < 1:
        raise ValueError("bins must be positive when supplied")
    if bins is None:
        labels, inverse = np.unique(forecasts, return_inverse=True)
    else:
        inverse = np.minimum((forecasts * bins).astype(int), bins - 1)
        labels = np.arange(bins)
    base_rate = float(targets.mean())
    reliability = resolution = within = covariance = 0.0
    for group in range(len(labels)):
        mask = inverse == group
        if not mask.any():
            continue
        weight = float(mask.mean())
        mean_forecast = float(forecasts[mask].mean())
        observed_rate = float(targets[mask].mean())
        reliability += weight * (mean_forecast - observed_rate) ** 2
        resolution += weight * (observed_rate - base_rate) ** 2
        within += weight * float(np.var(forecasts[mask]))
        covariance += weight * float(
            np.mean((forecasts[mask] - mean_forecast) * (targets[mask] - observed_rate))
        )
    uncertainty = base_rate * (1.0 - base_rate)
    brier = float(np.mean((forecasts - targets) ** 2))
    # Coarsening can correlate forecast deviations within a bin with outcomes.
    # Reporting that covariance makes the grouped decomposition reconstruct the
    # raw Brier score rather than a bin-mean approximation.
    reconstructed = reliability - resolution + uncertainty + within - 2.0 * covariance
    return BrierDecomposition(
        brier_score=brier,
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=float(uncertainty),
        within_bin_variance=float(within),
        within_bin_forecast_outcome_covariance=float(covariance),
        reconstructed_brier=float(reconstructed),
        observations=len(targets),
    )
