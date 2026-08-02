from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from reflexive_poker.phase1_offline_evidence import (
    brier_decomposition,
    d2_d1bm_post_switch_contrasts,
    expected_calibration_error,
    holm_adjust,
    offline_trajectory_inference,
    paired_trajectory_deltas,
    reliability_curve,
    trajectory_cluster_bootstrap,
    type_calibration_tables,
    within_trajectory_paired_permutation,
)


def _scores() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # Adaptive trajectories have a D2 post-switch reduction of 0.30; fixed
    # trajectories have a smaller 0.05 reduction.  Cases are deliberately
    # repeated within a trajectory so row-level pseudo-replication is detectable.
    for regime, delta in (("adaptive_shift", -0.30), ("fixed", -0.05)):
        for trajectory in range(3):
            for case in range(2):
                baseline = 0.60 + trajectory * 0.01 + case * 0.001
                rows.extend(
                    (
                        {
                            "trajectory_id": f"{regime}-{trajectory}",
                            "case_id": f"case-{case}",
                            "regime": regime,
                            "post_switch": True,
                            "treatment": "d1_budget_matched",
                            "action_brier": baseline,
                        },
                        {
                            "trajectory_id": f"{regime}-{trajectory}",
                            "case_id": f"case-{case}",
                            "regime": regime,
                            "post_switch": True,
                            "treatment": "recursive_d2",
                            "action_brier": baseline + delta,
                        },
                    )
                )
    return pd.DataFrame(rows)


def test_cluster_bootstrap_resamples_complete_trajectories() -> None:
    scores = pd.DataFrame(
        {"trajectory_id": ["short", "long", "long", "long"], "metric": [0.0, 2.0, 2.0, 2.0]}
    )
    result = trajectory_cluster_bootstrap(scores, value_col="metric", samples=400, seed=8)
    assert result.estimate == pytest.approx(1.5)
    assert result.clusters == 2
    assert result.observations == 4
    assert result.ci95_low <= result.estimate <= result.ci95_high


def test_paired_deltas_exclude_incomplete_cases_and_reject_duplicates() -> None:
    scores = _scores()
    incomplete = scores.loc[~((scores["trajectory_id"] == "fixed-0") & (scores["case_id"] == "case-1") & (scores["treatment"] == "recursive_d2"))]
    deltas = paired_trajectory_deltas(
        incomplete,
        metric="action_brier",
        treatment_a="recursive_d2",
        treatment_b="d1_budget_matched",
    )
    fixed_zero = deltas.loc[deltas["trajectory_id"] == "fixed-0"].iloc[0]
    assert fixed_zero["paired_observations"] == 1
    assert fixed_zero["delta"] == pytest.approx(-0.05)
    duplicated = pd.concat((scores, scores.iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="duplicate treatment rows"):
        paired_trajectory_deltas(
            duplicated,
            metric="action_brier",
            treatment_a="recursive_d2",
            treatment_b="d1_budget_matched",
        )


def test_within_trajectory_permutation_uses_exact_sign_flips() -> None:
    result = within_trajectory_paired_permutation(np.array([-1.0, -1.0, -1.0]))
    assert result.exact is True
    assert result.estimate == pytest.approx(-1.0)
    assert result.p_value == pytest.approx(0.25)


def test_holm_adjustment_preserves_named_hypotheses_and_nan() -> None:
    adjusted = holm_adjust({"h1": 0.01, "h2": 0.03, "h3": 0.04, "missing": math.nan})
    assert adjusted["h1"] == pytest.approx(0.03)
    assert adjusted["h2"] == pytest.approx(0.06)
    assert adjusted["h3"] == pytest.approx(0.06)
    assert math.isnan(adjusted["missing"])


def test_post_switch_contrasts_include_fixed_interaction_and_holm() -> None:
    result = d2_d1bm_post_switch_contrasts(
        _scores(), metric="action_brier", bootstrap_samples=300, permutation_samples=300, seed=6
    )
    assert result["contrast"].tolist() == [
        "D2-D1BM_post_switch_adaptive_shift",
        "D2-D1BM_post_switch_fixed",
        "D2-D1BM_post_switch_adaptive_minus_fixed",
    ]
    assert result.loc[0, "estimate"] == pytest.approx(-0.30)
    assert result.loc[1, "estimate"] == pytest.approx(-0.05)
    assert result.loc[2, "estimate"] == pytest.approx(-0.25)
    assert result["inference_unit"].eq("trajectory").all()
    assert result["holm_p"].between(0.0, 1.0).all()


def test_offline_inference_preserves_trajectory_deltas_and_applies_one_holm_family() -> None:
    result, deltas = offline_trajectory_inference(
        _scores().assign(provider="test", model="test"),
        metrics=("action_brier",),
        bootstrap_samples=100,
        permutation_samples=100,
        seed=7,
    )
    assert len(result) == 3
    assert result["metric"].eq("action_brier").all()
    assert result["holm_p"].between(0.0, 1.0).all()
    assert set(deltas["regime"]) == {"fixed", "adaptive_shift"}
    assert deltas["paired_observations"].eq(2).all()


def test_calibration_helpers_report_reliability_ece_and_exact_decomposition() -> None:
    observed = np.array([0.0, 0.0, 1.0, 1.0])
    probabilities = np.array([0.1, 0.1, 0.9, 0.9])
    curve = reliability_curve(observed, probabilities, bins=2)
    assert curve["n"].tolist() == [2, 2]
    assert expected_calibration_error(observed, probabilities, bins=2) == pytest.approx(0.1)
    decomposition = brier_decomposition(observed, probabilities)
    assert decomposition.brier_score == pytest.approx(0.01)
    assert decomposition.reliability == pytest.approx(0.01)
    assert decomposition.resolution == pytest.approx(0.25)
    assert decomposition.uncertainty == pytest.approx(0.25)
    assert decomposition.reconstructed_brier == pytest.approx(decomposition.brier_score)


def test_type_calibration_tables_use_observed_active_type_not_action_distribution() -> None:
    cases = [
        {"case_id": "a", "ground_truth": {"active_type": "rock"}},
        {"case_id": "b", "ground_truth": {"active_type": "tag"}},
    ]
    predictions = [
        {
            "case_id": "a",
            "provider": "provider",
            "model": "model",
            "method": "llm_recursive_d2",
            "treatment": "recursive_d2",
            "payload": {"type_probabilities": {"rock": 0.8, "tag": 0.2}},
        },
        {
            "case_id": "b",
            "provider": "provider",
            "model": "model",
            "method": "llm_recursive_d2",
            "treatment": "recursive_d2",
            "payload": {"type_probabilities": {"rock": 0.3, "tag": 0.7}},
        },
    ]
    summary, reliability = type_calibration_tables(cases, predictions, bins=2)
    assert set(summary["target"]) == {"observed_active_type_one_vs_rest"}
    assert set(summary["class"]) == {"rock", "tag"}
    assert summary["ece"].notna().all()
    assert set(reliability["class"]) == {"rock", "tag"}


def test_binned_decomposition_accounts_for_within_bin_forecast_variance() -> None:
    observed = np.array([0.0, 1.0])
    probabilities = np.array([0.2, 0.4])
    decomposition = brier_decomposition(observed, probabilities, bins=1)
    assert decomposition.within_bin_variance == pytest.approx(0.01)
    assert decomposition.reconstructed_brier == pytest.approx(decomposition.brier_score)
