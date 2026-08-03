from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from reflexive_poker.phase2_framework import (
    PHASE2_TREATMENTS,
    Phase2ExecutionError,
    Phase2OfflineRunConfig,
    Phase2PowerAnalysisConfig,
    Phase2SixMaxRunConfig,
    ServingSystem,
    analyze_phase2_six_max,
    assert_phase2_outcomes_ready,
    build_phase2_six_max_schedule,
    pareto_frontier,
    run_phase2_offline,
    systems_from_phase2,
    write_phase2_power_analysis,
)


def _phase2_and_readiness() -> tuple[dict[str, object], dict[str, object]]:
    phase2 = yaml.safe_load(Path("configs/phase2.yaml").read_text(encoding="utf-8"))["paper_phase2"]
    preflight: dict[str, object] = {}
    locks: dict[str, object] = {}
    for system in phase2["serving_systems"]:
        name = system["serving_system"]
        preflight[name] = {
            "valid": True,
            "expected_predictions": 20,
            "observed_predictions": 20,
        }
        locks[name] = {"status": "frozen", "matches_preflight": True}
    return phase2, {
        "protocol": "prbench-cross-model-v1",
        "ready_for_formal_outcomes": True,
        "preflight": preflight,
        "identity_locks": locks,
    }


def _valid_offline_result(config) -> dict[str, object]:
    rows = []
    for case in range(config.case_count):
        for treatment in PHASE2_TREATMENTS:
            rows.append(
                {
                    "case_id": f"case-{case}",
                    "trajectory_id": f"trajectory-{case // 2}",
                    "provider": config.provider,
                    "model": config.model,
                    "treatment": treatment,
                    "u_table_total": 0.7,
                    "type_brier": 0.4,
                    "action_brier": 0.3,
                    "decision_regret": 0.1,
                    "total_tokens": 120,
                    "latency_ms": 75.0,
                    "cost_usd": 0.001,
                }
            )
    expected = config.case_count * len(PHASE2_TREATMENTS)
    return {
        "scores_per_case": pd.DataFrame(rows),
        "provider_gate": {
            "valid": True,
            "expected_predictions": expected,
            "observed_predictions": expected,
            "zero_unresolved_failures": True,
            "actual_identity_matches": True,
            "model_identity_source_valid": True,
            "complete_token_accounting": True,
            "complete_model_version_attestation": True,
            "budget_match_valid": True,
            "ledger": {"fallbacks": 0},
        },
    }


def test_phase2_outcomes_stay_blocked_before_readiness_is_complete() -> None:
    phase2, readiness = _phase2_and_readiness()
    readiness["ready_for_formal_outcomes"] = False
    with pytest.raises(Phase2ExecutionError, match="blocked"):
        assert_phase2_outcomes_ready(phase2, readiness)


def test_power_analysis_is_pre_outcome_and_bound_to_phase1_lock(tmp_path: Path) -> None:
    payload = write_phase2_power_analysis(
        tmp_path / "power.json",
        Phase2PowerAnalysisConfig(
            phase1_outcome_lock="sha256:phase1-evidence",
            paired_seed_count=80,
            heads_up_hands=80,
            expected_return_delta=1.2,
            paired_seed_stddev=2.0,
        ),
    )
    assert payload["valid"] is True
    assert payload["analysis_unit"] == "paired_seed_block"
    assert payload["phase1_outcome_lock"] == "sha256:phase1-evidence"
    assert json.loads((tmp_path / "power.json").read_text()) == payload


def test_four_system_offline_runner_requires_all_live_gates_and_marks_return_pending(
    tmp_path: Path,
) -> None:
    phase2, readiness = _phase2_and_readiness()
    calls = []

    def fake_runner(config):
        calls.append((config.provider, config.model))
        return _valid_offline_result(config)

    result = run_phase2_offline(
        Phase2OfflineRunConfig(
            phase2=phase2,
            readiness=readiness,
            output_dir=tmp_path,
            case_count=2,
        ),
        offline_runner=fake_runner,
    )

    assert len(calls) == 4
    assert len(result["summary"]) == 4
    assert not result["pareto"]["pareto_eligible"].any()
    assert (tmp_path / "analysis" / "OFFLINE_OUTCOME_GATE.json").exists()


def test_pareto_refuses_missing_return_and_uses_all_three_dimensions() -> None:
    summary = pd.DataFrame(
        [
            {"serving_system": "a", "u_opponent": 0.9, "return_chips_per_100": 2.0, "usd_per_100_valid_decisions": 1.0, "return_available": True},
            {"serving_system": "b", "u_opponent": 0.8, "return_chips_per_100": 2.0, "usd_per_100_valid_decisions": 1.0, "return_available": True},
            {"serving_system": "c", "u_opponent": 0.95, "return_chips_per_100": 1.0, "usd_per_100_valid_decisions": 1.0, "return_available": True},
            {"serving_system": "d", "u_opponent": 0.7, "return_chips_per_100": 4.0, "usd_per_100_valid_decisions": 1.0, "return_available": False},
        ]
    )
    result = pareto_frontier(summary).set_index("serving_system")
    assert not result.loc["b", "pareto_nondominated"]
    assert result.loc["a", "pareto_nondominated"]
    assert result.loc["c", "pareto_nondominated"]
    assert not result.loc["d", "pareto_eligible"]


def _six_max_rows(systems: tuple[ServingSystem, ...]) -> pd.DataFrame:
    rows = []
    for system in systems:
        for treatment_index, treatment in enumerate(PHASE2_TREATMENTS):
            for seed in range(100, 141):
                for mirror in (0, 1):
                    for hand in (0, 1):
                        rows.append(
                            {
                                "serving_system": system.serving_system,
                                "provider": system.provider,
                                "model": system.model,
                                "treatment": treatment,
                                "seed": seed,
                                "seat_mirror": mirror,
                                "hand_index": hand,
                                "reward": float(treatment_index + mirror - hand),
                                "provider_gate_valid": True,
                                "fallback_count": 0,
                                "unresolved_failures": 0,
                                "formation_fork_hash": f"{system.serving_system}-{seed}-{mirror}",
                            }
                        )
    return pd.DataFrame(rows)


def test_six_max_schedule_and_analysis_are_four_system_paired_and_seat_mirrored() -> None:
    phase2, _ = _phase2_and_readiness()
    systems = systems_from_phase2(phase2)
    schedule = build_phase2_six_max_schedule(
        Phase2SixMaxRunConfig(
            systems=systems,
            treatments=PHASE2_TREATMENTS,
            seeds=tuple(range(100, 141)),
            hands=80,
            formation_hands=20,
        )
    )
    assert len(schedule) == 4 * 5 * 41 * 2
    inference, paired_hands = analyze_phase2_six_max(
        _six_max_rows(systems), bootstrap_samples=20, permutation_samples=20
    )
    assert len(inference) == 8
    assert inference["paired_seeds"].eq(41).all()
    assert inference["inference_unit"].eq("paired_seed_clustered_over_seat_mirror_hands").all()
    assert set(paired_hands["seat_mirror"]) == {0, 1}


def test_six_max_analysis_fails_closed_on_a_missing_seat_mirror_arm() -> None:
    phase2, _ = _phase2_and_readiness()
    systems = systems_from_phase2(phase2)
    rows = _six_max_rows(systems)
    bad = rows.loc[
        ~(
            (rows["serving_system"] == systems[0].serving_system)
            & (rows["treatment"] == "recursive_d2")
            & (rows["seed"] == 100)
            & (rows["seat_mirror"] == 1)
        )
    ]
    with pytest.raises(Phase2ExecutionError, match="incomplete paired six-max arm"):
        analyze_phase2_six_max(bad, bootstrap_samples=20, permutation_samples=20)
