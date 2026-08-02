from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflexive_poker.phase1_models import ReasoningTreatment
from reflexive_poker.phase1_offline import (
    OfflineBenchmarkConfig,
    _treatment_view,
    _validate_prediction,
    generate_offline_cases,
    run_offline_benchmark,
)


def test_offline_generator_is_balanced_and_deterministic() -> None:
    first = generate_offline_cases(17)
    second = generate_offline_cases(17)
    assert len(first) == 200
    assert [row["case_hash"] for row in first] == [row["case_hash"] for row in second]
    assert len({row["trajectory_id"] for row in first}) == 50
    assert {row["regime"] for row in first} == {"fixed", "adaptive_shift"}
    assert {row["table_state"]["street"] for row in first} == {
        "preflop",
        "flop",
        "turn",
        "river",
    }


def test_budget_matched_view_masks_recursive_information() -> None:
    case = generate_offline_cases(17)[0]
    control = _treatment_view(case, ReasoningTreatment.BUDGET_MATCHED_D1)
    recursive = _treatment_view(case, ReasoningTreatment.RECURSIVE_D2)
    assert control["recursive_public_summary"] == "__MASKED__"
    assert control["budget_match_control"]
    assert isinstance(recursive["recursive_public_summary"], dict)
    assert recursive["budget_match_control"] == "__MASKED__"


def test_prediction_validation_rejects_non_normalized_probabilities() -> None:
    case = generate_offline_cases(17)[0]
    payload = {
        "type_probabilities": {
            "rock": 1.0,
            "tag": 1.0,
            "lag": 0.0,
            "calling_station": 0.0,
            "myopic": 0.0,
        },
        "action_probabilities": {"fold": 0.2, "check_call": 0.5, "raise": 0.3},
        "hero_image_aggression": 0.5,
        "adaptation_probability": 0.0,
        "switch_detected": False,
        "recommended_action": case["table_state"]["legal_actions"][0],
        "confidence": 0.5,
        "audit_summary": "test",
    }
    with pytest.raises(ValueError, match="sum to one"):
        _validate_prediction(payload, case["table_state"]["legal_actions"])


def test_offline_mock_smoke_writes_raw_and_scored_artifacts(tmp_path: Path) -> None:
    result = run_offline_benchmark(
        OfflineBenchmarkConfig(
            output_dir=tmp_path,
            provider="mock",
            model="mock",
            case_count=8,
        )
    )
    assert len(result["cases"]) == 8
    assert len(result["predictions"]) == 8 * 10
    assert result["provider_gate"]["valid"] is True
    oracle = result["scores_per_case"][result["scores_per_case"]["method"] == "oracle"]
    assert oracle["type_brier"].max() == pytest.approx(0.0)
    assert oracle["action_brier"].max() == pytest.approx(0.0)
    assert (tmp_path / "cases.jsonl.gz").exists()
    assert (tmp_path / "predictions.jsonl.gz").exists()
    assert (tmp_path / "scores_per_case.csv").exists()
    assert json.loads((tmp_path / "provider_gate.json").read_text())["valid"] is True
