from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from reflexive_poker import phase1_offline
from reflexive_poker.llm_player import ProviderResponse
from reflexive_poker.phase1_models import ProviderBudget, ProviderLedger, ReasoningTreatment
from reflexive_poker.phase1_offline import (
    OfflineBenchmarkConfig,
    OfflineProgressConflict,
    _treatment_view,
    _validate_prediction,
    generate_offline_cases,
    run_offline_benchmark,
)


class _ResumableFakeProvider:
    name = "opencode_go"

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls = 0

    def structured(self, *, instructions, state, schema_name, schema):
        del instructions, schema_name, schema
        self.calls += 1
        table = {
            key: state["table_state"][key]
            for key in (
                "street",
                "position",
                "pot_bb",
                "effective_stack_bb",
                "spr",
                "pot_odds",
                "hand_class",
                "equity",
                "legal_actions",
            )
        }
        return ProviderResponse(
            payload={
                "table_state": table,
                "type_probabilities": {
                    "rock": 0.2,
                    "tag": 0.2,
                    "lag": 0.2,
                    "calling_station": 0.2,
                    "myopic": 0.2,
                },
                "action_probabilities": {"fold": 0.3, "check_call": 0.4, "raise": 0.3},
                "hero_image_aggression": 0.5,
                "adaptation_probability": 0.0,
                "switch_detected": False,
                "recommended_action": state["table_state"]["legal_actions"][0],
                "confidence": 0.5,
                "audit_summary": "test fake",
            },
            provider=self.name,
            model=self.model,
            latency_ms=1.0,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            cost_usd=0.01,
            actual_model=self.model,
            cost_observability="exact",
            model_identity_source="opencode_session_export",
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
        "table_state": {
            key: case["table_state"][key]
            for key in (
                "street", "position", "pot_bb", "effective_stack_bb", "spr", "pot_odds",
                "hand_class", "equity", "legal_actions",
            )
        },
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
    assert oracle["u_table_total"].min() == pytest.approx(1.0)
    assert oracle["type_brier"].max() == pytest.approx(0.0)
    assert oracle["action_brier"].max() == pytest.approx(0.0)
    assert (tmp_path / "cases.jsonl.gz").exists()
    assert (tmp_path / "predictions.jsonl.gz").exists()
    assert (tmp_path / "scores_per_case.csv").exists()
    assert {"checkpoint_index", "hand_index"}.issubset(result["scores_per_case"].columns)
    assert (tmp_path / "offline_trajectory_deltas.csv").exists()
    assert (tmp_path / "type_calibration_summary.csv").exists()
    assert (tmp_path / "type_calibration_reliability.csv").exists()
    assert json.loads((tmp_path / "provider_gate.json").read_text())["valid"] is True


def test_live_offline_predictions_resume_from_durable_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = OfflineBenchmarkConfig(
        output_dir=tmp_path,
        provider="opencode-go",
        model="resume-test",
        case_count=1,
        treatments=(
            ReasoningTreatment.STATE_ONLY,
            ReasoningTreatment.ACTION_PREDICTION,
        ),
        provider_budget=ProviderBudget(max_calls=4, max_primary_calls=2, max_retries=1),
    )
    cases = generate_offline_cases(17)[:1]
    first_treatment = config.treatments[0]
    first_payload = phase1_offline._mock_treatment_prediction(cases[0], first_treatment)
    first_row = phase1_offline._model_row(
        cases[0],
        first_treatment,
        {
            "provider": "opencode_go",
            "model": config.model,
            "latency_ms": 1.0,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "cost_usd": 0.01,
        },
        first_payload,
        1,
    )
    phase1_offline._append_journal_row(tmp_path / "live_predictions.jsonl", first_row)
    plan = phase1_offline._live_plan(cases, config)
    (tmp_path / "LIVE_PROGRESS.json").write_text(
        json.dumps({**plan, "completed_predictions": 1, "state": "running"}),
        encoding="utf-8",
    )
    (tmp_path / "live_provider_ledger.json").write_text(
        json.dumps(
            {
                "provider": "opencode_go",
                "model": config.model,
                "budget": asdict(config.provider_budget),
                "ledger": ProviderLedger(
                    calls=1,
                    input_tokens=1,
                    output_tokens=1,
                    total_tokens=2,
                    token_observed_calls=1,
                    latency_ms=1.0,
                    cost_usd=0.01,
                    cost_observed_calls=1,
                ).snapshot(),
            }
        ),
        encoding="utf-8",
    )
    fake = _ResumableFakeProvider(config.model)
    monkeypatch.setattr(phase1_offline, "_provider", lambda provider, model: fake)

    rows, ledger = phase1_offline.model_predictions(cases, config)

    assert len(rows) == 2
    assert fake.calls == 1
    assert ledger.calls == 2
    assert len(phase1_offline._read_journal(tmp_path / "live_predictions.jsonl")) == 2


def test_live_offline_predictions_fail_closed_for_unfinished_inflight_call(tmp_path: Path) -> None:
    config = OfflineBenchmarkConfig(
        output_dir=tmp_path,
        provider="opencode-go",
        model="resume-test",
        case_count=1,
    )
    (tmp_path / "LIVE_INFLIGHT.json").write_text("{}", encoding="utf-8")

    with pytest.raises(OfflineProgressConflict, match="in flight"):
        phase1_offline.model_predictions(generate_offline_cases(17)[:1], config)
