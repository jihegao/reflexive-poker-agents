from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from reflexive_poker.llm_player import DeterministicNarrativeProvider
from reflexive_poker.three_round_experiment import (
    ThreeRoundConfig,
    _lineups,
    run_three_round_experiment,
)


def test_three_round_mock_contract_and_resume(tmp_path: Path) -> None:
    factory = lambda kind, model, seed: DeterministicNarrativeProvider(seed=seed)
    config = ThreeRoundConfig(
        seeds=(9960,),
        hands=1,
        round3_lineup_count=1,
        gto_iterations=25,
        output_dir=tmp_path / "three_round",
        provider_factory=factory,
        model_specs=(
            ("deepseek", "mock", "mock-narrative-v1"),
            ("luna", "mock", "mock-narrative-v1"),
        ),
    )

    first = run_three_round_experiment(config)
    assert first["provider_gate"]["valid"] is True
    assert first["provider_gate"]["valid_match_count"] == 5

    decisions = first["match_summary"]
    assert set(decisions["round"]) == {1, 2, 3}
    r1 = json.loads((tmp_path / "three_round/matches/r1_seed9960_layout0.json").read_text())
    r2 = json.loads((tmp_path / "three_round/matches/r2_seed9960_layout0.json").read_text())
    r3 = json.loads((tmp_path / "three_round/matches/r3_seed9960_layout0.json").read_text())
    assert all("equity_estimate" not in row["state"] for row in r1["decisions"])
    assert all("gto_reference" in row["state"] for row in r2["decisions"])
    assert all("simulation_tool" in row["state"] for row in r3["decisions"])
    assert not r1["reflections"]
    assert r3["reflections"]
    assert first["evidence_gate"]["valid"] is True
    assert first["evidence_gate"]["formal_conclusion_allowed"] is False
    assert (tmp_path / "three_round/PLAN.json").exists()
    assert (tmp_path / "three_round/COMPLETED.json").exists()
    assert (tmp_path / "three_round/inference_summary.csv").exists()
    assert (tmp_path / "three_round/cost_summary.csv").exists()

    second = run_three_round_experiment(config)
    assert second["provider_gate"]["match_count"] == 5
    assert len(list((tmp_path / "three_round/matches").glob("*.json"))) == 5


def test_three_round_plan_mismatch_fails_closed(tmp_path: Path) -> None:
    factory = lambda kind, model, seed: DeterministicNarrativeProvider(seed=seed)
    config = ThreeRoundConfig(
        seeds=(9970,),
        hands=1,
        round3_lineup_count=1,
        gto_iterations=10,
        output_dir=tmp_path / "locked",
        provider_factory=factory,
        model_specs=(
            ("deepseek", "mock", "mock-narrative-v1"),
            ("luna", "mock", "mock-narrative-v1"),
        ),
    )
    run_three_round_experiment(config)
    with pytest.raises(ValueError, match="plan mismatch"):
        run_three_round_experiment(replace(config, hands=2))


def test_formal_schedule_is_complementary_and_can_complete(tmp_path: Path) -> None:
    lineups = _lineups(6)
    assert len(lineups) == 6
    assert all(sum(lineup[seat] == "deepseek" for lineup in lineups) == 3 for seat in range(6))

    factory = lambda kind, model, seed: DeterministicNarrativeProvider(seed=seed)
    config = ThreeRoundConfig(
        seeds=(9980, 9981),
        hands=1,
        round3_lineup_count=2,
        gto_iterations=10,
        evidence_tier="formal",
        minimum_formal_seeds=2,
        bootstrap_samples=20,
        permutation_samples=20,
        output_dir=tmp_path / "formal",
        provider_factory=factory,
        model_specs=(
            ("deepseek", "mock", "mock-narrative-v1"),
            ("luna", "mock", "mock-narrative-v1"),
        ),
    )
    result = run_three_round_experiment(config)
    assert result["spec_count"] == 12
    assert result["evidence_gate"]["formal_conclusion_allowed"] is True
    assert set(result["inference_summary"]["round"]) == {1, 2, 3}

    dirty = run_three_round_experiment(
        replace(config, output_dir=tmp_path / "formal_dirty", source_clean=False)
    )
    assert dirty["evidence_gate"]["formal_conclusion_allowed"] is False
    assert dirty["evidence_gate"]["requirements"]["clean_source_snapshot"] is False
