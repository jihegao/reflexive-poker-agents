from __future__ import annotations

from pathlib import Path

from reflexive_poker.expctl import _load_config

PHASE2_SYSTEMS = (
    ("opencode-go", "deepseek-v4-flash"),
    ("opencode-go", "qwen3.7-max"),
    ("opencode-go", "glm-5.2"),
    ("codex", "gpt-5.6-luna"),
)
ALL_TREATMENTS = (
    "state_only",
    "action_prediction",
    "d1_budget_matched",
    "recursive_d2",
    "recursive_d3",
)


def test_phase2_manifest_is_valid_and_freezes_four_serving_systems() -> None:
    config = _load_config(Path("configs/phase2.yaml").resolve())

    assert config["protocol"] == "prbench-cross-model-v1"
    systems = config["paper_phase2"]["serving_systems"]
    assert tuple((system["provider"], system["model"]) for system in systems) == PHASE2_SYSTEMS
    assert config["evidence_layers"]["llm_confirmation"]["models"] == systems


def test_phase2_manifest_freezes_treatments_and_six_max_external_validity() -> None:
    config = _load_config(Path("configs/phase2.yaml").resolve())
    phase2 = config["paper_phase2"]

    assert tuple(phase2["offline_understanding"]["treatments"]) == ALL_TREATMENTS
    assert tuple(phase2["external_validity"]["treatments"]) == ALL_TREATMENTS
    assert tuple(phase2["closed_loop"]["core_treatments"]) == (
        "state_only",
        "d1_budget_matched",
        "recursive_d2",
    )
    assert phase2["closed_loop"]["d3_extension"] == {
        "treatment": "recursive_d3",
        "compare_against": "recursive_d2",
        "preregister_before_outcomes": True,
    }

    six_max = phase2["external_validity"]
    assert six_max["arena"] == "six_max"
    assert six_max["composition"] == "heterogeneous_classic"
    assert six_max["lineup"] == ["llm", "tag", "lag", "rock", "calling_station", "myopic"]
    assert all(
        six_max[name] is True
        for name in (
            "shared_formation_checkpoint",
            "shared_deck_and_board",
            "shared_opponent_rng",
            "seat_mirroring",
        )
    )
    assert six_max["button_rotation"] == "required"
    assert six_max["longer_horizon_than_phase1"] == "required"
    assert six_max["larger_paired_seed_count_than_phase1"] == "required"
    assert six_max["return_power_analysis"] == "required_before_formal_run"
