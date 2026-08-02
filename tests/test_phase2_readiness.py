from __future__ import annotations

import json
from pathlib import Path

import yaml

from reflexive_poker.phase2_readiness import audit_phase2_readiness


def _system_artifact(path: Path, model: str) -> None:
    path.mkdir(parents=True)
    (path / "provider_gate.json").write_text(
        json.dumps(
            {
                "valid": True,
                "expected_predictions": 20,
                "observed_predictions": 20,
                "actual_identity_matches": True,
                "model_identity_source_valid": True,
                "observed_actual_models": [model],
            }
        ),
        encoding="utf-8",
    )


def test_phase2_readiness_fails_closed_until_identity_price_and_power_are_frozen(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(Path("configs/phase2.yaml").read_text(encoding="utf-8"))
    phase2 = config["paper_phase2"]
    preflight = tmp_path / "preflight"
    for system in phase2["serving_systems"]:
        _system_artifact(
            preflight
            / f"{system['provider']}__{system['model']}".replace("/", "_").replace(".", "_"),
            system["model"],
        )

    pending = audit_phase2_readiness(
        tmp_path / "pending",
        phase2=phase2,
        preflight_dir=preflight,
        pricing_manifest=None,
        power_analysis=None,
    )
    assert not pending["ready_for_formal_outcomes"]

    for lock in phase2["identity_locks"]:
        lock.update(
            {
                "status": "frozen",
                "returned_model_id": next(
                    system["model"]
                    for system in phase2["serving_systems"]
                    if system["serving_system"] == lock["serving_system"]
                ),
                "returned_version_id": "provider-returned-version",
            }
        )
    phase2["outcome_design"] = {
        "status": "frozen",
        "heads_up_hands": 80,
        "paired_seed_count": 80,
    }
    pricing = tmp_path / "pricing.json"
    pricing.write_text(json.dumps({"frozen": True, "entries": [{"model": "all"}]}), encoding="utf-8")
    power = tmp_path / "power.json"
    power.write_text(
        json.dumps({"valid": True, "protocol": config["protocol"]}), encoding="utf-8"
    )

    ready = audit_phase2_readiness(
        tmp_path / "ready",
        phase2=phase2,
        preflight_dir=preflight,
        pricing_manifest=pricing,
        power_analysis=power,
    )
    assert ready["ready_for_formal_outcomes"]
