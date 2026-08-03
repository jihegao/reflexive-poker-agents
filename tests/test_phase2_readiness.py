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
                "observed_model_versions": ["provider-returned-version"],
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
        "phase1_outcome_lock": "phase1-evidence-bundle-sha256",
        "heads_up_hands": 80,
        "paired_seed_count": 80,
    }
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": config["protocol"],
                "frozen": True,
                "frozen_at_utc": "2026-08-02T00:00:00+00:00",
                "entries": [
                    {
                        "provider": system["provider"],
                        "model": system["model"],
                        "cost_observability": "unavailable",
                        "unavailable_reason": "CLI subscription has no per-call bill",
                    }
                    for system in phase2["serving_systems"]
                ],
            }
        ),
        encoding="utf-8",
    )
    power = tmp_path / "power.json"
    power.write_text(
        json.dumps(
            {
                "valid": True,
                "protocol": config["protocol"],
                "analysis_unit": "paired_seed_block",
                "paired_seed_count": 80,
                "heads_up_hands": 80,
                "method": "paired bootstrap from locked Phase 1 pilot",
                "phase1_outcome_lock": "phase1-evidence-bundle-sha256",
            }
        ),
        encoding="utf-8",
    )

    ready = audit_phase2_readiness(
        tmp_path / "ready",
        phase2=phase2,
        preflight_dir=preflight,
        pricing_manifest=pricing,
        power_analysis=power,
    )
    assert ready["ready_for_formal_outcomes"]


def test_phase2_readiness_rejects_a_frozen_version_not_attested_by_preflight(
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
    for lock in phase2["identity_locks"]:
        lock.update(
            {
                "status": "frozen",
                "returned_model_id": lock["serving_system"].split("/", 1)[1],
                "returned_version_id": "different-version",
            }
        )
    result = audit_phase2_readiness(
        tmp_path / "readiness",
        phase2=phase2,
        preflight_dir=preflight,
        pricing_manifest=None,
        power_analysis=None,
    )
    assert not result["ready_for_formal_outcomes"]
    assert not all(item["version_matches_preflight"] for item in result["identity_locks"].values())
