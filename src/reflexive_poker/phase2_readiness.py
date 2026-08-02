"""Fail-closed readiness gate for the four-system Phase 2 extension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _slug(provider: str, model: str) -> str:
    return f"{provider}__{model}".replace("/", "_").replace(".", "_")


def audit_phase2_readiness(
    output_dir: Path,
    *,
    phase2: dict[str, Any],
    preflight_dir: Path,
    pricing_manifest: Path | None,
    power_analysis: Path | None,
) -> dict[str, Any]:
    """Audit prerequisites without interpreting or producing outcome results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = str(phase2.get("protocol", "prbench-cross-model-v1"))
    systems = phase2.get("serving_systems", [])
    locks = {
        str(lock.get("serving_system")): lock
        for lock in phase2.get("identity_locks", [])
        if isinstance(lock, dict)
    }
    preflight: dict[str, Any] = {}
    identity_locks: dict[str, Any] = {}
    for system in systems:
        if not isinstance(system, dict):
            continue
        provider, model = str(system.get("provider", "")), str(system.get("model", ""))
        name = str(system.get("serving_system", f"{provider}/{model}"))
        gate = _read_json(preflight_dir / _slug(provider, model) / "provider_gate.json")
        lock = locks.get(name, {})
        actual = gate.get("observed_actual_models", [])
        preflight[name] = {
            "valid": bool(gate.get("valid")),
            "expected_predictions": gate.get("expected_predictions"),
            "observed_predictions": gate.get("observed_predictions"),
            "identity_matches": bool(gate.get("actual_identity_matches")),
            "identity_source_valid": bool(gate.get("model_identity_source_valid")),
        }
        identity_locks[name] = {
            "status": lock.get("status"),
            "returned_model_id": lock.get("returned_model_id"),
            "returned_version_id": lock.get("returned_version_id"),
            "matches_preflight": (
                isinstance(lock.get("returned_model_id"), str)
                and lock.get("returned_model_id") in actual
            ),
        }
    pricing = _read_json(pricing_manifest) if pricing_manifest is not None else {}
    power = _read_json(power_analysis) if power_analysis is not None else {}
    frozen_design = phase2.get("outcome_design", {})
    expected_system_count = 4
    preflight_complete = (
        len(systems) == expected_system_count
        and len(preflight) == expected_system_count
        and all(
            item["valid"]
            and item["expected_predictions"] == 20
            and item["observed_predictions"] == 20
            and item["identity_matches"]
            and item["identity_source_valid"]
            for item in preflight.values()
        )
    )
    identities_complete = (
        len(identity_locks) == expected_system_count
        and all(
            item["status"] == "frozen"
            and isinstance(item["returned_model_id"], str)
            and isinstance(item["returned_version_id"], str)
            and item["matches_preflight"]
            for item in identity_locks.values()
        )
    )
    pricing_complete = bool(pricing.get("frozen")) and bool(pricing.get("entries"))
    design_complete = (
        frozen_design.get("status") == "frozen"
        and isinstance(frozen_design.get("heads_up_hands"), int)
        and isinstance(frozen_design.get("paired_seed_count"), int)
        and frozen_design["heads_up_hands"] > 20
        and frozen_design["paired_seed_count"] > 40
    )
    power_complete = bool(power.get("valid")) and power.get("protocol") == protocol
    ready = all(
        (
            preflight_complete,
            identities_complete,
            pricing_complete,
            design_complete,
            power_complete,
        )
    )
    result = {
        "protocol": protocol,
        "phase": 2,
        "ready_for_formal_outcomes": ready,
        "claim_status": "ready_for_formal_outcomes" if ready else "prepared_not_runnable",
        "preflight": preflight,
        "identity_locks": identity_locks,
        "pricing_manifest": {"path": str(pricing_manifest) if pricing_manifest else None, "complete": pricing_complete},
        "outcome_design": {"value": frozen_design, "complete": design_complete},
        "power_analysis": {"path": str(power_analysis) if power_analysis else None, "complete": power_complete},
    }
    (output_dir / "PHASE2_READINESS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
