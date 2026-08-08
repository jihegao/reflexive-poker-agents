from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .phase1_models import ProviderBudget, ReasoningTreatment
from .phase1_offline import OfflineBenchmarkConfig, run_offline_benchmark
from .phase1_pricing import PricingManifestError, resolve_phase1_pricing
from .phase1_resumable import (
    FullSimulationRunConfig,
    LLMConfirmationRunConfig,
    freeze_phase1_source_snapshot,
    run_full_simulation_matrix,
    run_llm_confirmation_resumable,
)
from .phase2_framework import Phase2OfflineRunConfig, run_phase2_offline
from .phase2_readiness import audit_phase2_readiness

SCHEMA_VERSION = 1
TERMINAL_STATES = {"completed", "failed", "cancelled"}
EXPERIMENTS = {
    "offline-baselines": "Generate the frozen 200-case dataset and deterministic controls.",
    "offline-model": "Run one configured live model across D0/D1/D1-BM/D2/D3.",
    "provider-preflight": "Run 20 structured calls per configured model across all treatments.",
    "simulation": "Run or resume the isolated rule-agent Phase 1 matrix.",
    "llm-confirmation": "Run or resume the paired two-model Heads-up confirmation.",
    "paper-phase1": "Run preflight, offline evidence, and paired closed-loop confirmation.",
    "paper-phase2-preflight": "Run the bounded four-system Phase 2 provider preflight only.",
    "paper-phase2-offline": "Run four-system Phase 2 offline evidence after the frozen readiness gate.",
    "three-round": "Run the resumable DeepSeek-vs-Luna three-round poker tournament.",
}


class ExpctlError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        exit_code: int = 2,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}
        self.exit_code = exit_code


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExpctlError("RUN_NOT_FOUND", f"Run metadata does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExpctlError("RUN_METADATA_INVALID", f"Run metadata is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ExpctlError("RUN_METADATA_INVALID", f"Run metadata is not an object: {path}")
    return payload


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExpctlError("CONFIG_NOT_FOUND", f"Config does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ExpctlError(
            "CONFIG_INVALID",
            f"Config is not valid YAML: {exc}",
            details={"path": str(path)},
        ) from exc
    if not isinstance(payload, dict):
        raise ExpctlError("CONFIG_INVALID", "Config root must be a mapping")
    required = {"protocol", "evidence_layers", "provider_gate", "resumable_execution"}
    missing = sorted(required - set(payload))
    if missing:
        raise ExpctlError(
            "CONFIG_INVALID",
            f"Config is missing required sections: {missing}",
            details={"missing": missing},
        )
    phase = payload.get("paper_phase1")
    if phase is not None:
        if not isinstance(phase, dict):
            raise ExpctlError("CONFIG_INVALID", "paper_phase1 must be a mapping")
        models = phase.get("models", [])
        if not isinstance(models, list) or len(models) < 2:
            raise ExpctlError(
                "CONFIG_INVALID",
                "paper_phase1.models must contain at least two serving systems",
                details={"field": "paper_phase1.models"},
            )
    phase2 = payload.get("paper_phase2")
    if phase2 is not None:
        if not isinstance(phase2, dict):
            raise ExpctlError("CONFIG_INVALID", "paper_phase2 must be a mapping")
        systems = phase2.get("serving_systems", [])
        if not isinstance(systems, list) or len(systems) != 4:
            raise ExpctlError(
                "CONFIG_INVALID",
                "paper_phase2.serving_systems must contain exactly four serving systems",
                details={"field": "paper_phase2.serving_systems"},
            )
        identities: set[tuple[str, str]] = set()
        for system in systems:
            if not isinstance(system, dict) or not system.get("provider") or not system.get("model"):
                raise ExpctlError(
                    "CONFIG_INVALID",
                    "Each paper_phase2 serving system needs provider and model fields",
                    details={"field": "paper_phase2.serving_systems"},
                )
            identity = (str(system["provider"]), str(system["model"]))
            if identity in identities:
                raise ExpctlError(
                    "CONFIG_INVALID",
                    "paper_phase2.serving_systems must not contain duplicate provider/model pairs",
                    details={"field": "paper_phase2.serving_systems"},
                )
            identities.add(identity)
    try:
        resolve_phase1_pricing(path.resolve(), payload)
    except PricingManifestError as exc:
        raise ExpctlError("PRICING_MANIFEST_INVALID", str(exc)) from exc
    return payload


def _config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _registry_root(value: str | None) -> Path:
    return Path(value or os.getenv("EXPCTL_ROOT", "results/experiments")).resolve()


def _run_dir(root: Path, run_id: str) -> Path:
    candidate = (root / run_id).resolve()
    if candidate.parent != root.resolve():
        raise ExpctlError("RUN_ID_INVALID", "run-id must not contain path components")
    return candidate


def _metadata_path(run_dir: Path) -> Path:
    return run_dir / "run.json"


def _events_path(run_dir: Path) -> Path:
    return run_dir / "events.jsonl"


def _event(run_dir: Path, event_type: str, **fields: Any) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _now(),
        "event": event_type,
        **fields,
    }
    with _events_path(run_dir).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _update_run(run_dir: Path, **updates: Any) -> dict[str, Any]:
    metadata = _load_json(_metadata_path(run_dir))
    metadata.update(updates)
    metadata["updated_at"] = _now()
    _atomic_json(_metadata_path(run_dir), metadata)
    return metadata


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"worker_command"}
    }


def _emit(payload: Any, output: str = "human") -> None:
    if output == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    if isinstance(payload, str):
        print(payload)
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _find_idempotent(root: Path, request_id: str) -> dict[str, Any] | None:
    if not root.exists():
        return None
    for path in root.glob("*/run.json"):
        try:
            metadata = _load_json(path)
        except ExpctlError:
            continue
        if metadata.get("request_id") == request_id:
            return metadata
    return None


def _paper_config(config: dict[str, Any]) -> dict[str, Any]:
    phase = config.get("paper_phase1")
    if not isinstance(phase, dict):
        llm = config["evidence_layers"].get("llm_confirmation", {})
        phase = {
            "case_count": 200,
            "preflight_cases": 4,
            "models": llm.get("models", []),
            "closed_loop": {
                "seed_start": 9700,
                "seed_count": 40,
                "hands": 20,
                "formation_hands": 5,
            },
        }
    return phase


def _models(config: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    models = _paper_config(config).get("models", [])
    parsed: list[tuple[str, str]] = []
    for item in models:
        if not isinstance(item, dict) or not item.get("provider") or not item.get("model"):
            raise ExpctlError(
                "CONFIG_INVALID",
                "Each paper_phase1 model needs provider and model fields",
                details={"field": "paper_phase1.models"},
            )
        parsed.append((str(item["provider"]), str(item["model"])))
    if len(parsed) < 2:
        raise ExpctlError("CONFIG_INVALID", "At least two Phase 1 models are required")
    return tuple(parsed)


def _phase2_preflight_models(config: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return the frozen Phase 2 systems without falling back to Phase 1 fields."""
    phase = config.get("paper_phase2")
    if not isinstance(phase, dict):
        raise ExpctlError("CONFIG_INVALID", "paper-phase2-preflight requires paper_phase2")
    systems = phase.get("serving_systems", [])
    if not isinstance(systems, list) or len(systems) != 4:
        raise ExpctlError(
            "CONFIG_INVALID",
            "paper-phase2-preflight requires exactly four paper_phase2 serving systems",
            details={"field": "paper_phase2.serving_systems"},
        )
    parsed: list[tuple[str, str]] = []
    for system in systems:
        if not isinstance(system, dict) or not system.get("provider") or not system.get("model"):
            raise ExpctlError(
                "CONFIG_INVALID",
                "Each paper_phase2 serving system needs provider and model fields",
                details={"field": "paper_phase2.serving_systems"},
            )
        parsed.append((str(system["provider"]), str(system["model"])))
    return tuple(parsed)


def _offline_config(
    config: dict[str, Any],
    output_dir: Path,
    *,
    provider: str,
    model: str,
    case_count: int,
    preregistered: bool,
) -> OfflineBenchmarkConfig:
    phase = _paper_config(config)
    budget = phase.get("offline_budget", {})
    return OfflineBenchmarkConfig(
        output_dir=output_dir,
        provider=provider,
        model=model,
        case_count=case_count,
        base_seed=int(phase.get("base_seed", 20260802)),
        provider_budget=ProviderBudget(
            max_calls=int(budget.get("max_calls", max(1_200, case_count * 6))),
            max_primary_calls=int(budget.get("max_primary_calls", case_count * 5)),
            max_retries=int(budget.get("max_retries", max(20, case_count))),
        ),
        preregistered=preregistered,
    )


def _freeze_run_sources(metadata: dict[str, Any], artifacts: Path) -> dict[str, Any]:
    """Archive the byte-for-byte price input copied before the worker spawned."""
    pricing_path = metadata.get("pricing_manifest_artifact")
    frozen_inputs: dict[str, Path] | None = None
    if pricing_path is not None:
        path = Path(str(pricing_path))
        expected = metadata.get("pricing_manifest_sha256")
        if not path.exists():
            raise ExpctlError("PRICING_MANIFEST_MISSING", "run-local pricing manifest is missing")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ExpctlError("PRICING_MANIFEST_MUTATED", "run-local pricing manifest hash changed")
        frozen_inputs = {"frozen_inputs/PRICE_MANIFEST.json": path}
    phase2_inputs = metadata.get("phase2_frozen_inputs")
    if phase2_inputs is not None:
        if not isinstance(phase2_inputs, dict):
            raise ExpctlError("PHASE2_INPUTS_INVALID", "Phase 2 frozen-input metadata is invalid")
        frozen_inputs = frozen_inputs or {}
        for archive_name, details in sorted(phase2_inputs.items()):
            if not isinstance(archive_name, str) or not isinstance(details, dict):
                raise ExpctlError("PHASE2_INPUTS_INVALID", "Phase 2 frozen input entry is invalid")
            path_value, expected = details.get("artifact_path"), details.get("sha256")
            path = Path(str(path_value))
            if not isinstance(expected, str) or not path.is_file():
                raise ExpctlError("PHASE2_INPUT_MISSING", f"Phase 2 frozen input is missing: {archive_name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ExpctlError("PHASE2_INPUT_MUTATED", f"Phase 2 frozen input changed: {archive_name}")
            frozen_inputs[archive_name] = path
    provenance = freeze_phase1_source_snapshot(
        artifacts,
        allow_dirty_worktree=bool(metadata.get("allow_dirty_worktree")),
        frozen_inputs=frozen_inputs,
    )
    if pricing_path is not None:
        provenance = {
            **provenance,
            "pricing_manifest_artifact": str(pricing_path),
            "pricing_manifest_sha256": metadata["pricing_manifest_sha256"],
            "pricing_manifest_frozen_at_utc": metadata["pricing_manifest_frozen_at_utc"],
        }
        _atomic_json(artifacts / "SOURCE_PROVENANCE.json", provenance)
    return provenance


def _copy_phase2_input(source: Path, destination: Path, *, label: str) -> dict[str, str]:
    """Copy a formal input before worker spawn and retain its digest."""
    if not source.is_file():
        raise ExpctlError("PHASE2_INPUT_MISSING", f"Phase 2 {label} is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {
        "artifact_path": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_path": str(source),
    }


def _freeze_phase2_inputs(
    run_dir: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Make the readiness evidence immutable before a Phase 2 worker exists."""
    supplied = {
        "phase2_preflight_dir": getattr(args, "phase2_preflight_dir", None),
        "phase2_pricing_manifest": getattr(args, "phase2_pricing_manifest", None),
        "phase2_power_analysis": getattr(args, "phase2_power_analysis", None),
    }
    missing = [name for name, value in supplied.items() if not value]
    if missing:
        raise ExpctlError(
            "PHASE2_READINESS_INPUT_MISSING",
            "paper-phase2-offline requires frozen preflight, pricing, and power artifacts",
            details={"missing": missing},
        )
    phase2 = config.get("paper_phase2")
    if not isinstance(phase2, dict):
        raise ExpctlError("CONFIG_INVALID", "paper-phase2-offline requires paper_phase2")
    preflight_source = Path(str(supplied["phase2_preflight_dir"])).resolve()
    if not preflight_source.is_dir():
        raise ExpctlError("PHASE2_INPUT_MISSING", f"Phase 2 preflight directory is missing: {preflight_source}")
    snapshot_root = run_dir / "frozen_inputs" / "phase2"
    frozen: dict[str, dict[str, str]] = {}
    for provider, model in _phase2_preflight_models(config):
        slug = f"{provider}__{model}".replace("/", "_").replace(".", "_")
        archive_name = f"frozen_inputs/phase2/preflight/{slug}/provider_gate.json"
        frozen[archive_name] = _copy_phase2_input(
            preflight_source / slug / "provider_gate.json",
            snapshot_root / "preflight" / slug / "provider_gate.json",
            label=f"preflight gate for {provider}/{model}",
        )
    for key, archive_name, filename, label in (
        (
            "phase2_pricing_manifest",
            "frozen_inputs/phase2/PHASE2_PRICE_MANIFEST.json",
            "PHASE2_PRICE_MANIFEST.json",
            "pricing manifest",
        ),
        (
            "phase2_power_analysis",
            "frozen_inputs/phase2/PHASE2_POWER_ANALYSIS.json",
            "PHASE2_POWER_ANALYSIS.json",
            "power analysis",
        ),
    ):
        frozen[archive_name] = _copy_phase2_input(
            Path(str(supplied[key])).resolve(),
            snapshot_root / filename,
            label=label,
        )
    return {
        "phase2_preflight_dir": str(snapshot_root / "preflight"),
        "phase2_pricing_manifest": str(snapshot_root / "PHASE2_PRICE_MANIFEST.json"),
        "phase2_power_analysis": str(snapshot_root / "PHASE2_POWER_ANALYSIS.json"),
        "phase2_frozen_inputs": frozen,
    }


def _run_offline_models(
    config: dict[str, Any],
    artifact_dir: Path,
    case_count: int,
    *,
    models: tuple[tuple[str, str], ...] | None = None,
) -> None:
    active_models = _models(config) if models is None else models
    run_offline_benchmark(
        _offline_config(
            config,
            artifact_dir / "baselines",
            provider="baselines",
            model="none",
            case_count=case_count,
            preregistered=True,
        )
    )
    for provider, model in active_models:
        slug = f"{provider}__{model}".replace("/", "_").replace(".", "_")
        result = run_offline_benchmark(
            _offline_config(
                config,
                artifact_dir / slug,
                provider=provider,
                model=model,
                case_count=case_count,
                preregistered=True,
            )
        )
        if not result["provider_gate"]["valid"]:
            raise ExpctlError(
                "PROVIDER_GATE_FAILED",
                f"Offline provider gate failed for {provider}:{model}",
                retryable=True,
            )


def _run_experiment(metadata: dict[str, Any], run_dir: Path) -> None:
    config = _load_config(Path(metadata["config_path"]))
    experiment = metadata["experiment"]
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    phase = _paper_config(config)
    if experiment == "offline-baselines":
        source_provenance = _freeze_run_sources(metadata, artifacts)
        _event(
            run_dir,
            "source_frozen",
            source_fingerprint=source_provenance["source_fingerprint"],
            source_snapshot_sha256=source_provenance["source_snapshot_sha256"],
        )
        run_offline_benchmark(
            _offline_config(
                config,
                artifacts / "offline_baselines",
                provider="baselines",
                model="none",
                case_count=int(phase.get("case_count", 200)),
                preregistered=True,
            )
        )
    elif experiment == "offline-model":
        provider = metadata.get("provider")
        model = metadata.get("model")
        if not provider or not model:
            raise ExpctlError(
                "MODEL_REQUIRED",
                "offline-model requires --provider and --model",
            )
        source_provenance = _freeze_run_sources(metadata, artifacts)
        _event(
            run_dir,
            "source_frozen",
            source_fingerprint=source_provenance["source_fingerprint"],
            source_snapshot_sha256=source_provenance["source_snapshot_sha256"],
        )
        result = run_offline_benchmark(
            _offline_config(
                config,
                artifacts / "offline_model",
                provider=str(provider),
                model=str(model),
                case_count=int(phase.get("case_count", 200)),
                preregistered=True,
            )
        )
        if not result["provider_gate"]["valid"]:
            raise ExpctlError(
                "PROVIDER_GATE_FAILED",
                f"Offline provider gate failed for {provider}:{model}",
                retryable=True,
                details={"provider_gate": result["provider_gate"]},
            )
    elif experiment == "provider-preflight":
        provider = metadata.get("provider")
        model = metadata.get("model")
        if bool(provider) != bool(model):
            raise ExpctlError(
                "MODEL_REQUIRED",
                "provider-preflight requires both --provider and --model when narrowed",
            )
        source_provenance = _freeze_run_sources(metadata, artifacts)
        _event(
            run_dir,
            "source_frozen",
            source_fingerprint=source_provenance["source_fingerprint"],
            source_snapshot_sha256=source_provenance["source_snapshot_sha256"],
        )
        _run_offline_models(
            config,
            artifacts / "preflight",
            int(phase.get("preflight_cases", 4)),
            models=((str(provider), str(model)),) if provider and model else None,
        )
    elif experiment == "paper-phase2-preflight":
        phase2 = config.get("paper_phase2")
        if not isinstance(phase2, dict):
            raise ExpctlError("CONFIG_INVALID", "paper-phase2-preflight requires paper_phase2")
        preflight_cases = int(phase2.get("preflight_cases", 4))
        if preflight_cases != 4:
            raise ExpctlError(
                "CONFIG_INVALID",
                "paper-phase2-preflight is fixed at four cases per treatment",
                details={"field": "paper_phase2.preflight_cases", "value": preflight_cases},
            )
        source_provenance = _freeze_run_sources(metadata, artifacts)
        _event(
            run_dir,
            "source_frozen",
            source_fingerprint=source_provenance["source_fingerprint"],
            source_snapshot_sha256=source_provenance["source_snapshot_sha256"],
        )
        _event(run_dir, "phase_started", phase="phase2_provider_preflight")
        _run_offline_models(
            config,
            artifacts / "preflight",
            preflight_cases,
            models=_phase2_preflight_models(config),
        )
        _event(run_dir, "phase_completed", phase="phase2_provider_preflight")
    elif experiment == "paper-phase2-offline":
        phase2 = config.get("paper_phase2")
        if not isinstance(phase2, dict):
            raise ExpctlError("CONFIG_INVALID", "paper-phase2-offline requires paper_phase2")
        required_inputs = (
            "phase2_preflight_dir",
            "phase2_pricing_manifest",
            "phase2_power_analysis",
        )
        missing = [name for name in required_inputs if not metadata.get(name)]
        if missing:
            raise ExpctlError(
                "PHASE2_READINESS_INPUT_MISSING",
                "paper-phase2-offline requires frozen preflight, pricing, and power artifacts",
                details={"missing": missing},
            )
        source_provenance = _freeze_run_sources(metadata, artifacts)
        _event(
            run_dir,
            "source_frozen",
            source_fingerprint=source_provenance["source_fingerprint"],
            source_snapshot_sha256=source_provenance["source_snapshot_sha256"],
        )
        readiness = audit_phase2_readiness(
            artifacts / "phase2_readiness",
            phase2={**phase2, "protocol": config.get("protocol")},
            preflight_dir=Path(str(metadata["phase2_preflight_dir"])),
            pricing_manifest=Path(str(metadata["phase2_pricing_manifest"])),
            power_analysis=Path(str(metadata["phase2_power_analysis"])),
        )
        _event(run_dir, "phase_started", phase="phase2_offline_understanding")
        run_phase2_offline(
            Phase2OfflineRunConfig(
                phase2={**phase2, "protocol": config.get("protocol")},
                readiness=readiness,
                output_dir=artifacts / "phase2_offline",
                case_count=int(phase2.get("offline_understanding", {}).get("case_count", 200)),
                base_seed=int(phase2.get("offline_understanding", {}).get("base_seed", 20260802)),
            )
        )
        _event(run_dir, "phase_completed", phase="phase2_offline_understanding")
    elif experiment == "simulation":
        simulation = config["evidence_layers"]["simulation"]["heads_up"]
        seeds = simulation.get("seeds", {})
        run_full_simulation_matrix(
            FullSimulationRunConfig(
                output_dir=artifacts / "simulation",
                seeds=tuple(
                    range(int(seeds.get("start", 9400)), int(seeds.get("start", 9400)) + int(seeds.get("count", 60)))
                ),
                horizon=int(simulation.get("hands", 400)),
                formation_hands=int(simulation.get("formation_hands", 100)),
                max_cells=metadata.get("max_cells"),
                max_seed_blocks=metadata.get("max_blocks"),
                allow_dirty_worktree=bool(metadata.get("allow_dirty_worktree")),
            )
        )
    elif experiment in {"llm-confirmation", "paper-phase1"}:
        if experiment == "paper-phase1":
            source_provenance = _freeze_run_sources(metadata, artifacts)
            _event(
                run_dir,
                "source_frozen",
                source_fingerprint=source_provenance["source_fingerprint"],
                source_snapshot_sha256=source_provenance["source_snapshot_sha256"],
            )
            _event(run_dir, "phase_started", phase="provider_preflight")
            _run_offline_models(
                config,
                artifacts / "preflight",
                int(phase.get("preflight_cases", 4)),
            )
            _event(run_dir, "phase_completed", phase="provider_preflight")
            _event(run_dir, "phase_started", phase="offline_understanding")
            _run_offline_models(
                config,
                artifacts / "offline_understanding",
                int(phase.get("case_count", 200)),
            )
            _event(run_dir, "phase_completed", phase="offline_understanding")
        closed = phase.get("closed_loop", {})
        _event(run_dir, "phase_started", phase="closed_loop")
        closed_loop_status = run_llm_confirmation_resumable(
            LLMConfirmationRunConfig(
                output_dir=artifacts / "llm_confirmation",
                selected_depth=ReasoningTreatment.RECURSIVE_D2,
                models=_models(config),
                seeds=tuple(
                    range(
                        int(closed.get("seed_start", 9700)),
                        int(closed.get("seed_start", 9700))
                        + int(closed.get("seed_count", 40)),
                    )
                ),
                horizon=int(closed.get("hands", 20)),
                formation_hands=int(closed.get("formation_hands", 5)),
                equity_samples=int(closed.get("equity_samples", 8)),
                max_calls_per_model=int(
                    config.get("llm_confirmation", {}).get("max_calls_per_model", 10_000)
                ),
                offline_call_budget=int(
                    config.get("llm_confirmation", {}).get("offline_understanding_calls", 1_600)
                ),
                preflight_retry_reserve=int(
                    config.get("llm_confirmation", {}).get("preflight_and_retry_reserve", 400)
                ),
                heads_up_contrast_calls=int(
                    config.get("llm_confirmation", {}).get("heads_up_contrast_calls", 8_000)
                ),
                max_primary_calls_per_paired_block=int(
                    closed.get("max_primary_calls_per_paired_block", 100)
                ),
                max_blocks=metadata.get("max_blocks"),
                allow_dirty_worktree=bool(metadata.get("allow_dirty_worktree")),
            )
        )
        if experiment == "paper-phase1":
            expected_blocks = int(closed.get("seed_count", 40))
            expected_models = {f"{provider}:{model}" for provider, model in _models(config)}
            observed_models = {
                f"{row.provider}:{row.model}"
                for row in closed_loop_status.itertuples(index=False)
            }
            incomplete = closed_loop_status.loc[
                closed_loop_status["valid_blocks"] < expected_blocks
            ]
            if observed_models != expected_models or not incomplete.empty:
                raise ExpctlError(
                    "FORMAL_COMPLETION_INVALID",
                    "Phase 1 closed-loop did not produce every required valid paired block",
                    retryable=True,
                    details={
                        "expected_blocks_per_model_regime": expected_blocks,
                        "observed_models": sorted(observed_models),
                        "incomplete": incomplete.to_dict(orient="records"),
                    },
                )
            paired_blocks = pd.read_csv(
                artifacts / "llm_confirmation" / "CROSS_MODEL_PAIRED_BLOCKS.csv"
            )
            completion_status = _load_json(
                artifacts / "llm_confirmation" / "CROSS_MODEL_PAIRED_BLOCK_STATUS.json"
            )
            if (
                completion_status.get("formal_completion_valid") is not True
                or len(paired_blocks) != expected_blocks
                or not bool(paired_blocks["valid"].all())
            ):
                raise ExpctlError(
                    "FORMAL_COMPLETION_INVALID",
                    "Phase 1 has no complete cross-model all-arm paired-block intersection",
                    retryable=True,
                    details={
                        "expected_blocks": expected_blocks,
                        "paired_blocks": paired_blocks.to_dict(orient="records"),
                        "completion_status": completion_status,
                    },
                )
        _event(run_dir, "phase_completed", phase="closed_loop")
    elif experiment == "three-round":
        from .three_round_experiment import ThreeRoundConfig, run_three_round_experiment

        source_provenance = _freeze_run_sources(metadata, artifacts)
        _event(
            run_dir,
            "source_frozen",
            source_fingerprint=source_provenance["source_fingerprint"],
            source_snapshot_sha256=source_provenance["source_snapshot_sha256"],
        )
        three_round = config.get("three_round")
        if not isinstance(three_round, dict):
            raise ExpctlError("CONFIG_INVALID", "three-round requires a three_round config section")
        models = three_round.get("models")
        if not isinstance(models, list) or len(models) != 2:
            raise ExpctlError("CONFIG_INVALID", "three_round.models must contain two serving systems")
        model_specs: list[tuple[str, str, str]] = []
        for item in models:
            if not isinstance(item, dict) or not item.get("label") or not item.get("provider") or not item.get("model"):
                raise ExpctlError(
                    "CONFIG_INVALID",
                    "three_round.models require label, provider, and model",
                )
            model_specs.append((str(item["label"]), str(item["provider"]), str(item["model"])))
        result = run_three_round_experiment(
            ThreeRoundConfig(
                seeds=tuple(
                    range(
                        int(three_round.get("seed_start", 9950)),
                        int(three_round.get("seed_start", 9950))
                        + int(three_round.get("seed_count", 1)),
                    )
                ),
                hands=int(three_round.get("hands", 1)),
                rounds=tuple(int(value) for value in three_round.get("rounds", (1, 2, 3))),
                round3_lineup_count=int(three_round.get("round3_lineup_count", 1)),
                gto_iterations=int(three_round.get("gto_iterations", 2_000)),
                equity_samples=int(three_round.get("equity_samples", 16)),
                memory_hands=int(three_round.get("memory_hands", 6)),
                evidence_tier=str(three_round.get("evidence_tier", "pilot")),
                minimum_formal_seeds=int(three_round.get("minimum_formal_seeds", 10)),
                bootstrap_samples=int(three_round.get("bootstrap_samples", 5_000)),
                permutation_samples=int(three_round.get("permutation_samples", 20_000)),
                source_clean=not bool(metadata.get("allow_dirty_worktree", False)),
                output_dir=artifacts / "three_round",
                model_specs=tuple(model_specs),
            )
        )
        _atomic_json(
            artifacts / "THREE_ROUND_RESULT.json",
            {
                "provider_gate": result["provider_gate"],
                "evidence_gate": result["evidence_gate"],
                "match_count": int(result["provider_gate"]["match_count"]),
                "valid_match_count": int(result["provider_gate"]["valid_match_count"]),
            },
        )
        if not result["provider_gate"]["valid"]:
            raise ExpctlError(
                "PROVIDER_GATE_FAILED",
                "three-round provider gate failed; artifacts are audit-only",
                retryable=True,
                details={"provider_gate": result["provider_gate"]},
            )
        if (
            result["evidence_gate"]["evidence_tier"] == "formal"
            and not result["evidence_gate"]["formal_conclusion_allowed"]
        ):
            raise ExpctlError(
                "EVIDENCE_GATE_FAILED",
                "three-round formal completion gate failed; artifacts are audit-only",
                retryable=False,
                details={"evidence_gate": result["evidence_gate"]},
            )
    else:
        raise ExpctlError("EXPERIMENT_UNKNOWN", f"Unknown experiment: {experiment}")


def _worker(run_dir: Path) -> int:
    try:
        try:
            os.nice(10)
        except OSError:
            pass
        metadata = _update_run(run_dir, state="running", started_at=_now(), pid=os.getpid())
        _event(run_dir, "run_started", run_id=metadata["run_id"], pid=os.getpid())
        _run_experiment(metadata, run_dir)
        _update_run(run_dir, state="completed", completed_at=_now(), exit_code=0)
        _event(run_dir, "run_completed", run_id=metadata["run_id"], exit_code=0)
        return 0
    except KeyboardInterrupt:
        _update_run(run_dir, state="cancelled", completed_at=_now(), exit_code=130)
        _event(run_dir, "run_cancelled", exit_code=130)
        return 130
    except BaseException as exc:  # noqa: BLE001 - the worker must persist every failure.
        code = exc.code if isinstance(exc, ExpctlError) else "WORKER_FAILED"
        _update_run(
            run_dir,
            state="failed",
            completed_at=_now(),
            exit_code=getattr(exc, "exit_code", 1),
            error={"code": code, "message": str(exc), "retryable": isinstance(exc, ExpctlError) and exc.retryable},
        )
        _event(run_dir, "run_failed", error_code=code, message=str(exc))
        return getattr(exc, "exit_code", 1)


def _refresh_status(run_dir: Path) -> dict[str, Any]:
    metadata = _load_json(_metadata_path(run_dir))
    if metadata.get("state") in {"queued", "running"} and not _pid_alive(metadata.get("pid")):
        metadata = _update_run(
            run_dir,
            state="failed",
            completed_at=_now(),
            exit_code=1,
            error={
                "code": "WORKER_DISAPPEARED",
                "message": "Worker process exited without recording a terminal state",
                "retryable": True,
            },
        )
        _event(run_dir, "run_failed", error_code="WORKER_DISAPPEARED")
    return metadata


def _spawn_background(run_dir: Path, worker_command: list[str]) -> subprocess.Popen[str]:
    stdout = (run_dir / "worker.stdout.log").open("a", encoding="utf-8")
    stderr = (run_dir / "worker.stderr.log").open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    try:
        return subprocess.Popen(
            worker_command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
            env=env,
            text=True,
        )
    finally:
        stdout.close()
        stderr.close()


def _start(args: argparse.Namespace) -> dict[str, Any]:
    root = _registry_root(args.root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    created_at = _now()
    try:
        pricing = resolve_phase1_pricing(config_path, config, run_created_at=created_at)
    except PricingManifestError as exc:
        raise ExpctlError("PRICING_MANIFEST_INVALID", str(exc)) from exc
    config_hash = _config_hash(config)
    if args.request_id:
        existing = _find_idempotent(root, args.request_id)
        if existing:
            if existing.get("config_hash") != config_hash or existing.get("experiment") != args.experiment:
                raise ExpctlError(
                    "IDEMPOTENCY_CONFLICT",
                    "request-id already exists with a different config or experiment",
                    details={"run_id": existing.get("run_id")},
                )
            return _public_metadata(existing)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{uuid.uuid4().hex[:10]}"
    run_dir = _run_dir(root, run_id)
    run_dir.mkdir(parents=True)
    phase2_inputs: dict[str, Any] = {}
    if args.experiment == "paper-phase2-offline":
        phase2_inputs = _freeze_phase2_inputs(run_dir, config, args)
    pricing_artifact: Path | None = None
    if pricing is not None:
        pricing_artifact = run_dir / "frozen_inputs" / "PRICE_MANIFEST.json"
        pricing_artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pricing.path, pricing_artifact)
        if hashlib.sha256(pricing_artifact.read_bytes()).hexdigest() != pricing.sha256:
            raise ExpctlError("PRICING_MANIFEST_COPY_FAILED", "pricing manifest artifact hash mismatch")
    worker_command = [
        sys.executable,
        "-m",
        "reflexive_poker.expctl",
        "--root",
        str(root),
        "_worker",
        "--run-dir",
        str(run_dir),
    ]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "request_id": args.request_id,
        "tag": args.tag,
        "experiment": args.experiment,
        "state": "created",
        "created_at": created_at,
        "updated_at": created_at,
        "config_path": str(config_path),
        "config_hash": config_hash,
        "artifact_dir": str(run_dir / "artifacts"),
        "pid": None,
        "provider": args.provider,
        "model": args.model,
        "max_blocks": args.max_blocks,
        "allow_dirty_worktree": args.allow_dirty_worktree,
        "phase2_preflight_dir": phase2_inputs.get("phase2_preflight_dir"),
        "phase2_pricing_manifest": phase2_inputs.get("phase2_pricing_manifest"),
        "phase2_power_analysis": phase2_inputs.get("phase2_power_analysis"),
        "phase2_frozen_inputs": phase2_inputs.get("phase2_frozen_inputs"),
        "worker_command": worker_command,
    }
    if pricing is not None and pricing_artifact is not None:
        metadata.update(
            {
                "pricing_manifest_artifact": str(pricing_artifact),
                "pricing_manifest_sha256": pricing.sha256,
                "pricing_manifest_frozen_at_utc": pricing.frozen_at_utc,
                "pricing_manifest_source_path": str(pricing.path),
            }
        )
    _atomic_json(_metadata_path(run_dir), metadata)
    _event(run_dir, "run_created", run_id=run_id, experiment=args.experiment)
    if pricing is not None:
        _event(
            run_dir,
            "pricing_frozen",
            pricing_manifest_sha256=pricing.sha256,
            pricing_manifest_frozen_at_utc=pricing.frozen_at_utc,
            pricing_manifest_artifact=str(pricing_artifact),
        )
    if args.foreground:
        _update_run(run_dir, state="queued", pid=os.getpid())
        exit_code = _worker(run_dir)
        if exit_code:
            raise ExpctlError(
                "WORKER_FAILED",
                f"Foreground worker failed with exit code {exit_code}",
                retryable=True,
                details={"run_id": run_id},
                exit_code=exit_code,
            )
        return _public_metadata(_load_json(_metadata_path(run_dir)))
    _update_run(run_dir, state="queued")
    process = _spawn_background(run_dir, worker_command)
    metadata = _update_run(run_dir, pid=process.pid)
    _event(run_dir, "run_queued", run_id=run_id, pid=process.pid)
    return _public_metadata(metadata)


def _signal_run(args: argparse.Namespace, action: str) -> dict[str, Any]:
    run_dir = _run_dir(_registry_root(args.root), args.run_id)
    metadata = _refresh_status(run_dir)
    state = metadata["state"]
    pid = metadata.get("pid")
    if action == "pause":
        if state != "running":
            raise ExpctlError("STATE_CONFLICT", f"Cannot pause a run in state {state}")
        os.killpg(pid, signal.SIGSTOP)
        metadata = _update_run(run_dir, state="paused")
        _event(run_dir, "run_paused", pid=pid)
    elif action == "resume":
        if state == "paused" and _pid_alive(pid):
            os.killpg(pid, signal.SIGCONT)
            metadata = _update_run(run_dir, state="running")
            _event(run_dir, "run_resumed", pid=pid)
        elif state in {"failed", "cancelled"}:
            worker_command = metadata.get("worker_command")
            if not isinstance(worker_command, list) or not all(
                isinstance(value, str) for value in worker_command
            ):
                raise ExpctlError("RUN_METADATA_INVALID", "Run has no reusable worker command")
            metadata = _update_run(
                run_dir,
                state="queued",
                completed_at=None,
                exit_code=None,
                error=None,
                resumed_at=_now(),
            )
            process = _spawn_background(run_dir, worker_command)
            metadata = _update_run(run_dir, pid=process.pid)
            _event(run_dir, "run_requeued", pid=process.pid)
        else:
            raise ExpctlError("STATE_CONFLICT", f"Cannot resume a run in state {state}")
    elif action == "stop":
        if state in TERMINAL_STATES:
            return _public_metadata(metadata)
        if not _pid_alive(pid):
            return _public_metadata(_refresh_status(run_dir))
        os.killpg(pid, signal.SIGTERM)
        metadata = _update_run(run_dir, state="cancelled", completed_at=_now(), exit_code=143)
        _event(run_dir, "run_cancelled", pid=pid, exit_code=143)
    return _public_metadata(metadata)


def _logs(args: argparse.Namespace) -> None:
    run_dir = _run_dir(_registry_root(args.root), args.run_id)
    path = _events_path(run_dir)
    if not path.exists():
        raise ExpctlError("RUN_NOT_FOUND", f"No event log exists for {args.run_id}")
    offset = 0
    while True:
        with path.open(encoding="utf-8") as handle:
            handle.seek(offset)
            for line in handle:
                if args.format == "jsonl":
                    print(line, end="")
                else:
                    event = json.loads(line)
                    print(f"{event['timestamp']} {event['event']} {json.dumps(event, ensure_ascii=False)}")
            offset = handle.tell()
        metadata = _refresh_status(run_dir)
        if not args.follow or metadata["state"] in TERMINAL_STATES:
            return
        time.sleep(0.5)


def _analyze(run_dir: Path) -> dict[str, Any]:
    metadata = _refresh_status(run_dir)
    artifacts = Path(metadata["artifact_dir"])
    csv_paths = sorted(artifacts.rglob("*.csv")) if artifacts.exists() else []
    tables: list[dict[str, Any]] = []
    for path in csv_paths:
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            continue
        numeric = frame.select_dtypes(include="number")
        tables.append(
            {
                "path": str(path.relative_to(run_dir)),
                "rows": len(frame),
                "columns": list(frame.columns),
                "numeric_means": {
                    name: float(value)
                    for name, value in numeric.mean(numeric_only=True).items()
                    if pd.notna(value)
                },
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": metadata["run_id"],
        "state": metadata["state"],
        "artifact_dir": str(artifacts),
        "tables": tables,
        "provider_gates": [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(artifacts.rglob("provider_gate.json"))
        ]
        if artifacts.exists()
        else [],
    }
    analysis_path = run_dir / "analysis" / "summary.json"
    _atomic_json(analysis_path, result)
    return result


def _export(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _run_dir(_registry_root(args.root), args.run_id)
    metadata = _refresh_status(run_dir)
    artifacts = Path(metadata["artifact_dir"])
    if not artifacts.exists():
        raise ExpctlError("ARTIFACTS_NOT_FOUND", "Run has no artifact directory")
    destination = Path(args.destination).resolve() if args.destination else run_dir / "exports"
    destination.mkdir(parents=True, exist_ok=True)
    if args.format == "tar.gz":
        target = destination / f"{args.run_id}.tar.gz"
        with tarfile.open(target, "w:gz") as archive:
            archive.add(artifacts, arcname="artifacts")
    elif args.format == "csv":
        target = destination / f"{args.run_id}-csv"
        target.mkdir(parents=True, exist_ok=True)
        for path in artifacts.rglob("*.csv"):
            relative = path.relative_to(artifacts)
            output_path = target / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output_path)
    else:
        target = destination / f"{args.run_id}-parquet"
        target.mkdir(parents=True, exist_ok=True)
        try:
            for path in artifacts.rglob("*.csv"):
                frame = pd.read_csv(path)
                output_path = (target / path.relative_to(artifacts)).with_suffix(".parquet")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(output_path, index=False)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ExpctlError(
                "PARQUET_UNAVAILABLE",
                "Install pyarrow or fastparquet to export Parquet",
                details={"destination": str(target)},
            ) from exc
    return {"run_id": args.run_id, "format": args.format, "path": str(target)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent-friendly isolated experiment controller")
    parser.add_argument("--root", help="Experiment registry root (default: results/experiments)")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check local experiment runtime prerequisites")
    doctor.add_argument("--output", choices=("human", "json"), default="human")

    experiment = sub.add_parser("experiment", help="Inspect available experiment types")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    experiment_list = experiment_sub.add_parser("list")
    experiment_list.add_argument("--output", choices=("human", "json"), default="human")

    config = sub.add_parser("config", help="Validate experiment configuration")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    validate = config_sub.add_parser("validate")
    validate.add_argument("path")
    validate.add_argument("--output", choices=("human", "json"), default="human")

    phase2_readiness = sub.add_parser(
        "phase2-readiness",
        help="Audit whether Phase 2 is frozen enough to permit outcome runs",
    )
    phase2_readiness.add_argument("--config", required=True)
    phase2_readiness.add_argument("--preflight-dir", required=True)
    phase2_readiness.add_argument("--pricing-manifest")
    phase2_readiness.add_argument("--power-analysis")
    phase2_readiness.add_argument("--output-dir", required=True)
    phase2_readiness.add_argument("--output", choices=("human", "json"), default="human")

    run = sub.add_parser("run", help="Manage background experiment runs")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    start = run_sub.add_parser("start")
    start.add_argument("--config", required=True)
    start.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="paper-phase1")
    start.add_argument("--tag")
    start.add_argument("--request-id")
    start.add_argument("--provider")
    start.add_argument("--model")
    start.add_argument("--max-blocks", type=int)
    start.add_argument(
        "--phase2-preflight-dir",
        help="Frozen four-system preflight artifact directory required by paper-phase2-offline",
    )
    start.add_argument(
        "--phase2-pricing-manifest",
        help="Frozen four-system price manifest required by paper-phase2-offline",
    )
    start.add_argument(
        "--phase2-power-analysis",
        help="Frozen paired-seed power analysis required by paper-phase2-offline",
    )
    start.add_argument("--allow-dirty-worktree", action="store_true")
    start.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)
    start.add_argument("--output", choices=("human", "json"), default="human")
    for name in ("status", "pause", "resume", "stop"):
        command = run_sub.add_parser(name)
        command.add_argument("run_id")
        command.add_argument("--output", choices=("human", "json"), default="human")
    logs = run_sub.add_parser("logs")
    logs.add_argument("run_id")
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--format", choices=("human", "jsonl"), default="jsonl")

    analyze = sub.add_parser("analyze", help="Build a machine-readable artifact summary")
    analyze.add_argument("run_id")
    analyze.add_argument("--output", choices=("human", "json"), default="human")

    export = sub.add_parser("export", help="Export raw experiment artifacts")
    export.add_argument("run_id")
    export.add_argument("--format", choices=("csv", "parquet", "tar.gz"), required=True)
    export.add_argument("--destination")
    export.add_argument("--output", choices=("human", "json"), default="human")

    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            root = _registry_root(args.root)
            payload = {
                "ok": True,
                "python": sys.version.split()[0],
                "registry_root": str(root),
                "registry_writable": os.access(root.parent if not root.exists() else root, os.W_OK),
                "opencode_available": shutil.which("opencode") is not None,
                "codex_available": shutil.which("codex") is not None,
                "resource_limits": {
                    "omp_threads": 1,
                    "openblas_threads": 1,
                    "mkl_threads": 1,
                    "worker_nice": 10,
                },
            }
            _emit(payload, args.output)
        elif args.command == "experiment":
            _emit(
                {"experiments": [{"name": name, "description": description} for name, description in EXPERIMENTS.items()]},
                args.output,
            )
        elif args.command == "config":
            config = _load_config(Path(args.path).resolve())
            _emit(
                {"ok": True, "path": str(Path(args.path).resolve()), "config_hash": _config_hash(config)},
                args.output,
            )
        elif args.command == "phase2-readiness":
            config = _load_config(Path(args.config).resolve())
            phase2 = config.get("paper_phase2")
            if not isinstance(phase2, dict):
                raise ExpctlError("CONFIG_INVALID", "phase2-readiness requires paper_phase2")
            result = audit_phase2_readiness(
                Path(args.output_dir).resolve(),
                phase2={**phase2, "protocol": config.get("protocol")},
                preflight_dir=Path(args.preflight_dir).resolve(),
                pricing_manifest=(
                    Path(args.pricing_manifest).resolve() if args.pricing_manifest else None
                ),
                power_analysis=Path(args.power_analysis).resolve() if args.power_analysis else None,
            )
            _emit(result, args.output)
        elif args.command == "run" and args.run_command == "start":
            _emit(_start(args), args.output)
        elif args.command == "run" and args.run_command == "status":
            run_dir = _run_dir(_registry_root(args.root), args.run_id)
            _emit(_public_metadata(_refresh_status(run_dir)), args.output)
        elif args.command == "run" and args.run_command == "logs":
            _logs(args)
        elif args.command == "run":
            _emit(_signal_run(args, args.run_command), args.output)
        elif args.command == "analyze":
            _emit(_analyze(_run_dir(_registry_root(args.root), args.run_id)), args.output)
        elif args.command == "export":
            _emit(_export(args), args.output)
        elif args.command == "_worker":
            raise SystemExit(_worker(args.run_dir.resolve()))
    except ExpctlError as exc:
        payload = {
            "ok": False,
            "error": {
                "code": exc.code,
                "message": str(exc),
                "retryable": exc.retryable,
                "details": exc.details,
            },
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
