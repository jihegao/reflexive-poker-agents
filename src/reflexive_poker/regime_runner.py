from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import traceback
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .regime_experiment import (
    REGIME_CONDITIONS,
    REGIME_MIRRORS,
    RegimeExperimentConfig,
    RegimeExperimentRow,
    run_regime_match,
    summarize_regime_experiment,
)
from .regime_statistics import build_regime_statistics, write_regime_statistics

REGIME_RUN_PROTOCOL = "regime-adaptation-resumable-v1"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RegimeRunError(RuntimeError):
    pass


class RegimeCheckpointError(RegimeRunError):
    pass


@dataclass(frozen=True)
class RegimeRunConfig:
    run_id: str
    output_dir: Path
    seeds: tuple[int, ...] = tuple(range(9300, 9330))
    conditions: tuple[str, ...] = REGIME_CONDITIONS
    mirrors: tuple[int, ...] = REGIME_MIRRORS
    hands: int = 320
    switch_hand: int = 160
    equity_samples: int = 4
    recovery_window: int = 32
    simulation_rollout_hands: int = 36
    simulation_equity_samples: int = 1
    formation_observations: int = 48
    calibration_observations: int = 32
    max_blocks: int | None = None

    def validate(self) -> None:
        if not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id must be 1-128 URL-safe characters")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if self.conditions != REGIME_CONDITIONS:
            raise ValueError(f"conditions must be exactly {REGIME_CONDITIONS}")
        if self.mirrors != REGIME_MIRRORS:
            raise ValueError(f"mirrors must be exactly {REGIME_MIRRORS}")
        if self.hands < 2 or self.switch_hand <= 0 or self.switch_hand >= self.hands:
            raise ValueError("switch_hand must fall inside the experiment horizon")
        if self.equity_samples < 1 or self.simulation_equity_samples < 1:
            raise ValueError("equity sample counts must be positive")
        if self.recovery_window < 1:
            raise ValueError("recovery_window must be positive")
        if self.simulation_rollout_hands < 2:
            raise ValueError("simulation_rollout_hands must be at least two")
        if self.formation_observations < 1 or self.calibration_observations < 1:
            raise ValueError("formation and calibration observations must be positive")
        if self.max_blocks is not None and self.max_blocks < 1:
            raise ValueError("max_blocks must be positive when provided")

    def experiment_config(self) -> RegimeExperimentConfig:
        return RegimeExperimentConfig(
            seeds=self.seeds,
            hands=self.hands,
            switch_hand=self.switch_hand,
            equity_samples=self.equity_samples,
            recovery_window=self.recovery_window,
            simulation_rollout_hands=self.simulation_rollout_hands,
            simulation_equity_samples=self.simulation_equity_samples,
            formation_observations=self.formation_observations,
            calibration_observations=self.calibration_observations,
        )


def regime_run_config_from_mapping(
    payload: Mapping[str, Any],
    *,
    output_dir: Path,
    run_id: str | None = None,
    max_blocks: int | None = None,
) -> RegimeRunConfig:
    section = payload.get("regime_adaptation")
    if not isinstance(section, Mapping):
        raise TypeError("config requires a regime_adaptation mapping")
    seed_spec = section.get("seeds")
    if isinstance(seed_spec, Mapping):
        seed_start = int(seed_spec.get("start", 9300))
        seed_count = int(seed_spec.get("count", 30))
        if seed_count < 1:
            raise ValueError("regime_adaptation.seeds.count must be positive")
        seeds = tuple(range(seed_start, seed_start + seed_count))
    elif isinstance(seed_spec, Sequence) and not isinstance(seed_spec, (str, bytes)):
        seeds = tuple(int(seed) for seed in seed_spec)
    else:
        raise TypeError("regime_adaptation.seeds must be a start/count mapping or a list")
    configured_run_id = section.get("run_id")
    resolved_run_id = run_id or (str(configured_run_id) if configured_run_id else None)
    if resolved_run_id is None:
        raise ValueError("regime_adaptation.run_id is required")
    config = RegimeRunConfig(
        run_id=resolved_run_id,
        output_dir=output_dir,
        seeds=seeds,
        conditions=tuple(str(value) for value in section.get("conditions", REGIME_CONDITIONS)),
        mirrors=tuple(int(value) for value in section.get("mirrors", REGIME_MIRRORS)),
        hands=int(section.get("hands", 320)),
        switch_hand=int(section.get("switch_hand", 160)),
        equity_samples=int(section.get("equity_samples", 4)),
        recovery_window=int(section.get("recovery_window", 32)),
        simulation_rollout_hands=int(section.get("simulation_rollout_hands", 36)),
        simulation_equity_samples=int(section.get("simulation_equity_samples", 1)),
        formation_observations=int(section.get("formation_observations", 48)),
        calibration_observations=int(section.get("calibration_observations", 32)),
        max_blocks=max_blocks,
    )
    config.validate()
    return config


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def _atomic_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegimeCheckpointError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RegimeCheckpointError(f"JSON artifact is not an object: {path}")
    return payload


def _block_id(condition: str, seed: int, mirror: int) -> str:
    return f"{condition}__seed_{seed}__mirror_{mirror}"


def _schedule(config: RegimeRunConfig) -> list[dict[str, Any]]:
    return [
        {
            "block_id": _block_id(condition, seed, mirror),
            "condition": condition,
            "seed": seed,
            "mirror": mirror,
        }
        for condition in config.conditions
        for seed in config.seeds
        for mirror in config.mirrors
    ]


def _plan_payload(config: RegimeRunConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": REGIME_RUN_PROTOCOL,
        "run_id": config.run_id,
        "conditions": list(config.conditions),
        "mirrors": list(config.mirrors),
        "seeds": list(config.seeds),
        "hands": config.hands,
        "switch_hand": config.switch_hand,
        "equity_samples": config.equity_samples,
        "recovery_window": config.recovery_window,
        "simulation_rollout_hands": config.simulation_rollout_hands,
        "simulation_equity_samples": config.simulation_equity_samples,
        "formation_observations": config.formation_observations,
        "calibration_observations": config.calibration_observations,
    }


def _freeze_or_verify(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = _load_json(path)
        if existing != payload:
            raise RegimeRunError(f"frozen run input differs from the existing artifact: {path}")
        return
    _atomic_json(path, payload)


def _checkpoint_path(output_dir: Path, block: Mapping[str, Any]) -> Path:
    return (
        output_dir
        / "checkpoints"
        / str(block["condition"])
        / f"seed_{block['seed']}__mirror_{block['mirror']}.json"
    )


def _read_checkpoint(
    path: Path,
    *,
    run_id: str,
    plan_hash: str,
    block: Mapping[str, Any],
) -> RegimeExperimentRow:
    payload = _load_json(path)
    expected_identity = {
        "block_id": block["block_id"],
        "condition": block["condition"],
        "seed": block["seed"],
        "mirror": block["mirror"],
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "completed"
        or payload.get("run_id") != run_id
        or payload.get("plan_hash") != plan_hash
        or payload.get("block") != expected_identity
    ):
        raise RegimeCheckpointError(f"checkpoint identity mismatch: {path}")
    row_payload = payload.get("row")
    if not isinstance(row_payload, dict):
        raise RegimeCheckpointError(f"checkpoint row is missing: {path}")
    expected_fields = {field.name for field in fields(RegimeExperimentRow)}
    if set(row_payload) != expected_fields or payload.get("row_sha256") != _payload_hash(row_payload):
        raise RegimeCheckpointError(f"checkpoint row checksum or schema mismatch: {path}")
    try:
        row = RegimeExperimentRow(**row_payload)
    except TypeError as exc:
        raise RegimeCheckpointError(f"checkpoint row cannot be decoded: {path}") from exc
    if (row.condition, row.seed, row.mirror) != (
        block["condition"],
        block["seed"],
        block["mirror"],
    ):
        raise RegimeCheckpointError(f"checkpoint row key mismatch: {path}")
    return row


def _scan_checkpoints(
    config: RegimeRunConfig,
    schedule: Sequence[dict[str, Any]],
    plan_hash: str,
) -> tuple[dict[str, RegimeExperimentRow], list[dict[str, str]]]:
    expected_paths = {
        _checkpoint_path(config.output_dir, block): block for block in schedule
    }
    rows: dict[str, RegimeExperimentRow] = {}
    corrupt: list[dict[str, str]] = []
    for path, block in expected_paths.items():
        if not path.exists():
            continue
        try:
            rows[str(block["block_id"])] = _read_checkpoint(
                path,
                run_id=config.run_id,
                plan_hash=plan_hash,
                block=block,
            )
        except RegimeCheckpointError as exc:
            corrupt.append({"path": str(path), "error": str(exc)})
    checkpoints_root = config.output_dir / "checkpoints"
    if checkpoints_root.exists():
        for path in checkpoints_root.rglob("*.json"):
            if path not in expected_paths:
                corrupt.append({"path": str(path), "error": "unexpected checkpoint path"})
    return rows, corrupt


def _failure_paths(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "failures").glob("*.json"))


def _record_failure(
    config: RegimeRunConfig,
    block: Mapping[str, Any],
    exc: BaseException,
    *,
    recovered_inflight: bool = False,
) -> Path:
    existing = list((config.output_dir / "failures").glob(f"{block['block_id']}__attempt_*.json"))
    attempt = len(existing) + 1
    path = config.output_dir / "failures" / f"{block['block_id']}__attempt_{attempt:03d}.json"
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "block": dict(block),
            "attempt": attempt,
            "failed_at": _now(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "recovered_inflight": recovered_inflight,
            "traceback": "" if recovered_inflight else "".join(traceback.format_exception(exc)),
        },
    )
    return path


def _recover_inflight(config: RegimeRunConfig, schedule: Sequence[dict[str, Any]]) -> None:
    indexed = {str(block["block_id"]): block for block in schedule}
    inflight_dir = config.output_dir / "inflight"
    if not inflight_dir.exists():
        return
    for path in sorted(inflight_dir.glob("*.json")):
        block = indexed.get(path.stem)
        if block is None:
            raise RegimeCheckpointError(f"unexpected inflight marker: {path}")
        checkpoint = _checkpoint_path(config.output_dir, block)
        if not checkpoint.exists():
            _record_failure(
                config,
                block,
                RuntimeError("process interrupted before an atomic checkpoint was committed"),
                recovered_inflight=True,
            )
        path.unlink()


def _publish_status(
    config: RegimeRunConfig,
    schedule: Sequence[dict[str, Any]],
    plan_hash: str,
    rows_by_block: Mapping[str, RegimeExperimentRow],
    corrupt: Sequence[dict[str, str]],
    *,
    state: str,
    completed_this_invocation: int,
) -> dict[str, Any]:
    completed_ids = set(rows_by_block)
    missing = [str(block["block_id"]) for block in schedule if block["block_id"] not in completed_ids]
    failure_paths = _failure_paths(config.output_dir)
    completion = {
        "schema_version": 1,
        "run_id": config.run_id,
        "plan_hash": plan_hash,
        "state": state,
        "total_blocks": len(schedule),
        "completed_blocks": len(completed_ids),
        "completed_this_invocation": completed_this_invocation,
        "missing_blocks": missing,
        "failure_records": [str(path.relative_to(config.output_dir)) for path in failure_paths],
        "corrupt_checkpoints": list(corrupt),
        "updated_at": _now(),
    }
    rows = [rows_by_block[str(block["block_id"])] for block in schedule if block["block_id"] in rows_by_block]
    row_payloads = [asdict(row) for row in rows]
    _atomic_csv(
        config.output_dir / "matches.csv",
        row_payloads,
        tuple(field.name for field in fields(RegimeExperimentRow)),
    )
    _atomic_json(
        config.output_dir / "summary.json",
        {"conditions": summarize_regime_experiment(rows), "partial": state != "completed"},
    )
    _atomic_json(config.output_dir / "COMPLETION_STATUS.json", completion)
    statistics = build_regime_statistics(
        rows,
        expected_seeds=config.seeds,
        expected_mirrors=config.mirrors,
        expected_conditions=config.conditions,
        completion_status=completion,
    )
    write_regime_statistics(config.output_dir, statistics)
    completion["formal_completion_valid"] = statistics["evidence_gate"]["valid"]
    completion["valid_paired_blocks"] = statistics["paired_summary"]["valid_paired_blocks"]
    _atomic_json(config.output_dir / "COMPLETION_STATUS.json", completion)
    manifest = _load_json(config.output_dir / "manifest.json")
    manifest["status"] = {
        "state": state,
        "completed": len(completed_ids),
        "total": len(schedule),
        "valid_paired_blocks": completion["valid_paired_blocks"],
        "formal_completion_valid": completion["formal_completion_valid"],
        "updated_at": completion["updated_at"],
    }
    _atomic_json(config.output_dir / "manifest.json", manifest)
    if completion["formal_completion_valid"]:
        _atomic_json(
            config.output_dir / "COMPLETED.json",
            {
                "schema_version": 1,
                "run_id": config.run_id,
                "plan_hash": plan_hash,
                "completed_blocks": len(completed_ids),
                "valid_paired_blocks": completion["valid_paired_blocks"],
                "completed_at": _now(),
            },
        )
    return completion


def run_regime_experiment_resumable(config: RegimeRunConfig) -> dict[str, Any]:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = _schedule(config)
    frozen_config = _plan_payload(config)
    seed_manifest = {
        "schema_version": 1,
        "run_id": config.run_id,
        "seeds": list(config.seeds),
        "conditions": list(config.conditions),
        "mirrors": list(config.mirrors),
        "blocks": schedule,
        "total_blocks": len(schedule),
    }
    plan_hash = _payload_hash({"config": frozen_config, "seed_manifest": seed_manifest})
    frozen_config = {**frozen_config, "plan_hash": plan_hash}
    seed_manifest = {**seed_manifest, "plan_hash": plan_hash}
    _freeze_or_verify(config.output_dir / "FROZEN_CONFIG.json", frozen_config)
    _freeze_or_verify(config.output_dir / "SEED_MANIFEST.json", seed_manifest)
    manifest_path = config.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if manifest.get("run_id") != config.run_id or manifest.get("plan_hash") != plan_hash:
            raise RegimeRunError("manifest differs from the existing resumable run")
    else:
        _atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "protocol": REGIME_RUN_PROTOCOL,
                "run_id": config.run_id,
                "plan_hash": plan_hash,
                "created_at": _now(),
                "frozen_config": "FROZEN_CONFIG.json",
                "seed_manifest": "SEED_MANIFEST.json",
                "checkpoint_unit": "condition_seed_mirror",
                "status": {
                    "state": "created",
                    "completed": 0,
                    "total": len(schedule),
                    "formal_completion_valid": False,
                },
            },
        )

    _recover_inflight(config, schedule)
    rows_by_block, corrupt = _scan_checkpoints(config, schedule, plan_hash)
    if corrupt:
        _publish_status(
            config,
            schedule,
            plan_hash,
            rows_by_block,
            corrupt,
            state="blocked_corrupt_checkpoint",
            completed_this_invocation=0,
        )
        raise RegimeCheckpointError("one or more regime checkpoints are corrupt")

    executed = 0
    experiment_config = config.experiment_config()
    for block in schedule:
        block_id = str(block["block_id"])
        if block_id in rows_by_block:
            continue
        if config.max_blocks is not None and executed >= config.max_blocks:
            break
        inflight_path = config.output_dir / "inflight" / f"{block_id}.json"
        _atomic_json(
            inflight_path,
            {
                "schema_version": 1,
                "run_id": config.run_id,
                "plan_hash": plan_hash,
                "block": block,
                "started_at": _now(),
            },
        )
        try:
            row = run_regime_match(
                experiment_config,
                str(block["condition"]),
                int(block["seed"]),
                int(block["mirror"]),
            )
            row_payload = asdict(row)
            _atomic_json(
                _checkpoint_path(config.output_dir, block),
                {
                    "schema_version": 1,
                    "status": "completed",
                    "run_id": config.run_id,
                    "plan_hash": plan_hash,
                    "block": block,
                    "row": row_payload,
                    "row_sha256": _payload_hash(row_payload),
                    "completed_at": _now(),
                },
            )
            rows_by_block[block_id] = row
            executed += 1
            inflight_path.unlink(missing_ok=True)
            _publish_status(
                config,
                schedule,
                plan_hash,
                rows_by_block,
                (),
                state="running",
                completed_this_invocation=executed,
            )
        except BaseException as exc:
            _record_failure(config, block, exc)
            inflight_path.unlink(missing_ok=True)
            state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            _publish_status(
                config,
                schedule,
                plan_hash,
                rows_by_block,
                (),
                state=state,
                completed_this_invocation=executed,
            )
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise RegimeRunError(f"regime block failed: {block_id}") from exc

    final_state = "completed" if len(rows_by_block) == len(schedule) else "incomplete"
    return _publish_status(
        config,
        schedule,
        plan_hash,
        rows_by_block,
        (),
        state=final_state,
        completed_this_invocation=executed,
    )
