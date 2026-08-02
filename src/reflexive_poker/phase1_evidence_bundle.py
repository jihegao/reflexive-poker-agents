"""Completion audit for the Phase 1 paper-minimum evidence package."""

from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

SEMANTIC_SOURCE_FILES = (
    "src/reflexive_poker/phase1_models.py",
    "src/reflexive_poker/phase1_experiment.py",
    "src/reflexive_poker/phase1_statistics.py",
    "src/reflexive_poker/phase1_protocol.py",
    "src/reflexive_poker/phase1_offline.py",
    "src/reflexive_poker/phase1_offline_evidence.py",
    "src/reflexive_poker/environment.py",
    "src/reflexive_poker/agents.py",
    "src/reflexive_poker/tournament_agents.py",
    "src/reflexive_poker/llm_player.py",
    "configs/phase1.yaml",
)
OFFLINE_TREATMENTS = (
    "state_only",
    "action_prediction",
    "d1_budget_matched",
    "recursive_d2",
    "recursive_d3",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_count(path: Path, *, llm_only: bool = False) -> int:
    if not path.exists():
        return 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle)
        if llm_only:
            return sum(str(row.get("method", "")).startswith("llm_") for row in rows)
        return sum(1 for _ in rows)


def _prediction_coverage(path: Path, *, expected_case_count: int) -> dict[str, Any]:
    cases_path = path / "cases.jsonl.gz"
    predictions_path = path / "predictions.jsonl.gz"
    if not cases_path.exists() or not predictions_path.exists():
        return {"valid": False, "case_count": 0, "prediction_keys": 0}
    try:
        with gzip.open(cases_path, "rt", encoding="utf-8") as handle:
            case_ids = [str(json.loads(line)["case_id"]) for line in handle if line.strip()]
        with gzip.open(predictions_path, "rt", encoding="utf-8") as handle:
            keys = [
                (str(row["case_id"]), str(row["treatment"]))
                for row in (json.loads(line) for line in handle if line.strip())
                if str(row.get("method", "")).startswith("llm_")
            ]
    except (KeyError, OSError, json.JSONDecodeError):
        return {"valid": False, "case_count": 0, "prediction_keys": 0}
    expected = {(case_id, treatment) for case_id in case_ids for treatment in OFFLINE_TREATMENTS}
    return {
        "valid": (
            len(case_ids) == expected_case_count
            and len(set(case_ids)) == expected_case_count
            and len(keys) == len(expected)
            and len(set(keys)) == len(expected)
            and set(keys) == expected
        ),
        "case_count": len(case_ids),
        "prediction_keys": len(keys),
    }


def _artifact_provenance(path: Path) -> dict[str, Any]:
    """Read run-level freeze data when an artifact belongs to an expctl run."""
    for candidate in (path, *path.parents):
        metadata_path = candidate / "run.json"
        if not metadata_path.exists():
            continue
        metadata = _read_json(metadata_path)
        snapshot = next(
            (
                value
                for value in (
                    candidate / "artifacts" / "SOURCE_PROVENANCE.json",
                    candidate / "SOURCE_PROVENANCE.json",
                )
                if value.exists()
            ),
            None,
        )
        source_archive = next(
            (
                value
                for value in (
                    candidate / "artifacts" / "SOURCE_SNAPSHOT.tar.gz",
                    candidate / "SOURCE_SNAPSHOT.tar.gz",
                )
                if value.exists()
            ),
            None,
        )
        snapshot_payload = _read_json(snapshot) if snapshot is not None else {}
        return {
            "run_id": metadata.get("run_id"),
            "config_hash": metadata.get("config_hash"),
            "source_provenance_present": snapshot is not None,
            "worktree_dirty": snapshot_payload.get("worktree_dirty"),
            "source_fingerprint": snapshot_payload.get("source_fingerprint"),
            "protocol_semantics_id": snapshot_payload.get("protocol_semantics_id")
            or _snapshot_protocol_semantics_id(source_archive),
            "protocol_semantics_fingerprint": snapshot_payload.get(
                "protocol_semantics_fingerprint"
            )
            or _semantic_snapshot_fingerprint(source_archive),
        }
    return {
        "run_id": None,
        "config_hash": None,
        "source_provenance_present": False,
        "worktree_dirty": None,
        "source_fingerprint": None,
        "protocol_semantics_id": None,
        "protocol_semantics_fingerprint": None,
    }


def _semantic_snapshot_fingerprint(source_archive: Path | None) -> str | None:
    if source_archive is None:
        return None
    digest = hashlib.sha256()
    try:
        with tarfile.open(source_archive, "r:gz") as archive:
            for relative in SEMANTIC_SOURCE_FILES:
                member = archive.extractfile(relative)
                if member is None:
                    return None
                digest.update(relative.encode())
                digest.update(member.read())
    except (OSError, tarfile.TarError):
        return None
    return digest.hexdigest()


def _snapshot_protocol_semantics_id(source_archive: Path | None) -> str | None:
    if source_archive is None:
        return None
    try:
        with tarfile.open(source_archive, "r:gz") as archive:
            member = archive.extractfile("configs/phase1.yaml")
            if member is None:
                return None
            for line in member.read().decode("utf-8").splitlines():
                if line.startswith("protocol:"):
                    return line.partition(":")[2].strip().strip('"')
    except (OSError, UnicodeDecodeError, tarfile.TarError):
        return None
    return None


def _provider_status(path: Path, *, expected_predictions: int) -> dict[str, Any]:
    gate_path = path / "provider_gate.json"
    predictions_path = path / "predictions.jsonl.gz"
    gate = _read_json(gate_path) if gate_path.exists() else {}
    observed = _jsonl_count(predictions_path, llm_only=expected_predictions > 0)
    coverage = _prediction_coverage(
        path, expected_case_count=expected_predictions // len(OFFLINE_TREATMENTS)
    )
    provenance = _artifact_provenance(path)
    return {
        "path": str(path),
        "gate_present": gate_path.exists(),
        "gate_valid": bool(gate.get("valid")),
        "expected_predictions": expected_predictions,
        "raw_predictions": observed,
        "complete": (
            bool(gate.get("valid"))
            and observed == expected_predictions
            and coverage["valid"]
            and provenance["source_provenance_present"]
            and provenance["worktree_dirty"] is False
        ),
        "identity_sources": gate.get("observed_model_identity_sources", []),
        "coverage": coverage,
        "provenance": provenance,
    }


def audit_phase1_evidence_bundle(
    output_dir: Path,
    *,
    deepseek_preflight: Path,
    codex_preflight: Path,
    baselines: Path,
    deepseek_offline: Path,
    codex_offline: Path,
    closed_loop: Path,
    case_count: int = 200,
    treatment_count: int = 5,
) -> dict[str, Any]:
    """Write a fail-closed checklist over the raw Phase 1 artifacts.

    The audit does not calculate outcomes. It only establishes whether the
    required evidence files exist and satisfy their own gates, so an incomplete
    provider run cannot be elevated into a paper claim by a summary report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_model_predictions = case_count * treatment_count
    baseline_cases = _jsonl_count(baselines / "cases.jsonl.gz")
    baseline_gate = _read_json(baselines / "provider_gate.json") if (baselines / "provider_gate.json").exists() else {}
    baseline_provenance = _artifact_provenance(baselines)
    statuses = {
        "deepseek_preflight": _provider_status(deepseek_preflight, expected_predictions=20),
        "codex_preflight": _provider_status(codex_preflight, expected_predictions=20),
        "baselines": {
            "path": str(baselines),
            "cases": baseline_cases,
            "gate_valid": bool(baseline_gate.get("valid")),
            "complete": (
                baseline_cases == case_count
                and bool(baseline_gate.get("valid"))
                and baseline_provenance["source_provenance_present"]
                and baseline_provenance["worktree_dirty"] is False
            ),
            "provenance": baseline_provenance,
        },
        "deepseek_offline": _provider_status(
            deepseek_offline, expected_predictions=expected_model_predictions
        ),
        "codex_offline": _provider_status(
            codex_offline, expected_predictions=expected_model_predictions
        ),
    }
    closed_loop_status = closed_loop / "CROSS_MODEL_PAIRED_BLOCK_STATUS.json"
    if closed_loop_status.exists():
        payload = _read_json(closed_loop_status)
        closed_loop_complete = (
            int(payload.get("target_seeds", 0)) > 0
            and int(payload.get("valid_paired_blocks", 0)) == int(payload.get("target_seeds", 0))
        )
    else:
        payload = {}
        closed_loop_complete = False
    statuses["closed_loop"] = {
        "path": str(closed_loop),
        "status_present": closed_loop_status.exists(),
        "status": payload,
        "complete": (
            closed_loop_complete
            and _artifact_provenance(closed_loop)["source_provenance_present"]
            and _artifact_provenance(closed_loop)["worktree_dirty"] is False
        ),
        "provenance": _artifact_provenance(closed_loop),
    }
    config_hashes = {
        section["provenance"]["config_hash"]
        for section in statuses.values()
        if section["provenance"]["config_hash"]
    }
    protocol_semantics_ids = {
        section["provenance"]["protocol_semantics_id"]
        for section in statuses.values()
        if section["provenance"]["protocol_semantics_id"]
    }
    provenance_complete = all(
        section["provenance"]["source_provenance_present"] for section in statuses.values()
    )
    clean_worktrees = all(
        section["provenance"]["worktree_dirty"] is False for section in statuses.values()
    )
    provenance_consistent = (
        provenance_complete
        and clean_worktrees
        and len(config_hashes) == 1
        and len(protocol_semantics_ids) == 1
    )
    complete = all(section["complete"] for section in statuses.values()) and provenance_consistent
    result = {
        "protocol": "prbench-cross-model-v1",
        "paper_phase": 1,
        "complete": complete,
        "claim_status": "ready_for_locked_analysis" if complete else "incomplete_no_paper_outcome_claim",
        "requirements": statuses,
        "provenance": {
            "complete": provenance_complete,
            "clean_worktrees": clean_worktrees,
            "consistent": provenance_consistent,
            "config_hashes": sorted(config_hashes),
            "protocol_semantics_ids": sorted(protocol_semantics_ids),
            "protocol_semantics_fingerprints": sorted(
                {
                    section["provenance"]["protocol_semantics_fingerprint"]
                    for section in statuses.values()
                    if section["provenance"]["protocol_semantics_fingerprint"]
                }
            ),
            "source_fingerprints": sorted(
                {
                    section["provenance"]["source_fingerprint"]
                    for section in statuses.values()
                    if section["provenance"]["source_fingerprint"]
                }
            ),
        },
    }
    (output_dir / "PHASE1_EVIDENCE_STATUS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Phase 1 论文最小证据包审计",
        "",
        f"- 完整：`{complete}`",
        f"- 结论状态：`{result['claim_status']}`",
        "",
        "| 证据项 | 完整 |",
        "|---|---:|",
    ]
    for name, section in statuses.items():
        lines.append(f"| {name} | {section['complete']} |")
    (output_dir / "PHASE1_EVIDENCE_AUDIT.zh-CN.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return result
