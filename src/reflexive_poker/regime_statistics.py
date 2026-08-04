from __future__ import annotations

import csv
import io
import json
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .regime_experiment import (
    REGIME_CONDITIONS,
    REGIME_MIRRORS,
    RegimeExperimentRow,
    paired_regime_effects,
    summarize_paired_regime_effects,
)


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


def build_regime_statistics(
    rows: Sequence[RegimeExperimentRow],
    *,
    expected_seeds: Sequence[int],
    expected_mirrors: Sequence[int] = REGIME_MIRRORS,
    expected_conditions: Sequence[str] = REGIME_CONDITIONS,
    completion_status: dict[str, Any],
) -> dict[str, Any]:
    """Build paired inference while keeping incomplete schedule blocks visible."""
    seeds = tuple(expected_seeds)
    mirrors = tuple(expected_mirrors)
    conditions = tuple(expected_conditions)
    if len(set(seeds)) != len(seeds):
        raise ValueError("expected_seeds must not contain duplicates")
    if len(set(mirrors)) != len(mirrors):
        raise ValueError("expected_mirrors must not contain duplicates")
    if len(set(conditions)) != len(conditions):
        raise ValueError("expected_conditions must not contain duplicates")

    expected_block_keys = {(seed, mirror) for seed in seeds for mirror in mirrors}
    grouped: dict[tuple[int, int], list[RegimeExperimentRow]] = defaultdict(list)
    unexpected_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (row.seed, row.mirror)
        if key not in expected_block_keys or row.condition not in conditions:
            unexpected_rows.append(
                {"condition": row.condition, "seed": row.seed, "mirror": row.mirror}
            )
            continue
        grouped[key].append(row)

    paired_blocks: list[dict[str, Any]] = []
    valid_rows: list[RegimeExperimentRow] = []
    for seed in seeds:
        for mirror in mirrors:
            block_rows = grouped[(seed, mirror)]
            counts = Counter(row.condition for row in block_rows)
            missing = [condition for condition in conditions if counts[condition] == 0]
            duplicate = [condition for condition in conditions if counts[condition] > 1]
            valid = not missing and not duplicate and len(block_rows) == len(conditions)
            paired_blocks.append(
                {
                    "seed": seed,
                    "mirror": mirror,
                    "valid": valid,
                    "observed_conditions": ",".join(sorted(counts)),
                    "missing_conditions": ",".join(missing),
                    "duplicate_conditions": ",".join(duplicate),
                }
            )
            if valid:
                valid_rows.extend(block_rows)

    effects = paired_regime_effects(valid_rows)
    expected_paired_blocks = len(expected_block_keys)
    valid_paired_blocks = sum(bool(block["valid"]) for block in paired_blocks)
    invalid_blocks = [
        {"seed": block["seed"], "mirror": block["mirror"]}
        for block in paired_blocks
        if not block["valid"]
    ]
    run_complete = (
        completion_status.get("state") == "completed"
        and completion_status.get("completed_blocks") == completion_status.get("total_blocks")
        and not completion_status.get("corrupt_checkpoints")
    )
    gate_reasons: list[str] = []
    if not run_complete:
        gate_reasons.append("run_incomplete")
    if valid_paired_blocks != expected_paired_blocks:
        gate_reasons.append("paired_block_intersection_incomplete")
    if unexpected_rows:
        gate_reasons.append("unexpected_rows")
    formal_completion_valid = not gate_reasons
    summary = {
        **summarize_paired_regime_effects(effects),
        "ci_method": "two_sided_student_t_95_percent",
        "inference_unit": "seed_and_seat_mirror_block",
        "expected_paired_blocks": expected_paired_blocks,
        "valid_paired_blocks": valid_paired_blocks,
        "formal_completion_valid": formal_completion_valid,
        "formal_conclusion_allowed": formal_completion_valid,
        "claim_status": (
            "ready_for_formal_interpretation"
            if formal_completion_valid
            else "blocked_incomplete_or_invalid_run"
        ),
    }
    evidence_gate = {
        "schema_version": 1,
        "gate": "regime_adaptation_formal_evidence_v1",
        "valid": formal_completion_valid,
        "formal_conclusion_allowed": formal_completion_valid,
        "reasons": gate_reasons,
        "expected_match_blocks": completion_status.get("total_blocks"),
        "completed_match_blocks": completion_status.get("completed_blocks"),
        "expected_paired_blocks": expected_paired_blocks,
        "valid_paired_blocks": valid_paired_blocks,
        "invalid_paired_blocks": invalid_blocks,
        "unexpected_rows": unexpected_rows,
        "completion_status": completion_status,
    }
    return {
        "paired_blocks": paired_blocks,
        "paired_effects": [asdict(effect) for effect in effects],
        "paired_summary": summary,
        "evidence_gate": evidence_gate,
    }


def write_regime_statistics(output_dir: Path, statistics: dict[str, Any]) -> None:
    paired_blocks = statistics["paired_blocks"]
    paired_effects = statistics["paired_effects"]
    _atomic_csv(
        output_dir / "paired_blocks.csv",
        paired_blocks,
        (
            "seed",
            "mirror",
            "valid",
            "observed_conditions",
            "missing_conditions",
            "duplicate_conditions",
        ),
    )
    _atomic_csv(
        output_dir / "paired_effects.csv",
        paired_effects,
        (
            "seed",
            "mirror",
            "treatment",
            "control",
            "total_reward_delta_bb",
            "post_switch_bb100_delta",
            "recovery_hands_delta",
        ),
    )
    _atomic_json(output_dir / "paired_summary.json", statistics["paired_summary"])
    _atomic_json(output_dir / "EVIDENCE_GATE.json", statistics["evidence_gate"])
