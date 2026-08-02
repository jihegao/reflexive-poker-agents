from __future__ import annotations

import gzip
import json
from pathlib import Path

from reflexive_poker.phase1_evidence_bundle import _artifact_provenance, audit_phase1_evidence_bundle


def _artifact(path: Path, *, valid: bool, predictions: int) -> None:
    path.mkdir(parents=True)
    (path / "provider_gate.json").write_text(
        json.dumps({"valid": valid, "attempt_audit": {"valid": True}}), encoding="utf-8"
    )
    (path / "run.json").write_text(json.dumps({"run_id": path.name, "config_hash": "frozen"}), encoding="utf-8")
    (path / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(
            {
                "source_fingerprint": "frozen-source",
                "worktree_dirty": False,
                "protocol_semantics_id": "prbench-cross-model-v1",
                "protocol_semantics_fingerprint": "frozen-semantics",
                "frozen_inputs": {
                    "frozen_inputs/PRICE_MANIFEST.json": {"sha256": "frozen-price"}
                },
            }
        ),
        encoding="utf-8",
    )
    with gzip.open(path / "cases.jsonl.gz", "wt", encoding="utf-8") as cases:
        for index in range(predictions // 5):
            cases.write(json.dumps({"case_id": f"case-{index:03d}"}) + "\n")
    treatments = (
        "state_only",
        "action_prediction",
        "d1_budget_matched",
        "recursive_d2",
        "recursive_d3",
    )
    with gzip.open(path / "predictions.jsonl.gz", "wt", encoding="utf-8") as handle:
        for index in range(predictions // 5):
            for treatment in treatments:
                handle.write(
                    json.dumps(
                        {
                            "case_id": f"case-{index:03d}",
                            "treatment": treatment,
                            "method": f"llm_{treatment}",
                        }
                    )
                    + "\n"
                )
        for _ in range(predictions % 5):
            handle.write('{"method":"llm_state_only"}\n')


def test_evidence_audit_fails_closed_until_every_requirement_exists(tmp_path: Path) -> None:
    deepseek_preflight = tmp_path / "deepseek_preflight"
    codex_preflight = tmp_path / "codex_preflight"
    baselines = tmp_path / "baselines"
    deepseek_offline = tmp_path / "deepseek_offline"
    codex_offline = tmp_path / "codex_offline"
    closed_loop = tmp_path / "closed_loop"
    _artifact(deepseek_preflight, valid=True, predictions=20)
    _artifact(codex_preflight, valid=True, predictions=20)
    _artifact(deepseek_offline, valid=True, predictions=1_000)
    _artifact(codex_offline, valid=True, predictions=1_000)
    _artifact(baselines, valid=True, predictions=0)
    with gzip.open(baselines / "cases.jsonl.gz", "wt", encoding="utf-8") as handle:
        for _ in range(200):
            handle.write("{}\n")

    result = audit_phase1_evidence_bundle(
        tmp_path / "audit",
        deepseek_preflight=deepseek_preflight,
        codex_preflight=codex_preflight,
        baselines=baselines,
        deepseek_offline=deepseek_offline,
        codex_offline=codex_offline,
        closed_loop=closed_loop,
    )

    assert not result["complete"]
    assert result["requirements"]["deepseek_offline"]["complete"]
    assert (tmp_path / "audit" / "PHASE1_EVIDENCE_AUDIT.zh-CN.md").exists()


def test_evidence_audit_requires_every_paired_closed_loop_seed(tmp_path: Path) -> None:
    names = (
        "deepseek_preflight",
        "codex_preflight",
        "deepseek_offline",
        "codex_offline",
    )
    paths = {name: tmp_path / name for name in names}
    for name in ("deepseek_preflight", "codex_preflight"):
        _artifact(paths[name], valid=True, predictions=20)
    for name in ("deepseek_offline", "codex_offline"):
        _artifact(paths[name], valid=True, predictions=1_000)
    baselines = tmp_path / "baselines"
    _artifact(baselines, valid=True, predictions=0)
    with gzip.open(baselines / "cases.jsonl.gz", "wt", encoding="utf-8") as handle:
        for _ in range(200):
            handle.write("{}\n")
    closed_loop = tmp_path / "closed_loop"
    _artifact(closed_loop, valid=True, predictions=0)
    (closed_loop / "CROSS_MODEL_PAIRED_BLOCK_STATUS.json").write_text(
        json.dumps(
            {"target_seeds": 40, "valid_paired_blocks": 39, "formal_completion_valid": False}
        ),
        encoding="utf-8",
    )

    incomplete = audit_phase1_evidence_bundle(
        tmp_path / "audit-incomplete",
        deepseek_preflight=paths["deepseek_preflight"],
        codex_preflight=paths["codex_preflight"],
        baselines=baselines,
        deepseek_offline=paths["deepseek_offline"],
        codex_offline=paths["codex_offline"],
        closed_loop=closed_loop,
    )
    assert not incomplete["complete"]

    (closed_loop / "CROSS_MODEL_PAIRED_BLOCK_STATUS.json").write_text(
        json.dumps(
            {"target_seeds": 40, "valid_paired_blocks": 40, "formal_completion_valid": True}
        ),
        encoding="utf-8",
    )
    complete = audit_phase1_evidence_bundle(
        tmp_path / "audit-complete",
        deepseek_preflight=paths["deepseek_preflight"],
        codex_preflight=paths["codex_preflight"],
        baselines=baselines,
        deepseek_offline=paths["deepseek_offline"],
        codex_offline=paths["codex_offline"],
        closed_loop=closed_loop,
    )
    assert complete["complete"]
    assert complete["claim_status"] == "ready_for_locked_analysis"

    mixed_semantics = json.loads((paths["codex_offline"] / "SOURCE_PROVENANCE.json").read_text())
    mixed_semantics["protocol_semantics_fingerprint"] = "different-semantics"
    (paths["codex_offline"] / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(mixed_semantics), encoding="utf-8"
    )
    inconsistent = audit_phase1_evidence_bundle(
        tmp_path / "audit-mixed-semantics",
        deepseek_preflight=paths["deepseek_preflight"],
        codex_preflight=paths["codex_preflight"],
        baselines=baselines,
        deepseek_offline=paths["deepseek_offline"],
        codex_offline=paths["codex_offline"],
        closed_loop=closed_loop,
    )
    assert not inconsistent["complete"]
    assert not inconsistent["provenance"]["consistent"]


def test_evidence_audit_rejects_duplicate_prediction_keys_with_matching_row_count(
    tmp_path: Path,
) -> None:
    deepseek_preflight = tmp_path / "deepseek_preflight"
    _artifact(deepseek_preflight, valid=True, predictions=20)
    with gzip.open(deepseek_preflight / "predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    rows[-1] = dict(rows[0])
    with gzip.open(deepseek_preflight / "predictions.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    unavailable = tmp_path / "unavailable"

    result = audit_phase1_evidence_bundle(
        tmp_path / "audit",
        deepseek_preflight=deepseek_preflight,
        codex_preflight=unavailable,
        baselines=unavailable,
        deepseek_offline=unavailable,
        codex_offline=unavailable,
        closed_loop=unavailable,
    )

    assert not result["requirements"]["deepseek_preflight"]["coverage"]["valid"]
    assert not result["requirements"]["deepseek_preflight"]["complete"]


def test_evidence_audit_rejects_missing_or_mixed_price_snapshots(tmp_path: Path) -> None:
    paths = {name: tmp_path / name for name in ("deepseek", "codex", "baselines", "closed")}
    _artifact(paths["deepseek"], valid=True, predictions=20)
    _artifact(paths["codex"], valid=True, predictions=20)
    _artifact(paths["baselines"], valid=True, predictions=0)
    with gzip.open(paths["baselines"] / "cases.jsonl.gz", "wt", encoding="utf-8") as handle:
        for _ in range(200):
            handle.write("{}\n")
    _artifact(paths["closed"], valid=True, predictions=0)
    (paths["closed"] / "CROSS_MODEL_PAIRED_BLOCK_STATUS.json").write_text(
        json.dumps({"target_seeds": 1, "valid_paired_blocks": 1}), encoding="utf-8"
    )
    # Re-use the small complete artifacts for both 1,000-call requirements only
    # to assert provenance failure before outcome coverage is considered.
    source = json.loads((paths["codex"] / "SOURCE_PROVENANCE.json").read_text())
    source["frozen_inputs"]["frozen_inputs/PRICE_MANIFEST.json"]["sha256"] = "different-price"
    (paths["codex"] / "SOURCE_PROVENANCE.json").write_text(json.dumps(source), encoding="utf-8")
    result = audit_phase1_evidence_bundle(
        tmp_path / "audit",
        deepseek_preflight=paths["deepseek"],
        codex_preflight=paths["codex"],
        baselines=paths["baselines"],
        deepseek_offline=tmp_path / "missing-deepseek-offline",
        codex_offline=tmp_path / "missing-codex-offline",
        closed_loop=paths["closed"],
    )
    assert not result["provenance"]["consistent"]


def test_evidence_audit_discovers_resumable_closed_loop_snapshot_below_artifacts(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    closed_loop = run / "artifacts" / "llm_confirmation"
    closed_loop.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "run_id": "closed-loop",
                "config_hash": "frozen",
                "pricing_manifest_sha256": "frozen-price",
                "pricing_manifest_frozen_at_utc": "2026-08-02T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (closed_loop / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(
            {
                "source_fingerprint": "frozen-source",
                "worktree_dirty": False,
                "protocol_semantics_id": "prbench-cross-model-v1",
                "frozen_inputs": {
                    "frozen_inputs/PRICE_MANIFEST.json": {"sha256": "frozen-price"}
                },
            }
        ),
        encoding="utf-8",
    )
    provenance = _artifact_provenance(closed_loop)
    assert provenance["source_provenance_present"]
    assert provenance["worktree_dirty"] is False
    assert provenance["pricing_manifest_present"]
    assert provenance["pricing_manifest_sha256"] == "frozen-price"
