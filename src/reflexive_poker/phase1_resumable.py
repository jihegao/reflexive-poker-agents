from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .phase1_experiment import (
    ConfirmationJob,
    Phase1ExperimentConfig,
    Phase1LLMConfirmationPlan,
    _attach_paired_large_pot_sensitivity,
    _jsonable,
    _paired,
    _paired_hand_deltas,
    build_llm_confirmation_plan,
    phase1_simulation_matrix,
    run_phase1_experiment,
)
from .phase1_models import ProviderBudget, ReasoningTreatment
from .phase1_offline import OfflineBenchmarkConfig, run_offline_benchmark
from .phase1_protocol import (
    canonical_checkpoint_id,
    valid_paired_block_intersection,
    validate_closed_loop_completion,
)
from .phase1_statistics import inference_table

REQUIRED_BLOCK_FILES = (
    "manifest.json",
    "per_hand.csv",
    "per_seed.csv",
    "paired.csv",
    "inference.csv",
    "forks.csv",
    "provider_gate.json",
)

PROTOCOL_SEMANTIC_SOURCE_FILES = (
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
PROTOCOL_SEMANTICS_ID = "prbench-cross-model-v1"

PROTOCOL_SOURCE_FILES = PROTOCOL_SEMANTIC_SOURCE_FILES + (
    "src/reflexive_poker/phase1_evidence_bundle.py",
    "src/reflexive_poker/phase2_readiness.py",
    "src/reflexive_poker/phase1_resumable.py",
    "src/reflexive_poker/expctl.py",
    "scripts/run_phase1_experiment.py",
    "scripts/run_phase1_resumable.py",
    "configs/phase2.yaml",
    "src/reflexive_poker/regime_adaptation.py",
    "src/reflexive_poker/regime_agents.py",
    "src/reflexive_poker/regime_detection.py",
    "src/reflexive_poker/regime_experiment.py",
    "src/reflexive_poker/regime_runner.py",
    "src/reflexive_poker/regime_simulation.py",
    "src/reflexive_poker/regime_statistics.py",
    "scripts/run_regime_resumable.py",
    "configs/regime_pilot.yaml",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _code_provenance(allow_dirty_worktree: bool) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = bool(status.strip())
    if dirty and not allow_dirty_worktree:
        raise RuntimeError(
            "formal Phase 1 execution requires a clean worktree; commit the implementation "
            "or use --allow-dirty-worktree only for a bounded rehearsal"
        )
    digest = hashlib.sha256()
    for relative in PROTOCOL_SOURCE_FILES:
        path = repository / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return {
        "git_commit": commit,
        "worktree_dirty": dirty,
        "source_fingerprint": digest.hexdigest(),
        "protocol_semantics_id": PROTOCOL_SEMANTICS_ID,
        "protocol_semantics_fingerprint": _source_fingerprint(
            repository, PROTOCOL_SEMANTIC_SOURCE_FILES
        ),
    }


def _source_fingerprint(repository: Path, source_files: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in source_files:
        path = repository / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_source_snapshot(
    output_dir: Path,
    provenance: dict[str, Any],
    *,
    frozen_inputs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Archive the exact protocol sources used by a formal dirty-worktree run.

    A clean commit remains preferred, but a local research run must not become
    irreproducible merely because its validated implementation has not yet been
    committed. The artifact contains the same files included in the source
    fingerprint and is recorded before any provider calls.
    """
    repository = Path(__file__).resolve().parents[2]
    frozen_inputs = frozen_inputs or {}
    for arcname, input_path in frozen_inputs.items():
        if Path(arcname).is_absolute() or ".." in Path(arcname).parts:
            raise ValueError("frozen source input archive name must be relative")
        if not input_path.exists():
            raise FileNotFoundError(f"frozen source input does not exist: {input_path}")
    if frozen_inputs:
        digest = hashlib.sha256()
        digest.update(str(provenance["source_fingerprint"]).encode())
        for arcname, input_path in sorted(frozen_inputs.items()):
            digest.update(arcname.encode())
            digest.update(input_path.read_bytes())
        provenance = {
            **provenance,
            "source_fingerprint": digest.hexdigest(),
            "frozen_inputs": {
                arcname: {
                    "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    "artifact_path": str(input_path),
                }
                for arcname, input_path in sorted(frozen_inputs.items())
            },
        }
    snapshot = output_dir / "SOURCE_SNAPSHOT.tar.gz"
    provenance_path = output_dir / "SOURCE_PROVENANCE.json"
    if provenance_path.exists() and snapshot.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing.get("source_fingerprint") == provenance["source_fingerprint"]:
            return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(snapshot, "w:gz") as archive:
        for relative in PROTOCOL_SOURCE_FILES:
            archive.add(repository / relative, arcname=relative, recursive=False)
        for arcname, input_path in sorted(frozen_inputs.items()):
            archive.add(input_path, arcname=arcname, recursive=False)
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    payload = {
        **provenance,
        "source_snapshot": snapshot.name,
        "source_snapshot_sha256": digest,
        "source_snapshot_files": [*PROTOCOL_SOURCE_FILES, *sorted(frozen_inputs)],
    }
    _atomic_json(provenance_path, payload)
    return payload


def freeze_phase1_source_snapshot(
    output_dir: Path,
    *,
    allow_dirty_worktree: bool,
    frozen_inputs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Freeze the protocol implementation before the first provider call."""
    return _write_source_snapshot(
        output_dir,
        _code_provenance(allow_dirty_worktree),
        frozen_inputs=frozen_inputs,
    )


def _archive_interrupted(path: Path, reason: str) -> Path:
    interrupted = path.parent / "interrupted"
    interrupted.mkdir(parents=True, exist_ok=True)
    target = interrupted / f"{path.name}__{int(time.time() * 1000)}"
    shutil.move(str(path), str(target))
    _atomic_json(target / "INTERRUPTED.json", {"reason": reason, "source": path.name})
    return target


def _completion_valid(path: Path, expected_hash: str) -> bool:
    marker = path / "COMPLETED.json"
    if not marker.exists() or any(not (path / name).exists() for name in REQUIRED_BLOCK_FILES):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("block_hash") == expected_hash


def _prepare_running(final: Path, expected_hash: str) -> tuple[Path, bool]:
    if final.exists():
        if _completion_valid(final, expected_hash):
            return final, True
        _archive_interrupted(final, "invalid_or_incomplete_final_directory")
    running = final.with_name(final.name + ".running")
    if running.exists():
        _archive_interrupted(running, "process_interrupted_before_atomic_completion")
    running.mkdir(parents=True)
    return running, False


def _finish_running(running: Path, final: Path, marker: dict[str, Any]) -> None:
    _atomic_json(running / "COMPLETED.json", marker)
    os.replace(running, final)


def _aggregate_seed_blocks(
    block_dirs: list[Path],
    output_dir: Path,
    treatments: tuple[ReasoningTreatment, ...],
    bootstrap_samples: int,
    permutation_samples: int,
) -> dict[str, int]:
    per_hand = pd.concat([pd.read_csv(path / "per_hand.csv") for path in block_dirs])
    per_seed = pd.concat([pd.read_csv(path / "per_seed.csv") for path in block_dirs])
    paired_hands = _paired_hand_deltas(per_hand, treatments)
    paired = _attach_paired_large_pot_sensitivity(_paired(per_seed, treatments), paired_hands)
    inference = pd.concat(
        [
            inference_table(
                paired,
                metric="decision_regret_reduction",
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
            ),
            inference_table(
                paired,
                metric="chips_per_100_delta",
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
            ),
            inference_table(
                paired,
                metric="trimmed_delta",
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
            ),
            inference_table(
                paired,
                metric="leave_largest_pot_out_chips_per_100_delta",
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
            ),
        ],
        ignore_index=True,
    )
    _atomic_csv(output_dir / "per_hand.csv", per_hand)
    _atomic_csv(output_dir / "per_seed.csv", per_seed)
    _atomic_csv(output_dir / "paired.csv", paired)
    _atomic_csv(output_dir / "paired_hand_deltas.csv", paired_hands)
    _atomic_csv(output_dir / "inference.csv", inference)
    summary = {
        "completed_seed_blocks": len(block_dirs),
        "per_hand_rows": len(per_hand),
        "paired_rows": len(paired),
    }
    _atomic_json(output_dir / "AGGREGATE_STATUS.json", summary)
    return summary


@dataclass(frozen=True)
class FullSimulationRunConfig:
    output_dir: Path = Path("results/phase1/full_simulation")
    seeds: tuple[int, ...] = tuple(range(9400, 9460))
    horizon: int = 400
    formation_hands: int = 100
    equity_samples: int = 8
    mccfr_iterations: int = 20_000
    bootstrap_samples: int = 5_000
    permutation_samples: int = 20_000
    max_cells: int | None = None
    max_seed_blocks: int | None = None
    allow_dirty_worktree: bool = False


def run_full_simulation_matrix(config: FullSimulationRunConfig) -> pd.DataFrame:
    cells = phase1_simulation_matrix()
    selected_cells = cells[: config.max_cells] if config.max_cells is not None else cells
    source_provenance = _write_source_snapshot(
        config.output_dir, _code_provenance(config.allow_dirty_worktree)
    )
    plan_payload = {
        "protocol": "phase1-full-simulation-v1",
        "cells": cells,
        "seeds": config.seeds,
        "horizon": config.horizon,
        "formation_hands": config.formation_hands,
        "equity_samples": config.equity_samples,
        "mccfr_iterations": config.mccfr_iterations,
        "code_provenance": source_provenance,
    }
    plan_hash = _payload_hash(plan_payload)
    plan_path = config.output_dir / "FULL_SIMULATION_PLAN.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("plan_hash") != plan_hash:
            raise RuntimeError("full simulation plan differs from the existing resumable run")
    else:
        _atomic_json(plan_path, {**_jsonable(plan_payload), "plan_hash": plan_hash})

    statuses: list[dict[str, Any]] = []
    executed = 0
    for cell_index, overrides in enumerate(selected_cells):
        cell_dir = config.output_dir / f"cell_{cell_index:03d}"
        treatments = tuple(overrides["treatments"])
        completed_dirs: list[Path] = []
        for seed in config.seeds:
            block_payload = {
                "plan_hash": plan_hash,
                "cell": cell_index,
                "seed": seed,
                "overrides": overrides,
                "horizon": config.horizon,
                "formation_hands": config.formation_hands,
                "equity_samples": config.equity_samples,
            }
            block_hash = _payload_hash(block_payload)
            final = cell_dir / "seeds" / f"seed_{seed}"
            if _completion_valid(final, block_hash):
                completed_dirs.append(final)
                continue
            if config.max_seed_blocks is not None and executed >= config.max_seed_blocks:
                continue
            running, already_complete = _prepare_running(final, block_hash)
            if already_complete:
                completed_dirs.append(final)
                continue
            result = run_phase1_experiment(
                Phase1ExperimentConfig(
                    **overrides,
                    seeds=(seed,),
                    horizon=config.horizon,
                    formation_hands=config.formation_hands,
                    equity_samples=config.equity_samples,
                    mccfr_iterations=config.mccfr_iterations,
                    bootstrap_samples=max(20, min(config.bootstrap_samples, 200)),
                    permutation_samples=max(20, min(config.permutation_samples, 200)),
                    output_dir=running,
                    preregistered=True,
                )
            )
            if not bool(result["forks"]["identical"].all()):
                raise RuntimeError(f"fork gate failed for simulation cell {cell_index}, seed {seed}")
            _finish_running(
                running,
                final,
                {
                    "block_hash": block_hash,
                    "manifest_hash": result["manifest"]["manifest_hash"],
                    "seed": seed,
                    "valid": True,
                },
            )
            completed_dirs.append(final)
            executed += 1
            _atomic_json(
                config.output_dir / "RUN_STATUS.json",
                {
                    "plan_hash": plan_hash,
                    "last_completed_cell": cell_index,
                    "last_completed_seed": seed,
                    "completed_this_invocation": executed,
                },
            )
        if completed_dirs:
            aggregate = _aggregate_seed_blocks(
                completed_dirs,
                cell_dir / "aggregate",
                treatments,
                config.bootstrap_samples,
                config.permutation_samples,
            )
        else:
            aggregate = {"completed_seed_blocks": 0, "per_hand_rows": 0, "paired_rows": 0}
        statuses.append(
            {
                "cell": cell_index,
                "completed_seeds": len(completed_dirs),
                "target_seeds": len(config.seeds),
                "complete": len(completed_dirs) == len(config.seeds),
                **aggregate,
            }
        )
    frame = pd.DataFrame(statuses)
    _atomic_csv(config.output_dir / "FULL_SIMULATION_STATUS.csv", frame)
    return frame


@dataclass(frozen=True)
class LLMConfirmationRunConfig:
    output_dir: Path = Path("results/phase1/llm_confirmation")
    selected_depth: ReasoningTreatment = ReasoningTreatment.RECURSIVE_D2
    models: tuple[tuple[str, str], ...] = (
        ("opencode-go", "deepseek-v4-flash"),
        ("codex", "gpt-5.6-luna"),
    )
    seeds: tuple[int, ...] = tuple(range(9700, 9900))
    horizon: int = 20
    formation_hands: int = 5
    equity_samples: int = 8
    max_calls_per_model: int = 10_000
    offline_call_budget: int = 1_600
    preflight_retry_reserve: int = 400
    heads_up_contrast_calls: int = 8_000
    minimum_primary_calls_to_start_block: int = 20
    # Frozen all-or-nothing allowance for one seed across the three treatment
    # branches. A block exceeding this allowance fails closed.
    max_primary_calls_per_paired_block: int = 100
    max_blocks: int | None = None
    allow_dirty_worktree: bool = False


def _attempt_usage(path: Path) -> dict[str, int]:
    attempt = path / "ATTEMPT.json"
    if not attempt.exists():
        return {"calls": 0, "retries": 0, "primary_calls": 0}
    payload = json.loads(attempt.read_text(encoding="utf-8"))
    calls = int(payload.get("calls", 0))
    retries = int(payload.get("retries", 0))
    return {"calls": calls, "retries": retries, "primary_calls": calls - retries}


def _recover_running_attempts(model_dir: Path) -> None:
    running_paths = list(model_dir.glob("jobs/*/blocks/seed_*.running"))
    preflight_running = model_dir / "preflight.running"
    if preflight_running.exists():
        running_paths.append(preflight_running)
    for running in sorted(running_paths):
        ledger_path = running / "live_provider_ledger.json"
        ledger = {"calls": 0, "retries": 0}
        if ledger_path.exists():
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger.update(payload.get("ledger", {}))
        recovered = _archive_interrupted(running, "llm_block_interrupted")
        _atomic_json(
            recovered / "ATTEMPT.json",
            {
                "valid": False,
                "interrupted": True,
                "calls": int(ledger.get("calls", 0)),
                "retries": int(ledger.get("retries", 0)),
            },
        )


def _all_attempt_dirs(model_dir: Path) -> list[Path]:
    attempts = [
        path.parent
        for path in model_dir.glob("jobs/**/ATTEMPT.json")
        if ".running" not in path.parent.name
    ]
    preflight = model_dir / "preflight" / "ATTEMPT.json"
    if preflight.exists():
        attempts.append(preflight.parent)
    attempts.extend(path.parent for path in (model_dir / "interrupted").glob("*/ATTEMPT.json"))
    return attempts


def _job_attempt_dirs(job_dir: Path) -> list[Path]:
    completed = [path for path in (job_dir / "blocks").glob("seed_*") if path.is_dir()]
    interrupted = [
        path
        for path in (job_dir / "blocks" / "interrupted").glob("seed_*__*")
        if path.is_dir()
    ]
    return completed + interrupted


def _llm_block_config(
    run_config: LLMConfirmationRunConfig,
    job: ConfirmationJob,
    provider: str,
    model: str,
    seed: int,
    output_dir: Path,
    primary_remaining: int,
    retry_remaining: int,
    formation_protocol_hash: str,
) -> Phase1ExperimentConfig:
    return Phase1ExperimentConfig(
        arena=job.arena,
        treatments=job.treatments,
        opponent_type=job.opponent_type,
        opponent_composition=job.opponent_composition,
        stability=job.stability,
        epsilon=job.epsilon,
        seeds=(seed,),
        horizon=run_config.horizon,
        formation_hands=run_config.formation_hands,
        equity_samples=run_config.equity_samples,
        provider=provider,
        model=model,
        provider_budget=ProviderBudget(
            max_calls=primary_remaining + retry_remaining,
            max_primary_calls=primary_remaining,
            max_retries=retry_remaining,
        ),
        bootstrap_samples=20,
        permutation_samples=20,
        output_dir=output_dir,
        preregistered=True,
        formation_protocol_hash=formation_protocol_hash,
    )


def _block_primary_call_upper_bound(config: LLMConfirmationRunConfig, job: ConfirmationJob) -> int:
    # max_raises_per_street=2 means Hero can act at most three times on each of four streets.
    decisions_per_hero_hand = 12
    exploitation_hands = config.horizon - config.formation_hands
    return decisions_per_hero_hand * (
        config.formation_hands + exploitation_hands * len(job.treatments)
    )


def _write_cross_model_paired_blocks(
    config: LLMConfirmationRunConfig,
    confirmation_plan: Phase1LLMConfirmationPlan,
    plan_hash: str,
) -> pd.DataFrame:
    """Emit the fail-closed intersection used by paper-level inference.

    Individual serving systems can have valid blocks while the cross-model
    paper comparison is incomplete.  This file records exactly that distinction
    and keeps missing blocks visible instead of silently intersecting only the
    convenient seeds.
    """
    rows: list[dict[str, Any]] = []
    provider_ids = tuple(f"{provider}:{model}" for provider, model in config.models)
    regimes = tuple(job.stability.value for job in confirmation_plan.jobs)
    treatments = tuple(treatment.value for treatment in confirmation_plan.jobs[0].treatments)
    for (provider, model), provider_id in zip(config.models, provider_ids, strict=True):
        model_slug = f"{provider}__{model}".replace("/", "_").replace(".", "_")
        model_dir = config.output_dir / "models" / model_slug
        for job in confirmation_plan.jobs:
            for block in sorted((model_dir / "jobs" / job.name / "blocks").glob("seed_*")):
                attempt_path = block / "ATTEMPT.json"
                if not attempt_path.exists():
                    continue
                attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
                seed = int(attempt.get("seed", block.name.removeprefix("seed_")))
                checkpoint_id = attempt.get("checkpoint_id")
                if not isinstance(checkpoint_id, str):
                    checkpoint_id = canonical_checkpoint_id(plan_hash, seed)
                for treatment in job.treatments:
                    rows.append(
                        {
                            "seed": seed,
                            "provider": provider_id,
                            "treatment": treatment.value,
                            "regime": job.stability.value,
                            "checkpoint_id": checkpoint_id,
                            "valid": bool(attempt.get("valid")),
                        }
                    )
    raw = pd.DataFrame(
        rows,
        columns=["seed", "provider", "treatment", "regime", "checkpoint_id", "valid"],
    )
    if raw.empty:
        paired = pd.DataFrame(
            columns=[
                "seed",
                "checkpoint_id",
                "mirror_seat",
                "valid",
                "missing_or_duplicate_arms",
                "provider_gate_failure",
                "checkpoint_mismatch",
            ]
        )
    else:
        paired = valid_paired_block_intersection(
            raw,
            providers=provider_ids,
            treatments=treatments,
            regimes=regimes,
        )
    present = set(paired.get("seed", pd.Series(dtype=int)).astype(int))
    missing_rows = [
        {
            "seed": seed,
            "checkpoint_id": canonical_checkpoint_id(plan_hash, seed),
            "mirror_seat": seed % 2,
            "valid": False,
            "missing_or_duplicate_arms": True,
            "provider_gate_failure": False,
            "checkpoint_mismatch": False,
        }
        for seed in config.seeds
        if seed not in present
    ]
    if missing_rows:
        paired = pd.concat([paired, pd.DataFrame(missing_rows)], ignore_index=True)
    paired = paired.sort_values("seed").reset_index(drop=True)
    _atomic_csv(config.output_dir / "CROSS_MODEL_PAIRED_BLOCKS.csv", paired)
    _atomic_csv(config.output_dir / "CROSS_MODEL_PAIRED_BLOCK_ARMS.csv", raw)
    completion = validate_closed_loop_completion(
        raw,
        providers=provider_ids,
        treatments=treatments,
        regimes=regimes,
        target_seeds=config.seeds,
    )
    _atomic_json(
        config.output_dir / "CROSS_MODEL_PAIRED_BLOCK_STATUS.json",
        {
            "plan_hash": plan_hash,
            "providers": list(provider_ids),
            "treatments": list(treatments),
            "regimes": list(regimes),
            **completion,
        },
    )
    return paired


def _ensure_model_preflight(
    config: LLMConfirmationRunConfig,
    provider: str,
    model: str,
    model_dir: Path,
    plan_hash: str,
    reserve_remaining: int,
) -> dict[str, Any]:
    block_payload = {
        "plan_hash": plan_hash,
        "kind": "formal_model_preflight",
        "provider": provider,
        "model": model,
        "case_count": 4,
    }
    block_hash = _payload_hash(block_payload)
    final = model_dir / "preflight"
    if _completion_valid(final, block_hash):
        return json.loads((final / "ATTEMPT.json").read_text(encoding="utf-8"))
    if reserve_remaining <= 0:
        return {"valid": False, "calls": 0, "retries": 0, "reason": "no_preflight_reserve"}
    running, already_complete = _prepare_running(final, block_hash)
    if already_complete:
        return json.loads((final / "ATTEMPT.json").read_text(encoding="utf-8"))
    result = run_offline_benchmark(
        OfflineBenchmarkConfig(
            output_dir=running,
            provider=provider,
            model=model,
            case_count=4,
            provider_budget=ProviderBudget(
                max_calls=reserve_remaining,
                max_primary_calls=min(20, reserve_remaining),
                max_retries=reserve_remaining,
            ),
            preregistered=True,
        )
    )
    ledger = result["provider_gate"]["ledger"]
    attempt = {
        "block_hash": block_hash,
        "valid": bool(result["provider_gate"]["valid"]),
        "calls": int(ledger["calls"]),
        "retries": int(ledger["retries"]),
        "provider_gate": result["provider_gate"],
    }
    _atomic_json(running / "ATTEMPT.json", attempt)
    _finish_running(running, final, attempt)
    return attempt


def run_llm_confirmation_resumable(config: LLMConfirmationRunConfig) -> pd.DataFrame:
    confirmation_plan = build_llm_confirmation_plan(
        config.selected_depth,
        max_calls_per_model=config.max_calls_per_model,
        offline_call_budget=config.offline_call_budget,
        preflight_retry_reserve=config.preflight_retry_reserve,
        heads_up_contrast_calls=config.heads_up_contrast_calls,
    )
    for job in confirmation_plan.jobs:
        block_primary_upper_bound = _block_primary_call_upper_bound(config, job)
        if config.max_primary_calls_per_paired_block < block_primary_upper_bound:
            raise ValueError(
                "max_primary_calls_per_paired_block cannot complete every legal paired block: "
                f"cap={config.max_primary_calls_per_paired_block}, "
                f"required_upper_bound={block_primary_upper_bound}, job={job.name}"
            )
        required_job_primary_calls = block_primary_upper_bound * len(config.seeds)
        if job.call_budget < required_job_primary_calls:
            raise ValueError(
                "closed-loop job budget cannot cover every frozen paired seed: "
                f"job_budget={job.call_budget}, required_upper_bound="
                f"{required_job_primary_calls}, job={job.name}"
            )
    source_provenance = _write_source_snapshot(
        config.output_dir, _code_provenance(config.allow_dirty_worktree)
    )
    plan_payload = {
        "protocol": "phase1-llm-confirmation-resumable-v1",
        "selected_depth": config.selected_depth,
        "models": config.models,
        "jobs": tuple(asdict(job) for job in confirmation_plan.jobs),
        "seeds": config.seeds,
        "horizon": config.horizon,
        "formation_hands": config.formation_hands,
        "max_calls_per_model": confirmation_plan.max_calls_per_model,
        "offline_call_budget": confirmation_plan.offline_call_budget,
        "retry_reserve": confirmation_plan.preflight_retry_reserve,
        "code_provenance": source_provenance,
    }
    plan_hash = _payload_hash(plan_payload)
    plan_path = config.output_dir / "LLM_RESUMABLE_PLAN.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("plan_hash") != plan_hash:
            raise RuntimeError("LLM confirmation plan differs from the existing resumable run")
    else:
        _atomic_json(plan_path, {**_jsonable(plan_payload), "plan_hash": plan_hash})

    statuses: list[dict[str, Any]] = []
    executed = 0
    for provider, model in config.models:
        model_slug = f"{provider}__{model}".replace("/", "_").replace(".", "_")
        model_dir = config.output_dir / "models" / model_slug
        _recover_running_attempts(model_dir)
        existing_usage = [_attempt_usage(path) for path in _all_attempt_dirs(model_dir)]
        used_before_preflight = sum(item["calls"] for item in existing_usage)
        preflight = _ensure_model_preflight(
            config,
            provider,
            model,
            model_dir,
            plan_hash,
            max(0, confirmation_plan.preflight_retry_reserve - used_before_preflight),
        )
        if not preflight.get("valid"):
            statuses.append(
                {
                    "provider": provider,
                    "model": model,
                    "job": "formal_preflight",
                    "attempted_blocks": 1,
                    "valid_blocks": 0,
                    "job_primary_calls_used": 0,
                    "job_primary_calls_remaining": 0,
                    "model_total_calls_remaining": max(
                        0, confirmation_plan.max_calls_per_model - int(preflight.get("calls", 0))
                    ),
                    "retry_calls_remaining": 0,
                }
            )
            continue
        for job in confirmation_plan.jobs:
            job_dir = model_dir / "jobs" / job.name
            attempts = _job_attempt_dirs(job_dir)
            usage = [_attempt_usage(path) for path in attempts]
            job_primary_used = sum(item["primary_calls"] for item in usage)
            model_usage = [_attempt_usage(path) for path in _all_attempt_dirs(model_dir)]
            model_calls_used = sum(item["calls"] for item in model_usage)
            job_attempt_usage = [
                _attempt_usage(path.parent)
                for path in model_dir.glob("jobs/**/ATTEMPT.json")
            ]
            experimental_retries_used = sum(item["retries"] for item in job_attempt_usage)
            preflight_calls = _attempt_usage(model_dir / "preflight")["calls"]
            primary_remaining = max(0, job.call_budget - job_primary_used)
            retry_remaining = max(
                0,
                confirmation_plan.preflight_retry_reserve
                - preflight_calls
                - experimental_retries_used,
            )
            total_remaining = max(0, confirmation_plan.max_calls_per_model - model_calls_used)
            attempted_seeds = {
                int(
                    path.name.split("__", 1)[0]
                    .removeprefix("seed_")
                    .removesuffix(".running")
                )
                for path in attempts
            }
            block_primary_upper_bound = _block_primary_call_upper_bound(config, job)
            minimum_to_start = max(
                config.minimum_primary_calls_to_start_block,
                config.max_primary_calls_per_paired_block,
            )
            for seed in config.seeds:
                if seed in attempted_seeds:
                    continue
                if config.max_blocks is not None and executed >= config.max_blocks:
                    break
                if (
                    primary_remaining < minimum_to_start
                    or total_remaining < minimum_to_start
                ):
                    break
                block_payload = {
                    "plan_hash": plan_hash,
                    "provider": provider,
                    "model": model,
                    "job": asdict(job),
                    "seed": seed,
                    "horizon": config.horizon,
                    "formation_hands": config.formation_hands,
                }
                block_hash = _payload_hash(block_payload)
                final = job_dir / "blocks" / f"seed_{seed}"
                running, already_complete = _prepare_running(final, block_hash)
                if already_complete:
                    continue
                result = run_phase1_experiment(
                    # Keep retry reserve separate from the preregistered primary-call allocation.
                    _llm_block_config(
                        config,
                        job,
                        provider,
                        model,
                        seed,
                        running,
                        min(
                            config.max_primary_calls_per_paired_block,
                            primary_remaining,
                            total_remaining,
                        ),
                        min(
                            retry_remaining,
                            max(0, total_remaining - min(primary_remaining, total_remaining)),
                        ),
                        plan_hash,
                    )
                )
                ledger = result["provider_gate"]["ledger"]
                attempt = {
                    "block_hash": block_hash,
                    "seed": seed,
                    "valid": bool(result["provider_gate"]["valid"]),
                    "calls": int(ledger["calls"]),
                    "retries": int(ledger["retries"]),
                    "provider_gate": result["provider_gate"],
                    "checkpoint_id": str(result["forks"].iloc[0]["checkpoint_id"]),
                }
                _atomic_json(running / "ATTEMPT.json", attempt)
                _finish_running(running, final, attempt)
                executed += 1
                primary_spent = attempt["calls"] - attempt["retries"]
                primary_remaining = max(0, primary_remaining - primary_spent)
                retry_remaining = max(0, retry_remaining - attempt["retries"])
                total_remaining = max(0, total_remaining - attempt["calls"])
                _atomic_json(
                    model_dir / "MODEL_BUDGET_STATUS.json",
                    {
                        "plan_hash": plan_hash,
                        "calls_used": confirmation_plan.max_calls_per_model - total_remaining,
                        "calls_remaining": total_remaining,
                        "retries_remaining": retry_remaining,
                        "last_job": job.name,
                        "last_seed": seed,
                    },
                )
            valid_blocks = [
                path
                for path in (job_dir / "blocks").glob("seed_*")
                if path.is_dir()
                and json.loads((path / "ATTEMPT.json").read_text(encoding="utf-8")).get("valid")
            ]
            if valid_blocks:
                _aggregate_seed_blocks(
                    valid_blocks,
                    job_dir / "aggregate_valid_blocks",
                    job.treatments,
                    5_000,
                    20_000,
                )
            statuses.append(
                {
                    "provider": provider,
                    "model": model,
                    "job": job.name,
                    "attempted_blocks": len(_job_attempt_dirs(job_dir)),
                    "valid_blocks": len(valid_blocks),
                    "job_primary_calls_used": job.call_budget - primary_remaining,
                    "job_primary_calls_remaining": primary_remaining,
                    "model_total_calls_remaining": total_remaining,
                    "retry_calls_remaining": retry_remaining,
                    "block_primary_call_upper_bound": block_primary_upper_bound,
                    "block_primary_call_cap": config.max_primary_calls_per_paired_block,
                }
            )
    frame = pd.DataFrame(statuses)
    _atomic_csv(config.output_dir / "LLM_CONFIRMATION_STATUS.csv", frame)
    _write_cross_model_paired_blocks(config, confirmation_plan, plan_hash)
    return frame
