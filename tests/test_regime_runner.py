from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from reflexive_poker import expctl
from reflexive_poker.expctl import ExpctlError, _load_config, _start
from reflexive_poker.regime_experiment import REGIME_CONDITIONS, RegimeExperimentRow
from reflexive_poker.regime_runner import (
    RegimeCheckpointError,
    RegimeRunConfig,
    regime_run_config_from_mapping,
    run_regime_experiment_resumable,
)
from reflexive_poker.regime_statistics import build_regime_statistics


def _row(condition: str, seed: int, mirror: int) -> RegimeExperimentRow:
    treatment_bonus = 5.0 if condition == "reflection_simulation" else 0.0
    recovery = 7 if condition == "reflection_simulation" else 10
    return RegimeExperimentRow(
        condition=condition,
        seed=seed,
        mirror=mirror,
        total_reward_bb=float(seed + mirror) + treatment_bonus,
        pre_switch_reward_bb=0.0,
        post_switch_reward_bb=float(seed + mirror) + treatment_bonus,
        post_switch_bb100=float(seed + mirror) + treatment_bonus,
        recovery_hands=recovery,
        detected_change_hand=12 if condition == "reflection_simulation" else None,
        detection_delay_hands=2 if condition == "reflection_simulation" else None,
        hypothesis_calls=1 if condition == "reflection_simulation" else 0,
        simulation_calls=1 if condition == "reflection_simulation" else 0,
        simulated_hands=12 if condition == "reflection_simulation" else 0,
        final_response_policy=(
            "balanced" if condition == "reflection_simulation" else None
        ),
        surprise_threshold=0.1 if condition == "reflection_simulation" else None,
        calibration_complete=(condition == "reflection_simulation"),
    )


def _run_config(tmp_path: Path, *, max_blocks: int | None = None) -> RegimeRunConfig:
    return RegimeRunConfig(
        run_id="test-regime-v1",
        output_dir=tmp_path / "run",
        seeds=(101,),
        hands=20,
        switch_hand=10,
        equity_samples=1,
        recovery_window=2,
        simulation_rollout_hands=2,
        simulation_equity_samples=1,
        formation_observations=2,
        calibration_observations=1,
        max_blocks=max_blocks,
    )


def test_resumable_runner_records_interruption_and_skips_completed_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from reflexive_poker import regime_runner

    calls: list[tuple[str, int, int]] = []
    interrupted = False

    def interrupt_once(config, condition: str, seed: int, mirror: int):
        nonlocal interrupted
        key = (condition, seed, mirror)
        calls.append(key)
        if key == ("baseline", 101, 1) and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return _row(condition, seed, mirror)

    monkeypatch.setattr(regime_runner, "run_regime_match", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        run_regime_experiment_resumable(_run_config(tmp_path))

    interrupted_status = json.loads(
        (tmp_path / "run" / "COMPLETION_STATUS.json").read_text(encoding="utf-8")
    )
    assert interrupted_status["state"] == "interrupted"
    assert interrupted_status["completed_blocks"] == 1
    assert len(interrupted_status["failure_records"]) == 1

    completed = run_regime_experiment_resumable(_run_config(tmp_path))

    assert completed["state"] == "completed"
    assert completed["formal_completion_valid"] is True
    assert calls.count(("baseline", 101, 0)) == 1
    assert (tmp_path / "run" / "COMPLETED.json").exists()


def test_corrupt_checkpoint_fails_closed_without_rerunning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from reflexive_poker import regime_runner

    calls: list[tuple[str, int, int]] = []

    def fake_match(config, condition: str, seed: int, mirror: int):
        calls.append((condition, seed, mirror))
        return _row(condition, seed, mirror)

    monkeypatch.setattr(regime_runner, "run_regime_match", fake_match)
    partial = run_regime_experiment_resumable(_run_config(tmp_path, max_blocks=1))
    assert partial["state"] == "incomplete"
    checkpoint = next((tmp_path / "run" / "checkpoints").rglob("*.json"))
    checkpoint.write_text("{damaged", encoding="utf-8")
    calls_before_resume = list(calls)

    with pytest.raises(RegimeCheckpointError, match="corrupt"):
        run_regime_experiment_resumable(_run_config(tmp_path))

    assert calls == calls_before_resume
    gate = json.loads((tmp_path / "run" / "EVIDENCE_GATE.json").read_text())
    assert gate["valid"] is False
    assert "run_incomplete" in gate["reasons"]


def test_statistics_use_only_complete_paired_blocks() -> None:
    rows = [
        _row(condition, seed, mirror)
        for seed in (1, 2)
        for mirror in (0, 1)
        for condition in REGIME_CONDITIONS
    ]
    completion = {
        "state": "completed",
        "total_blocks": 12,
        "completed_blocks": 12,
        "corrupt_checkpoints": [],
    }
    complete = build_regime_statistics(
        rows,
        expected_seeds=(1, 2),
        completion_status=completion,
    )
    assert complete["paired_summary"]["valid_paired_blocks"] == 4
    assert complete["paired_summary"]["formal_conclusion_allowed"] is True
    assert complete["paired_summary"]["post_switch_bb100_delta"]["mean"] == 5.0
    assert complete["paired_summary"]["recovery_hands_delta"]["mean"] == -3.0

    incomplete = build_regime_statistics(
        rows[:-1],
        expected_seeds=(1, 2),
        completion_status={**completion, "state": "incomplete", "completed_blocks": 11},
    )
    assert incomplete["paired_summary"]["valid_paired_blocks"] == 3
    assert len(incomplete["paired_effects"]) == 3
    assert incomplete["evidence_gate"]["valid"] is False


def test_regime_expctl_request_id_is_idempotent_and_conflicts_on_new_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(expctl, "_run_experiment", lambda metadata, run_dir: None)
    args = argparse.Namespace(
        root=str(tmp_path / "registry"),
        config="configs/regime_pilot.yaml",
        experiment="regime-adaptation",
        tag="test",
        request_id="regime-request-1",
        provider=None,
        model=None,
        max_blocks=1,
        allow_dirty_worktree=True,
        foreground=True,
    )
    first = _start(args)
    second = _start(args)
    assert first["run_id"] == second["run_id"]
    frozen = Path(first["frozen_config_path"])
    assert frozen.exists()
    assert first["frozen_config_sha256"]

    changed = yaml.safe_load(Path("configs/regime_pilot.yaml").read_text(encoding="utf-8"))
    changed["regime_adaptation"]["hands"] = 321
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    args.config = str(changed_path)
    with pytest.raises(ExpctlError) as exc_info:
        _start(args)
    assert exc_info.value.code == "IDEMPOTENCY_CONFLICT"


def test_regime_expctl_rejects_partial_runner_as_formal_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen_config = tmp_path / "CONFIG.yaml"
    frozen_config.write_bytes(Path("configs/regime_pilot.yaml").read_bytes())
    monkeypatch.setattr(
        expctl,
        "_freeze_run_sources",
        lambda metadata, artifacts: {
            "source_fingerprint": "source-hash",
            "source_snapshot_sha256": "snapshot-hash",
        },
    )
    monkeypatch.setattr(
        expctl,
        "run_regime_experiment_resumable",
        lambda config: {
            "state": "incomplete",
            "completed_blocks": 1,
            "total_blocks": 180,
            "formal_completion_valid": False,
        },
    )
    metadata = {
        "run_id": "registry-run-id",
        "experiment": "regime-adaptation",
        "config_path": str(frozen_config),
        "frozen_config_path": str(frozen_config),
        "allow_dirty_worktree": True,
        "max_blocks": 1,
    }

    with pytest.raises(ExpctlError) as exc_info:
        expctl._run_experiment(metadata, tmp_path / "registry-run-id")

    assert exc_info.value.code == "FORMAL_COMPLETION_INVALID"
    assert exc_info.value.retryable is True


def test_formal_pilot_config_freezes_thirty_seeds() -> None:
    payload = _load_config(Path("configs/regime_pilot.yaml").resolve())
    config = regime_run_config_from_mapping(payload, output_dir=Path("unused"))
    assert len(config.seeds) == 30
    assert config.seeds[0] == 9300
    assert config.seeds[-1] == 9329
    assert len(config.conditions) * len(config.seeds) * len(config.mirrors) == 180
