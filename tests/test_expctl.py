from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from reflexive_poker import expctl
from reflexive_poker.expctl import _load_config, _start


def test_expctl_validates_phase1_config() -> None:
    config = _load_config(Path("configs/phase1.yaml").resolve())
    assert config["paper_phase1"]["case_count"] == 200
    assert len(config["paper_phase1"]["models"]) == 2


def test_expctl_foreground_run_is_idempotent(tmp_path: Path) -> None:
    args = argparse.Namespace(
        root=str(tmp_path / "registry"),
        config="configs/phase1.yaml",
        experiment="offline-baselines",
        tag="test",
        request_id="request-1",
        provider=None,
        model=None,
        max_blocks=None,
        allow_dirty_worktree=True,
        foreground=True,
    )
    first = _start(args)
    second = _start(args)
    assert first["run_id"] == second["run_id"]
    assert first["state"] == "completed"
    run_dir = Path(first["artifact_dir"]).parent
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "run_completed"
    assert (Path(first["artifact_dir"]) / "offline_baselines" / "cases.jsonl.gz").exists()


def test_provider_preflight_can_target_one_frozen_serving_system(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_offline_benchmark(config):
        calls.append(config)
        return {"provider_gate": {"valid": True}}

    monkeypatch.setattr(expctl, "run_offline_benchmark", fake_offline_benchmark)
    args = argparse.Namespace(
        root=str(tmp_path / "registry"),
        config="configs/phase1.yaml",
        experiment="provider-preflight",
        tag="single-provider-preflight",
        request_id="single-provider-preflight-1",
        provider="codex",
        model="gpt-5.6-luna",
        max_blocks=None,
        allow_dirty_worktree=True,
        foreground=True,
    )

    result = _start(args)

    assert result["state"] == "completed"
    assert [(call.provider, call.model) for call in calls] == [
        ("baselines", "none"),
        ("codex", "gpt-5.6-luna"),
    ]


def test_phase2_preflight_runs_only_the_frozen_four_systems(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_offline_benchmark(config):
        calls.append(config)
        return {"provider_gate": {"valid": True}}

    monkeypatch.setattr(expctl, "run_offline_benchmark", fake_offline_benchmark)
    phase2 = yaml.safe_load(Path("configs/phase2.yaml").read_text(encoding="utf-8"))
    phase2["evidence_layers"]["llm_confirmation"]["models"] = [
        {"provider": "mock", "model": "phase1-fallback-must-not-run"}
    ]
    config_path = tmp_path / "phase2.yaml"
    config_path.write_text(yaml.safe_dump(phase2), encoding="utf-8")
    args = argparse.Namespace(
        root=str(tmp_path / "registry"),
        config=str(config_path),
        experiment="paper-phase2-preflight",
        tag="test-phase2-preflight",
        request_id="phase2-preflight-request-1",
        provider=None,
        model=None,
        max_blocks=None,
        allow_dirty_worktree=True,
        foreground=True,
    )

    result = _start(args)

    assert result["state"] == "completed"
    assert [(call.provider, call.model) for call in calls] == [
        ("baselines", "none"),
        ("opencode-go", "deepseek-v4-flash"),
        ("opencode-go", "qwen3.7-max"),
        ("opencode-go", "glm-5.2"),
        ("codex", "gpt-5.6-luna"),
    ]
    assert all(call.case_count == 4 for call in calls)
    assert all(len(call.treatments) == 5 for call in calls)
    assert all("preflight" in str(call.output_dir) for call in calls)
    run_dir = Path(result["artifact_dir"]).parent
    phase_events = [
        json.loads(line)["phase"]
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if json.loads(line)["event"] in {"phase_started", "phase_completed"}
    ]
    assert phase_events == ["phase2_provider_preflight", "phase2_provider_preflight"]


def test_phase2_readiness_cli_writes_fail_closed_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    expctl.main(
        [
            "phase2-readiness",
            "--config",
            "configs/phase2.yaml",
            "--preflight-dir",
            str(tmp_path / "missing-preflight"),
            "--output-dir",
            str(tmp_path / "readiness"),
            "--output",
            "json",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert not result["ready_for_formal_outcomes"]
    assert (tmp_path / "readiness" / "PHASE2_READINESS.json").exists()
