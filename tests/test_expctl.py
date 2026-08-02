from __future__ import annotations

import argparse
import json
from pathlib import Path

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
