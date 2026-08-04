from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reflexive_poker.coalition_experiment import CoalitionConfig, run_coalition_experiment


def test_coalition_smoke_emits_all_eight_arms_and_public_only_audit(tmp_path: Path) -> None:
    result = run_coalition_experiment(
        CoalitionConfig(seeds=(9400,), hands=4, output_dir=tmp_path / "run")
    )
    summary = result["summary"]
    assert isinstance(summary, pd.DataFrame)
    assert set(summary["condition"]) == {f"t{t}r{r}s{s}" for t in (0, 1) for r in (0, 1) for s in (0, 1)}
    assert summary["mean_pair_chips_per_100"].nunique() > 1
    assert summary.loc[summary["condition"] == "t0r0s1", "mean_simulation_calls"].item() > 0
    assert summary.loc[summary["condition"] == "t0r0s0", "mean_simulation_calls"].item() == 0
    metadata = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert metadata["formal_conclusion_allowed"] is False
    assert metadata["information_boundary"]["private_cards_shared"] is False
    assert metadata["information_boundary"]["private_information_accesses"] == 0


def test_coalition_smoke_is_deterministic(tmp_path: Path) -> None:
    first = run_coalition_experiment(
        CoalitionConfig(seeds=(9401,), hands=4, output_dir=tmp_path / "first")
    )["per_seed"]
    second = run_coalition_experiment(
        CoalitionConfig(seeds=(9401,), hands=4, output_dir=tmp_path / "second")
    )["per_seed"]
    assert isinstance(first, pd.DataFrame)
    assert isinstance(second, pd.DataFrame)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
