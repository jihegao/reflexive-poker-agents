from __future__ import annotations

import pandas as pd

from reflexive_poker.phase1_protocol import (
    canonical_checkpoint_id,
    mirror_assignment,
    valid_paired_block_intersection,
)


def _rows() -> pd.DataFrame:
    rows = []
    for seed in (10, 11):
        checkpoint = canonical_checkpoint_id("abc123" * 12, seed)
        for provider in ("deepseek", "codex"):
            for treatment in ("state_only", "recursive_d2"):
                for regime in ("fixed", "adaptive"):
                    rows.append(
                        {
                            "seed": seed,
                            "provider": provider,
                            "treatment": treatment,
                            "regime": regime,
                            "checkpoint_id": checkpoint,
                            "valid": True,
                        }
                    )
    return pd.DataFrame(rows)


def test_valid_paired_block_requires_every_arm_and_one_checkpoint() -> None:
    rows = _rows()
    rows.loc[
        (rows["seed"] == 11)
        & (rows["provider"] == "codex")
        & (rows["treatment"] == "recursive_d2"),
        "valid",
    ] = False
    summary = valid_paired_block_intersection(
        rows,
        providers=("deepseek", "codex"),
        treatments=("state_only", "recursive_d2"),
        regimes=("fixed", "adaptive"),
    )
    assert summary["valid"].tolist() == [True, False]
    assert summary.loc[0, "mirror_seat"] == mirror_assignment(10)


def test_paired_block_rejects_duplicate_or_missing_arms() -> None:
    rows = _rows().query("seed == 10").copy()
    duplicated = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    summary = valid_paired_block_intersection(
        duplicated,
        providers=("deepseek", "codex"),
        treatments=("state_only", "recursive_d2"),
        regimes=("fixed", "adaptive"),
    )
    assert not bool(summary.loc[0, "valid"])
    assert bool(summary.loc[0, "missing_or_duplicate_arms"])
