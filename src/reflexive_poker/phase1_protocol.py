"""Frozen pairing rules shared by the Phase 1 and Phase 2 runners.

These helpers deliberately contain no provider or simulator calls.  They make
the unit of analysis explicit: a seed is valid only when every predeclared arm
for that seed and its seat mirror completed its provider gate.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def mirror_assignment(seed: int) -> int:
    """Deterministically alternate the focal serving system's physical seat."""
    return seed % 2


def canonical_checkpoint_id(protocol_hash: str, seed: int) -> str:
    """Stable ID used to require a single provider-independent formation fork."""
    return f"{protocol_hash[:16]}-seed-{seed:05d}-mirror-{mirror_assignment(seed)}"


def valid_paired_block_intersection(
    rows: pd.DataFrame,
    *,
    providers: Iterable[str],
    treatments: Iterable[str],
    regimes: Iterable[str],
) -> pd.DataFrame:
    """Select only seeds with complete, valid, same-checkpoint arms.

    Expected rows must have ``seed``, ``provider``, ``treatment``, ``regime``,
    ``checkpoint_id`` and ``valid`` columns. Duplicate arm rows are rejected
    rather than silently deduplicated.
    """
    required = {"seed", "provider", "treatment", "regime", "checkpoint_id", "valid"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"paired block rows are missing columns: {missing}")
    expected = {
        (provider, treatment, regime)
        for provider in providers
        for treatment in treatments
        for regime in regimes
    }
    accepted: list[dict[str, object]] = []
    for seed, group in rows.groupby("seed", sort=True):
        observed = list(zip(group["provider"], group["treatment"], group["regime"], strict=True))
        checkpoint_ids = set(group["checkpoint_id"])
        complete = set(observed) == expected and len(observed) == len(expected)
        valid = bool(group["valid"].all())
        same_checkpoint = len(checkpoint_ids) == 1
        accepted.append(
            {
                "seed": int(seed),
                "checkpoint_id": next(iter(checkpoint_ids)) if same_checkpoint else None,
                "mirror_seat": mirror_assignment(int(seed)),
                "valid": complete and valid and same_checkpoint,
                "missing_or_duplicate_arms": not complete,
                "provider_gate_failure": not valid,
                "checkpoint_mismatch": not same_checkpoint,
            }
        )
    return pd.DataFrame(accepted)


def validate_closed_loop_completion(
    rows: pd.DataFrame,
    *,
    providers: Iterable[str],
    treatments: Iterable[str],
    regimes: Iterable[str],
    target_seeds: Iterable[int],
) -> dict[str, object]:
    """Fail closed before a closed-loop run can be called formally complete.

    A resumable worker may have useful partial artifacts, but it is not a
    completed paper outcome until every requested seed has every provider,
    treatment and regime exactly once, has a single shared formation checkpoint,
    and every arm passes its provider gate.
    """
    required = {"seed", "provider", "treatment", "regime", "checkpoint_id", "valid"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"closed-loop rows are missing columns: {missing}")
    seed_values = tuple(int(seed) for seed in target_seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("target_seeds must be non-empty and unique")
    expected = {
        (provider, treatment, regime)
        for provider in providers
        for treatment in treatments
        for regime in regimes
    }
    if not expected:
        raise ValueError("providers, treatments, and regimes must be non-empty")
    statuses: list[dict[str, object]] = []
    for seed in seed_values:
        group = rows.loc[rows["seed"] == seed]
        observed = list(zip(group["provider"], group["treatment"], group["regime"], strict=True))
        checkpoints = set(group["checkpoint_id"])
        complete_arms = set(observed) == expected and len(observed) == len(expected)
        provider_valid = bool(group["valid"].all()) and not group.empty
        checkpoint_valid = len(checkpoints) == 1
        statuses.append(
            {
                "seed": seed,
                "valid": complete_arms and provider_valid and checkpoint_valid,
                "missing_or_duplicate_arms": not complete_arms,
                "provider_gate_failure": not provider_valid,
                "checkpoint_mismatch": not checkpoint_valid,
            }
        )
    unexpected_seeds = sorted(set(rows["seed"].astype(int)) - set(seed_values))
    valid_blocks = sum(bool(status["valid"]) for status in statuses)
    complete = valid_blocks == len(seed_values) and not unexpected_seeds
    return {
        "target_seeds": len(seed_values),
        "valid_paired_blocks": valid_blocks,
        "formal_completion_valid": complete,
        "claim_status": "formal_closed_loop_complete" if complete else "incomplete_no_paper_outcome_claim",
        "unexpected_seeds": unexpected_seeds,
        "seed_statuses": statuses,
    }
