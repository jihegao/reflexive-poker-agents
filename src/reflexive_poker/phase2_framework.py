"""Fail-closed execution and analysis primitives for the Phase 2 extension.

This module deliberately contains no provider fallback and no mock-to-live
conversion.  A Phase 2 outcome run is admitted only after the independent
four-system readiness audit has passed.  The six-max engine is an injected
runner interface because its treatment-specific prompt contract must be
frozen after the Phase 1 outcome lock; this prevents the legacy boolean
``reflexive_enabled`` pilot from being relabelled as D0--D3 evidence.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from .phase1_models import ProviderBudget, ReasoningTreatment
from .phase1_offline import OfflineBenchmarkConfig, run_offline_benchmark
from .phase1_offline_evidence import holm_adjust
from .phase1_statistics import (
    large_pot_sensitivity,
    paired_bootstrap_interval,
    paired_sign_permutation_p,
)

PHASE2_TREATMENTS = (
    ReasoningTreatment.STATE_ONLY.value,
    ReasoningTreatment.ACTION_PREDICTION.value,
    ReasoningTreatment.BUDGET_MATCHED_D1.value,
    ReasoningTreatment.RECURSIVE_D2.value,
    ReasoningTreatment.RECURSIVE_D3.value,
)
PHASE2_SYSTEM_COUNT = 4


class Phase2ExecutionError(RuntimeError):
    """A formal Phase 2 invariant was not satisfied; output is not evidence."""


@dataclass(frozen=True)
class ServingSystem:
    serving_system: str
    provider: str
    model: str


@dataclass(frozen=True)
class Phase2OfflineRunConfig:
    phase2: Mapping[str, Any]
    readiness: Mapping[str, Any]
    output_dir: Path
    case_count: int = 200
    base_seed: int = 20260802
    max_calls_per_system: int = 1_200
    max_retries_per_system: int = 200


@dataclass(frozen=True)
class Phase2PowerAnalysisConfig:
    """Pre-outcome paired-return power calculation bound to Phase 1 evidence.

    The caller provides an effect-size and standard-deviation estimate extracted
    from the immutable Phase 1 bundle.  This function does not inspect, fit, or
    alter Phase 2 outcome rows, so its manifest can be frozen before any Phase
    2 provider call.
    """

    phase1_outcome_lock: str
    paired_seed_count: int
    heads_up_hands: int
    expected_return_delta: float
    paired_seed_stddev: float
    alpha: float = 0.05
    target_power: float = 0.80


@dataclass(frozen=True)
class Phase2SixMaxRunConfig:
    """Frozen schedule passed to a treatment-aware six-max arm runner.

    ``arm_runner`` is intentionally not supplied here.  The caller must bind a
    runner that can prove all five prompt/treatment contracts; the old pilot's
    two-state switch cannot satisfy this contract.
    """

    systems: tuple[ServingSystem, ...]
    treatments: tuple[str, ...]
    seeds: tuple[int, ...]
    hands: int
    formation_hands: int


def systems_from_phase2(phase2: Mapping[str, Any]) -> tuple[ServingSystem, ...]:
    raw = phase2.get("serving_systems")
    if not isinstance(raw, list) or len(raw) != PHASE2_SYSTEM_COUNT:
        raise Phase2ExecutionError("Phase 2 requires exactly four serving systems")
    systems: list[ServingSystem] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise Phase2ExecutionError("Phase 2 serving system is not an object")
        provider, model = str(item.get("provider", "")), str(item.get("model", ""))
        name = str(item.get("serving_system", f"{provider}/{model}"))
        if not provider or not model or provider in {"mock", "baselines"}:
            raise Phase2ExecutionError("Phase 2 outcome systems must be named real providers")
        if (provider, model) in seen:
            raise Phase2ExecutionError("Phase 2 provider/model pairs must be unique")
        seen.add((provider, model))
        systems.append(ServingSystem(name, provider, model))
    return tuple(systems)


def assert_phase2_outcomes_ready(
    phase2: Mapping[str, Any], readiness: Mapping[str, Any]
) -> tuple[ServingSystem, ...]:
    """Validate the *current* admission result, not merely a previous preflight."""
    systems = systems_from_phase2(phase2)
    if readiness.get("protocol") != "prbench-cross-model-v1":
        raise Phase2ExecutionError("Phase 2 readiness protocol does not match the preregistration")
    if readiness.get("ready_for_formal_outcomes") is not True:
        raise Phase2ExecutionError("Phase 2 outcomes are blocked until readiness is fully frozen")
    preflight = readiness.get("preflight")
    locks = readiness.get("identity_locks")
    if not isinstance(preflight, Mapping) or not isinstance(locks, Mapping):
        raise Phase2ExecutionError("Phase 2 readiness lacks provider preflight or identity locks")
    for system in systems:
        gate = preflight.get(system.serving_system)
        lock = locks.get(system.serving_system)
        if not isinstance(gate, Mapping) or not isinstance(lock, Mapping):
            raise Phase2ExecutionError(f"readiness missing {system.serving_system}")
        if not (gate.get("valid") and gate.get("expected_predictions") == 20 and gate.get("observed_predictions") == 20):
            raise Phase2ExecutionError(f"provider preflight is incomplete: {system.serving_system}")
        if not (lock.get("status") == "frozen" and lock.get("matches_preflight") is True):
            raise Phase2ExecutionError(f"provider identity is not frozen: {system.serving_system}")
    return systems


def _slug(system: ServingSystem) -> str:
    return f"{system.provider}__{system.model}".replace("/", "_").replace(".", "_")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_phase2_power_analysis(
    config: Phase2PowerAnalysisConfig,
    *,
    protocol: str = "prbench-cross-model-v1",
) -> dict[str, Any]:
    """Create a transparent normal-approximation paired-block power manifest.

    It is deliberately an *assumption manifest*, not a result: `valid` only
    means the locked design reaches the stated target power under its supplied
    Phase-1-derived effect-size assumptions.
    """

    if not config.phase1_outcome_lock:
        raise Phase2ExecutionError("power analysis requires a Phase 1 outcome lock")
    if config.paired_seed_count <= 40 or config.heads_up_hands <= 20:
        raise Phase2ExecutionError("power analysis requires the larger Phase 2 design")
    if not 0.0 < config.alpha < 1.0 or not 0.0 < config.target_power < 1.0:
        raise Phase2ExecutionError("alpha and target power must lie strictly between zero and one")
    if not np.isfinite(config.expected_return_delta) or not np.isfinite(config.paired_seed_stddev):
        raise Phase2ExecutionError("power inputs must be finite")
    if config.paired_seed_stddev <= 0.0:
        raise Phase2ExecutionError("paired-seed standard deviation must be positive")

    normal = NormalDist()
    z_critical = normal.inv_cdf(1.0 - config.alpha / 2.0)
    noncentrality = abs(config.expected_return_delta) * np.sqrt(config.paired_seed_count) / config.paired_seed_stddev
    achieved_power = normal.cdf(-z_critical - noncentrality) + (1.0 - normal.cdf(z_critical - noncentrality))
    return {
        "schema_version": 1,
        "valid": bool(achieved_power >= config.target_power),
        "protocol": protocol,
        "analysis_unit": "paired_seed_block",
        "paired_seed_count": config.paired_seed_count,
        "heads_up_hands": config.heads_up_hands,
        "phase1_outcome_lock": config.phase1_outcome_lock,
        "method": "two_sided_normal_approximation_for_paired_seed_mean",
        "alpha": config.alpha,
        "target_power": config.target_power,
        "achieved_power": float(achieved_power),
        "expected_return_delta": config.expected_return_delta,
        "paired_seed_stddev": config.paired_seed_stddev,
        "standardized_effect": config.expected_return_delta / config.paired_seed_stddev,
        "assumption_source": "locked_phase1_evidence_bundle",
    }


def write_phase2_power_analysis(
    path: Path,
    config: Phase2PowerAnalysisConfig,
    *,
    protocol: str = "prbench-cross-model-v1",
) -> dict[str, Any]:
    """Persist the pre-outcome power manifest atomically for readiness input."""

    payload = build_phase2_power_analysis(config, protocol=protocol)
    _atomic_json(path, payload)
    return payload


def _validate_offline_result(
    result: Mapping[str, Any], system: ServingSystem, case_count: int
) -> pd.DataFrame:
    gate = result.get("provider_gate")
    scores = result.get("scores_per_case")
    expected = case_count * len(PHASE2_TREATMENTS)
    if not isinstance(gate, Mapping) or gate.get("valid") is not True:
        raise Phase2ExecutionError(f"provider gate failed: {system.serving_system}")
    required_gate = {
        "expected_predictions": expected,
        "observed_predictions": expected,
        "zero_unresolved_failures": True,
        "actual_identity_matches": True,
        "model_identity_source_valid": True,
        "complete_token_accounting": True,
        "complete_model_version_attestation": True,
        "budget_match_valid": True,
    }
    if any(gate.get(key) != value for key, value in required_gate.items()):
        raise Phase2ExecutionError(f"incomplete provider evidence: {system.serving_system}")
    ledger = gate.get("ledger", {})
    if not isinstance(ledger, Mapping) or int(ledger.get("fallbacks", 0)) != 0:
        raise Phase2ExecutionError(f"fallback makes the block invalid: {system.serving_system}")
    if not isinstance(scores, pd.DataFrame):
        raise Phase2ExecutionError(f"missing per-case scores: {system.serving_system}")
    required = {"case_id", "trajectory_id", "provider", "model", "treatment", "u_table_total", "type_brier", "action_brier", "decision_regret", "total_tokens", "latency_ms", "cost_usd"}
    missing = required.difference(scores.columns)
    if missing:
        raise Phase2ExecutionError(f"scores missing required columns: {sorted(missing)}")
    real = scores.loc[(scores["provider"] == system.provider) & (scores["model"] == system.model)].copy()
    if len(real) != expected or set(real["treatment"]) != set(PHASE2_TREATMENTS):
        raise Phase2ExecutionError(f"unbalanced treatment coverage: {system.serving_system}")
    if real.duplicated(["case_id", "treatment"]).any():
        raise Phase2ExecutionError(f"duplicate case/treatment row: {system.serving_system}")
    return real


def _cost_per_100(scores: pd.DataFrame) -> tuple[float, str]:
    costs = pd.to_numeric(scores["cost_usd"], errors="coerce")
    if costs.notna().all():
        return float(costs.sum() * 100.0 / len(scores)), "observed"
    return float("nan"), "unavailable"


def summarize_phase2_offline(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarise four-system offline evidence without inventing return results."""
    required = {"serving_system", "provider", "model", "u_table_total", "type_brier", "action_brier", "decision_regret", "total_tokens", "latency_ms", "cost_usd"}
    missing = required.difference(scores.columns)
    if missing:
        raise Phase2ExecutionError(f"offline scores missing required columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for name, group in scores.groupby("serving_system", sort=False):
        if len(group) == 0:
            continue
        cost, cost_status = _cost_per_100(group)
        tokens = pd.to_numeric(group["total_tokens"], errors="coerce")
        latency = pd.to_numeric(group["latency_ms"], errors="coerce")
        opponent_u = 1.0 - float((group["type_brier"].mean() + group["action_brier"].mean()) / 4.0)
        rows.append(
            {
                "serving_system": name,
                "provider": group.iloc[0]["provider"],
                "model": group.iloc[0]["model"],
                "u_table": float(group["u_table_total"].mean()),
                "u_opponent": opponent_u,
                "decision_regret": float(group["decision_regret"].mean()),
                "tokens_per_valid_decision": float(tokens.mean()) if tokens.notna().all() else float("nan"),
                "latency_p95_ms": float(latency.quantile(0.95)) if latency.notna().all() else float("nan"),
                "usd_per_100_valid_decisions": cost,
                "cost_observability": cost_status,
                "return_available": False,
                "pareto_eligible": False,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != PHASE2_SYSTEM_COUNT:
        raise Phase2ExecutionError("offline summary does not contain all four serving systems")
    return result


def pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    """Mark nondominated systems only when all required outcomes are observed.

    The three preregistered dimensions are opponent understanding, return, and
    cost.  Missing or unavailable cost/return excludes a row rather than
    treating it as zero or assigning it a rank.
    """
    required = {"serving_system", "u_opponent", "return_chips_per_100", "usd_per_100_valid_decisions", "return_available"}
    missing = required.difference(summary.columns)
    if missing:
        raise Phase2ExecutionError(f"Pareto summary missing columns: {sorted(missing)}")
    result = summary.copy()
    result["pareto_eligible"] = (
        result["return_available"].astype(bool)
        & pd.to_numeric(result["u_opponent"], errors="coerce").notna()
        & pd.to_numeric(result["return_chips_per_100"], errors="coerce").notna()
        & pd.to_numeric(result["usd_per_100_valid_decisions"], errors="coerce").notna()
    )
    result["pareto_nondominated"] = False
    eligible = result.index[result["pareto_eligible"]].tolist()
    for index in eligible:
        row = result.loc[index]
        dominated = False
        for other_index in eligible:
            if index == other_index:
                continue
            other = result.loc[other_index]
            weak = (
                other["u_opponent"] >= row["u_opponent"]
                and other["return_chips_per_100"] >= row["return_chips_per_100"]
                and other["usd_per_100_valid_decisions"] <= row["usd_per_100_valid_decisions"]
            )
            strict = (
                other["u_opponent"] > row["u_opponent"]
                or other["return_chips_per_100"] > row["return_chips_per_100"]
                or other["usd_per_100_valid_decisions"] < row["usd_per_100_valid_decisions"]
            )
            if weak and strict:
                dominated = True
                break
        result.loc[index, "pareto_nondominated"] = not dominated
    return result


def run_phase2_offline(
    config: Phase2OfflineRunConfig,
    *,
    offline_runner: Callable[[OfflineBenchmarkConfig], Mapping[str, Any]] = run_offline_benchmark,
) -> dict[str, pd.DataFrame]:
    """Run exactly four real systems across all five treatments after readiness.

    This performs real calls only when invoked by the caller.  Tests inject a
    runner; no fallback/mock path is accepted by this function.
    """
    systems = assert_phase2_outcomes_ready(config.phase2, config.readiness)
    if not 1 <= config.case_count <= 200:
        raise Phase2ExecutionError("Phase 2 case_count must be in [1, 200]")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    all_scores: list[pd.DataFrame] = []
    gate_rows: list[dict[str, Any]] = []
    for system in systems:
        result = offline_runner(
            OfflineBenchmarkConfig(
                output_dir=config.output_dir / "offline_understanding" / _slug(system),
                provider=system.provider,
                model=system.model,
                case_count=config.case_count,
                base_seed=config.base_seed,
                treatments=tuple(ReasoningTreatment(value) for value in PHASE2_TREATMENTS),
                provider_budget=ProviderBudget(
                    max_calls=config.max_calls_per_system,
                    max_primary_calls=config.case_count * len(PHASE2_TREATMENTS),
                    max_retries=config.max_retries_per_system,
                ),
                preregistered=True,
            )
        )
        scores = _validate_offline_result(result, system, config.case_count)
        scores.insert(0, "serving_system", system.serving_system)
        all_scores.append(scores)
        gate = dict(result["provider_gate"])
        gate["serving_system"] = system.serving_system
        gate_rows.append(gate)
    combined = pd.concat(all_scores, ignore_index=True)
    summary = summarize_phase2_offline(combined)
    # Offline evidence has no long-horizon return, so its Pareto view is
    # intentionally ineligible until the closed-loop/Six-max join is present.
    offline_pareto = pareto_frontier(
        summary.assign(return_chips_per_100=np.nan, return_available=False)
    )
    target = config.output_dir / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target / "offline_scores_per_case.csv", index=False)
    summary.to_csv(target / "offline_summary.csv", index=False)
    offline_pareto.to_csv(target / "pareto_frontier.pending_return.csv", index=False)
    _atomic_json(
        target / "OFFLINE_OUTCOME_GATE.json",
        {
            "phase": 2,
            "evidence_class": "preregistered_live_outcome",
            "all_systems_valid": True,
            "systems": gate_rows,
            "pareto_status": "pending_closed_loop_return",
        },
    )
    return {"scores_per_case": combined, "summary": summary, "pareto": offline_pareto}


def build_phase2_six_max_schedule(config: Phase2SixMaxRunConfig) -> pd.DataFrame:
    """Create the complete pre-outcome arm grid, including true seat mirrors."""
    if len(config.systems) != PHASE2_SYSTEM_COUNT:
        raise Phase2ExecutionError("six-max schedule requires all four serving systems")
    if set(config.treatments) != set(PHASE2_TREATMENTS):
        raise Phase2ExecutionError("six-max schedule requires all five frozen treatments")
    if len(config.seeds) <= 40 or len(set(config.seeds)) != len(config.seeds):
        raise Phase2ExecutionError("six-max requires more than 40 unique paired seeds")
    if config.hands <= 20 or config.formation_hands < 1:
        raise Phase2ExecutionError("six-max horizon/formation must be frozen above Phase 1 minimum")
    rows = [
        {
            "serving_system": system.serving_system,
            "provider": system.provider,
            "model": system.model,
            "treatment": treatment,
            "seed": seed,
            "seat_mirror": mirror,
            "button_rotation": "required",
            "hands": config.hands,
            "formation_hands": config.formation_hands,
        }
        for system in config.systems
        for treatment in config.treatments
        for seed in config.seeds
        for mirror in (0, 1)
    ]
    return pd.DataFrame(rows)


def analyze_phase2_six_max(
    per_hand: pd.DataFrame,
    *,
    contrasts: Sequence[tuple[str, str]] | None = None,
    bootstrap_samples: int = 5_000,
    permutation_samples: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired seed/mirror return analysis with large-pot robustness and Holm.

    The input must contain every paired hand for every arm.  This is deliberately
    an analyser, not a way to backfill missing calls or label an old mock pilot
    as evidence.
    """
    required = {"serving_system", "provider", "model", "treatment", "seed", "seat_mirror", "hand_index", "reward", "provider_gate_valid", "fallback_count", "unresolved_failures", "formation_fork_hash"}
    missing = required.difference(per_hand.columns)
    if missing:
        raise Phase2ExecutionError(f"six-max rows missing required columns: {sorted(missing)}")
    if per_hand.empty:
        raise Phase2ExecutionError("six-max rows are empty")
    if not per_hand["provider_gate_valid"].astype(bool).all() or (pd.to_numeric(per_hand["fallback_count"], errors="raise") != 0).any() or (pd.to_numeric(per_hand["unresolved_failures"], errors="raise") != 0).any():
        raise Phase2ExecutionError("invalid provider block cannot enter six-max analysis")
    keys = ["serving_system", "treatment", "seed", "seat_mirror", "hand_index"]
    if per_hand.duplicated(keys).any():
        raise Phase2ExecutionError("duplicate six-max arm/hand rows")
    arm_keys = ["serving_system", "seed", "seat_mirror"]
    fork_counts = per_hand.groupby(arm_keys, sort=False)["formation_fork_hash"].nunique()
    if not (fork_counts == 1).all():
        raise Phase2ExecutionError("formation fork hash changes within a paired block")
    # Same seed/mirror must fork from one identical formation across every treatment.
    by_pair = per_hand.groupby(["serving_system", "seed", "seat_mirror"], sort=False)["formation_fork_hash"].nunique()
    if not (by_pair == 1).all():
        raise Phase2ExecutionError("treatments do not share a formation checkpoint")
    treatments = set(per_hand["treatment"])
    if set(PHASE2_TREATMENTS) != treatments:
        raise Phase2ExecutionError("six-max analysis requires all frozen treatments")
    compare = tuple(contrasts or (("recursive_d2", "d1_budget_matched"), ("recursive_d3", "recursive_d2")))
    rows: list[dict[str, Any]] = []
    paired_rows: list[pd.DataFrame] = []
    for system, group in per_hand.groupby("serving_system", sort=False):
        expected_pairs = set(group[["seed", "seat_mirror", "hand_index"]].itertuples(index=False, name=None))
        for treatment in PHASE2_TREATMENTS:
            observed = set(group.loc[group["treatment"] == treatment, ["seed", "seat_mirror", "hand_index"]].itertuples(index=False, name=None))
            if observed != expected_pairs:
                raise Phase2ExecutionError(f"incomplete paired six-max arm: {system}/{treatment}")
        for treatment_a, treatment_b in compare:
            if treatment_a not in treatments or treatment_b not in treatments:
                raise Phase2ExecutionError("requested contrast is not a frozen treatment")
            wide = group.loc[group["treatment"].isin((treatment_a, treatment_b))].pivot(
                index=["seed", "seat_mirror", "hand_index"], columns="treatment", values="reward"
            )
            delta = (wide[treatment_a] - wide[treatment_b]).rename("reward_delta").reset_index()
            # Independent unit is the paired seed/fork; its mirror hands remain clustered.
            seed_deltas = delta.groupby("seed", as_index=False).agg(
                reward_delta=("reward_delta", "mean"), paired_hands=("reward_delta", "size")
            )
            values = seed_deltas["reward_delta"].to_numpy(dtype=float)
            low, high = paired_bootstrap_interval(values, samples=bootstrap_samples)
            robust = large_pot_sensitivity(delta.rename(columns={"reward_delta": "reward"}))
            without_largest = np.delete(values, np.argmax(np.abs(values))) if len(values) > 1 else values
            rows.append(
                {
                    "serving_system": system,
                    "contrast": f"{treatment_a}-{treatment_b}",
                    "paired_seeds": len(seed_deltas),
                    "mean_return_delta": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "permutation_p": paired_sign_permutation_p(values, samples=permutation_samples),
                    "positive_seed_rate": float((values > 0).mean()),
                    "leave_largest_paired_seed_out_mean": float(without_largest.mean()),
                    "top_1pct_abs_share": robust["top_1pct_abs_share"],
                    "trimmed_1pct_return_delta": robust["trimmed_1pct_reward"],
                    "inference_unit": "paired_seed_clustered_over_seat_mirror_hands",
                }
            )
            delta.insert(0, "serving_system", system)
            delta.insert(1, "contrast", f"{treatment_a}-{treatment_b}")
            paired_rows.append(delta)
    inference = pd.DataFrame(rows)
    inference["holm_p"] = holm_adjust(inference["permutation_p"].tolist())
    return inference, pd.concat(paired_rows, ignore_index=True)
