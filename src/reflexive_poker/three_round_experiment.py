"""Auditable three-round cross-model poker tournament.

The runner deliberately reuses :class:`LLMPlayer` and the repository's
``HoldemEnvironment``.  It adds only the missing orchestration for the
DeepSeek-vs-Luna heads-up rounds and a six-seat 3-vs-3 table.  Every match is
written as an atomic JSON checkpoint so a terminated run can be resumed by
rerunning the exact command.
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .agents import AgentStyle
from .environment import EnvironmentConfig, HoldemEnvironment
from .llm_player import (
    CodexProvider,
    DeterministicNarrativeProvider,
    LLMPlayer,
    OpenCodeGoProvider,
    ProviderResponse,
)
from .models import HandRecord
from .phase1_models import AbstractMCCFRPolicy
from .phase1_statistics import paired_bootstrap_interval, paired_sign_permutation_p

MODEL_LABELS = ("deepseek", "luna")
MODEL_SPECS = (
    ("deepseek", "opencode-go", "deepseek-v4-flash"),
    ("luna", "codex", "gpt-5.6-luna"),
)


@dataclass(frozen=True)
class ThreeRoundConfig:
    seeds: tuple[int, ...] = (9950,)
    hands: int = 2
    rounds: tuple[int, ...] = (1, 2, 3)
    round3_lineup_count: int = 2
    gto_iterations: int = 2_000
    equity_samples: int = 16
    memory_hands: int = 6
    evidence_tier: str = "pilot"
    minimum_formal_seeds: int = 10
    bootstrap_samples: int = 5_000
    permutation_samples: int = 20_000
    source_clean: bool = True
    output_dir: Path = Path("results/three_round/pilot")
    model_specs: tuple[tuple[str, str, str], ...] = MODEL_SPECS
    provider_factory: Callable[[str, str, int], Any] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.hands <= 0:
            raise ValueError("hands must be positive")
        if not self.rounds or any(round_id not in {1, 2, 3} for round_id in self.rounds):
            raise ValueError("rounds must contain only 1, 2, or 3")
        if self.round3_lineup_count <= 0 or self.round3_lineup_count > 20:
            raise ValueError("round3_lineup_count must be in [1, 20]")
        if self.gto_iterations <= 0:
            raise ValueError("gto_iterations must be positive")
        if self.evidence_tier not in {"pilot", "formal"}:
            raise ValueError("evidence_tier must be pilot or formal")
        if self.minimum_formal_seeds <= 1:
            raise ValueError("minimum_formal_seeds must be greater than one")
        if self.bootstrap_samples <= 0 or self.permutation_samples <= 0:
            raise ValueError("bootstrap and permutation samples must be positive")
        if self.evidence_tier == "formal" and self.round3_lineup_count % 2:
            raise ValueError("formal round 3 requires complementary lineup pairs")


class ContextProvider:
    """Add an immutable, auditable context payload to provider calls."""

    def __init__(self, base: Any, augment: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.base = base
        self.augment = augment
        self.name = base.name
        self.model = base.model

    def _state(self, state: dict[str, Any]) -> dict[str, Any]:
        # Mutate the per-call state object so LLMPlayer's persisted trace
        # proves which frozen reference/tool context was actually supplied.
        return self.augment(state)

    def decide(self, state: dict[str, Any]) -> ProviderResponse:
        return self.base.decide(self._state(state))

    def reflect(self, state: dict[str, Any]) -> ProviderResponse:
        return self.base.reflect(self._state(state))

    def structured(
        self,
        *,
        instructions: str,
        state: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        return self.base.structured(
            instructions=instructions,
            state=self._state(state),
            schema_name=schema_name,
            schema=schema,
        )


def _provider(kind: str, model: str, seed: int) -> Any:
    if kind == "mock":
        return DeterministicNarrativeProvider(seed=seed)
    if kind == "opencode-go":
        return OpenCodeGoProvider(model=model)
    if kind == "codex":
        return CodexProvider(model=model)
    raise ValueError(f"unknown provider: {kind}")


def _policy_payload(policy: AbstractMCCFRPolicy) -> dict[str, Any]:
    return {
        "label": policy.report.label,
        "iterations": policy.report.iterations,
        "infosets": policy.report.infosets,
        "policy_hash": policy.report.policy_hash,
        "empirical_exploitability": policy.report.empirical_exploitability,
        "policy": policy.policy,
        "interpretation": (
            "bucketed approximate-equilibrium reference over the repository action abstraction; "
            "not a solver-grade full no-limit GTO policy"
        ),
    }


def _reference_provider(base: Any, policy_payload: dict[str, Any]) -> ContextProvider:
    def augment(state: dict[str, Any]) -> dict[str, Any]:
        state["gto_reference"] = policy_payload
        state["gto_reference_instruction"] = (
            "This is a frozen strategy reference. You may use it as prior guidance, but choose "
            "only from the legal actions and visible information. Do not claim exact GTO."
        )
        return state

    return ContextProvider(base, augment)


def _simulation_provider(base: Any, equity_samples: int) -> ContextProvider:
    def augment(state: dict[str, Any]) -> dict[str, Any]:
        state["simulation_tool"] = {
            "name": "bounded_equity_simulator",
            "available": True,
            "samples": equity_samples,
            "result_field": "equity_estimate",
            "private_information_boundary": "own_cards_and_public_board_only",
        }
        return state

    return ContextProvider(base, augment)


def _model_spec_map(config: ThreeRoundConfig) -> dict[str, tuple[str, str]]:
    return {label: (provider, model) for label, provider, model in config.model_specs}


def _make_player(
    *,
    label: str,
    seat: int,
    seed: int,
    opponents: tuple[str, ...],
    round_id: int,
    config: ThreeRoundConfig,
    gto_payload: dict[str, Any] | None,
) -> LLMPlayer:
    spec = _model_spec_map(config)[label]
    factory = config.provider_factory or _provider
    provider = factory(spec[0], spec[1], seed * 1009 + seat + 1)
    if round_id == 2:
        if gto_payload is None:
            raise ValueError("round 2 requires a frozen GTO reference payload")
        provider = _reference_provider(provider, gto_payload)
    elif round_id == 3:
        provider = _simulation_provider(provider, config.equity_samples)
    player = LLMPlayer(
        f"{label}_seat_{seat}",
        seed * 1009 + seat + 1,
        provider,
        AgentStyle(
            aggression=0.47,
            risk_margin=0.055,
            belief_sensitivity=0.24,
            social_learning_rate=0.20,
            equity_samples=config.equity_samples,
        ),
        opponents=opponents,
        memory_hands=config.memory_hands,
        reflexive_enabled=round_id == 3,
        reflection_enabled=round_id == 3,
        simulation_enabled=round_id == 3,
    )
    player.condition = {
        1: "round1_raw_llm",
        2: "round2_gto_reference",
        3: "round3_reflection_simulation",
    }[round_id]
    return player


def _lineups(count: int) -> tuple[tuple[str, ...], ...]:
    combinations = list(itertools.combinations(range(6), 3))
    remaining = set(combinations)
    values: list[tuple[str, ...]] = []
    # Emit each layout next to its color-swapped complement.  Every even
    # prefix therefore gives both model families the same seat exposure.
    for deepseek_seats in combinations:
        if deepseek_seats not in remaining:
            continue
        complement = tuple(seat for seat in range(6) if seat not in deepseek_seats)
        for selected in (deepseek_seats, complement):
            deepseek = set(selected)
            values.append(tuple("deepseek" if seat in deepseek else "luna" for seat in range(6)))
        remaining.remove(deepseek_seats)
        remaining.remove(complement)
    return tuple(values[:count])


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _match_id(round_id: int, seed: int, index: int) -> str:
    return f"r{round_id}_seed{seed}_layout{index}"


def _expected_specs(config: ThreeRoundConfig) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for round_id in config.rounds:
        if round_id in {1, 2}:
            layouts = [(0, ("deepseek", "luna")), (1, ("luna", "deepseek"))]
        else:
            layouts = list(enumerate(_lineups(config.round3_lineup_count)))
        for seed in config.seeds:
            for layout_index, seat_labels in layouts:
                specs.append(
                    {
                        "match_id": _match_id(round_id, seed, layout_index),
                        "round": round_id,
                        "seed": seed,
                        "layout_index": layout_index,
                        "seat_labels": list(seat_labels),
                        "hands": config.hands,
                    }
                )
    return specs


def _plan_payload(config: ThreeRoundConfig) -> dict[str, Any]:
    config_payload = asdict(config)
    config_payload["output_dir"] = str(config.output_dir)
    config_payload["provider_factory"] = None
    payload = {
        "protocol": "three-round-cross-model-poker-v2",
        "config": config_payload,
        "expected_specs": _expected_specs(config),
    }
    # Normalize tuples and other JSON-compatible containers before comparing a
    # live plan to the deserialized PLAN.json on resume.
    return json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _plan_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _lock_plan(config: ThreeRoundConfig) -> tuple[dict[str, Any], str]:
    payload = _plan_payload(config)
    digest = _plan_hash(payload)
    locked = payload | {"plan_hash": digest}
    path = config.output_dir / "PLAN.json"
    matches_dir = config.output_dir / "matches"
    if path.exists():
        existing = _json(path)
        if existing != locked:
            raise ValueError(
                "three-round plan mismatch: resume requires the exact same frozen config"
            )
    elif matches_dir.exists() and any(matches_dir.glob("*.json")):
        raise ValueError(
            "existing match checkpoints have no frozen PLAN.json; refusing unsafe resume"
        )
    else:
        _atomic_json(path, locked)
    return locked, digest


def _hand_rows(
    records: list[HandRecord],
    seat_labels: tuple[str, ...],
    *,
    round_id: int,
    seed: int,
    layout_index: int,
    valid: bool,
) -> list[dict[str, Any]]:
    names = tuple(f"{label}_seat_{seat}" for seat, label in enumerate(seat_labels))
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "round": round_id,
            "seed": seed,
            "layout_index": layout_index,
            "hand_index": record.hand_index,
            "valid": valid,
            "showdown": record.showdown,
            "action_count": len(record.actions),
            "deepseek_reward": sum(
                record.rewards.get(name, 0.0) for name in names if name.startswith("deepseek_")
            ),
            "luna_reward": sum(
                record.rewards.get(name, 0.0) for name in names if name.startswith("luna_")
            ),
        }
        row["deepseek_minus_luna_reward"] = row["deepseek_reward"] - row["luna_reward"]
        rows.append(row)
    return rows


def _match_gate(players: list[LLMPlayer], expected: dict[str, tuple[str, str]]) -> dict[str, Any]:
    traces = [
        trace for player in players for trace in player.decision_traces + player.reflection_traces
    ]
    failures = sum(player.provider_failures for player in players)
    fallbacks = sum(
        bool(trace.get("final_decision", {}).get("fallback_used"))
        for player in players
        for trace in player.decision_traces
    )
    invalid = sum(player.invalid_actions for player in players)
    missing_identity: list[dict[str, Any]] = []
    identity_mismatch: list[dict[str, Any]] = []
    provider_mismatch: list[dict[str, Any]] = []

    def _provider_key(value: str) -> str:
        normalized = value.lower().replace("-", "_")
        # The adapters expose implementation-specific trace names while the
        # experiment config uses the public provider selectors.
        return {"codex_exec": "codex", "opencode_go": "opencode_go"}.get(normalized, normalized)

    for player in players:
        wanted = expected[player.name.split("_", 1)[0]]
        for trace in player.decision_traces + player.reflection_traces:
            observed_provider = _provider_key(str(trace.get("provider") or ""))
            expected_provider = _provider_key(wanted[0])
            if expected_provider != "mock" and observed_provider != expected_provider:
                provider_mismatch.append(
                    {"agent": player.name, "expected": wanted[0], "actual": trace.get("provider")}
                )
            actual = trace.get("actual_model")
            if not actual:
                # Deterministic fixtures intentionally have no provider-side
                # attestation; live providers must attest the selected model.
                if expected_provider != "mock":
                    missing_identity.append(
                        {"agent": player.name, "trace_type": trace.get("trace_type")}
                    )
            else:
                normalized_actual = str(actual).lower().replace("-", "")
                normalized_expected = wanted[1].lower().replace("-", "")
                if not normalized_actual.startswith(normalized_expected):
                    identity_mismatch.append(
                        {"agent": player.name, "expected": wanted[1], "actual": actual}
                    )
    return {
        "valid": not failures
        and not fallbacks
        and not invalid
        and not missing_identity
        and not identity_mismatch
        and not provider_mismatch,
        "provider_failure_count": failures,
        "fallback_count": fallbacks,
        "invalid_action_count": invalid,
        "trace_count": len(traces),
        "missing_identity": missing_identity,
        "identity_mismatch": identity_mismatch,
        "provider_mismatch": provider_mismatch,
        "cost_observability": {
            "exact": sum(trace.get("cost_observability") == "exact" for trace in traces),
            "estimated": sum(trace.get("cost_observability") == "estimated" for trace in traces),
            "unavailable": sum(
                trace.get("cost_observability") == "unavailable" for trace in traces
            ),
        },
    }


def _run_match(
    *,
    round_id: int,
    seed: int,
    layout_index: int,
    seat_labels: tuple[str, ...],
    config: ThreeRoundConfig,
    gto_payload: dict[str, Any] | None,
    plan_hash: str,
) -> dict[str, Any]:
    names = tuple(f"{label}_seat_{seat}" for seat, label in enumerate(seat_labels))
    players = [
        _make_player(
            label=label,
            seat=seat,
            seed=seed,
            opponents=tuple(name for name in names if name != names[seat]),
            round_id=round_id,
            config=config,
            gto_payload=gto_payload,
        )
        for seat, label in enumerate(seat_labels)
    ]
    environment = HoldemEnvironment(
        players,
        seed=seed,
        config=EnvironmentConfig(
            starting_stack=100.0,
            max_raises_per_street=None,
            regime_switch_hand=config.hands + 1,
        ),
    )
    records = environment.play(config.hands)
    expected = _model_spec_map(config)
    gate = _match_gate(players, expected)
    decisions = [
        {"round": round_id, "seed": seed, "layout_index": layout_index, **trace}
        for player in players
        for trace in player.decision_traces
    ]
    reflections = [
        {"round": round_id, "seed": seed, "layout_index": layout_index, **trace}
        for player in players
        for trace in player.reflection_traces
    ]
    rewards = {
        label: sum(
            record.rewards.get(f"{label}_seat_{seat}", 0.0)
            for seat, assigned in enumerate(seat_labels)
            if assigned == label
            for record in records
        )
        for label in MODEL_LABELS
    }
    return {
        "match_id": _match_id(round_id, seed, layout_index),
        "plan_hash": plan_hash,
        "round": round_id,
        "seed": seed,
        "layout_index": layout_index,
        "seat_labels": list(seat_labels),
        "hands": config.hands,
        "valid": gate["valid"],
        "gate": gate,
        "rewards": rewards,
        "chips_per_100": {label: 100.0 * value / config.hands for label, value in rewards.items()},
        "hand_rows": _hand_rows(
            records,
            seat_labels,
            round_id=round_id,
            seed=seed,
            layout_index=layout_index,
            valid=gate["valid"],
        ),
        "decisions": decisions,
        "reflections": reflections,
    }


def _validate_checkpoint(path: Path, spec: dict[str, Any], plan_hash: str) -> dict[str, Any]:
    payload = _json(path)
    required = {
        "match_id": spec["match_id"],
        "round": spec["round"],
        "seed": spec["seed"],
        "layout_index": spec["layout_index"],
        "seat_labels": spec["seat_labels"],
        "hands": spec["hands"],
        "plan_hash": plan_hash,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint mismatch for {path.name}: {mismatches}")
    if len(payload.get("hand_rows", [])) != spec["hands"]:
        raise ValueError(f"checkpoint hand count mismatch for {path.name}")
    return payload


def _source_fingerprint() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _tool_contract(match: dict[str, Any]) -> dict[str, Any]:
    round_id = int(match["round"])
    decisions = match.get("decisions", [])
    reflections = match.get("reflections", [])
    states = [row.get("state", {}) for row in decisions]
    if round_id == 1:
        valid = (
            bool(decisions)
            and not reflections
            and all(
                "equity_estimate" not in state
                and "gto_reference" not in state
                and "simulation_tool" not in state
                for state in states
            )
        )
    elif round_id == 2:
        valid = (
            bool(decisions)
            and not reflections
            and all(
                "gto_reference" in state
                and "equity_estimate" not in state
                and "simulation_tool" not in state
                for state in states
            )
        )
    else:
        valid = (
            bool(decisions)
            and len(reflections) == 6 * int(match["hands"])
            and all("equity_estimate" in state and "simulation_tool" in state for state in states)
            and all("simulation_tool" in row.get("state", {}) for row in reflections)
        )
    return {
        "valid": valid,
        "decision_count": len(decisions),
        "reflection_count": len(reflections),
        "expected_reflection_count": 6 * int(match["hands"]) if round_id == 3 else 0,
    }


def _token_accounting_complete(config: ThreeRoundConfig, matches: list[dict[str, Any]]) -> bool:
    live_labels = {label for label, provider, _model in config.model_specs if provider != "mock"}
    if not live_labels:
        return True
    traces = [
        trace
        for match in matches
        for trace in match.get("decisions", []) + match.get("reflections", [])
        if str(trace.get("agent", "")).split("_", 1)[0] in live_labels
    ]
    return bool(traces) and all(int(trace.get("total_tokens") or 0) > 0 for trace in traces)


def _cost_summary(matches: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        trace
        for match in matches
        for trace in match.get("decisions", []) + match.get("reflections", [])
    ]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in ("total_tokens", "observed_billed_cost", "estimated_api_equivalent_cost"):
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["exact_cost_call"] = frame.get("cost_observability", "") == "exact"
    return (
        frame.groupby(["round", "provider", "actual_model"], dropna=False, as_index=False)
        .agg(
            calls=("trace_type", "size"),
            total_tokens=("total_tokens", "sum"),
            observed_billed_cost=("observed_billed_cost", "sum"),
            estimated_api_equivalent_cost=("estimated_api_equivalent_cost", "sum"),
            exact_cost_calls=("exact_cost_call", "sum"),
        )
        .sort_values(["round", "provider"], kind="stable")
    )


def _inference_summary(config: ThreeRoundConfig, paired_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for round_id, group in paired_frame.groupby("round", sort=True):
        values = group["mean_deepseek_minus_luna_chips_per_100"].to_numpy(dtype=float)
        low, high = paired_bootstrap_interval(
            values,
            samples=config.bootstrap_samples,
            seed=20260805 + int(round_id),
        )
        rows.append(
            {
                "round": int(round_id),
                "seed_pairs": len(values),
                "mean_delta": float(values.mean()),
                "median_delta": float(pd.Series(values).median()),
                "ci95_low": low,
                "ci95_high": high,
                "permutation_p": paired_sign_permutation_p(
                    values,
                    samples=config.permutation_samples,
                    seed=20260815 + int(round_id),
                ),
                "deepseek_positive_seed_rate": float((values > 0).mean()),
                "direction": (
                    "deepseek"
                    if math.isfinite(low) and low > 0
                    else "luna"
                    if math.isfinite(high) and high < 0
                    else "inconclusive"
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate(
    config: ThreeRoundConfig,
    match_paths: list[Path],
    gto_payload: dict[str, Any],
    *,
    plan: dict[str, Any],
    plan_hash: str,
) -> dict[str, Any]:
    matches = [_json(path) for path in match_paths]
    hand_rows = [row for match in matches for row in match["hand_rows"]]
    decision_rows = [row for match in matches for row in match["decisions"]]
    reflection_rows = [row for match in matches for row in match["reflections"]]
    match_rows = []
    for match in matches:
        match_rows.append(
            {
                "match_id": match["match_id"],
                "round": match["round"],
                "seed": match["seed"],
                "layout_index": match["layout_index"],
                "valid": match["valid"],
                "deepseek_chips_per_100": match["chips_per_100"]["deepseek"],
                "luna_chips_per_100": match["chips_per_100"]["luna"],
                "deepseek_minus_luna_chips_per_100": (
                    match["chips_per_100"]["deepseek"] - match["chips_per_100"]["luna"]
                ),
                "provider_failures": match["gate"]["provider_failure_count"],
                "fallbacks": match["gate"]["fallback_count"],
                "invalid_actions": match["gate"]["invalid_action_count"],
            }
        )
    match_frame = pd.DataFrame(match_rows)
    hand_frame = pd.DataFrame(hand_rows)
    valid_hand_frame = (
        hand_frame[hand_frame["valid"]].copy() if not hand_frame.empty else hand_frame
    )
    if valid_hand_frame.empty:
        seed_hand_frame = pd.DataFrame()
    else:
        seed_hand_frame = valid_hand_frame.groupby(
            ["round", "seed", "hand_index"], as_index=False
        ).agg(
            layout_count=("layout_index", "nunique"),
            mean_deepseek_minus_luna_reward=("deepseek_minus_luna_reward", "mean"),
            max_abs_deepseek_minus_luna_reward=(
                "deepseek_minus_luna_reward",
                lambda values: float(values.abs().max()),
            ),
        )
        seed_hand_frame["deepseek_minus_luna_chips_per_100"] = (
            100.0 * seed_hand_frame["mean_deepseek_minus_luna_reward"]
        )
    if match_frame.empty:
        paired_frame = pd.DataFrame()
    else:
        valid = match_frame[match_frame["valid"]]
        paired_rows: list[dict[str, Any]] = []
        for (round_id, seed), group in valid.groupby(["round", "seed"]):
            paired_rows.append(
                {
                    "round": round_id,
                    "seed": seed,
                    "valid_matches": len(group),
                    "mean_deepseek_minus_luna_chips_per_100": group[
                        "deepseek_minus_luna_chips_per_100"
                    ].mean(),
                    "mean_deepseek_chips_per_100": group["deepseek_chips_per_100"].mean(),
                    "mean_luna_chips_per_100": group["luna_chips_per_100"].mean(),
                }
            )
        paired_frame = pd.DataFrame(paired_rows)
    inference_frame = (
        _inference_summary(config, paired_frame) if not paired_frame.empty else pd.DataFrame()
    )
    sensitivity_rows: list[dict[str, Any]] = []
    if not seed_hand_frame.empty:
        for (round_id, seed), group in seed_hand_frame.groupby(["round", "seed"], sort=True):
            ordered = group.sort_values("hand_index", kind="stable")
            values = ordered["deepseek_minus_luna_chips_per_100"].to_numpy(dtype=float)
            trim_count = max(1, math.ceil(len(values) * 0.01))
            remove = set(abs(pd.Series(values)).nlargest(trim_count).index.tolist())
            kept = [value for index, value in enumerate(values) if index not in remove]
            largest = int(abs(pd.Series(values)).idxmax())
            leave_one = [value for index, value in enumerate(values) if index != largest]
            sensitivity_rows.append(
                {
                    "round": int(round_id),
                    "seed": int(seed),
                    "hands": len(values),
                    "raw_mean_delta": float(values.mean()),
                    "trimmed_top_1pct_mean_delta": float(pd.Series(kept).mean())
                    if kept
                    else float("nan"),
                    "leave_largest_hand_out_mean_delta": (
                        float(pd.Series(leave_one).mean()) if leave_one else float("nan")
                    ),
                }
            )
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    cost_frame = _cost_summary(matches)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    match_frame.to_csv(output / "match_summary.csv", index=False)
    hand_frame.to_csv(output / "hand_rows.csv", index=False)
    paired_frame.to_csv(output / "paired_summary.csv", index=False)
    seed_hand_frame.to_csv(output / "seed_hand_summary.csv", index=False)
    inference_frame.to_csv(output / "inference_summary.csv", index=False)
    sensitivity_frame.to_csv(output / "large_pot_sensitivity.csv", index=False)
    cost_frame.to_csv(output / "cost_summary.csv", index=False)
    _write_jsonl_gz(output / "decision_traces.jsonl.gz", decision_rows)
    _write_jsonl_gz(output / "reflection_traces.jsonl.gz", reflection_rows)
    gates = [
        match["gate"] | {"match_id": match["match_id"], "round": match["round"]}
        for match in matches
    ]
    provider_gate = {
        "valid": bool(matches) and all(match["valid"] for match in matches),
        "match_count": len(matches),
        "valid_match_count": sum(match["valid"] for match in matches),
        "matches": gates,
    }
    _atomic_json(output / "provider_gate.json", provider_gate)
    expected_ids = {spec["match_id"] for spec in plan["expected_specs"]}
    actual_ids = {match["match_id"] for match in matches}
    tool_contracts = {match["match_id"]: _tool_contract(match) for match in matches}
    token_accounting = _token_accounting_complete(config, matches)
    balanced_round3 = config.round3_lineup_count % 2 == 0 and all(
        sum(lineup[seat] == "deepseek" for lineup in _lineups(config.round3_lineup_count))
        == config.round3_lineup_count // 2
        for seat in range(6)
    )
    execution_complete = (
        actual_ids == expected_ids
        and len(matches) == len(plan["expected_specs"])
        and all(len(match.get("hand_rows", [])) == config.hands for match in matches)
    )
    requirements = {
        "plan_hash_locked": True,
        "all_expected_matches_completed": execution_complete,
        "all_provider_gates_valid": provider_gate["valid"],
        "all_round_tool_contracts_valid": all(
            contract["valid"] for contract in tool_contracts.values()
        ),
        "complete_token_accounting": token_accounting,
        "all_three_rounds_present": set(config.rounds) == {1, 2, 3},
        "minimum_seed_count_met": len(config.seeds) >= config.minimum_formal_seeds,
        "round3_complementary_seat_balance": balanced_round3,
        "clean_source_snapshot": config.source_clean,
    }
    formal_allowed = config.evidence_tier == "formal" and all(requirements.values())
    evidence_gate = {
        "valid": all(
            value
            for key, value in requirements.items()
            if key
            not in {
                "minimum_seed_count_met",
                "round3_complementary_seat_balance",
                "clean_source_snapshot",
            }
            or config.evidence_tier == "formal"
        ),
        "evidence_tier": config.evidence_tier,
        "formal_conclusion_allowed": formal_allowed,
        "plan_hash": plan_hash,
        "expected_match_count": len(plan["expected_specs"]),
        "completed_match_count": len(matches),
        "requirements": requirements,
        "tool_contracts": tool_contracts,
        "limitations": [
            "comparison identifies model plus serving stack, not an immutable model snapshot",
            "round 2 uses a bucketed repository abstraction, not solver-grade full no-limit GTO",
        ],
    }
    _atomic_json(output / "evidence_gate.json", evidence_gate)
    manifest = {
        "protocol": "three-round-cross-model-poker-v2",
        "source_commit": _source_fingerprint(),
        "plan_hash": plan_hash,
        "config": asdict(config) | {"output_dir": str(config.output_dir), "provider_factory": None},
        "gto_reference": gto_payload,
        "match_count": len(matches),
        "completed_match_count": len(matches),
        "valid_match_count": sum(match["valid"] for match in matches),
    }
    _atomic_json(output / "manifest.json", manifest)
    if execution_complete:
        _atomic_json(
            output / "COMPLETED.json",
            {
                "plan_hash": plan_hash,
                "expected_match_count": len(plan["expected_specs"]),
                "completed_match_count": len(matches),
                "provider_gate_valid": provider_gate["valid"],
                "evidence_gate_valid": evidence_gate["valid"],
                "formal_conclusion_allowed": formal_allowed,
            },
        )
    return {
        "match_summary": match_frame,
        "hand_rows": hand_frame,
        "paired_summary": paired_frame,
        "seed_hand_summary": seed_hand_frame,
        "inference_summary": inference_frame,
        "large_pot_sensitivity": sensitivity_frame,
        "cost_summary": cost_frame,
        "provider_gate": provider_gate,
        "evidence_gate": evidence_gate,
    }


def run_three_round_experiment(config: ThreeRoundConfig) -> dict[str, Any]:
    """Run or resume the configured three-round experiment."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    plan, plan_hash = _lock_plan(config)
    policy = AbstractMCCFRPolicy.train(config.gto_iterations, seed=min(config.seeds))
    gto_payload = _policy_payload(policy)
    gto_raw = json.dumps(gto_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    gto_payload["pack_sha256"] = hashlib.sha256(gto_raw.encode()).hexdigest()
    gto_path = config.output_dir / "gto_reference.json"
    if gto_path.exists() and _json(gto_path) != gto_payload:
        raise ValueError("frozen GTO reference mismatch; refusing unsafe resume")
    if not gto_path.exists():
        _atomic_json(gto_path, gto_payload)
    matches_dir = config.output_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    match_paths: list[Path] = []
    for spec in plan["expected_specs"]:
        path = matches_dir / f"{spec['match_id']}.json"
        if not path.exists():
            match = _run_match(
                round_id=int(spec["round"]),
                seed=int(spec["seed"]),
                layout_index=int(spec["layout_index"]),
                seat_labels=tuple(spec["seat_labels"]),
                config=config,
                gto_payload=gto_payload if int(spec["round"]) == 2 else None,
                plan_hash=plan_hash,
            )
            _atomic_json(path, match)
        _validate_checkpoint(path, spec, plan_hash)
        match_paths.append(path)
    result = _aggregate(
        config,
        match_paths,
        gto_payload,
        plan=plan,
        plan_hash=plan_hash,
    )
    result["spec_count"] = len(plan["expected_specs"])
    return result
