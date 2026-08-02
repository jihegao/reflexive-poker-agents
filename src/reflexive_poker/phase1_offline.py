from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .llm_player import CodexProvider, OpenCodeGoProvider
from .phase1_models import (
    ACTION_NAMES,
    TYPE_ACTION_LIKELIHOODS,
    AbstractMCCFRPolicy,
    BudgetedRetryProvider,
    ProviderBudget,
    ProviderLedger,
    ReasoningTreatment,
)
from .phase1_offline_evidence import d2_d1bm_post_switch_contrasts

PAPER_OPPONENT_TYPES = ("rock", "tag", "lag", "calling_station", "myopic")
OFFLINE_TREATMENTS = (
    ReasoningTreatment.STATE_ONLY,
    ReasoningTreatment.ACTION_PREDICTION,
    ReasoningTreatment.BUDGET_MATCHED_D1,
    ReasoningTreatment.RECURSIVE_D2,
    ReasoningTreatment.RECURSIVE_D3,
)
CHECKPOINTS = (10, 40, 55, 80)

_PROBABILITY = {"type": "number", "minimum": 0.0, "maximum": 1.0}
_TABLE_STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "street", "position", "pot_bb", "effective_stack_bb", "spr", "pot_odds",
        "hand_class", "equity", "legal_actions",
    ],
    "properties": {
        "street": {"type": "string", "enum": ["preflop", "flop", "turn", "river"]},
        "position": {"type": "string", "enum": ["button", "big_blind"]},
        "pot_bb": {"type": "number", "minimum": 0.0},
        "effective_stack_bb": {"type": "number", "minimum": 0.0},
        "spr": {"type": "number", "minimum": 0.0},
        "pot_odds": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "hand_class": {"type": "string"},
        "equity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "legal_actions": {"type": "array", "items": {"type": "string", "enum": list(ACTION_NAMES)}},
    },
}
PHASE1_PREDICTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "table_state",
        "type_probabilities",
        "action_probabilities",
        "hero_image_aggression",
        "adaptation_probability",
        "switch_detected",
        "recommended_action",
        "confidence",
        "audit_summary",
    ],
    "properties": {
        "table_state": _TABLE_STATE_SCHEMA,
        "type_probabilities": {
            "type": "object",
            "additionalProperties": False,
            "required": list(PAPER_OPPONENT_TYPES),
            "properties": {name: dict(_PROBABILITY) for name in PAPER_OPPONENT_TYPES},
        },
        "action_probabilities": {
            "type": "object",
            "additionalProperties": False,
            "required": list(ACTION_NAMES),
            "properties": {name: dict(_PROBABILITY) for name in ACTION_NAMES},
        },
        "hero_image_aggression": dict(_PROBABILITY),
        "adaptation_probability": dict(_PROBABILITY),
        "switch_detected": {"type": "boolean"},
        "recommended_action": {"type": "string", "enum": list(ACTION_NAMES)},
        "confidence": dict(_PROBABILITY),
        "audit_summary": {"type": "string"},
    },
}


@dataclass(frozen=True)
class OfflineBenchmarkConfig:
    output_dir: Path = Path("results/phase1/offline_understanding")
    provider: str = "baselines"
    model: str = "none"
    case_count: int = 200
    base_seed: int = 20260802
    treatments: tuple[ReasoningTreatment, ...] = OFFLINE_TREATMENTS
    provider_budget: ProviderBudget = field(
        default_factory=lambda: ProviderBudget(max_calls=1_200, max_retries=100)
    )
    preregistered: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.case_count <= 200:
            raise ValueError("case_count must be in [1, 200]")
        if self.provider not in {"baselines", "mock", "opencode-go", "codex"}:
            raise ValueError(f"unsupported offline provider: {self.provider}")
        if not self.treatments:
            raise ValueError("at least one offline treatment is required")


def _normalise(values: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(1e-9, float(value)) for key, value in values.items()}
    total = sum(clipped.values())
    return {key: value / total for key, value in clipped.items()}


def _policy(player_type: str, hero_image: float, adaptive: bool) -> dict[str, float]:
    values = dict(TYPE_ACTION_LIKELIHOODS[player_type])
    if adaptive:
        pressure = 0.24 * (hero_image - 0.5)
        values["raise"] += pressure
        values["fold"] -= pressure * 0.65
        values["check_call"] -= pressure * 0.35
    return _normalise(values)


def _sample_action(rng: random.Random, distribution: dict[str, float]) -> str:
    return rng.choices(tuple(distribution), weights=tuple(distribution.values()), k=1)[0]


def _table_state(rng: random.Random, trajectory_index: int, checkpoint_index: int) -> dict[str, Any]:
    street = ("preflop", "flop", "turn", "river")[(trajectory_index + checkpoint_index) % 4]
    to_call = round(rng.choice((0.0, 0.5, 1.0, 2.0, 4.0)), 2)
    pot = round(rng.uniform(3.0, 35.0), 2)
    stack = round(rng.uniform(35.0, 100.0), 2)
    legal = ["check_call", "raise"] if to_call == 0.0 else list(ACTION_NAMES)
    hand_strength = round(rng.uniform(0.12, 0.88), 4)
    hand_class = (
        "strong_made_hand" if hand_strength >= 0.72
        else "top_or_overpair" if hand_strength >= 0.55
        else "medium_showdown_value" if hand_strength >= 0.35
        else "draw_or_weak_hand"
    )
    return {
        "street": street,
        "position": "button" if trajectory_index % 2 == 0 else "big_blind",
        "pot_bb": pot,
        "effective_stack_bb": stack,
        "spr": round(stack / max(pot, 1e-9), 4),
        "pot_odds": round(to_call / max(pot + to_call, 1e-9), 4),
        "to_call_bb": to_call,
        "hand_strength": hand_strength,
        "hand_class": hand_class,
        "equity": hand_strength,
        "legal_actions": legal,
    }


def generate_offline_cases(base_seed: int = 20260802) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    trajectory_index = 0
    for type_index, initial_type in enumerate(PAPER_OPPONENT_TYPES):
        shifted_type = PAPER_OPPONENT_TYPES[(type_index + 2) % len(PAPER_OPPONENT_TYPES)]
        for regime in ("fixed", "adaptive_shift"):
            for replication in range(5):
                trajectory_id = f"{initial_type}-{regime}-{replication:02d}"
                rng = random.Random(base_seed + trajectory_index * 100_003)
                opponent_actions: list[str] = []
                hero_actions: list[str] = []
                next_checkpoint = 0
                for hand_index in range(1, CHECKPOINTS[-1] + 1):
                    hero_raise_probability = min(0.75, 0.20 + 0.10 * replication + hand_index / 800)
                    hero_action = "raise" if rng.random() < hero_raise_probability else "check_call"
                    hero_actions.append(hero_action)
                    hero_image = (hero_actions.count("raise") + 1) / (len(hero_actions) + 2)
                    # ``post_switch`` is the matched late-history checkpoint
                    # indicator used by the pre-registered D2-D1BM interaction.
                    # Only adaptive_shift changes latent type at this point.
                    post_switch = hand_index >= 50
                    shifted = regime == "adaptive_shift" and post_switch
                    active_type = shifted_type if shifted else initial_type
                    action_distribution = _policy(active_type, hero_image, shifted)
                    opponent_actions.append(_sample_action(rng, action_distribution))
                    if hand_index != CHECKPOINTS[next_checkpoint]:
                        continue
                    checkpoint_index = next_checkpoint
                    recent = opponent_actions[-50:]
                    counts = Counter(recent)
                    action_counts = {name: int(counts[name]) for name in ACTION_NAMES}
                    table_state = _table_state(rng, trajectory_index, checkpoint_index)
                    ground_truth = {
                        "active_type": active_type,
                        "type_probabilities": {
                            name: float(name == active_type) for name in PAPER_OPPONENT_TYPES
                        },
                        "action_probabilities": action_distribution,
                        "hero_image_aggression": hero_image,
                        "adaptation_probability": 1.0 if shifted else 0.0,
                        "switch_detected": shifted,
                    }
                    case_payload = {
                        "case_id": f"case-{trajectory_index:03d}-{checkpoint_index}",
                        "trajectory_id": trajectory_id,
                        "trajectory_index": trajectory_index,
                        "checkpoint_index": checkpoint_index,
                        "hand_index": hand_index,
                        "switch_hand": 50 if regime == "adaptive_shift" else None,
                        "regime": regime,
                        "initial_type": initial_type,
                        "post_switch": post_switch,
                        "table_state": table_state,
                        "public_history": {
                            "opponent_action_counts": action_counts,
                            "recent_opponent_actions": recent,
                            "hero_raise_count": hero_actions.count("raise"),
                            "hero_passive_count": len(hero_actions) - hero_actions.count("raise"),
                            "observations": len(opponent_actions),
                        },
                        "recursive_public_summary": {
                            "opponent_view_of_hero": hero_image,
                            "anticipated_adjustment": max(-1.0, min(1.0, 2.0 * (hero_image - 0.5))),
                        },
                        "ground_truth": ground_truth,
                    }
                    canonical = json.dumps(case_payload, sort_keys=True, separators=(",", ":"))
                    case_payload["case_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
                    cases.append(case_payload)
                    next_checkpoint += 1
                    if next_checkpoint == len(CHECKPOINTS):
                        break
                trajectory_index += 1
    if len(cases) != 200:
        raise AssertionError(f"offline generator produced {len(cases)} cases instead of 200")
    return cases


def _bayesian_type_posterior(actions: list[str], transition: float = 0.0) -> dict[str, float]:
    posterior = {name: 1.0 / len(PAPER_OPPONENT_TYPES) for name in PAPER_OPPONENT_TYPES}
    for action in actions:
        if transition:
            retained = 1.0 - transition
            incoming = transition / len(PAPER_OPPONENT_TYPES)
            posterior = {name: retained * value + incoming for name, value in posterior.items()}
        posterior = _normalise(
            {
                name: probability * TYPE_ACTION_LIKELIHOODS[name][action]
                for name, probability in posterior.items()
            }
        )
    return posterior


def _mixture_action(
    type_probabilities: dict[str, float],
    *,
    hero_image: float = 0.5,
    adaptive: bool = False,
) -> dict[str, float]:
    return _normalise(
        {
            action: sum(
                type_probabilities[player_type]
                * _policy(player_type, hero_image, adaptive)[action]
                for player_type in PAPER_OPPONENT_TYPES
            )
            for action in ACTION_NAMES
        }
    )


def _action_values(case: dict[str, Any], opponent_distribution: dict[str, float]) -> dict[str, float]:
    table = case["table_state"]
    strength = float(table["hand_strength"])
    pot_odds = float(table["pot_odds"])
    return {
        action: sum(
            probability
            * AbstractMCCFRPolicy._utility(action, opponent_action, strength, pot_odds)
            for opponent_action, probability in opponent_distribution.items()
        )
        for action in table["legal_actions"]
    }


def _recommended_action(case: dict[str, Any], distribution: dict[str, float]) -> str:
    values = _action_values(case, distribution)
    return max(values, key=values.get)


def _prediction(
    case: dict[str, Any],
    *,
    type_probabilities: dict[str, float],
    action_probabilities: dict[str, float],
    hero_image: float,
    adaptation_probability: float,
    switch_detected: bool,
    audit_summary: str,
) -> dict[str, Any]:
    return {
        "table_state": {
            key: case["table_state"][key]
            for key in _TABLE_STATE_SCHEMA["required"]
        },
        "type_probabilities": _normalise(type_probabilities),
        "action_probabilities": _normalise(action_probabilities),
        "hero_image_aggression": min(1.0, max(0.0, hero_image)),
        "adaptation_probability": min(1.0, max(0.0, adaptation_probability)),
        "switch_detected": bool(switch_detected),
        "recommended_action": _recommended_action(case, action_probabilities),
        "confidence": 0.75,
        "audit_summary": audit_summary,
    }


def baseline_predictions(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    uniform_types = {name: 1.0 / len(PAPER_OPPONENT_TYPES) for name in PAPER_OPPONENT_TYPES}
    uniform_actions = {name: 1.0 / len(ACTION_NAMES) for name in ACTION_NAMES}
    for case in cases:
        history = case["public_history"]
        recent = list(history["recent_opponent_actions"])
        total = sum(history["opponent_action_counts"].values()) + len(ACTION_NAMES)
        frequency_actions = {
            name: (history["opponent_action_counts"][name] + 1.0) / total
            for name in ACTION_NAMES
        }
        bayes_types = _bayesian_type_posterior(recent)
        hmm_types = _bayesian_type_posterior(recent, transition=0.035)
        truth = case["ground_truth"]
        definitions = {
            "uniform": _prediction(
                case,
                type_probabilities=uniform_types,
                action_probabilities=uniform_actions,
                hero_image=0.5,
                adaptation_probability=0.5,
                switch_detected=False,
                audit_summary="uninformative sanity baseline",
            ),
            "frequency": _prediction(
                case,
                type_probabilities=uniform_types,
                action_probabilities=frequency_actions,
                hero_image=(history["hero_raise_count"] + 1)
                / (history["hero_raise_count"] + history["hero_passive_count"] + 2),
                adaptation_probability=0.0,
                switch_detected=False,
                audit_summary="Laplace-smoothed public action frequency",
            ),
            "bayesian_filter": _prediction(
                case,
                type_probabilities=bayes_types,
                action_probabilities=_mixture_action(bayes_types),
                hero_image=case["recursive_public_summary"]["opponent_view_of_hero"],
                adaptation_probability=0.0,
                switch_detected=False,
                audit_summary="fixed-type Bayesian filter",
            ),
            "hmm_filter": _prediction(
                case,
                type_probabilities=hmm_types,
                action_probabilities=_mixture_action(
                    hmm_types,
                    hero_image=case["recursive_public_summary"]["opponent_view_of_hero"],
                    adaptive=True,
                ),
                hero_image=case["recursive_public_summary"]["opponent_view_of_hero"],
                adaptation_probability=max(hmm_types.values()) < 0.75,
                switch_detected=(
                    history["observations"] >= 50 and max(hmm_types.values()) < 0.75
                ),
                audit_summary="Bayesian filter with frozen type-transition probability",
            ),
            "oracle": _prediction(
                case,
                type_probabilities=truth["type_probabilities"],
                action_probabilities=truth["action_probabilities"],
                hero_image=truth["hero_image_aggression"],
                adaptation_probability=truth["adaptation_probability"],
                switch_detected=truth["switch_detected"],
                audit_summary="instrumented upper-bound oracle",
            ),
        }
        for method, payload in definitions.items():
            rows.append(
                {
                    "case_id": case["case_id"],
                    "trajectory_id": case["trajectory_id"],
                    "method": method,
                    "provider": "deterministic_baseline",
                    "model": method,
                    "treatment": "baseline",
                    "payload": payload,
                    "latency_ms": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            )
    return rows


def _treatment_view(case: dict[str, Any], treatment: ReasoningTreatment) -> dict[str, Any]:
    view: dict[str, Any] = {
        "case_id": case["case_id"],
        "treatment": treatment.value,
        "table_state": case["table_state"],
        "public_history": "__MASKED__",
        "recursive_public_summary": "__MASKED__",
        "anticipated_adjustment": "__MASKED__",
        "budget_match_control": "__MASKED__",
    }
    if treatment is not ReasoningTreatment.STATE_ONLY:
        view["public_history"] = case["public_history"]
    if treatment in {ReasoningTreatment.RECURSIVE_D2, ReasoningTreatment.RECURSIVE_D3}:
        view["recursive_public_summary"] = {
            "opponent_view_of_hero": case["recursive_public_summary"]["opponent_view_of_hero"]
        }
    if treatment is ReasoningTreatment.RECURSIVE_D3:
        view["anticipated_adjustment"] = case["recursive_public_summary"][
            "anticipated_adjustment"
        ]
    if treatment is ReasoningTreatment.BUDGET_MATCHED_D1:
        view["budget_match_control"] = ""
        target_length = len(
            json.dumps(
                _treatment_view(case, ReasoningTreatment.RECURSIVE_D2),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        control_length = len(json.dumps(view, ensure_ascii=False, sort_keys=True))
        view["budget_match_control"] = "0" * max(0, target_length - control_length)
    return view


def _validate_prediction(payload: dict[str, Any], legal_actions: list[str]) -> None:
    missing = [name for name in PHASE1_PREDICTION_SCHEMA["required"] if name not in payload]
    if missing:
        raise ValueError(f"phase1 prediction missing fields: {missing}")
    table_state = payload["table_state"]
    expected_table_fields = set(_TABLE_STATE_SCHEMA["required"])
    if not isinstance(table_state, dict) or set(table_state) != expected_table_fields:
        raise ValueError("table_state must contain exactly the audited fields")
    if not isinstance(table_state["legal_actions"], list):
        raise TypeError("table_state.legal_actions must be a list")
    if not 0.0 <= float(table_state["equity"]) <= 1.0:
        raise ValueError("table_state.equity must be in [0, 1]")
    for field_name, names in (
        ("type_probabilities", PAPER_OPPONENT_TYPES),
        ("action_probabilities", ACTION_NAMES),
    ):
        values = payload.get(field_name)
        if not isinstance(values, dict) or set(values) != set(names):
            raise ValueError(f"{field_name} must contain exactly {names}")
        probabilities = [float(values[name]) for name in names]
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError(f"{field_name} contains a value outside [0, 1]")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
            raise ValueError(f"{field_name} must sum to one")
    for field_name in ("hero_image_aggression", "adaptation_probability", "confidence"):
        if not 0.0 <= float(payload[field_name]) <= 1.0:
            raise ValueError(f"{field_name} must be in [0, 1]")
    if payload["recommended_action"] not in legal_actions:
        raise ValueError("recommended_action is not legal for this case")
    if not isinstance(payload["switch_detected"], bool):
        raise TypeError("switch_detected must be boolean")


def _mock_treatment_prediction(case: dict[str, Any], treatment: ReasoningTreatment) -> dict[str, Any]:
    recent = case["public_history"]["recent_opponent_actions"]
    if treatment is ReasoningTreatment.STATE_ONLY:
        type_probabilities = {
            name: 1.0 / len(PAPER_OPPONENT_TYPES) for name in PAPER_OPPONENT_TYPES
        }
    elif treatment in {
        ReasoningTreatment.ACTION_PREDICTION,
        ReasoningTreatment.BUDGET_MATCHED_D1,
    }:
        type_probabilities = _bayesian_type_posterior(recent)
    else:
        type_probabilities = _bayesian_type_posterior(recent, transition=0.035)
    action_probabilities = _mixture_action(type_probabilities)
    hero_image = (
        case["recursive_public_summary"]["opponent_view_of_hero"]
        if treatment in {ReasoningTreatment.RECURSIVE_D2, ReasoningTreatment.RECURSIVE_D3}
        else 0.5
    )
    adaptation = (
        abs(case["recursive_public_summary"]["anticipated_adjustment"])
        if treatment is ReasoningTreatment.RECURSIVE_D3
        else 0.0
    )
    return _prediction(
        case,
        type_probabilities=type_probabilities,
        action_probabilities=action_probabilities,
        hero_image=hero_image,
        adaptation_probability=adaptation,
        switch_detected=adaptation > 0.25,
        audit_summary="deterministic mock for contract validation only",
    )


def _provider(kind: str, model: str):
    if kind == "opencode-go":
        return OpenCodeGoProvider(model=model)
    if kind == "codex":
        return CodexProvider(model=model)
    raise ValueError(f"provider {kind} does not support live offline calls")


class OfflineProgressConflict(RuntimeError):
    """Persisted call accounting cannot be safely reconciled with raw rows."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _live_plan(cases: list[dict[str, Any]], config: OfflineBenchmarkConfig) -> dict[str, Any]:
    payload = {
        "provider": config.provider,
        "model": config.model,
        "case_hashes": [case["case_hash"] for case in cases],
        "treatments": [treatment.value for treatment in config.treatments],
        "schema": PHASE1_PREDICTION_SCHEMA,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "plan_hash": hashlib.sha256(canonical.encode()).hexdigest(),
        "expected_primary_predictions": len(cases) * len(config.treatments),
    }


def _read_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OfflineProgressConflict(
                    f"invalid live prediction journal at {path}:{line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise OfflineProgressConflict(
                    f"live prediction journal row is not an object at {path}:{line_number}"
                )
            rows.append(payload)
    return rows


def _append_journal_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _prediction_key(row: dict[str, Any]) -> tuple[str, str]:
    try:
        return str(row["case_id"]), str(row["treatment"])
    except KeyError as exc:
        raise OfflineProgressConflict("live prediction journal row lacks case_id or treatment") from exc


def _load_live_ledger(
    checkpoint_path: Path, config: OfflineBenchmarkConfig
) -> ProviderLedger:
    if not checkpoint_path.exists():
        return ProviderLedger()
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        ledger_payload = payload["ledger"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise OfflineProgressConflict(f"invalid provider ledger checkpoint: {checkpoint_path}") from exc
    if payload.get("model") != config.model or payload.get("budget") != asdict(config.provider_budget):
        raise OfflineProgressConflict("provider ledger checkpoint does not match the frozen offline plan")
    try:
        return ProviderLedger(**ledger_payload)
    except TypeError as exc:
        raise OfflineProgressConflict("provider ledger checkpoint has an invalid ledger shape") from exc


def _live_progress(
    cases: list[dict[str, Any]], config: OfflineBenchmarkConfig
) -> tuple[list[dict[str, Any]], ProviderLedger, Path, Path, Path]:
    journal_path = config.output_dir / "live_predictions.jsonl"
    progress_path = config.output_dir / "LIVE_PROGRESS.json"
    inflight_path = config.output_dir / "LIVE_INFLIGHT.json"
    ledger_path = config.output_dir / "live_provider_ledger.json"
    plan = _live_plan(cases, config)
    if inflight_path.exists():
        raise OfflineProgressConflict(
            "a provider call was in flight during interruption; its result is not auditable"
        )
    existing_progress = (
        json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None
    )
    if existing_progress is not None and existing_progress.get("plan_hash") != plan["plan_hash"]:
        raise OfflineProgressConflict("live prediction journal does not match the frozen offline plan")
    rows = _read_journal(journal_path)
    keys = [_prediction_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise OfflineProgressConflict("live prediction journal contains duplicate case/treatment rows")
    expected_keys = {
        (case["case_id"], treatment.value)
        for case in cases
        for treatment in config.treatments
    }
    if not set(keys).issubset(expected_keys):
        raise OfflineProgressConflict("live prediction journal contains rows outside the frozen plan")
    ledger = _load_live_ledger(ledger_path, config)
    primary_calls = ledger.calls - ledger.retries
    if primary_calls != len(rows):
        raise OfflineProgressConflict(
            "provider ledger and journal disagree; unrecorded provider calls cannot be reused"
        )
    _atomic_json(
        progress_path,
        {
            **plan,
            "completed_predictions": len(rows),
            "state": "running",
        },
    )
    return rows, ledger, journal_path, progress_path, inflight_path


def _model_row(
    case: dict[str, Any],
    treatment: ReasoningTreatment,
    response: dict[str, Any],
    payload: dict[str, Any],
    prompt_chars: int,
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "trajectory_id": case["trajectory_id"],
        "method": f"llm_{treatment.value}",
        "provider": response["provider"],
        "model": response["model"],
        "treatment": treatment.value,
        "prompt_chars": prompt_chars,
        "payload": payload,
        **{
            key: response.get(key)
            for key in (
                "latency_ms",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cost_usd",
                "response_id",
                "actual_model",
                "model_version",
                "observed_billed_cost",
                "estimated_api_equivalent_cost",
                "cost_observability",
                "model_identity_source",
                "serving_stack_version",
            )
        },
    }


def model_predictions(
    cases: list[dict[str, Any]], config: OfflineBenchmarkConfig
) -> tuple[list[dict[str, Any]], ProviderLedger]:
    if config.provider == "mock":
        rows: list[dict[str, Any]] = []
        for case in cases:
            for treatment in config.treatments:
                treatment_view = _treatment_view(case, treatment)
                payload = _mock_treatment_prediction(case, treatment)
                encoded_input = json.dumps(treatment_view, ensure_ascii=False)
                encoded_output = json.dumps(payload, ensure_ascii=False)
                rows.append(
                    _model_row(
                        case,
                        treatment,
                        {
                            "provider": "deterministic_mock",
                            "model": "phase1-offline-mock-v1",
                            "latency_ms": 0.0,
                            "input_tokens": len(encoded_input) // 4,
                            "output_tokens": len(encoded_output) // 4,
                            "total_tokens": (len(encoded_input) + len(encoded_output)) // 4,
                            "cost_usd": None,
                            "response_id": f"mock-{case['case_id']}-{treatment.value}",
                        },
                        payload,
                        len(json.dumps(treatment_view, ensure_ascii=False, sort_keys=True)),
                    )
                )
        return rows, ProviderLedger()
    if config.provider == "baselines":
        raise ValueError("model_predictions only supports live providers or mock")
    rows, ledger, journal_path, progress_path, inflight_path = _live_progress(cases, config)
    completed = {_prediction_key(row) for row in rows}
    budgeted = BudgetedRetryProvider(
        _provider(config.provider, config.model),
        config.provider_budget,
        ledger,
        checkpoint_path=config.output_dir / "live_provider_ledger.json",
        attempt_log_path=config.output_dir / "live_provider_attempts.jsonl",
    )
    for case in cases:
        for treatment in config.treatments:
            key = (case["case_id"], treatment.value)
            if key in completed:
                continue
            treatment_view = _treatment_view(case, treatment)
            # Provider cache accounting is a cost signal, not a controlled
            # prompt-length measurement. The preregistered D1-BM control is
            # checked on these canonical bytes before any serving-stack cache.
            prompt_chars = len(
                json.dumps(treatment_view, ensure_ascii=False, sort_keys=True)
            )
            _atomic_json(
                inflight_path,
                {
                    "plan_hash": _live_plan(cases, config)["plan_hash"],
                    "case_id": case["case_id"],
                    "treatment": treatment.value,
                },
            )
            provider_response = budgeted.structured(
                    instructions=(
                        "First reproduce the visible table state, then estimate the simulated opponent's latent type, next-action distribution, "
                        "view of Hero, and adaptation probability from only the unmasked fields. "
                        "Choose one legal Hero action. Return concise auditable fields; do not "
                        "invent hidden cards or provide hidden chain-of-thought."
                    ),
                    state=treatment_view,
                    schema_name="phase1_opponent_prediction",
                    schema=PHASE1_PREDICTION_SCHEMA,
                    validator=lambda value, legal=case["table_state"]["legal_actions"]: (
                        _validate_prediction(value, legal)
                    ),
            )
            payload = provider_response.payload
            response = asdict(provider_response)
            response.pop("payload")
            _validate_prediction(payload, case["table_state"]["legal_actions"])
            row = _model_row(case, treatment, response, payload, prompt_chars)
            _append_journal_row(journal_path, row)
            rows.append(row)
            completed.add(key)
            _atomic_json(
                progress_path,
                {
                    **_live_plan(cases, config),
                    "completed_predictions": len(rows),
                    "state": "running",
                },
            )
            inflight_path.unlink()
    return rows, ledger


def score_predictions(
    cases: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> pd.DataFrame:
    by_case = {case["case_id"]: case for case in cases}
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        case = by_case[prediction["case_id"]]
        payload = prediction["payload"]
        truth = case["ground_truth"]
        true_action = truth["action_probabilities"]
        predicted_action = payload["action_probabilities"]
        true_type = truth["active_type"]
        expected_table = case["table_state"]
        observed_table = payload["table_state"]
        basic_fields = ("street", "position", "pot_bb", "effective_stack_bb", "spr", "pot_odds")
        basic_errors = [
            float(observed_table[field] != expected_table[field])
            if field in {"street", "position"}
            else min(1.0, abs(float(observed_table[field]) - float(expected_table[field])) / max(abs(float(expected_table[field])), 1.0))
            for field in basic_fields
        ]
        legal_set_accuracy = float(
            set(observed_table["legal_actions"]) == set(expected_table["legal_actions"])
        )
        table_basic = 1.0 - (sum(basic_errors) / len(basic_errors))
        table_hand = float(observed_table["hand_class"] == expected_table["hand_class"])
        table_equity_mae = abs(float(observed_table["equity"]) - float(expected_table["equity"]))
        table_total = (table_basic + legal_set_accuracy + table_hand + (1.0 - table_equity_mae)) / 4.0
        type_brier = sum(
            (float(payload["type_probabilities"][name]) - float(name == true_type)) ** 2
            for name in PAPER_OPPONENT_TYPES
        )
        action_brier = sum(
            (float(predicted_action[name]) - float(true_action[name])) ** 2
            for name in ACTION_NAMES
        )
        true_values = _action_values(case, true_action)
        chosen = payload["recommended_action"]
        regret = max(true_values.values()) - true_values[chosen]
        rows.append(
            {
                "case_id": case["case_id"],
                "trajectory_id": case["trajectory_id"],
                "regime": case["regime"],
                "post_switch": case["post_switch"],
                "method": prediction["method"],
                "provider": prediction["provider"],
                "model": prediction["model"],
                "treatment": prediction["treatment"],
                "u_table_basic": table_basic,
                "u_table_legal_actions": legal_set_accuracy,
                "u_table_hand": table_hand,
                "u_table_equity_mae": table_equity_mae,
                "u_table_total": table_total,
                "type_brier": type_brier,
                "type_log_loss": -math.log(
                    max(1e-9, float(payload["type_probabilities"][true_type]))
                ),
                "type_correct": max(
                    payload["type_probabilities"], key=payload["type_probabilities"].get
                )
                == true_type,
                "action_brier": action_brier,
                "action_cross_entropy": -sum(
                    float(true_action[name])
                    * math.log(max(1e-9, float(predicted_action[name])))
                    for name in ACTION_NAMES
                ),
                "hero_image_mae": abs(
                    float(payload["hero_image_aggression"])
                    - float(truth["hero_image_aggression"])
                ),
                "adaptation_mae": abs(
                    float(payload["adaptation_probability"])
                    - float(truth["adaptation_probability"])
                ),
                "switch_correct": bool(payload["switch_detected"])
                == bool(truth["switch_detected"]),
                "decision_regret": regret,
                "input_tokens": prediction.get("input_tokens"),
                "output_tokens": prediction.get("output_tokens"),
                "total_tokens": prediction.get("total_tokens"),
                "latency_ms": prediction.get("latency_ms"),
                "cost_usd": prediction.get("cost_usd"),
                "prompt_chars": prediction.get("prompt_chars"),
            }
        )
    return pd.DataFrame(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _attempt_audit(path: Path, ledger: ProviderLedger, *, applicable: bool) -> dict[str, Any]:
    if not applicable:
        return {"applicable": False, "valid": True, "attempts": 0, "raw_failures": 0}
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"applicable": True, "valid": False, "attempts": 0, "raw_failures": 0}
    failures = sum(row.get("outcome") == "failed" for row in rows)
    retries = sum(bool(row.get("retry")) for row in rows)
    valid = (
        len(rows) == ledger.calls
        and failures == ledger.raw_failures
        and retries == ledger.retries
        and all(row.get("outcome") in {"succeeded", "failed"} for row in rows)
    )
    return {
        "applicable": True,
        "valid": valid,
        "path": str(path),
        "attempts": len(rows),
        "raw_failures": failures,
        "retries": retries,
    }


def _report(config: OfflineBenchmarkConfig, summary: pd.DataFrame, gate: dict[str, Any]) -> str:
    lines = [
        "# Phase 1 离线对手理解基准",
        "",
        f"- 证据类别：`{'preregistered' if config.preregistered else 'exploratory_or_smoke'}`",
        f"- Cases：`{config.case_count}`；provider：`{config.provider}`；model：`{config.model}`",
        f"- Provider gate：`{gate['valid']}`",
        (
            f"- provider attested model ID：`{', '.join(gate.get('observed_actual_models', [])) or 'n/a'}`；"
            f"identity source：`{', '.join(gate.get('observed_model_identity_sources', [])) or 'n/a'}`；"
            f"独立版本字段完整：`{gate.get('complete_model_version_attestation', False)}`"
        ),
        "- Mock 与 deterministic baseline 只验证协议和指标，不能作为真实模型能力证据。",
        "",
        "## 汇总",
        "",
        "| Method | Regime | N | Table U ↑ | Type Brier ↓ | Action Brier ↓ | Regret ↓ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['method']} | {row['regime']} | {int(row['n'])} | {row['u_table_total']:.4f} | "
            f"{row['type_brier']:.4f} | {row['action_brier']:.4f} | "
            f"{row['decision_regret']:.4f} |"
        )
    return "\n".join(lines)


def run_offline_benchmark(config: OfflineBenchmarkConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cases = generate_offline_cases(config.base_seed)[: config.case_count]
    predictions = baseline_predictions(cases)
    ledger = ProviderLedger()
    if config.provider != "baselines":
        model_rows, ledger = model_predictions(cases, config)
        predictions.extend(model_rows)
        if config.provider != "mock":
            _atomic_json(
                config.output_dir / "LIVE_PROGRESS.json",
                {
                    **_live_plan(cases, config),
                    "completed_predictions": len(model_rows),
                    "state": "completed",
                },
            )
    scores = score_predictions(cases, predictions)
    summary = scores.groupby(["method", "regime"], as_index=False).agg(
        n=("case_id", "size"),
        trajectories=("trajectory_id", "nunique"),
        u_table_basic=("u_table_basic", "mean"),
        u_table_legal_actions=("u_table_legal_actions", "mean"),
        u_table_hand=("u_table_hand", "mean"),
        u_table_equity_mae=("u_table_equity_mae", "mean"),
        u_table_total=("u_table_total", "mean"),
        type_brier=("type_brier", "mean"),
        type_log_loss=("type_log_loss", "mean"),
        type_accuracy=("type_correct", "mean"),
        action_brier=("action_brier", "mean"),
        action_cross_entropy=("action_cross_entropy", "mean"),
        hero_image_mae=("hero_image_mae", "mean"),
        adaptation_mae=("adaptation_mae", "mean"),
        switch_accuracy=("switch_correct", "mean"),
        decision_regret=("decision_regret", "mean"),
        total_tokens=("total_tokens", "sum"),
        latency_ms=("latency_ms", "sum"),
    )
    evidence_rows: list[pd.DataFrame] = []
    for (provider, model), group in scores.groupby(["provider", "model"], dropna=False):
        if {
            ReasoningTreatment.BUDGET_MATCHED_D1.value,
            ReasoningTreatment.RECURSIVE_D2.value,
        }.issubset(set(group["treatment"])) and {"fixed", "adaptive_shift"}.issubset(
            set(group["regime"])
        ):
            contrast = d2_d1bm_post_switch_contrasts(group, metric="action_brier")
            contrast.insert(0, "model", model)
            contrast.insert(0, "provider", provider)
            evidence_rows.append(contrast)
    offline_inference = (
        pd.concat(evidence_rows, ignore_index=True)
        if evidence_rows
        else pd.DataFrame()
    )
    expected_model_predictions = (
        config.case_count * len(config.treatments) if config.provider != "baselines" else 0
    )
    observed_model_predictions = sum(row["method"].startswith("llm_") for row in predictions)
    model_rows = [row for row in predictions if row["method"].startswith("llm_")]
    observed_identities = sorted({(row["provider"], row["model"]) for row in model_rows})
    attempt_audit = _attempt_audit(
        config.output_dir / "live_provider_attempts.jsonl",
        ledger,
        applicable=config.provider not in {"baselines", "mock"},
    )
    observed_actual_models = sorted(
        {str(row["actual_model"]) for row in model_rows if row.get("actual_model")}
    )
    observed_model_versions = sorted(
        {str(row["model_version"]) for row in model_rows if row.get("model_version")}
    )
    expected_identity = {
        "opencode-go": ("opencode_go", config.model),
        "codex": ("codex_exec", config.model),
        "mock": ("deterministic_mock", "phase1-offline-mock-v1"),
    }.get(config.provider)
    complete_token_accounting = (
        config.provider in {"baselines", "mock"}
        or ledger.token_observed_calls == ledger.calls
    )
    complete_cost_observability = all(
        row.get("cost_observability") in {"exact", "estimated", "unavailable"}
        for row in model_rows
    )
    actual_identity_matches = (
        config.provider in {"baselines", "mock"}
        or observed_actual_models == [config.model]
    )
    allowed_identity_sources = {
        "opencode-go": {"provider_stream", "opencode_session_export"},
        "codex": {"provider_stream", "cli_selected_model"},
    }
    observed_identity_sources = sorted(
        {str(row["model_identity_source"]) for row in model_rows if row.get("model_identity_source")}
    )
    identity_source_valid = (
        config.provider in {"baselines", "mock"}
        or (
            bool(model_rows)
            and all(
                row.get("model_identity_source") in allowed_identity_sources.get(config.provider, set())
                for row in model_rows
            )
        )
    )
    complete_model_version_attestation = (
        config.provider in {"baselines", "mock"}
        or (
            len(observed_model_versions) == 1
            and all(bool(row.get("model_version")) for row in model_rows)
        )
    )
    gate = {
        "applicable": config.provider not in {"baselines", "mock"},
        "valid": (
            observed_model_predictions == expected_model_predictions
            and ledger.unresolved_failures == 0
            and observed_identities == ([expected_identity] if expected_identity else [])
            and complete_token_accounting
            and complete_cost_observability
            and actual_identity_matches
            and identity_source_valid
            and attempt_audit["valid"]
        ),
        "provider": config.provider,
        "model": config.model,
        "expected_predictions": expected_model_predictions,
        "observed_predictions": observed_model_predictions,
        "zero_unresolved_failures": ledger.unresolved_failures == 0,
        "expected_identity": list(expected_identity) if expected_identity else None,
        "observed_identities": [list(value) for value in observed_identities],
        "observed_actual_models": observed_actual_models,
        "observed_model_versions": observed_model_versions,
        "actual_identity_matches": actual_identity_matches,
        "observed_model_identity_sources": observed_identity_sources,
        "model_identity_source_valid": identity_source_valid,
        "complete_model_version_attestation": complete_model_version_attestation,
        "complete_token_accounting": complete_token_accounting,
        "complete_cost_observability": complete_cost_observability,
        "attempt_audit": attempt_audit,
        "ledger": ledger.snapshot(),
    }
    if config.provider in {"baselines", "mock"}:
        gate["valid"] = True
    manifest_payload = {
        "protocol": "prbench-cross-model-v1",
        "configuration": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "treatments": [value.value for value in config.treatments],
            "provider_budget": asdict(config.provider_budget),
        },
        "case_hashes": [case["case_hash"] for case in cases],
        "evidence_class": "preregistered" if config.preregistered else "exploratory_or_smoke",
    }
    canonical = json.dumps(manifest_payload, ensure_ascii=False, sort_keys=True, default=str)
    manifest_payload["manifest_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    _write_jsonl(config.output_dir / "cases.jsonl.gz", cases)
    _write_jsonl(config.output_dir / "predictions.jsonl.gz", predictions)
    scores.to_csv(config.output_dir / "scores_per_case.csv", index=False)
    summary.to_csv(config.output_dir / "scores_per_model.csv", index=False)
    offline_inference.to_csv(config.output_dir / "offline_inference.csv", index=False)
    llm_scores = scores[scores["method"].str.startswith("llm_")]
    token_diagnostics = pd.DataFrame(
        columns=["case_id", "d1_budget_matched_prompt_chars", "recursive_d2_prompt_chars", "ratio"]
    )
    if not llm_scores.empty:
        token_pivot = llm_scores.pivot_table(
            index="case_id",
            columns="treatment",
            values="prompt_chars",
            aggfunc="first",
        )
        required = {
            ReasoningTreatment.BUDGET_MATCHED_D1.value,
            ReasoningTreatment.RECURSIVE_D2.value,
        }
        if required.issubset(token_pivot.columns):
            token_diagnostics = token_pivot.reset_index()[
                [
                    "case_id",
                    ReasoningTreatment.BUDGET_MATCHED_D1.value,
                    ReasoningTreatment.RECURSIVE_D2.value,
                ]
            ].rename(
                columns={
                    ReasoningTreatment.BUDGET_MATCHED_D1.value: "d1_budget_matched_prompt_chars",
                    ReasoningTreatment.RECURSIVE_D2.value: "recursive_d2_prompt_chars",
                }
            )
            denominator = token_diagnostics["recursive_d2_prompt_chars"].replace(0, float("nan"))
            token_diagnostics["ratio"] = (
                token_diagnostics["d1_budget_matched_prompt_chars"] / denominator
            )
    token_diagnostics.to_csv(config.output_dir / "budget_match_diagnostics.csv", index=False)
    observed_ratios = token_diagnostics["ratio"].dropna()
    ratio_median = float(observed_ratios.median()) if not observed_ratios.empty else None
    budget_match_valid = (
        config.provider in {"baselines", "mock"}
        or (
            ratio_median is not None
            and 0.95 <= ratio_median <= 1.05
            and bool(((observed_ratios >= 0.90) & (observed_ratios <= 1.10)).all())
        )
    )
    gate["budget_match_ratio_median"] = ratio_median
    gate["budget_match_metric"] = "canonical_prompt_chars"
    gate["budget_match_valid"] = budget_match_valid
    gate["valid"] = bool(gate["valid"] and budget_match_valid)
    (config.output_dir / "provider_ledger.json").write_text(
        json.dumps(ledger.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.output_dir / "provider_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (config.output_dir / "REPORT.zh-CN.md").write_text(
        _report(config, summary, gate), encoding="utf-8"
    )
    return {
        "manifest": manifest_payload,
        "cases": cases,
        "predictions": predictions,
        "scores_per_case": scores,
        "scores_per_model": summary,
        "offline_inference": offline_inference,
        "provider_gate": gate,
    }
