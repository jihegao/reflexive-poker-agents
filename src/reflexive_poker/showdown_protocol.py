from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_ARTIFACTS = {
    "protocol.yaml",
    "model_manifest.json",
    "prompt_and_gto_hashes.json",
    "provider_gate.json",
    "hands.jsonl",
    "decisions.jsonl",
    "reflections.jsonl",
    "tool_calls.jsonl",
    "cost_ledger.csv",
    "summary.json",
    "public_replay.jsonl",
    "private_audit.jsonl",
}

EXPECTED_MODELS = {
    ("opencode-go", "deepseek-v4-flash"),
    ("codex", "gpt-5.6-luna"),
}

EXPECTED_R2_CONDITIONS = {"raw", "masked_gto", "gto"}
EXPECTED_R3_CONDITIONS = {
    "tools_off",
    "reflection_only",
    "reflection_plus_simulation",
}
EXPECTED_RAISE_SIZES = {"0.5_pot", "0.75_pot", "1.0_pot", "all_in"}


class ShowdownProtocolError(ValueError):
    """Raised when a showdown protocol violates the frozen experiment contract."""


def load_showdown_protocol(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ShowdownProtocolError(f"Protocol config does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ShowdownProtocolError(f"Protocol config is not valid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise ShowdownProtocolError("Protocol root must be a mapping")
    return payload


def protocol_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _mapping(payload: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be a mapping")
        return {}
    return value


def _require_true(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
    prefix: str,
    errors: list[str],
) -> None:
    for key in keys:
        if mapping.get(key) is not True:
            errors.append(f"{prefix}.{key} must be true")


def validate_showdown_protocol(payload: dict[str, Any], *, formal: bool = False) -> list[str]:
    errors: list[str] = []

    protocol = _mapping(payload, "protocol", errors)
    if protocol.get("id") != "deepseek-v4-flash-vs-gpt-5.6-luna-v1":
        errors.append("protocol.id must be deepseek-v4-flash-vs-gpt-5.6-luna-v1")
    if formal and protocol.get("status") != "frozen":
        errors.append("protocol.status must be frozen for a formal run")

    models = payload.get("models")
    if not isinstance(models, list) or len(models) != 2:
        errors.append("models must contain exactly two serving systems")
        models = []
    identities = {
        (str(item.get("provider")), str(item.get("requested_model")))
        for item in models
        if isinstance(item, dict)
    }
    if identities != EXPECTED_MODELS:
        errors.append("models must exactly match DeepSeek V4 Flash and GPT-5.6 Luna providers")

    rules = _mapping(payload, "table_rules", errors)
    if rules.get("rules_engine") != "pokerkit":
        errors.append("table_rules.rules_engine must be pokerkit")
    if rules.get("starting_stack_bb") != 100:
        errors.append("table_rules.starting_stack_bb must equal 100")
    if rules.get("ante_bb") != 0.0:
        errors.append("table_rules.ante_bb must equal 0.0")
    if rules.get("rake") is not False:
        errors.append("table_rules.rake must be false")
    if set(rules.get("allowed_raise_sizes", [])) != EXPECTED_RAISE_SIZES:
        errors.append("table_rules.allowed_raise_sizes must match the frozen discrete sizes")
    _require_true(
        rules,
        (
            "model_returns_action_intent_only",
            "engine_owns_dealing_legality_side_pots_showdown_and_chip_conservation",
        ),
        "table_rules",
        errors,
    )

    pairing = _mapping(payload, "pairing", errors)
    _require_true(
        pairing,
        ("identical_deck_sequence", "seat_mirroring", "mirrored_branches_isolated"),
        "pairing",
        errors,
    )
    if pairing.get("statistical_unit") != "paired_deck_block":
        errors.append("pairing.statistical_unit must be paired_deck_block")

    gate = _mapping(payload, "provider_gate", errors)
    _require_true(
        gate,
        (
            "require_exact_model_identity",
            "require_zero_unresolved_timeouts",
            "require_zero_unresolved_provider_errors",
            "require_zero_fallbacks",
            "require_zero_wrong_model_responses",
            "require_zero_version_drift",
            "invalidate_entire_paired_block_on_failure",
        ),
        "provider_gate",
        errors,
    )
    if gate.get("rule_bot_takeover_allowed") is not False:
        errors.append("provider_gate.rule_bot_takeover_allowed must be false")

    rounds = _mapping(payload, "rounds", errors)
    round_1 = _mapping(rounds, "round_1_raw_heads_up", errors)
    if round_1.get("players") != 2 or round_1.get("tools") != []:
        errors.append("round 1 must be heads-up with no tools")
    if round_1.get("external_reference") != "none":
        errors.append("round 1 must not include an external reference")
    if _mapping(round_1, "pilot", errors).get("mirrored_hands") != 100:
        errors.append("round 1 pilot must contain 100 mirrored hands")
    if _mapping(round_1, "confirmatory", errors).get("mirrored_hands") != 1000:
        errors.append("round 1 confirmatory run must contain at least 1000 mirrored hands")

    round_2 = _mapping(rounds, "round_2_frozen_gto_reference", errors)
    if set(round_2.get("conditions", [])) != EXPECTED_R2_CONDITIONS:
        errors.append("round 2 conditions must be raw, masked_gto and gto")
    if round_2.get("budget_matched") is not True:
        errors.append("round 2 must use a budget-matched control")
    if round_2.get("realtime_retrieval_allowed") is not False:
        errors.append("round 2 must prohibit realtime retrieval")
    if round_2.get("solver_tool_allowed") is not False:
        errors.append("round 2 must prohibit a live solver tool")
    reference_pack = _mapping(round_2, "reference_pack", errors)
    if formal:
        if reference_pack.get("status") != "frozen":
            errors.append("round 2 GTO reference pack must be frozen for a formal run")
        if not reference_pack.get("path"):
            errors.append("round 2 GTO reference pack path is required for a formal run")
        sha256 = reference_pack.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            errors.append("round 2 GTO reference pack sha256 must be a 64-character hash")

    round_3 = _mapping(rounds, "round_3_six_max_tools", errors)
    if round_3.get("players") != 6 or round_3.get("players_per_camp") != 3:
        errors.append("round 3 must be a 3-vs-3 six-max game")
    if round_3.get("agent_scheduler") != "mesa" or round_3.get("rules_engine") != "pokerkit":
        errors.append("round 3 must use Mesa scheduling and PokerKit rules")
    if round_3.get("all_camp_seat_assignments") != 20:
        errors.append("round 3 must enumerate all C(6,3)=20 camp seat assignments")
    if set(round_3.get("shadow_conditions", [])) != EXPECTED_R3_CONDITIONS:
        errors.append("round 3 shadow conditions must match the frozen three-way ablation")
    if round_3.get("shadow_fork_from_identical_checkpoint") is not True:
        errors.append("round 3 shadow conditions must fork from an identical checkpoint")
    tools = _mapping(round_3, "tools", errors)
    simulation = _mapping(tools, "equity_simulate", errors)
    if simulation.get("rollouts_per_call") != 5000:
        errors.append("equity_simulate must use 5000 rollouts per call")
    if simulation.get("max_calls_per_decision") != 1:
        errors.append("equity_simulate must allow at most one call per decision")
    if _mapping(round_3, "pilot", errors).get("total_hands") != 800:
        errors.append("round 3 pilot must contain 800 hands")

    artifacts = payload.get("required_artifacts")
    if not isinstance(artifacts, list):
        errors.append("required_artifacts must be a list")
    else:
        missing = sorted(REQUIRED_ARTIFACTS - set(artifacts))
        if missing:
            errors.append(f"required_artifacts is missing: {missing}")

    winner_policy = _mapping(payload, "winner_policy", errors)
    _require_true(
        winner_policy,
        (
            "report_paired_bb_per_100_with_95_percent_interval",
            "interval_crosses_zero_means_tie",
            "composite_ai_iq_score_prohibited",
        ),
        "winner_policy",
        errors,
    )

    return errors


def assert_valid_showdown_protocol(payload: dict[str, Any], *, formal: bool = False) -> None:
    errors = validate_showdown_protocol(payload, formal=formal)
    if errors:
        raise ShowdownProtocolError("; ".join(errors))
