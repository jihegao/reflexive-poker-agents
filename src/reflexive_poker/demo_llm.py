from __future__ import annotations

import json
import random
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any

from .demo_engine import DEFAULT_LLM_MODEL, RAISE_SCALES, DemoTable
from .equity import estimate_equity
from .llm_player import DeterministicNarrativeProvider, OpenCodeGoProvider, ProviderResponse

LLM_TIMEOUT_SECONDS = 60
MODEL_LIST_TIMEOUT_SECONDS = 10
FALLBACK_OPENCODE_GO_MODELS = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "glm-5.1",
    "glm-5.2",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "minimax-m2.7",
    "minimax-m3",
    "qwen3.6-plus",
    "qwen3.7-max",
    "qwen3.7-plus",
)


@dataclass(frozen=True)
class DemoDecision:
    action: str
    raise_scale: float
    advice: dict[str, Any]
    response: ProviderResponse


@dataclass(frozen=True)
class DemoReflection:
    reflection: dict[str, Any]
    patch: dict[str, Any]
    response: ProviderResponse


def list_opencode_go_models() -> tuple[str, ...]:
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "aliyun_99",
                "opencode",
                "models",
                "opencode-go",
            ],
            capture_output=True,
            text=True,
            timeout=MODEL_LIST_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("opencode-go model discovery timed out") from exc
    if result.returncode:
        detail = result.stderr.strip()[-500:] or "remote model discovery failed"
        raise RuntimeError(detail)
    models = tuple(
        line.removeprefix("opencode-go/").strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("opencode-go/")
    )
    if not models:
        raise RuntimeError("opencode-go model discovery returned no models")
    return tuple(dict.fromkeys(models))


def _aliyun_runner(prompt: str, model: str = DEFAULT_LLM_MODEL) -> str:
    model_id = f"opencode-go/{model.removeprefix('opencode-go/')}"
    remote_script = f"""
import subprocess, sys, tempfile
prompt = sys.stdin.read()
with tempfile.TemporaryDirectory(prefix='poker-demo-') as directory:
    result = subprocess.run(
        ['opencode', 'run', '--pure', '--format', 'json', '--dir', directory,
         '--model', {model_id!r}, prompt],
        capture_output=True, text=True, timeout=60, check=False,
    )
if result.returncode:
    sys.stderr.write(result.stderr[-1000:])
    raise SystemExit(result.returncode)
sys.stdout.write(result.stdout)
""".strip()
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "aliyun_99",
                "python3",
                "-c",
                shlex.quote(remote_script),
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=LLM_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("aliyun_99 provider timed out after 60 seconds") from exc
    if result.returncode:
        detail = result.stderr.strip()[-500:] or "remote provider failed"
        raise RuntimeError(f"aliyun_99 provider failed: {detail}")
    return result.stdout


def provider_for(table: DemoTable, actor: int = 0):
    if table.config.provider_mode == "live_aliyun":
        model = table.model_for(actor)
        return OpenCodeGoProvider(
            model=model,
            run=lambda prompt: _aliyun_runner(prompt, model),
        )
    return DeterministicNarrativeProvider(seed=table.config.seed)


def enriched_decision_state(table: DemoTable, actor: int | None = None) -> dict[str, Any]:
    if table.hand is None:
        raise ValueError("no_active_hand")
    actor = table.hero_seat if actor is None else actor
    state = table.decision_state(actor)
    hand = table.hand
    board_count = (0, 3, 4, 5)[hand.street_index]
    rng = random.Random(table.config.seed * 65537 + hand.hand_index * 257 + len(hand.actions))
    equity = estimate_equity(
        tuple(hand.holes[actor]),
        tuple(hand.board[:board_count]),
        max(1, sum(hand.active) - 1),
        rng,
        samples=max(64, table.config.equity_samples),
    )
    pot = max(float(state["pot"]), 0.01)
    to_call = float(state["to_call"])
    opponent_actions = [item for item in hand.actions if item["seat"] != actor]
    folds = sum(item["action"] == "fold" for item in opponent_actions)
    raises = sum(item["action"] == "raise" for item in opponent_actions)
    hero_actions = [item for item in hand.actions if item["seat"] == actor]
    hero_raises = sum(item["action"] == "raise" for item in hero_actions)
    recent_rewards = [
        float(item["rewards"].get(table.names[actor], 0.0))
        for item in table.completed_hands[-6:]
    ]
    state.update(
        {
            "equity_estimate": round(equity, 4),
            "pot_odds": round(to_call / max(pot + to_call, 0.01), 4),
            "predicted_all_fold": round((folds + 1) / (len(opponent_actions) + 3), 4),
            "recent_reward_mean": round(
                sum(recent_rewards) / max(1, len(recent_rewards)), 4
            ),
            "self_image_estimate": round((hero_raises + 1) / (len(hero_actions) + 2), 4),
            "opponent_aggression_mean": round(
                (raises + 1) / (len(opponent_actions) + 2), 4
            ),
            "opponent_fold_mean": round((folds + 1) / (len(opponent_actions) + 2), 4),
        }
    )
    return state


def decide(table: DemoTable, actor: int | None = None) -> DemoDecision:
    actor = table.hero_seat if actor is None else actor
    state = enriched_decision_state(table, actor)
    response = provider_for(table, actor).decide(state)
    payload = response.payload
    action = str(payload.get("action"))
    if action not in state["legal_actions"]:
        raise ValueError("provider_returned_illegal_action")
    requested_scale = float(payload.get("raise_scale", 0.5))
    raise_scale = min(RAISE_SCALES, key=lambda value: abs(value - requested_scale))
    advice = {
        "action": action,
        "raiseScale": raise_scale,
        "confidence": round(float(payload.get("confidence", 0.0)), 3),
        "summary": str(payload.get("situation_summary", ""))[:300],
        "rationale": str(payload.get("rationale", ""))[:500],
        "selfModel": str(payload.get("self_model", ""))[:300],
        "opponentModel": str(payload.get("opponent_model", ""))[:300],
        "riskFlags": [str(value)[:160] for value in payload.get("risk_flags", [])[:4]],
        "nextStep": str(payload.get("next_step", ""))[:300],
        "provider": response.provider,
        "model": response.model,
        "readOnly": table.controller_for(actor) == "human",
        "state": {
            "street": str(state["street"]),
            "board": list(state["community_cards"]),
            "potBb": round(float(state["pot"]), 2),
            "toCallBb": round(float(state["to_call"]), 2),
            "stackBb": round(float(state["stack"]), 2),
            "activePlayers": int(state["active_players"]),
            "equityEstimate": round(float(state["equity_estimate"]), 4),
            "potOdds": round(float(state["pot_odds"]), 4),
            "predictedAllFold": round(float(state["predicted_all_fold"]), 4),
            "opponentAggressionMean": round(
                float(state["opponent_aggression_mean"]), 4
            ),
            "opponentFoldMean": round(float(state["opponent_fold_mean"]), 4),
        },
    }
    return DemoDecision(action, raise_scale, advice, response)


def reflect_and_patch(table: DemoTable, actor: int = 0) -> DemoReflection:
    if table.hand is None or not table.hand.complete:
        raise ValueError("hand_not_complete")
    hand_index = table.hand.hand_index
    decisions = [
        {
            "action": item["action"],
            "confidence": 0.65,
            "street": item["street"],
        }
        for item in table.hand.actions
        if item["seat"] == actor and item.get("controller") == "llm_closed_loop"
    ]
    reward = float(table.hand.rewards[actor])
    current = table.strategy_for(actor)
    state = {
        "hand_index": hand_index,
        "controlled_seat": actor,
        "player_name": table.names[actor],
        "base_persona": current["basePersona"],
        "reward": reward,
        "decisions": decisions,
        "public_actions": table.hand.actions,
        "showdown": table.hand.showdown,
        "strategy": current,
        "recent_reflections": table.reflection_memory_for(actor)[
            -int(current["memoryHands"]) :
        ],
    }
    response = provider_for(table, actor).reflect(state)
    payload = response.payload
    direction = 1.0 if reward > 0 else -1.0 if reward < 0 else 0.0
    aggression = max(-0.20, min(0.20, float(current["aggressionBias"]) + 0.02 * direction))
    risk = max(-0.10, min(0.10, float(current["riskMarginDelta"]) - 0.01 * direction))
    changes = {
        "aggressionBias": round(aggression, 3),
        "riskMarginDelta": round(risk, 3),
        "notes": [str(payload.get("strategy_adjustment", "保持当前策略"))[:120]],
    }
    patch = {
        "patchId": f"patch_{uuid.uuid4().hex[:10]}",
        "baseStrategyVersion": int(current["version"]),
        "author": "llm_closed_loop",
        "reason": str(payload.get("strategy_adjustment", "post-hand reflection"))[:240],
        "changes": changes,
    }
    reflection = {
        "handIndex": hand_index,
        "seat": actor,
        "basePersona": current["basePersona"],
        "outcomeSummary": str(payload.get("outcome_summary", ""))[:300],
        "decisionReview": str(payload.get("decision_review", ""))[:500],
        "whatWorked": [str(value)[:160] for value in payload.get("what_worked", [])[:4]],
        "whatFailed": [str(value)[:160] for value in payload.get("what_failed", [])[:4]],
        "beliefUpdates": [
            str(value)[:160] for value in payload.get("belief_updates", [])[:4]
        ],
        "strategyAdjustment": str(payload.get("strategy_adjustment", ""))[:300],
        "calibrationNote": str(payload.get("calibration_note", ""))[:300],
        "confidenceAfter": round(float(payload.get("confidence_after", 0.0)), 3),
        "provider": response.provider,
        "model": response.model,
        "rawBytes": len(json.dumps(payload, ensure_ascii=False)),
    }
    return DemoReflection(reflection, patch, response)
