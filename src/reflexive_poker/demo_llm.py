from __future__ import annotations

import json
import random
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any

from .demo_engine import RAISE_SCALES, DemoTable
from .equity import estimate_equity
from .llm_player import DeterministicNarrativeProvider, OpenCodeGoProvider, ProviderResponse

LIVE_MODEL = "deepseek-v4-flash"


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


def _aliyun_runner(prompt: str) -> str:
    remote_script = """
import subprocess, sys, tempfile
prompt = sys.stdin.read()
with tempfile.TemporaryDirectory(prefix='poker-demo-') as directory:
    result = subprocess.run(
        ['opencode', 'run', '--pure', '--format', 'json', '--dir', directory,
         '--model', 'opencode-go/deepseek-v4-flash', prompt],
        capture_output=True, text=True, timeout=14, check=False,
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
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("aliyun_99 provider timed out after 15 seconds") from exc
    if result.returncode:
        detail = result.stderr.strip()[-500:] or "remote provider failed"
        raise RuntimeError(f"aliyun_99 provider failed: {detail}")
    return result.stdout


def provider_for(table: DemoTable):
    if table.config.provider_mode == "live_aliyun":
        return OpenCodeGoProvider(model=LIVE_MODEL, run=_aliyun_runner)
    return DeterministicNarrativeProvider(seed=table.config.seed)


def enriched_decision_state(table: DemoTable) -> dict[str, Any]:
    if table.hand is None:
        raise ValueError("no_active_hand")
    state = table.decision_state()
    hand = table.hand
    board_count = (0, 3, 4, 5)[hand.street_index]
    rng = random.Random(table.config.seed * 65537 + hand.hand_index * 257 + len(hand.actions))
    equity = estimate_equity(
        tuple(hand.holes[table.hero_seat]),
        tuple(hand.board[:board_count]),
        max(1, sum(hand.active) - 1),
        rng,
        samples=max(64, table.config.equity_samples),
    )
    pot = max(float(state["pot"]), 0.01)
    to_call = float(state["to_call"])
    opponent_actions = [item for item in hand.actions if item["seat"] != table.hero_seat]
    folds = sum(item["action"] == "fold" for item in opponent_actions)
    raises = sum(item["action"] == "raise" for item in opponent_actions)
    hero_actions = [item for item in hand.actions if item["seat"] == table.hero_seat]
    hero_raises = sum(item["action"] == "raise" for item in hero_actions)
    recent_rewards = [
        float(item["rewards"].get("hero", 0.0)) for item in table.completed_hands[-6:]
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


def decide(table: DemoTable) -> DemoDecision:
    state = enriched_decision_state(table)
    response = provider_for(table).decide(state)
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
        "riskFlags": [str(value)[:160] for value in payload.get("risk_flags", [])[:4]],
        "provider": response.provider,
        "model": response.model,
        "readOnly": table.controller == "human",
    }
    return DemoDecision(action, raise_scale, advice, response)


def reflect_and_patch(table: DemoTable) -> DemoReflection:
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
        if item["seat"] == table.hero_seat
    ]
    reward = float(table.hand.rewards[table.hero_seat])
    state = {
        "hand_index": hand_index,
        "reward": reward,
        "decisions": decisions,
        "public_actions": table.hand.actions,
        "showdown": table.hand.showdown,
        "strategy": table.strategy,
    }
    response = provider_for(table).reflect(state)
    payload = response.payload
    current = table.strategy
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
        "outcomeSummary": str(payload.get("outcome_summary", ""))[:300],
        "decisionReview": str(payload.get("decision_review", ""))[:500],
        "whatWorked": [str(value)[:160] for value in payload.get("what_worked", [])[:4]],
        "whatFailed": [str(value)[:160] for value in payload.get("what_failed", [])[:4]],
        "strategyAdjustment": str(payload.get("strategy_adjustment", ""))[:300],
        "provider": response.provider,
        "model": response.model,
        "rawBytes": len(json.dumps(payload, ensure_ascii=False)),
    }
    return DemoReflection(reflection, patch, response)
