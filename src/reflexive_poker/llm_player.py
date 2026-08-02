from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .agents import AgentStyle, PokerAgent
from .cards import cards_to_str
from .depth import AdaptiveDepthController
from .equity import estimate_equity
from .models import ActionEvent, ActionType, Decision, DecisionContext, HandRecord

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action",
        "raise_scale",
        "confidence",
        "situation_summary",
        "rationale",
        "self_model",
        "opponent_model",
        "risk_flags",
        "next_step",
    ],
    "properties": {
        "action": {"type": "string", "enum": ["fold", "check_call", "raise"]},
        "raise_scale": {"type": "number", "minimum": 0.25, "maximum": 1.25},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "situation_summary": {"type": "string"},
        "rationale": {"type": "string"},
        "self_model": {"type": "string"},
        "opponent_model": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "next_step": {"type": "string"},
    },
}

REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "outcome_summary",
        "decision_review",
        "what_worked",
        "what_failed",
        "belief_updates",
        "strategy_adjustment",
        "calibration_note",
        "confidence_after",
    ],
    "properties": {
        "outcome_summary": {"type": "string"},
        "decision_review": {"type": "string"},
        "what_worked": {"type": "array", "items": {"type": "string"}},
        "what_failed": {"type": "array", "items": {"type": "string"}},
        "belief_updates": {"type": "array", "items": {"type": "string"}},
        "strategy_adjustment": {"type": "string"},
        "calibration_note": {"type": "string"},
        "confidence_after": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


@dataclass(frozen=True)
class ProviderResponse:
    payload: dict[str, Any]
    provider: str
    model: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    response_id: str | None = None


class LLMProvider(Protocol):
    name: str
    model: str

    def decide(self, state: dict[str, Any]) -> ProviderResponse: ...

    def reflect(self, state: dict[str, Any]) -> ProviderResponse: ...

    def structured(
        self,
        *,
        instructions: str,
        state: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> ProviderResponse: ...


class OpenAIResponsesProvider:
    """OpenAI Responses API adapter using strict Structured Outputs.

    The prompt asks for concise, audit-friendly rationale. It intentionally does not ask
    for hidden chain-of-thought. The SDK import is lazy so rule-only experiments do not
    require the optional dependency.
    """

    name = "openai_responses"

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: str | None = None,
        max_output_tokens: int = 700,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI  # type: ignore
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "Install the optional OpenAI SDK with `pip install -e '.[llm]'`."
                ) from exc
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is required for the real LLM provider.")
            client = OpenAI(api_key=key)
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _usage(response: Any) -> tuple[int | None, int | None, int | None]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None, None
        return (
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            getattr(usage, "total_tokens", None),
        )

    def _call(
        self,
        *,
        instructions: str,
        state: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        started = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=json.dumps(state, ensure_ascii=False, sort_keys=True),
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
            max_output_tokens=self.max_output_tokens,
        )
        latency_ms = 1000.0 * (time.perf_counter() - started)
        payload = json.loads(response.output_text)
        input_tokens, output_tokens, total_tokens = self._usage(response)
        return ProviderResponse(
            payload=payload,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            response_id=getattr(response, "id", None),
        )

    def decide(self, state: dict[str, Any]) -> ProviderResponse:
        legal = ", ".join(state["legal_actions"])
        return self._call(
            instructions=(
                "You are a bounded Texas Hold'em decision agent in a reproducible simulator. "
                f"Choose exactly one legal action from: {legal}. Use the supplied equity and pot "
                "odds rather than inventing card probabilities. Return a concise audit rationale, "
                "assumptions, opponent model, self-model, risk flags, and next-step plan. A "
                "raise_scale of 1.25 means all-in. Do not "
                "produce hidden chain-of-thought or long private deliberation."
            ),
            state=state,
            schema_name="poker_decision",
            schema=DECISION_SCHEMA,
        )

    def reflect(self, state: dict[str, Any]) -> ProviderResponse:
        return self._call(
            instructions=(
                "Review one completed simulated poker hand. Produce a concise post-hand audit: "
                "outcome, decision quality, what worked, what failed, belief updates, strategy "
                "adjustment, and confidence calibration. Do not invent unseen hole cards and do "
                "not provide hidden chain-of-thought."
            ),
            state=state,
            schema_name="poker_reflection",
            schema=REFLECTION_SCHEMA,
        )

    def structured(
        self,
        *,
        instructions: str,
        state: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        return self._call(
            instructions=instructions,
            state=state,
            schema_name=schema_name,
            schema=schema,
        )


class OpenCodeGoProvider:
    """OpenCode Go adapter that delegates authenticated calls to the local CLI."""

    name = "opencode_go"

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        command: str = "opencode",
        run: Any | None = None,
    ) -> None:
        self.model = model
        self.command = command
        self.run = run or self._run

    def _run(self, prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="reflexive-poker-opencode-") as directory:
            try:
                completed = subprocess.run(
                    [
                        self.command,
                        "run",
                        "--pure",
                        "--format",
                        "json",
                        "--dir",
                        directory,
                        "--model",
                        f"opencode-go/{self.model}",
                        prompt,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "`opencode` CLI is required for the opencode-go provider."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("OpenCode Go request timed out after 180 seconds.") from exc
        if completed.returncode != 0:
            raise RuntimeError(f"OpenCode Go request failed: {completed.stderr[-500:]}")
        return completed.stdout

    @staticmethod
    def _json_output(output: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        for start, character in enumerate(output):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(output[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise ValueError("OpenCode Go response did not contain a JSON object.")

    @classmethod
    def _result(
        cls, output: str
    ) -> tuple[dict[str, Any], dict[str, int | None], float | None, str | None]:
        final_text: str | None = None
        response_id: str | None = None
        cost_usd: float | None = None
        usage: dict[str, int | None] = {"input": None, "output": None, "total": None}
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            response_id = event.get("sessionID") or response_id
            if event.get("type") == "text":
                part = event.get("part", {})
                if isinstance(part.get("text"), str):
                    final_text = part["text"]
            if event.get("type") == "step_finish":
                part = event.get("part", {})
                tokens = part.get("tokens", {})
                usage = {
                    "input": tokens.get("input"),
                    "output": tokens.get("output"),
                    "total": tokens.get("total"),
                }
                reported_cost = part.get("cost")
                if isinstance(reported_cost, int | float):
                    cost_usd = float(reported_cost)
        if final_text is None:
            return cls._json_output(output), usage, cost_usd, response_id
        payload = cls._json_output(final_text)
        return payload, usage, cost_usd, response_id

    def _call(
        self,
        *,
        instructions: str,
        state: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        prompt = "\n\n".join(
            [
                instructions,
                "Return exactly one JSON object and no Markdown or prose.",
                f"JSON schema name: {schema_name}",
                f"JSON schema: {json.dumps(schema, ensure_ascii=False, sort_keys=True)}",
                f"Simulator state: {json.dumps(state, ensure_ascii=False, sort_keys=True)}",
            ]
        )
        started = time.perf_counter()
        output = self.run(prompt)
        latency_ms = 1000.0 * (time.perf_counter() - started)
        payload, usage, cost_usd, response_id = self._result(output)
        return ProviderResponse(
            payload=payload,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=usage["input"],
            output_tokens=usage["output"],
            total_tokens=usage["total"],
            cost_usd=cost_usd,
            response_id=response_id,
        )

    def decide(self, state: dict[str, Any]) -> ProviderResponse:
        legal = ", ".join(state["legal_actions"])
        return self._call(
            instructions=(
                "You are a bounded Texas Hold'em decision agent in a reproducible simulator. "
                f"Choose exactly one legal action from: {legal}. Use the supplied equity and pot "
                "odds rather than inventing card probabilities. A raise_scale of 1.25 means "
                "all-in. Return concise audit fields only."
            ),
            state=state,
            schema_name="poker_decision",
            schema=DECISION_SCHEMA,
        )

    def reflect(self, state: dict[str, Any]) -> ProviderResponse:
        return self._call(
            instructions=(
                "Review one completed simulated poker hand. Return a concise audit of outcome, "
                "decision quality, belief updates, strategy adjustment, and confidence calibration. "
                "Do not invent unseen hole cards."
            ),
            state=state,
            schema_name="poker_reflection",
            schema=REFLECTION_SCHEMA,
        )

    def structured(
        self,
        *,
        instructions: str,
        state: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        return self._call(
            instructions=instructions,
            state=state,
            schema_name=schema_name,
            schema=schema,
        )


class CodexProvider:
    """Codex CLI provider using the locally authenticated account and JSON Schema output."""

    name = "codex_exec"

    def __init__(
        self,
        model: str = "current",
        command: str = "codex",
        run: Any | None = None,
    ) -> None:
        self.model = model
        self.command = command
        self.run = run or self._run

    def _run(self, prompt: str, schema: dict[str, Any]) -> str:
        with tempfile.TemporaryDirectory(prefix="reflexive-poker-codex-") as directory:
            schema_path = Path(directory) / "output-schema.json"
            schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
            command = [
                self.command,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--cd",
                directory,
                "--json",
                "--color",
                "never",
                "--output-schema",
                str(schema_path),
            ]
            if self.model != "current":
                command.extend(["--model", self.model])
            command.append(prompt)
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("`codex` CLI is required for the codex provider.") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Codex request timed out after 300 seconds.") from exc
        if completed.returncode != 0:
            raise RuntimeError(f"Codex request failed: {completed.stderr[-500:]}")
        return completed.stdout

    @staticmethod
    def _result(output: str) -> tuple[dict[str, Any], dict[str, int | None]]:
        final_message: str | None = None
        usage: dict[str, int | None] = {"input": None, "output": None, "total": None}
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                    final_message = item["text"]
            if event.get("type") == "turn.completed":
                reported = event.get("usage", {})
                input_tokens = reported.get("input_tokens")
                output_tokens = reported.get("output_tokens")
                usage = {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": (
                        input_tokens + output_tokens
                        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
                        else None
                    ),
                }
        if final_message is None:
            raise ValueError("Codex response did not include a final agent message.")
        return json.loads(final_message), usage

    def _call(
        self,
        *,
        instructions: str,
        state: dict[str, Any],
        schema: dict[str, Any],
    ) -> ProviderResponse:
        prompt = "\n\n".join(
            [
                instructions,
                "Return only the JSON object required by the supplied output schema.",
                f"Simulator state: {json.dumps(state, ensure_ascii=False, sort_keys=True)}",
            ]
        )
        started = time.perf_counter()
        output = self.run(prompt, schema)
        latency_ms = 1000.0 * (time.perf_counter() - started)
        payload, usage = self._result(output)
        return ProviderResponse(
            payload=payload,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=usage["input"],
            output_tokens=usage["output"],
            total_tokens=usage["total"],
        )

    def decide(self, state: dict[str, Any]) -> ProviderResponse:
        legal = ", ".join(state["legal_actions"])
        return self._call(
            instructions=(
                "You are a bounded Texas Hold'em decision agent in a reproducible simulator. "
                f"Choose exactly one legal action from: {legal}. Use the supplied equity and pot "
                "odds rather than inventing card probabilities. A raise_scale of 1.25 means "
                "all-in. Return concise audit fields only."
            ),
            state=state,
            schema=DECISION_SCHEMA,
        )

    def reflect(self, state: dict[str, Any]) -> ProviderResponse:
        return self._call(
            instructions=(
                "Review one completed simulated poker hand. Return a concise audit of outcome, "
                "decision quality, belief updates, strategy adjustment, and confidence calibration. "
                "Do not invent unseen hole cards."
            ),
            state=state,
            schema=REFLECTION_SCHEMA,
        )

    def structured(
        self,
        *,
        instructions: str,
        state: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        del schema_name
        return self._call(instructions=instructions, state=state, schema=schema)


class DeterministicNarrativeProvider:
    """Offline integration provider that mimics the structured LLM contract.

    This provider is deliberately labelled as a mock. It is useful for testing logging,
    validation, replay, and evaluation plumbing without making claims about LLM capability.
    """

    name = "deterministic_mock"
    model = "mock-narrative-v1"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _rng(self, state: dict[str, Any], salt: int) -> random.Random:
        hand = int(state.get("hand_index", 0))
        street = str(state.get("street", "none"))
        street_value = sum(ord(char) for char in street)
        return random.Random(self.seed * 1000003 + hand * 9176 + street_value * 31 + salt)

    def decide(self, state: dict[str, Any]) -> ProviderResponse:
        started = time.perf_counter()
        equity = float(state["equity_estimate"])
        pot_odds = float(state["pot_odds"])
        fold_prob = float(state.get("predicted_all_fold", 0.5))
        to_call = float(state["to_call"])
        legal = set(state["legal_actions"])
        rng = self._rng(state, 11)

        if to_call > 0:
            if "raise" in legal and (equity >= 0.72 or (fold_prob >= 0.62 and equity >= 0.38)):
                action = "raise"
            elif "check_call" in legal and equity + 0.035 >= pot_odds:
                action = "check_call"
            else:
                action = "fold"
        else:
            pressure = 0.04 if fold_prob > 0.55 else 0.0
            threshold = 0.58 - pressure
            if "raise" in legal and (equity >= threshold or rng.random() < 0.025 * fold_prob):
                action = "raise"
            else:
                action = "check_call"
        if action not in legal:
            action = "check_call" if "check_call" in legal else min(legal)

        margin = equity - pot_odds
        confidence = min(0.94, max(0.51, 0.62 + abs(margin) * 0.42))
        risks: list[str] = []
        if 0.43 <= equity <= 0.58:
            risks.append("边缘牌力，决策对对手范围假设敏感")
        if fold_prob < 0.30 and action == "raise":
            risks.append("弃牌率预测偏低，诈唬收益有限")
        if state.get("recent_reward_mean", 0.0) < -2.0:
            risks.append("近期亏损可能诱发过度补偿")

        payload = {
            "action": action,
            "raise_scale": 0.78 if equity > 0.76 else 0.52,
            "confidence": confidence,
            "situation_summary": (
                f"{state['street']}阶段，牌力估计{equity:.2f}，底池赔率{pot_odds:.2f}，"
                f"预测全部弃牌概率{fold_prob:.2f}。"
            ),
            "rationale": (
                f"选择{action}：比较牌力与底池赔率，并将预测弃牌率作为加注的附加收益；"
                "不假设未知底牌。"
            ),
            "self_model": (
                f"近期平均收益{state.get('recent_reward_mean', 0.0):.2f}；"
                f"当前公共激进度估计{state.get('self_image_estimate', 0.5):.2f}。"
            ),
            "opponent_model": (
                f"对手平均激进度{state.get('opponent_aggression_mean', 0.5):.2f}，"
                f"平均弃牌倾向{state.get('opponent_fold_mean', 0.5):.2f}。"
            ),
            "risk_flags": risks,
            "next_step": "观察本街响应，并在下一决策点更新对手弃牌倾向与自身形象。",
        }
        latency_ms = 1000.0 * (time.perf_counter() - started)
        text = json.dumps(payload, ensure_ascii=False)
        return ProviderResponse(
            payload=payload,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=len(json.dumps(state, ensure_ascii=False)) // 4,
            output_tokens=len(text) // 4,
            total_tokens=(len(json.dumps(state, ensure_ascii=False)) + len(text)) // 4,
            response_id=f"mock-decision-{state['hand_index']}-{state['street']}",
        )

    def reflect(self, state: dict[str, Any]) -> ProviderResponse:
        started = time.perf_counter()
        reward = float(state["reward"])
        decisions = state["decisions"]
        avg_confidence = (
            sum(float(decision["confidence"]) for decision in decisions) / len(decisions)
            if decisions
            else 0.5
        )
        positive = reward >= 0
        payload = {
            "outcome_summary": (
                f"本手净收益{reward:+.2f}筹码；"
                f"共记录{len(decisions)}个决策点，平均置信度{avg_confidence:.2f}。"
            ),
            "decision_review": (
                "决策遵循牌力、底池赔率和预测弃牌率的结构化比较。"
                if positive
                else "结果为负，需要区分牌运损失与范围或弃牌率估计偏差。"
            ),
            "what_worked": [
                "行动均满足合法动作约束",
                "使用已提供的牌力估计而非虚构未知信息",
            ],
            "what_failed": [] if positive else ["需要重新校准边缘牌的继续范围"],
            "belief_updates": [
                f"将本手对手公开响应纳入下一手的弃牌率估计，当前收益信号为{reward:+.2f}。"
            ],
            "strategy_adjustment": (
                "保持当前价值下注范围，并避免因单手胜负过度调整。"
                if positive
                else "略微收紧低置信度跟注，并要求更高的诈唬弃牌率阈值。"
            ),
            "calibration_note": (
                "盈利结果与置信度方向一致。"
                if positive and avg_confidence >= 0.6
                else "单手结果不足以验证置信度；继续累积样本。"
            ),
            "confidence_after": min(
                0.90, max(0.45, avg_confidence + (0.02 if positive else -0.03))
            ),
        }
        latency_ms = 1000.0 * (time.perf_counter() - started)
        text = json.dumps(payload, ensure_ascii=False)
        return ProviderResponse(
            payload=payload,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=len(json.dumps(state, ensure_ascii=False)) // 4,
            output_tokens=len(text) // 4,
            total_tokens=(len(json.dumps(state, ensure_ascii=False)) + len(text)) // 4,
            response_id=f"mock-reflection-{state['hand_index']}",
        )

    def structured(
        self,
        *,
        instructions: str,
        state: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        del instructions, schema
        if schema_name != "phase1_closed_loop_decision":
            raise ValueError(f"deterministic mock does not implement schema {schema_name}")
        base = self.decide(state)
        bounded = state.get("bounded_opponent_model", {})
        type_probabilities = bounded.get("strategy_type")
        if not isinstance(type_probabilities, dict):
            names = ("rock", "tag", "lag", "calling_station", "myopic", "adaptive")
            type_probabilities = {name: 1.0 / len(names) for name in names}
        action_probabilities = bounded.get("action_prediction")
        if not isinstance(action_probabilities, dict):
            action_probabilities = {name: 1.0 / 3.0 for name in ("fold", "check_call", "raise")}
        hero_image = bounded.get("opponent_view_of_hero")
        if not isinstance(hero_image, int | float):
            hero_image = 0.5
        adjustment = bounded.get("anticipated_adjustment")
        if not isinstance(adjustment, int | float):
            adjustment = 0.0
        payload = {
            **base.payload,
            "opponent_state": {
                "type_probabilities": type_probabilities,
                "action_probabilities": action_probabilities,
                "hero_image_aggression": float(hero_image),
                "adaptation_probability": min(1.0, abs(float(adjustment))),
                "switch_detected": abs(float(adjustment)) > 0.25,
            },
        }
        text = json.dumps(payload, ensure_ascii=False)
        return ProviderResponse(
            payload=payload,
            provider=base.provider,
            model=base.model,
            latency_ms=base.latency_ms,
            input_tokens=base.input_tokens,
            output_tokens=len(text) // 4,
            total_tokens=(base.input_tokens or 0) + len(text) // 4,
            response_id=base.response_id,
        )


def _validate_payload(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    required = schema["required"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"provider payload missing fields: {missing}")
    if schema is DECISION_SCHEMA:
        if payload["action"] not in {"fold", "check_call", "raise"}:
            raise ValueError(f"invalid provider action: {payload['action']}")
        if not 0.0 <= float(payload["confidence"]) <= 1.0:
            raise ValueError("decision confidence outside [0, 1]")
        if not 0.25 <= float(payload["raise_scale"]) <= 1.25:
            raise ValueError("raise scale outside supported range")
    else:
        if not 0.0 <= float(payload["confidence_after"]) <= 1.0:
            raise ValueError("reflection confidence outside [0, 1]")


class LLMPlayer(PokerAgent):
    """Provider-backed poker agent with auditable decision and reflection traces."""

    condition = "llm_player"

    def __init__(
        self,
        name: str,
        seed: int,
        provider: LLMProvider,
        style: AgentStyle | None = None,
        trace_dir: Path | None = None,
        reflection_memory_size: int = 6,
        *,
        opponents: tuple[str, ...] = (),
        memory_hands: int | None = None,
        reflexive_enabled: bool = True,
    ) -> None:
        super().__init__(name, seed, style)
        self.opponents = opponents
        self.provider = provider
        self.reflexive_enabled = reflexive_enabled
        self.condition = "llm_reflexive_on" if reflexive_enabled else "llm_reflexive_off"
        self.depth_controller = AdaptiveDepthController(opponents)
        self.trace_dir = trace_dir
        self.reflection_memory_size = memory_hands or reflection_memory_size
        self.decision_traces: list[dict[str, Any]] = []
        self.reflection_traces: list[dict[str, Any]] = []
        self._hand_decisions: dict[int, list[dict[str, Any]]] = {}
        self.recent_reflections: list[dict[str, Any]] = []
        self.recent_rewards: list[float] = []
        self.provider_failures = 0
        self.illegal_action_count = 0
        self.invalid_actions = 0
        self.public_aggressive_actions = 0
        self.public_passive_actions = 0
        # Compatibility aliases retained for the published v0.5.0 trace contract.
        self.llm_decision_log = self.decision_traces
        self.llm_reflection_log = self.reflection_traces
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pot_odds(context: DecisionContext) -> float:
        if context.to_call <= 0:
            return 0.0
        return context.to_call / max(context.pot + context.to_call, 1e-9)

    def _estimated_self_image(self) -> float:
        actions = self.public_aggressive_actions + self.public_passive_actions
        return self.public_aggressive_actions / actions if actions else 0.5

    def _opponent_features(self, context: DecisionContext) -> tuple[float, float]:
        aggression: list[float] = []
        folds: list[float] = []
        for opponent in context.opponents:
            counts = self.depth_controller.opponent_counts[opponent]
            total = sum(counts.values())
            if total:
                aggression.append(counts["raise"] / total)
                folds.append(counts["fold"] / total)
            else:
                aggression.append(0.5)
                folds.append(0.5)
        return (
            sum(aggression) / len(aggression) if aggression else 0.5,
            sum(folds) / len(folds) if folds else 0.5,
        )

    def observe_action(self, event: ActionEvent) -> None:
        super().observe_action(event)
        if event.actor == self.name:
            if event.action is ActionType.RAISE:
                self.public_aggressive_actions += 1
            else:
                self.public_passive_actions += 1
            return
        self.depth_controller.opponent_counts[event.actor][event.action.value] += 1

    def _decision_state(self, context: DecisionContext, equity: float) -> dict[str, Any]:
        depth = self.depth_controller.choose_depth()
        prediction = self.depth_controller.predict(depth, context.opponents, {})
        opponent_aggression, opponent_fold = self._opponent_features(context)
        recent_reward_mean = (
            sum(self.recent_rewards[-10:]) / len(self.recent_rewards[-10:])
            if self.recent_rewards
            else 0.0
        )
        state = {
            "task": "choose_poker_action",
            "reasoning_mode": "second_order" if self.reflexive_enabled else "first_order",
            "hand_index": context.hand_index,
            "street": context.street.value,
            "hole_cards": cards_to_str(context.hole_cards),
            "community_cards": cards_to_str(context.board),
            "position_index": context.button_distance,
            "active_players": context.active_players,
            "pot": context.pot,
            "to_call": context.to_call,
            "stack": context.stack,
            "legal_actions": [action.value for action in context.legal_actions],
            "equity_estimate": equity,
            "pot_odds": self._pot_odds(context),
            "public_history": [],
        }
        if self.reflexive_enabled:
            state.update(
                {
                    "predicted_all_fold": prediction.all_fold_probability,
                    "reflexive_tools": {
                        "multiway_equity": equity,
                        "pot_odds": self._pot_odds(context),
                        "self_public_image": self._estimated_self_image(),
                        "opponent_aggression_mean": opponent_aggression,
                        "opponent_fold_mean": opponent_fold,
                        "all_opponents_fold_probability": prediction.all_fold_probability,
                    },
                    "reasoning_depth": depth,
                    "self_image_estimate": self._estimated_self_image(),
                    "opponent_aggression_mean": opponent_aggression,
                    "opponent_fold_mean": opponent_fold,
                    "recent_reward_mean": recent_reward_mean,
                    "recent_reflections": self.recent_reflections[-self.reflection_memory_size :],
                }
            )
        return state

    @staticmethod
    def _decision_from_payload(
        payload: dict[str, Any],
        context: DecisionContext,
        fallback: Decision,
    ) -> tuple[Decision, str | None]:
        mapping = {
            "fold": ActionType.FOLD,
            "check_call": ActionType.CHECK_CALL,
            "raise": ActionType.RAISE,
        }
        action = mapping[payload["action"]]
        if action not in context.legal_actions:
            return fallback, f"provider chose illegal action {action.value}"
        if action is ActionType.RAISE:
            return Decision(
                action, raise_scale=float(payload["raise_scale"]), reasoning_depth=0
            ), None
        return Decision(action, reasoning_depth=0), None

    def _write_jsonl(self, name: str, record: dict[str, Any]) -> None:
        if self.trace_dir is None:
            return
        with (self.trace_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def act(self, context: DecisionContext) -> Decision:
        equity = estimate_equity(
            context.hole_cards,
            context.board,
            context.active_players - 1,
            self.rng,
            samples=self.style.equity_samples,
        )
        fallback = self._policy(
            context,
            aggression_shift=0.0,
            reasoning_depth=1,
            predicted_all_fold=None,
            metadata={"phase": "llm_fallback"},
        )
        state = self._decision_state(context, equity)
        error: str | None = None
        provider_response: ProviderResponse | None = None
        try:
            provider_response = self.provider.decide(state)
            _validate_payload(provider_response.payload, DECISION_SCHEMA)
            decision, error = self._decision_from_payload(
                provider_response.payload, context, fallback
            )
            if error:
                self.illegal_action_count += 1
                self.invalid_actions += 1
        except Exception as exc:  # noqa: BLE001 - trace provider failures and continue safely.
            self.provider_failures += 1
            error = f"{type(exc).__name__}: {exc}"
            decision = fallback

        payload = (
            provider_response.payload
            if provider_response is not None
            else {
                "action": fallback.action.value,
                "raise_scale": 0.5,
                "confidence": 0.0,
                "situation_summary": "provider failure; fallback policy used",
                "rationale": "provider output unavailable",
                "self_model": "unavailable",
                "opponent_model": "unavailable",
                "risk_flags": ["provider_failure"],
                "next_step": "retry provider on next decision",
            }
        )
        trace = {
            "trace_type": "decision",
            "hand_index": context.hand_index,
            "street": context.street.value,
            "agent": self.name,
            "condition": self.condition,
            "state": state,
            "provider_output": payload,
            "final_decision": {
                "action": decision.action.value,
                "raise_scale": decision.raise_scale,
                "fallback_used": error is not None,
                "error": error,
            },
            "provider": provider_response.provider if provider_response else self.provider.name,
            "model": provider_response.model if provider_response else self.provider.model,
            "latency_ms": provider_response.latency_ms if provider_response else None,
            "input_tokens": provider_response.input_tokens if provider_response else None,
            "output_tokens": provider_response.output_tokens if provider_response else None,
            "total_tokens": provider_response.total_tokens if provider_response else None,
            "cost_usd": provider_response.cost_usd if provider_response else None,
            "response_id": provider_response.response_id if provider_response else None,
        }
        trace["final_action"] = decision.action.value
        trace["output"] = payload
        self.decision_traces.append(trace)
        self._hand_decisions.setdefault(context.hand_index, []).append(trace)
        self._write_jsonl("decision_traces.jsonl", trace)
        return decision

    def on_hand_end(self, record: HandRecord) -> None:
        super().observe_hand_end(record)
        reward = float(record.rewards.get(self.name, 0.0))
        self.recent_rewards.append(reward)
        decisions = self._hand_decisions.pop(record.hand_index, [])
        state = {
            "task": "reflect_on_completed_hand",
            "reasoning_mode": "second_order" if self.reflexive_enabled else "first_order",
            "hand_index": record.hand_index,
            "community_cards": cards_to_str(record.board),
            "showdown": record.showdown,
            "reward": reward,
            "winners": list(record.winners),
            "public_actions": [
                {
                    "street": event.street.value,
                    "actor": event.actor,
                    "action": event.action.value,
                    "paid": event.paid,
                }
                for event in record.actions
            ],
            "decisions": [
                {
                    "street": trace["state"]["street"],
                    "action": trace["final_decision"]["action"],
                    "confidence": trace["provider_output"]["confidence"],
                    "rationale": trace["provider_output"]["rationale"],
                }
                for trace in decisions
            ],
            "recent_reward_mean": (sum(self.recent_rewards[-10:]) / len(self.recent_rewards[-10:])),
        }
        try:
            response = self.provider.reflect(state)
            _validate_payload(response.payload, REFLECTION_SCHEMA)
            payload = response.payload
            error = None
        except Exception as exc:  # noqa: BLE001
            self.provider_failures += 1
            response = None
            error = f"{type(exc).__name__}: {exc}"
            payload = {
                "outcome_summary": f"provider reflection failed after reward {reward:+.2f}",
                "decision_review": "unavailable",
                "what_worked": [],
                "what_failed": ["provider_failure"],
                "belief_updates": [],
                "strategy_adjustment": "retain fallback policy",
                "calibration_note": "reflection unavailable",
                "confidence_after": 0.0,
            }
        trace = {
            "trace_type": "reflection",
            "hand_index": record.hand_index,
            "agent": self.name,
            "condition": self.condition,
            "state": state,
            "provider_output": payload,
            "provider": response.provider if response else self.provider.name,
            "model": response.model if response else self.provider.model,
            "latency_ms": response.latency_ms if response else None,
            "input_tokens": response.input_tokens if response else None,
            "output_tokens": response.output_tokens if response else None,
            "total_tokens": response.total_tokens if response else None,
            "cost_usd": response.cost_usd if response else None,
            "response_id": response.response_id if response else None,
            "error": error,
        }
        self.reflection_traces.append(trace)
        self.recent_reflections.append(
            {
                "hand_index": record.hand_index,
                "outcome_summary": payload["outcome_summary"],
                "belief_updates": payload["belief_updates"],
                "strategy_adjustment": payload["strategy_adjustment"],
                "confidence_after": payload["confidence_after"],
            }
        )
        self.recent_reflections = self.recent_reflections[-self.reflection_memory_size :]
        self._write_jsonl("reflection_traces.jsonl", trace)

    def snapshot(self) -> dict[str, Any]:
        return {
            **super().snapshot(),
            "provider": self.provider.name,
            "model": self.provider.model,
            "decision_trace_count": len(self.decision_traces),
            "reflection_trace_count": len(self.reflection_traces),
            "provider_failures": self.provider_failures,
            "illegal_action_count": self.illegal_action_count,
            "self_public_image": self._estimated_self_image(),
            "recent_reflections": self.recent_reflections[-3:],
        }
