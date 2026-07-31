from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .equity import estimate_equity
from .models import ActionEvent, ActionType, Decision, DecisionContext, HandRecord, ResponseEvent


@dataclass
class _Belief:
    aggression_total: float = 1.0
    aggression_raises: float = 0.5

    @property
    def aggression(self) -> _Belief:
        return self

    @property
    def mean(self) -> float:
        return self.aggression_raises / self.aggression_total


@dataclass
class AgentStyle:
    aggression: float = 0.45
    risk_margin: float = 0.06
    belief_sensitivity: float = 0.22
    social_learning_rate: float = 0.20
    equity_samples: int = 18


class PokerAgent:
    condition = "base"

    def __init__(self, name: str, seed: int, style: AgentStyle | None = None) -> None:
        self.name, self.rng, self.style = name, random.Random(seed), style or AgentStyle()
        self.beliefs: dict[str, _Belief] = defaultdict(_Belief)
        self.cumulative_reward = 0.0
        self.recent_rewards: deque[float] = deque(maxlen=20)
        self.decision_log: list[dict[str, Any]] = []

    def _policy(
        self,
        context: DecisionContext,
        aggression_shift: float = 0.0,
        reasoning_depth: int = 0,
        predicted_all_fold: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Decision:
        equity = estimate_equity(
            context.hole_cards,
            context.board,
            max(1, context.active_players - 1),
            self.rng,
            self.style.equity_samples,
        )
        pot_odds = context.to_call / max(context.pot + context.to_call, 1e-9)
        aggression = min(max(self.style.aggression + aggression_shift, 0.03), 0.97)
        can_raise = ActionType.RAISE in context.legal_actions
        if context.to_call > 0:
            action = (
                ActionType.RAISE
                if can_raise and equity > 0.78 - aggression * 0.12
                else (
                    ActionType.CHECK_CALL
                    if equity + 0.02 >= pot_odds - self.style.risk_margin
                    else ActionType.FOLD
                )
            )
        else:
            action = (
                ActionType.RAISE
                if can_raise and equity > 0.66 - aggression * 0.16
                else ActionType.CHECK_CALL
            )
        decision = Decision(
            action,
            0.65 if equity > 0.75 else 0.5,
            equity,
            predicted_all_fold or 0.0,
            reasoning_depth,
            (1, 3, 7, 15)[reasoning_depth],
            metadata or {},
        )
        self.decision_log.append(
            {
                "hand_index": context.hand_index,
                "street": context.street.value,
                "action": action.value,
                "equity": equity,
                "reasoning_depth": reasoning_depth,
                **(metadata or {}),
            }
        )
        return decision

    def act(self, context: DecisionContext) -> Decision:
        return self._policy(context)

    def observe_action(self, event: ActionEvent) -> None:
        if event.actor != self.name:
            belief = self.beliefs[event.actor]
            belief.aggression_total += self.style.social_learning_rate
            if event.action is ActionType.RAISE:
                belief.aggression_raises += self.style.social_learning_rate

    def observe_response(self, event: ResponseEvent) -> None:
        del event

    def on_own_action(self, event: ActionEvent) -> None:
        del event

    def observe_hand_end(self, record: HandRecord) -> None:
        reward = record.rewards.get(self.name, 0.0)
        self.cumulative_reward += reward
        self.recent_rewards.append(reward)

    def on_hand_end(self, record: HandRecord) -> None:
        self.observe_hand_end(record)

    def snapshot(self) -> dict[str, Any]:
        return {"condition": self.condition, "cumulative_reward": self.cumulative_reward}
