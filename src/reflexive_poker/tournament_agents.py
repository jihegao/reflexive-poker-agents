from __future__ import annotations

import random
from dataclasses import dataclass

from .agents import (
    AgentStyle,
    MyopicControlAgent,
    OpenLoopImageShapingAgent,
    PassiveImageTrackingAgent,
    PokerAgent,
    SituatedReflectionAgent,
)
from .depth import AdaptiveDepthController
from .equity import estimate_equity
from .models import ActionType, Decision, DecisionContext


TYPE_NAMES = (
    "rock",
    "tag",
    "lag",
    "calling_station",
    "myopic",
    "passive_tracker",
    "open_loop_shaper",
    "closed_loop_shaper",
)

TYPE_LABELS = {
    "rock": "Rock",
    "tag": "TAG",
    "lag": "LAG",
    "calling_station": "Calling station",
    "myopic": "Myopic control",
    "passive_tracker": "Passive tracker",
    "open_loop_shaper": "Open-loop shaper",
    "closed_loop_shaper": "Closed-loop shaper",
}


@dataclass(frozen=True)
class TypeSpec:
    aggression: float
    risk_margin: float
    belief_sensitivity: float
    social_learning_rate: float


BASE_SPECS = {
    "rock": TypeSpec(0.18, 0.115, 0.12, 0.12),
    "tag": TypeSpec(0.43, 0.060, 0.24, 0.18),
    "lag": TypeSpec(0.72, 0.025, 0.32, 0.22),
    "calling_station": TypeSpec(0.22, -0.020, 0.08, 0.10),
    "myopic": TypeSpec(0.47, 0.055, 0.22, 0.18),
    "passive_tracker": TypeSpec(0.47, 0.055, 0.24, 0.20),
    "open_loop_shaper": TypeSpec(0.47, 0.055, 0.24, 0.20),
    "closed_loop_shaper": TypeSpec(0.47, 0.055, 0.24, 0.20),
}


class TypedPolicyAgent(PokerAgent):
    """Behaviorally distinct Rock, TAG, and LAG archetypes."""

    def __init__(self, name: str, seed: int, player_type: str, style: AgentStyle) -> None:
        super().__init__(name, seed, style)
        self.player_type = player_type
        self.condition = player_type

    def act(self, context: DecisionContext) -> Decision:
        equity = estimate_equity(
            context.hole_cards,
            context.board,
            max(1, context.active_players - 1),
            self.rng,
            samples=self.style.equity_samples,
        )
        if context.last_raiser and context.last_raiser != self.name:
            equity += self.style.belief_sensitivity * (
                self.beliefs[context.last_raiser].aggression.mean - 0.5
            )
        equity = min(max(equity, 0.0), 1.0)
        pot_odds = context.to_call / max(context.pot + context.to_call, 1e-9)
        can_raise = ActionType.RAISE in context.legal_actions
        random_value = self.rng.random()

        if self.player_type == "rock":
            if context.to_call > 0:
                if can_raise and equity > 0.84:
                    action = ActionType.RAISE
                elif equity > pot_odds + 0.10:
                    action = ActionType.CHECK_CALL
                else:
                    action = ActionType.FOLD
            else:
                action = ActionType.RAISE if can_raise and equity > 0.76 else ActionType.CHECK_CALL
            raise_scale = 0.72
            depth = 0
        elif self.player_type == "tag":
            if context.to_call > 0:
                if can_raise and (equity > 0.73 or random_value < 0.018):
                    action = ActionType.RAISE
                elif equity > pot_odds + 0.025:
                    action = ActionType.CHECK_CALL
                else:
                    action = ActionType.FOLD
            else:
                action = (
                    ActionType.RAISE
                    if can_raise and (equity > 0.61 or random_value < 0.045)
                    else ActionType.CHECK_CALL
                )
            raise_scale = 0.62
            depth = 1
        else:
            if context.to_call > 0:
                if can_raise and (equity > 0.58 or random_value < 0.12):
                    action = ActionType.RAISE
                elif equity > pot_odds - 0.055 or random_value < 0.12:
                    action = ActionType.CHECK_CALL
                else:
                    action = ActionType.FOLD
            else:
                action = (
                    ActionType.RAISE
                    if can_raise and (equity > 0.46 or random_value < 0.18)
                    else ActionType.CHECK_CALL
                )
            raise_scale = 0.52
            depth = 1

        decision = Decision(
            action=action,
            raise_scale=raise_scale,
            equity=equity,
            predicted_all_fold=0.0,
            reasoning_depth=depth,
            reasoning_ops=AdaptiveDepthController.OPS[depth],
            metadata={"player_type": self.player_type, "phase": "fixed"},
        )
        self.decision_log.append(
            {
                "hand_index": context.hand_index,
                "street": context.street.value,
                "action": action.value,
                "equity": equity,
                "predicted_all_fold": 0.0,
                "reasoning_depth": depth,
                "reasoning_ops": decision.reasoning_ops,
                "environment_regime": context.environment_regime,
                "player_type": self.player_type,
                "phase": "fixed",
            }
        )
        return decision

    def snapshot(self) -> dict[str, object]:
        return {**super().snapshot(), "player_type": self.player_type, "phase": "fixed"}


class CallingStationAgent(PokerAgent):
    """Calls too widely, rarely bluffs, and raises mainly for strong value."""

    condition = "calling_station"

    def act(self, context: DecisionContext) -> Decision:
        equity = estimate_equity(
            context.hole_cards,
            context.board,
            max(1, context.active_players - 1),
            self.rng,
            samples=self.style.equity_samples,
        )
        pot_odds = context.to_call / max(context.pot + context.to_call, 1e-9)
        can_raise = ActionType.RAISE in context.legal_actions
        if context.to_call > 0:
            if can_raise and equity > 0.86 and self.rng.random() < 0.22:
                action = ActionType.RAISE
            elif equity + 0.16 >= pot_odds or self.rng.random() < 0.72:
                action = ActionType.CHECK_CALL
            else:
                action = ActionType.FOLD
        else:
            if can_raise and equity > 0.82 and self.rng.random() < 0.28:
                action = ActionType.RAISE
            else:
                action = ActionType.CHECK_CALL
        decision = Decision(
            action=action,
            raise_scale=0.55,
            equity=equity,
            predicted_all_fold=0.0,
            reasoning_depth=0,
            reasoning_ops=AdaptiveDepthController.OPS[0],
            metadata={"player_type": self.condition, "phase": "fixed"},
        )
        self.decision_log.append(
            {
                "hand_index": context.hand_index,
                "street": context.street.value,
                "action": action.value,
                "equity": equity,
                "predicted_all_fold": 0.0,
                "reasoning_depth": 0,
                "reasoning_ops": decision.reasoning_ops,
                "environment_regime": context.environment_regime,
                "player_type": self.condition,
                "phase": "fixed",
            }
        )
        return decision

    def snapshot(self) -> dict[str, object]:
        return {**super().snapshot(), "player_type": self.condition, "phase": "fixed"}


def sampled_style(player_type: str, seed: int, equity_samples: int) -> AgentStyle:
    """Generate a seed-specific instance within a narrow, pre-frozen type envelope."""
    spec = BASE_SPECS[player_type]
    rng = random.Random(seed * 1009 + sum(ord(char) for char in player_type) * 9176)
    return AgentStyle(
        aggression=min(max(spec.aggression + rng.uniform(-0.035, 0.035), 0.03), 0.95),
        risk_margin=spec.risk_margin + rng.uniform(-0.012, 0.012),
        belief_sensitivity=min(max(spec.belief_sensitivity + rng.uniform(-0.035, 0.035), 0.0), 0.60),
        social_learning_rate=min(max(spec.social_learning_rate + rng.uniform(-0.035, 0.035), 0.03), 0.40),
        equity_samples=equity_samples,
    )


def make_tournament_agent(
    player_type: str,
    name: str,
    opponents: tuple[str, ...],
    seed: int,
    equity_samples: int = 2,
) -> PokerAgent:
    if player_type not in TYPE_NAMES:
        raise ValueError(f"Unknown player type: {player_type}")
    style = sampled_style(player_type, seed, equity_samples)
    if player_type in {"rock", "tag", "lag"}:
        return TypedPolicyAgent(name, seed, player_type, style)
    if player_type == "calling_station":
        return CallingStationAgent(name, seed, style)
    if player_type == "myopic":
        agent = MyopicControlAgent(name, seed, style)
        agent.condition = player_type
        return agent
    if player_type == "passive_tracker":
        agent = PassiveImageTrackingAgent(name, seed, opponents, style)
        agent.condition = player_type
        return agent
    if player_type == "open_loop_shaper":
        agent = OpenLoopImageShapingAgent(
            name,
            seed,
            opponents,
            style,
            fixed_signal_hands=30,
        )
        agent.condition = player_type
        return agent
    agent = SituatedReflectionAgent(
        name,
        seed,
        opponents=opponents,
        style=style,
        max_signal_hands=85,
        use_response_feedback=True,
        condition_label=player_type,
    )
    return agent
