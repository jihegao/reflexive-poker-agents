from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .agents import AgentStyle, PokerAgent
from .equity import estimate_equity
from .llm_player import (
    DECISION_SCHEMA,
    LLMPlayer,
    LLMProvider,
    ProviderResponse,
    _validate_payload,
)
from .models import ActionEvent, ActionType, Decision, DecisionContext, HandRecord, ResponseEvent
from .tournament_agents import TYPE_NAMES, make_tournament_agent, sampled_style


class Arena(str, Enum):
    HEADS_UP = "heads_up"
    SIX_MAX = "six_max"


class Stability(str, Enum):
    FIXED = "fixed"
    MIDPOINT_SHIFT = "midpoint_shift"
    ADAPTIVE = "adaptive"


class OpponentComposition(str, Enum):
    HOMOGENEOUS_TAG = "homogeneous_tag"
    HETEROGENEOUS_CLASSIC = "heterogeneous_classic"


class ReasoningTreatment(str, Enum):
    STATE_ONLY = "state_only"
    HISTORY_STATISTICS = "history_statistics"
    ACTION_PREDICTION = "action_prediction"
    STRATEGY_TYPE = "strategy_type"
    RECURSIVE_D2 = "recursive_d2"
    RECURSIVE_D3 = "recursive_d3"

    @property
    def depth(self) -> int:
        return {
            self.STATE_ONLY: 0,
            self.HISTORY_STATISTICS: 0,
            self.ACTION_PREDICTION: 1,
            self.STRATEGY_TYPE: 1,
            self.RECURSIVE_D2: 2,
            self.RECURSIVE_D3: 3,
        }[self]


ACTION_NAMES = tuple(action.value for action in ActionType)
MODEL_TYPES = ("rock", "tag", "lag", "calling_station", "myopic", "adaptive")
TYPE_ACTION_LIKELIHOODS: dict[str, dict[str, float]] = {
    "rock": {"fold": 0.45, "check_call": 0.45, "raise": 0.10},
    "tag": {"fold": 0.32, "check_call": 0.45, "raise": 0.23},
    "lag": {"fold": 0.20, "check_call": 0.42, "raise": 0.38},
    "calling_station": {"fold": 0.12, "check_call": 0.78, "raise": 0.10},
    "myopic": {"fold": 0.30, "check_call": 0.50, "raise": 0.20},
    "adaptive": {"fold": 0.28, "check_call": 0.44, "raise": 0.28},
}


@dataclass
class OpponentBeliefState:
    """A bounded, deterministic opponent model used by every Phase 1 arm."""

    opponent: str
    window_size: int = 50
    action_counts: dict[str, float] = field(
        default_factory=lambda: {name: 1.0 for name in ACTION_NAMES}
    )
    type_posterior: dict[str, float] = field(
        default_factory=lambda: {name: 1.0 / len(MODEL_TYPES) for name in MODEL_TYPES}
    )
    hero_aggressive: float = 1.0
    hero_passive: float = 1.0
    response_counts: dict[str, float] = field(
        default_factory=lambda: {name: 1.0 for name in ACTION_NAMES}
    )
    observations: int = 0
    last_updated_hand: int = -1
    recent_actions: deque[str] = field(default_factory=lambda: deque(maxlen=50))

    def __post_init__(self) -> None:
        if self.recent_actions.maxlen != self.window_size:
            self.recent_actions = deque(self.recent_actions, maxlen=self.window_size)

    def observe_opponent_action(self, action: ActionType, hand_index: int) -> None:
        value = action.value
        self.action_counts[value] += 1.0
        self.recent_actions.append(value)
        weighted = {
            name: probability * TYPE_ACTION_LIKELIHOODS[name][value]
            for name, probability in self.type_posterior.items()
        }
        normalizer = sum(weighted.values())
        if normalizer > 0:
            self.type_posterior = {
                name: value_ / normalizer for name, value_ in weighted.items()
            }
        self.observations += 1
        self.last_updated_hand = hand_index

    def observe_hero_action(self, action: ActionType, hand_index: int) -> None:
        if action is ActionType.RAISE:
            self.hero_aggressive += 1.0
        else:
            self.hero_passive += 1.0
        self.last_updated_hand = hand_index

    def observe_response(self, action: ActionType, hand_index: int) -> None:
        self.response_counts[action.value] += 1.0
        self.last_updated_hand = hand_index

    def observe_reward(self, reward: float, hand_index: int) -> None:
        del reward
        self.last_updated_hand = hand_index

    @property
    def action_distribution(self) -> dict[str, float]:
        if self.recent_actions:
            counts = Counter(self.recent_actions)
            denominator = len(self.recent_actions) + len(ACTION_NAMES)
            return {name: (counts[name] + 1.0) / denominator for name in ACTION_NAMES}
        denominator = sum(self.action_counts.values())
        return {name: self.action_counts[name] / denominator for name in ACTION_NAMES}

    @property
    def hero_public_image(self) -> float:
        return self.hero_aggressive / (self.hero_aggressive + self.hero_passive)

    @property
    def conditional_response_model(self) -> dict[str, float]:
        denominator = sum(self.response_counts.values())
        return {name: self.response_counts[name] / denominator for name in ACTION_NAMES}

    @property
    def confidence(self) -> float:
        return 1.0 - math.exp(-self.observations / 20.0)

    @property
    def anticipated_adjustment(self) -> float:
        response = self.conditional_response_model
        pressure_response = response["raise"] - response["fold"]
        return max(-1.0, min(1.0, pressure_response * self.confidence))

    def snapshot(self) -> dict[str, Any]:
        return {
            "opponent": self.opponent,
            "window_size": self.window_size,
            "action_counts": dict(self.action_counts),
            "action_distribution": self.action_distribution,
            "type_posterior": dict(self.type_posterior),
            "hero_public_image": self.hero_public_image,
            "conditional_response_model": self.conditional_response_model,
            "anticipated_adjustment": self.anticipated_adjustment,
            "observations": self.observations,
            "last_updated_hand": self.last_updated_hand,
            "confidence": self.confidence,
        }

    def digest(self) -> str:
        payload = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ProviderBudget:
    max_calls: int = 10_000
    max_primary_calls: int | None = None
    max_total_tokens: int | None = None
    max_input_tokens_per_call: int | None = 50_000
    max_output_tokens_per_call: int | None = 2_000
    max_retries: int = 400
    max_latency_ms: float | None = None
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")


@dataclass
class ProviderLedger:
    calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_observed_calls: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cost_observed_calls: int = 0
    raw_failures: int = 0
    unresolved_failures: int = 0

    def snapshot(self) -> dict[str, int | float]:
        return asdict(self)


class ProviderBudgetExceeded(RuntimeError):
    pass


class BudgetedRetryProvider:
    """Shared provider ledger with one schema-repair retry per decision."""

    def __init__(
        self,
        provider: LLMProvider,
        budget: ProviderBudget,
        ledger: ProviderLedger | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.provider = provider
        self.budget = budget
        self.ledger = ledger or ProviderLedger()
        self.name = provider.name
        self.model = provider.model
        self.checkpoint_path = checkpoint_path

    def _checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": self.name,
            "model": self.model,
            "budget": asdict(self.budget),
            "ledger": self.ledger.snapshot(),
        }
        temporary = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, self.checkpoint_path)

    def __deepcopy__(self, memo: dict[int, Any]) -> BudgetedRetryProvider:
        del memo
        return self

    def _preflight(self, retry: bool) -> None:
        if self.ledger.calls >= self.budget.max_calls:
            raise ProviderBudgetExceeded("provider call budget exhausted")
        if retry and self.ledger.retries >= self.budget.max_retries:
            raise ProviderBudgetExceeded("provider retry budget exhausted")
        if (
            not retry
            and self.budget.max_primary_calls is not None
            and self.ledger.calls - self.ledger.retries >= self.budget.max_primary_calls
        ):
            raise ProviderBudgetExceeded("provider primary-call budget exhausted")
        if (
            self.budget.max_total_tokens is not None
            and self.ledger.total_tokens >= self.budget.max_total_tokens
        ):
            raise ProviderBudgetExceeded("provider token budget exhausted")
        if (
            self.budget.max_latency_ms is not None
            and self.ledger.latency_ms >= self.budget.max_latency_ms
        ):
            raise ProviderBudgetExceeded("provider latency budget exhausted")
        if self.budget.max_cost_usd is not None and self.ledger.cost_usd >= self.budget.max_cost_usd:
            raise ProviderBudgetExceeded("provider cost budget exhausted")

    def _record(self, response: ProviderResponse) -> None:
        if response.input_tokens is not None:
            self.ledger.input_tokens += int(response.input_tokens)
        if response.output_tokens is not None:
            self.ledger.output_tokens += int(response.output_tokens)
        if response.total_tokens is not None:
            self.ledger.total_tokens += int(response.total_tokens)
            self.ledger.token_observed_calls += 1
        self.ledger.latency_ms += float(response.latency_ms)
        if response.cost_usd is not None:
            self.ledger.cost_usd += float(response.cost_usd)
            self.ledger.cost_observed_calls += 1
        if (
            self.budget.max_input_tokens_per_call is not None
            and response.input_tokens is not None
            and response.input_tokens > self.budget.max_input_tokens_per_call
        ):
            raise ValueError("provider input token cap exceeded")
        if (
            self.budget.max_output_tokens_per_call is not None
            and response.output_tokens is not None
            and response.output_tokens > self.budget.max_output_tokens_per_call
        ):
            raise ValueError("provider output token cap exceeded")

    def decide(self, state: dict[str, Any]) -> ProviderResponse:
        last_error: Exception | None = None
        for attempt in range(2):
            retry = attempt == 1
            self._preflight(retry)
            self.ledger.calls += 1
            if retry:
                self.ledger.retries += 1
            self._checkpoint()
            try:
                response = self.provider.decide(state)
                self._record(response)
                self._checkpoint()
                _validate_payload(response.payload, DECISION_SCHEMA)
                return response
            except ProviderBudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - failures are part of the audit contract.
                self.ledger.raw_failures += 1
                last_error = exc
                self._checkpoint()
        self.ledger.unresolved_failures += 1
        self._checkpoint()
        assert last_error is not None
        raise last_error

    def reflect(self, state: dict[str, Any]) -> ProviderResponse:
        raise RuntimeError("Phase 1 uses deterministic belief updates, not LLM reflection calls")


class TreatmentStateMixin:
    treatment: ReasoningTreatment
    belief_states: dict[str, OpponentBeliefState]

    def set_treatment(self, treatment: ReasoningTreatment) -> None:
        self.treatment = treatment
        self.condition = treatment.value

    def _belief(self, opponent: str) -> OpponentBeliefState:
        return self.belief_states.setdefault(opponent, OpponentBeliefState(opponent))

    def _aggregate_belief(self, opponents: tuple[str, ...]) -> dict[str, Any]:
        states = [self._belief(name) for name in opponents]
        if not states:
            return {
                "action_distribution": {name: 1.0 / 3.0 for name in ACTION_NAMES},
                "type_posterior": {name: 1.0 / len(MODEL_TYPES) for name in MODEL_TYPES},
                "hero_public_image": 0.5,
                "conditional_response_model": {name: 1.0 / 3.0 for name in ACTION_NAMES},
                "anticipated_adjustment": 0.0,
                "confidence": 0.0,
            }
        return {
            "action_distribution": {
                action: sum(state.action_distribution[action] for state in states) / len(states)
                for action in ACTION_NAMES
            },
            "type_posterior": {
                player_type: sum(state.type_posterior[player_type] for state in states)
                / len(states)
                for player_type in MODEL_TYPES
            },
            "hero_public_image": sum(state.hero_public_image for state in states) / len(states),
            "conditional_response_model": {
                action: sum(state.conditional_response_model[action] for state in states)
                / len(states)
                for action in ACTION_NAMES
            },
            "anticipated_adjustment": sum(state.anticipated_adjustment for state in states)
            / len(states),
            "confidence": sum(state.confidence for state in states) / len(states),
        }

    def treatment_features(self, opponents: tuple[str, ...]) -> dict[str, Any]:
        aggregate = self._aggregate_belief(opponents)
        masked = "__MASKED__"
        features: dict[str, Any] = {
            "history_statistics": masked,
            "action_prediction": masked,
            "strategy_type": masked,
            "opponent_view_of_hero": masked,
            "conditional_response_model": masked,
            "anticipated_adjustment": masked,
        }
        if self.treatment is not ReasoningTreatment.STATE_ONLY:
            features["history_statistics"] = {
                name: self._belief(name).observations for name in opponents
            }
        if self.treatment in {
            ReasoningTreatment.ACTION_PREDICTION,
            ReasoningTreatment.RECURSIVE_D2,
            ReasoningTreatment.RECURSIVE_D3,
        }:
            features["action_prediction"] = aggregate["action_distribution"]
        if self.treatment in {
            ReasoningTreatment.STRATEGY_TYPE,
            ReasoningTreatment.RECURSIVE_D2,
            ReasoningTreatment.RECURSIVE_D3,
        }:
            features["strategy_type"] = aggregate["type_posterior"]
        if self.treatment in {ReasoningTreatment.RECURSIVE_D2, ReasoningTreatment.RECURSIVE_D3}:
            features["opponent_view_of_hero"] = aggregate["hero_public_image"]
            features["conditional_response_model"] = aggregate["conditional_response_model"]
        if self.treatment is ReasoningTreatment.RECURSIVE_D3:
            features["anticipated_adjustment"] = aggregate["anticipated_adjustment"]
        return features

    def observe_phase1_action(self, event: ActionEvent) -> None:
        if event.actor == self.name:
            for state in self.belief_states.values():
                state.observe_hero_action(event.action, event.hand_index)
        else:
            self._belief(event.actor).observe_opponent_action(event.action, event.hand_index)

    def observe_phase1_response(self, event: ResponseEvent) -> None:
        self._belief(event.responder).observe_response(
            ActionType.RAISE if event.reraised else ActionType.FOLD if event.folded else ActionType.CHECK_CALL,
            event.hand_index,
        )

    def belief_digest(self) -> str:
        payload = {name: state.snapshot() for name, state in sorted(self.belief_states.items())}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class Phase1RuleHero(TreatmentStateMixin, PokerAgent):
    def __init__(
        self,
        name: str,
        seed: int,
        treatment: ReasoningTreatment,
        opponents: tuple[str, ...],
        style: AgentStyle | None = None,
    ) -> None:
        super().__init__(name, seed, style)
        self.treatment = treatment
        self.condition = treatment.value
        self.belief_states = {name_: OpponentBeliefState(name_) for name_ in opponents}
        self.phase1_traces: list[dict[str, Any]] = []

    def act(self, context: DecisionContext) -> Decision:
        aggregate = self._aggregate_belief(context.opponents)
        shift = 0.0
        if self.treatment is ReasoningTreatment.HISTORY_STATISTICS:
            shift = 0.04 * (aggregate["confidence"] - 0.5)
        elif self.treatment is ReasoningTreatment.ACTION_PREDICTION:
            shift = 0.20 * (aggregate["action_distribution"]["fold"] - 1.0 / 3.0)
        elif self.treatment is ReasoningTreatment.STRATEGY_TYPE:
            type_aggression = {"rock": 0.18, "tag": 0.43, "lag": 0.72, "calling_station": 0.22, "myopic": 0.47, "adaptive": 0.50}
            expected = sum(
                aggregate["type_posterior"][name] * value for name, value in type_aggression.items()
            )
            shift = 0.12 * (0.5 - expected)
        elif self.treatment in {ReasoningTreatment.RECURSIVE_D2, ReasoningTreatment.RECURSIVE_D3}:
            response = aggregate["conditional_response_model"]
            shift = 0.18 * (response["fold"] - response["raise"])
            shift += 0.10 * (0.5 - aggregate["hero_public_image"])
            if self.treatment is ReasoningTreatment.RECURSIVE_D3:
                shift -= 0.12 * aggregate["anticipated_adjustment"]
        prediction = aggregate["action_distribution"]
        decision = self._policy(
            context,
            aggression_shift=shift,
            reasoning_depth=self.treatment.depth,
            predicted_all_fold=prediction["fold"],
            metadata={"phase1_treatment": self.treatment.value},
        )
        self.phase1_traces.append(
            {
                "hand_index": context.hand_index,
                "street": context.street.value,
                "treatment": self.treatment.value,
                "reasoning_depth": self.treatment.depth,
                "visible_features": self.treatment_features(context.opponents),
                "action_prediction": prediction,
                "belief_state_hash": self.belief_digest(),
                "final_action": decision.action.value,
            }
        )
        return decision

    def observe_action(self, event: ActionEvent) -> None:
        PokerAgent.observe_action(self, event)
        self.observe_phase1_action(event)

    def observe_response(self, event: ResponseEvent) -> None:
        self.observe_phase1_response(event)

    def on_hand_end(self, record: HandRecord) -> None:
        PokerAgent.observe_hand_end(self, record)
        reward = float(record.rewards.get(self.name, 0.0))
        for state in self.belief_states.values():
            state.observe_reward(reward, record.hand_index)

    def snapshot(self) -> dict[str, Any]:
        return {
            **super().snapshot(),
            "phase1_treatment": self.treatment.value,
            "reasoning_depth": self.treatment.depth,
            "belief_state_hash": self.belief_digest(),
        }


class Phase1LLMHero(TreatmentStateMixin, LLMPlayer):
    """LLM decision agent with deterministic bounded belief updates and no reflection call."""

    def __init__(
        self,
        name: str,
        seed: int,
        provider: LLMProvider,
        treatment: ReasoningTreatment,
        opponents: tuple[str, ...],
        style: AgentStyle | None = None,
    ) -> None:
        super().__init__(
            name,
            seed,
            provider,
            style,
            opponents=opponents,
            reflection_memory_size=0,
            reflexive_enabled=False,
        )
        self.treatment = treatment
        self.condition = treatment.value
        self.belief_states = {name_: OpponentBeliefState(name_) for name_ in opponents}

    def _decision_state(self, context: DecisionContext, equity: float) -> dict[str, Any]:
        return {
            "task": "choose_phase1_poker_action",
            "phase1_treatment": self.treatment.value,
            "reasoning_depth": self.treatment.depth,
            "hand_index": context.hand_index,
            "street": context.street.value,
            "hole_cards": list(context.hole_cards),
            "community_cards": list(context.board),
            "pot": context.pot,
            "to_call": context.to_call,
            "stack": context.stack,
            "legal_actions": [action.value for action in context.legal_actions],
            "active_players": context.active_players,
            "opponents": list(context.opponents),
            "equity_estimate": equity,
            "pot_odds": self._pot_odds(context),
            "bounded_opponent_model": self.treatment_features(context.opponents),
            "belief_state_hash": self.belief_digest(),
        }

    def act(self, context: DecisionContext) -> Decision:
        decision = super().act(context)
        trace = self.decision_traces[-1]
        trace.update(
            {
                "phase1_treatment": self.treatment.value,
                "reasoning_depth": self.treatment.depth,
                "belief_state_hash": self.belief_digest(),
                "budget": (
                    self.provider.ledger.snapshot()
                    if isinstance(self.provider, BudgetedRetryProvider)
                    else None
                ),
            }
        )
        return decision

    def observe_action(self, event: ActionEvent) -> None:
        PokerAgent.observe_action(self, event)
        self.observe_phase1_action(event)

    def observe_response(self, event: ResponseEvent) -> None:
        self.observe_phase1_response(event)

    def on_hand_end(self, record: HandRecord) -> None:
        PokerAgent.observe_hand_end(self, record)
        reward = float(record.rewards.get(self.name, 0.0))
        self.recent_rewards.append(reward)
        for state in self.belief_states.values():
            state.observe_reward(reward, record.hand_index)

    def snapshot(self) -> dict[str, Any]:
        return {
            **PokerAgent.snapshot(self),
            "phase1_treatment": self.treatment.value,
            "reasoning_depth": self.treatment.depth,
            "belief_state_hash": self.belief_digest(),
        }


class ExperimentalOpponent(PokerAgent):
    """Frozen, noisy, switching, or adaptive opponent under one auditable interface."""

    def __init__(
        self,
        name: str,
        seed: int,
        player_type: str,
        opponents: tuple[str, ...],
        epsilon: float,
        stability: Stability,
        switch_hand: int,
        switch_type: str = "lag",
        equity_samples: int = 2,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if player_type not in TYPE_NAMES:
            raise ValueError(f"unknown experimental opponent type: {player_type}")
        super().__init__(name, seed, sampled_style(player_type, seed, equity_samples))
        self.player_type = player_type
        self.switch_type = switch_type
        self.epsilon = epsilon
        self.stability = stability
        self.switch_hand = switch_hand
        self.opponents = opponents
        self.hero_name = opponents[0] if opponents else "hero"
        self.hero_actions: deque[str] = deque(maxlen=50)
        self._delegates = {
            value: make_tournament_agent(
                value,
                name,
                opponents,
                seed + index * 100_003,
                equity_samples=equity_samples,
            )
            for index, value in enumerate({player_type, switch_type})
        }
        self.condition = f"experimental_{player_type}"

    def active_type(self, hand_index: int) -> str:
        if self.stability is Stability.MIDPOINT_SHIFT and hand_index >= self.switch_hand:
            return self.switch_type
        return self.player_type

    def act(self, context: DecisionContext) -> Decision:
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(list(context.legal_actions))
            decision = Decision(
                action=action,
                raise_scale=0.5,
                reasoning_depth=0,
                metadata={"phase": "epsilon", "true_type": self.active_type(context.hand_index)},
            )
        elif self.stability is Stability.ADAPTIVE:
            raises = sum(value == ActionType.RAISE.value for value in self.hero_actions)
            hero_raise_rate = raises / len(self.hero_actions) if self.hero_actions else 0.5
            shift = 0.20 * (0.5 - hero_raise_rate)
            decision = self._policy(
                context,
                aggression_shift=shift,
                metadata={
                    "phase": "adaptive",
                    "true_type": "adaptive",
                    "hero_raise_rate": hero_raise_rate,
                },
            )
        else:
            decision = self._delegates[self.active_type(context.hand_index)].act(context)
        self.decision_log.append(
            {
                "hand_index": context.hand_index,
                "street": context.street.value,
                "action": decision.action.value,
                "true_type": self.active_type(context.hand_index),
                "stability": self.stability.value,
                "epsilon": self.epsilon,
            }
        )
        return decision

    def observe_action(self, event: ActionEvent) -> None:
        super().observe_action(event)
        if event.actor == self.hero_name:
            self.hero_actions.append(event.action.value)
        for delegate in self._delegates.values():
            delegate.observe_action(event)

    def on_hand_end(self, record: HandRecord) -> None:
        super().observe_hand_end(record)

    def snapshot(self) -> dict[str, Any]:
        return {
            **super().snapshot(),
            "player_type": self.player_type,
            "active_type": self.active_type(self.decision_log[-1]["hand_index"] if self.decision_log else 0),
            "stability": self.stability.value,
            "epsilon": self.epsilon,
        }


@dataclass(frozen=True)
class MCCFRTrainingReport:
    iterations: int
    infosets: int
    mean_positive_regret: float
    empirical_exploitability: float
    policy_hash: str
    label: str = "abstract_external_sampling_mccfr_reference"


class AbstractMCCFRPolicy:
    """External-sampling regret matching on the repository's compact action abstraction.

    This is deliberately an approximate-equilibrium reference over bucketed decision
    states, not a solver-grade equilibrium for full no-limit Texas Hold'em.
    """

    ACTIONS = ACTION_NAMES

    def __init__(self, policy: dict[str, dict[str, float]], report: MCCFRTrainingReport) -> None:
        self.policy = policy
        self.report = report

    @staticmethod
    def key(strength_bucket: int, pot_odds_bucket: int, facing_raise: bool) -> str:
        return f"s{strength_bucket}:p{pot_odds_bucket}:f{int(facing_raise)}"

    @staticmethod
    def _utility(action: str, opponent_action: str, strength: float, pot_odds: float) -> float:
        if action == "fold":
            return -pot_odds if opponent_action == "raise" else -0.02
        if action == "check_call":
            return (strength - pot_odds) * (1.4 if opponent_action == "raise" else 0.8)
        fold_bonus = 0.45 if opponent_action == "fold" else 0.0
        contest = (2.0 * strength - 1.0) * (1.3 if opponent_action == "raise" else 1.0)
        return fold_bonus + contest - 0.08

    @classmethod
    def train(cls, iterations: int = 20_000, seed: int = 20260802) -> AbstractMCCFRPolicy:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        rng = random.Random(seed)
        regrets: dict[str, dict[str, float]] = {}
        strategy_sums: dict[str, dict[str, float]] = {}
        keys = [cls.key(s, p, facing) for s in range(5) for p in range(4) for facing in (False, True)]
        for key in keys:
            regrets[key] = {action: 0.0 for action in cls.ACTIONS}
            strategy_sums[key] = {action: 0.0 for action in cls.ACTIONS}
        for _ in range(iterations):
            strength_bucket = rng.randrange(5)
            pot_odds_bucket = rng.randrange(4)
            facing_raise = bool(rng.randrange(2))
            key = cls.key(strength_bucket, pot_odds_bucket, facing_raise)
            positive = {action: max(value, 0.0) for action, value in regrets[key].items()}
            normalizer = sum(positive.values())
            strategy = (
                {action: positive[action] / normalizer for action in cls.ACTIONS}
                if normalizer
                else {action: 1.0 / len(cls.ACTIONS) for action in cls.ACTIONS}
            )
            opponent_action = rng.choices(cls.ACTIONS, weights=(0.28, 0.48, 0.24), k=1)[0]
            strength = (strength_bucket + 0.5) / 5.0
            pot_odds = (pot_odds_bucket + 0.5) / 4.0
            utilities = {
                action: cls._utility(action, opponent_action, strength, pot_odds)
                for action in cls.ACTIONS
            }
            sampled_value = sum(strategy[action] * utilities[action] for action in cls.ACTIONS)
            for action in cls.ACTIONS:
                regrets[key][action] += utilities[action] - sampled_value
                strategy_sums[key][action] += strategy[action]
        policy: dict[str, dict[str, float]] = {}
        for key, values in strategy_sums.items():
            normalizer = sum(values.values())
            policy[key] = (
                {action: values[action] / normalizer for action in cls.ACTIONS}
                if normalizer
                else {action: 1.0 / len(cls.ACTIONS) for action in cls.ACTIONS}
            )
        payload = json.dumps(policy, sort_keys=True, separators=(",", ":"))
        positive_regrets = [max(value, 0.0) for values in regrets.values() for value in values.values()]
        opponent_weights = {"fold": 0.28, "check_call": 0.48, "raise": 0.24}
        exploitability: list[float] = []
        for key, strategy in policy.items():
            parts = key.split(":")
            strength = (int(parts[0][1:]) + 0.5) / 5.0
            pot_odds = (int(parts[1][1:]) + 0.5) / 4.0
            expected_by_action = {
                action: sum(
                    weight * cls._utility(action, opponent_action, strength, pot_odds)
                    for opponent_action, weight in opponent_weights.items()
                )
                for action in cls.ACTIONS
            }
            policy_value = sum(
                strategy[action] * expected_by_action[action] for action in cls.ACTIONS
            )
            exploitability.append(max(expected_by_action.values()) - policy_value)
        report = MCCFRTrainingReport(
            iterations=iterations,
            infosets=len(keys),
            mean_positive_regret=sum(positive_regrets) / len(positive_regrets) / iterations,
            empirical_exploitability=sum(exploitability) / len(exploitability),
            policy_hash=hashlib.sha256(payload.encode()).hexdigest(),
        )
        return cls(policy, report)

    def distribution(self, equity: float, pot_odds: float, facing_raise: bool) -> dict[str, float]:
        strength_bucket = min(4, max(0, int(equity * 5)))
        pot_odds_bucket = min(3, max(0, int(pot_odds * 4)))
        return self.policy[self.key(strength_bucket, pot_odds_bucket, facing_raise)]


class ApproximateEquilibriumOpponent(PokerAgent):
    condition = "approximate_equilibrium"

    def __init__(
        self,
        name: str,
        seed: int,
        policy: AbstractMCCFRPolicy,
        epsilon: float = 0.05,
        equity_samples: int = 8,
    ) -> None:
        super().__init__(name, seed, AgentStyle(equity_samples=equity_samples))
        self.policy = policy
        self.epsilon = epsilon

    def act(self, context: DecisionContext) -> Decision:
        equity = estimate_equity(
            context.hole_cards,
            context.board,
            max(1, context.active_players - 1),
            self.rng,
            samples=self.style.equity_samples,
        )
        pot_odds = context.to_call / max(context.pot + context.to_call, 1e-9)
        distribution = self.policy.distribution(equity, pot_odds, context.to_call > 0)
        legal = [action.value for action in context.legal_actions]
        legal_weights = [distribution[action] + self.epsilon / len(legal) for action in legal]
        action_name = self.rng.choices(legal, weights=legal_weights, k=1)[0]
        action = ActionType(action_name)
        decision = Decision(
            action=action,
            raise_scale=0.5,
            equity=equity,
            metadata={
                "policy_label": self.policy.report.label,
                "policy_hash": self.policy.report.policy_hash,
            },
        )
        self.decision_log.append(
            {
                "hand_index": context.hand_index,
                "street": context.street.value,
                "action": action.value,
                "policy_hash": self.policy.report.policy_hash,
            }
        )
        return decision


def deepcopy_environment(value: Any) -> Any:
    """Named helper so tests can assert that provider ledgers remain shared across forks."""
    return copy.deepcopy(value)
