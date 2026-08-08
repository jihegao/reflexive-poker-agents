from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from .agents import AgentStyle, PokerAgent
from .environment import EnvironmentConfig, HoldemEnvironment
from .equity import estimate_equity
from .models import ActionType, Decision, DecisionContext
from .regime_detection import OpponentObservation, OpponentWorld

RESPONSE_POLICIES = ("pressure", "balanced", "bluff_catch")


def response_policy_decision(
    agent: PokerAgent,
    context: DecisionContext,
    response_policy: str,
    *,
    metadata: dict[str, object] | None = None,
) -> Decision:
    """Apply the same response policy in live play and counterfactual rollout."""

    if response_policy not in RESPONSE_POLICIES:
        raise ValueError(f"Unknown response policy: {response_policy}")
    equity = estimate_equity(
        context.hole_cards,
        context.board,
        max(1, context.active_players - 1),
        agent.rng,
        agent.style.equity_samples,
    )
    pot_odds = context.to_call / max(context.pot + context.to_call, 1e-9)
    can_raise = ActionType.RAISE in context.legal_actions
    aggression = agent.style.aggression
    call_margin = agent.style.risk_margin
    raise_scale = 0.54
    if response_policy == "pressure":
        aggression = min(aggression + 0.22, 0.97)
        call_margin += 0.01
        raise_scale = 0.56
    elif response_policy == "bluff_catch":
        aggression = max(aggression - 0.08, 0.04)
        if context.to_call > 0:
            call_margin += 0.13
        raise_scale = 0.48

    if context.to_call > 0:
        raise_threshold = 0.80 - 0.11 * aggression
        if response_policy == "bluff_catch":
            raise_threshold += 0.08
        if can_raise and equity > raise_threshold:
            action = ActionType.RAISE
        elif equity + 0.02 >= pot_odds - call_margin:
            action = ActionType.CHECK_CALL
        else:
            action = ActionType.FOLD
    else:
        threshold = 0.66 - 0.18 * aggression
        if response_policy == "pressure":
            threshold -= 0.10
        elif response_policy == "bluff_catch":
            threshold += 0.06
        action = (
            ActionType.RAISE
            if can_raise and equity > threshold
            else ActionType.CHECK_CALL
        )

    return Decision(
        action=action,
        raise_scale=raise_scale,
        equity=equity,
        reasoning_depth=2,
        reasoning_ops=7,
        metadata=metadata or {},
    )


class _ResponsePolicyAgent(PokerAgent):
    def __init__(
        self,
        name: str,
        seed: int,
        response_policy: str,
        style: AgentStyle,
    ) -> None:
        super().__init__(name, seed, style)
        self.response_policy = response_policy
        self.condition = f"rollout_{response_policy}"

    def act(self, context: DecisionContext) -> Decision:
        return response_policy_decision(self, context, self.response_policy)


class _WorldPolicyOpponent(PokerAgent):
    def __init__(
        self,
        name: str,
        seed: int,
        world: OpponentWorld,
        style: AgentStyle,
    ) -> None:
        super().__init__(name, seed, style)
        self.world = world
        self.condition = f"rollout_world_{world.name}"

    def _weights(self, context: DecisionContext) -> dict[ActionType, float]:
        facing_bet = context.to_call > 1e-9
        base = self.world.action_probabilities(facing_bet)
        return {
            action: max(base[action], 1e-9)
            for action in context.legal_actions
        }

    def act(self, context: DecisionContext) -> Decision:
        weights = self._weights(context)
        draw = self.rng.random() * sum(weights.values())
        cumulative = 0.0
        action = ActionType.CHECK_CALL
        for candidate in context.legal_actions:
            cumulative += weights[candidate]
            if draw <= cumulative:
                action = candidate
                break
        return Decision(
            action=action,
            raise_scale=0.55,
            equity=0.5,
            reasoning_depth=0,
            reasoning_ops=1,
            metadata={"rollout_world": self.world.name},
        )


@dataclass(frozen=True)
class SimulationResult:
    world_name: str
    posterior: float
    best_response: str
    expected_bb_per_decision: float
    response_values: dict[str, float]
    rollout_hands: int
    response_samples: dict[str, tuple[float, ...]] = field(repr=False)
    simulation_unit: str = "full_hand"

    @property
    def expected_bb_per_hand(self) -> float:
        return self.expected_bb_per_decision


class WorldSimulator:
    """Evaluate candidate worlds with paired full-hand Hold'em rollouts.

    Every response policy receives the same deck and seat seeds for a given
    candidate world. The simulator therefore compares complete hand outcomes,
    including folds, betting, side pots, and showdown, rather than a static
    action-payoff table.
    """

    def __init__(
        self,
        rollouts: int = 48,
        seed: int = 0,
        *,
        equity_samples: int = 1,
        starting_stack: float = 100.0,
        max_raises_per_street: int | None = 2,
    ) -> None:
        if rollouts < 2:
            raise ValueError("rollouts must be at least two")
        if equity_samples < 1:
            raise ValueError("equity_samples must be positive")
        self.rollouts = rollouts
        self.seed = seed
        self.equity_samples = equity_samples
        self.starting_stack = starting_stack
        self.max_raises_per_street = max_raises_per_street
        self.calls = 0
        self.simulated_hands = 0

    @staticmethod
    def _posterior(
        worlds: Sequence[OpponentWorld],
        observations: Sequence[OpponentObservation],
    ) -> list[float]:
        log_weights = [
            math.log(world.prior)
            + sum(
                math.log(world.probability(observation))
                for observation in observations
            )
            for world in worlds
        ]
        anchor = max(log_weights)
        weights = [math.exp(value - anchor) for value in log_weights]
        total = sum(weights)
        return [weight / total for weight in weights]

    @staticmethod
    def _stable_name_seed(name: str) -> int:
        return sum(
            (index + 1) * ord(character)
            for index, character in enumerate(name)
        )

    def _rollout_rewards(
        self,
        world: OpponentWorld,
        response_policy: str,
    ) -> tuple[float, ...]:
        first_mirror_hands = (self.rollouts + 1) // 2
        rewards: list[float] = []
        world_seed = self._stable_name_seed(world.name)
        response_index = RESPONSE_POLICIES.index(response_policy)
        for mirror in (0, 1):
            hands = (
                first_mirror_hands
                if mirror == 0
                else self.rollouts - first_mirror_hands
            )
            if hands <= 0:
                continue
            environment_seed = self.seed + world_seed * 101 + mirror * 100_003
            hero = _ResponsePolicyAgent(
                name="hero",
                seed=environment_seed + 7_001 + response_index * 997,
                response_policy=response_policy,
                style=AgentStyle(
                    aggression=0.40,
                    risk_margin=-0.045,
                    equity_samples=self.equity_samples,
                ),
            )
            opponent = _WorldPolicyOpponent(
                name="opponent",
                seed=environment_seed + 11_003,
                world=world,
                style=AgentStyle(equity_samples=self.equity_samples),
            )
            agents = [hero, opponent] if mirror == 0 else [opponent, hero]
            records = HoldemEnvironment(
                agents,
                seed=environment_seed,
                config=EnvironmentConfig(
                    starting_stack=self.starting_stack,
                    small_blind=0.5,
                    big_blind=1.0,
                    max_raises_per_street=self.max_raises_per_street,
                    regime_switch_hand=hands + 1,
                ),
            ).play(hands)
            rewards.extend(record.rewards["hero"] for record in records)
        self.simulated_hands += len(rewards)
        return tuple(rewards)

    def evaluate(
        self,
        worlds: Sequence[OpponentWorld],
        observations: Sequence[OpponentObservation],
    ) -> list[SimulationResult]:
        if not worlds:
            raise ValueError("At least one world is required")
        self.calls += 1
        posteriors = self._posterior(worlds, observations)
        results: list[SimulationResult] = []
        for world, posterior in zip(worlds, posteriors, strict=True):
            samples = {
                response: self._rollout_rewards(world, response)
                for response in RESPONSE_POLICIES
            }
            values = {
                response: statistics.fmean(response_samples)
                for response, response_samples in samples.items()
            }
            best_response = max(values, key=values.get)
            results.append(
                SimulationResult(
                    world_name=world.name,
                    posterior=posterior,
                    best_response=best_response,
                    expected_bb_per_decision=values[best_response],
                    response_values=values,
                    rollout_hands=self.rollouts,
                    response_samples=samples,
                )
            )
        return sorted(results, key=lambda result: result.posterior, reverse=True)

    @staticmethod
    def choose_response(
        results: Sequence[SimulationResult],
    ) -> tuple[str, float]:
        if not results:
            return "balanced", 0.0
        aggregate: Counter[str] = Counter()
        for result in results:
            for response, value in result.response_values.items():
                aggregate[response] += result.posterior * value
        best = max(aggregate, key=aggregate.get)
        return best, aggregate[best]

    @staticmethod
    def choose_response_robust(
        results: Sequence[SimulationResult],
        *,
        safe_policy: str = "balanced",
        confidence_z: float = 1.645,
        minimum_improvement: float = 0.0,
    ) -> tuple[str, float, float]:
        """Choose a response only when its paired lower bound beats the safe policy."""

        if safe_policy not in RESPONSE_POLICIES:
            raise ValueError(f"Unknown safe policy: {safe_policy}")
        if not results:
            return safe_policy, 0.0, 0.0
        aggregate: Counter[str] = Counter()
        for result in results:
            for response, value in result.response_values.items():
                aggregate[response] += result.posterior * value
        candidates: list[tuple[float, str, float]] = []
        for response in RESPONSE_POLICIES:
            if response == safe_policy:
                continue
            sample_count = min(
                len(result.response_samples[response])
                for result in results
            )
            deltas = [
                sum(
                    result.posterior
                    * (
                        result.response_samples[response][index]
                        - result.response_samples[safe_policy][index]
                    )
                    for result in results
                )
                for index in range(sample_count)
            ]
            mean_delta = statistics.fmean(deltas)
            standard_error = (
                statistics.stdev(deltas) / math.sqrt(len(deltas))
                if len(deltas) > 1
                else math.inf
            )
            lower_bound = mean_delta - confidence_z * standard_error
            if lower_bound > minimum_improvement:
                candidates.append((aggregate[response], response, lower_bound))
        if not candidates:
            return safe_policy, aggregate[safe_policy], 0.0
        expected_value, response, lower_bound = max(candidates)
        return response, expected_value, lower_bound
