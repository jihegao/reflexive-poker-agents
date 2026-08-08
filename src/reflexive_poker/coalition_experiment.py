from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .agents import AgentStyle, PokerAgent
from .environment import EnvironmentConfig, HoldemEnvironment
from .models import (
    ActionEvent,
    ActionType,
    Decision,
    DecisionContext,
    HandRecord,
)
from .tournament_agents import make_tournament_agent

COALITION_CONDITIONS = tuple(
    (team_reward, reflection, joint_simulation)
    for team_reward in (False, True)
    for reflection in (False, True)
    for joint_simulation in (False, True)
)
BASELINE_CONDITION = (False, False, False)
BASELINE_TYPES = ("tag", "lag", "rock", "calling_station")


@dataclass(frozen=True)
class CoalitionConfig:
    """Small, paired coalition pilot configuration.

    This runner is intentionally rule-based. It tests the information and
    accounting contracts before any provider calls are introduced.
    """

    seeds: tuple[int, ...] = (9400,)
    hands: int = 12
    seat_mirrors: tuple[int, ...] = (0, 1)
    equity_samples: int = 32
    lambda_partner: float = 0.5
    output_dir: Path = Path("results/coalition/mock_smoke")


@dataclass(frozen=True)
class CoalitionCondition:
    team_reward: bool
    reflection: bool
    joint_simulation: bool

    @property
    def condition_id(self) -> str:
        return f"t{int(self.team_reward)}r{int(self.reflection)}s{int(self.joint_simulation)}"


def condition_grid() -> tuple[CoalitionCondition, ...]:
    return tuple(CoalitionCondition(*values) for values in COALITION_CONDITIONS)


class CoalitionAgent(PokerAgent):
    """A public-action-only partner agent for the mechanism smoke test.

    The joint simulation is a bounded deterministic proxy, not a claim of a
    poker solver. It scores legal actions using the agent's own equity and a
    small public partner-signal bonus. The bonus is only active when the team
    objective is enabled, making the 2x2x2 attribution explicit.
    """

    condition = "coalition"

    def __init__(
        self,
        name: str,
        seed: int,
        style: AgentStyle,
        *,
        partner_name: str,
        team_reward: bool,
        reflection: bool,
        joint_simulation: bool,
        lambda_partner: float,
    ) -> None:
        super().__init__(name, seed, style)
        self.partner_name = partner_name
        self.team_reward = team_reward
        self.reflection_enabled = reflection
        self.joint_simulation_enabled = joint_simulation
        self.lambda_partner = lambda_partner
        self.public_history: list[dict[str, Any]] = []
        self.partner_actions_by_hand: dict[int, list[str]] = {}
        self.partner_aggression = 0.5
        self.reflection_count = 0
        self.simulation_calls = 0
        self.coordination_actions = 0
        self.private_information_accesses = 0

    @property
    def condition_id(self) -> str:
        return (
            f"t{int(self.team_reward)}r{int(self.reflection_enabled)}"
            f"s{int(self.joint_simulation_enabled)}"
        )

    def observe_action(self, event: ActionEvent) -> None:
        super().observe_action(event)
        self.public_history.append(
            {
                "hand_index": event.hand_index,
                "street": event.street.value,
                "actor": event.actor,
                "action": event.action.value,
                "paid": event.paid,
            }
        )
        if event.actor == self.partner_name:
            self.partner_actions_by_hand.setdefault(event.hand_index, []).append(
                event.action.value
            )

    def _partner_raised_this_hand(self, hand_index: int) -> bool:
        return ActionType.RAISE.value in self.partner_actions_by_hand.get(hand_index, [])

    def _expected_utility(
        self, context: DecisionContext, action: ActionType, equity: float
    ) -> float:
        call = context.to_call
        if action is ActionType.FOLD:
            return -call
        if action is ActionType.CHECK_CALL:
            return equity * (context.pot + call) - call
        raise_cost = max(context.pot * 0.50, context.to_call + 1.0)
        return equity * (context.pot + call + raise_cost) - call - raise_cost

    def _joint_action(self, context: DecisionContext, equity: float, fallback: Decision) -> Decision:
        self.simulation_calls += 1
        legal = context.legal_actions
        partner_signal = self._partner_raised_this_hand(context.hand_index)
        scored: list[tuple[float, ActionType]] = []
        for action in legal:
            own_value = self._expected_utility(context, action, equity)
            partner_value = 0.0
            if self.team_reward:
                partner_value = self.lambda_partner * (
                    0.03 * max(context.pot, 1.0)
                    if action is not ActionType.FOLD
                    else -0.03 * max(context.pot, 1.0)
                )
                if partner_signal and action is ActionType.RAISE:
                    partner_value += self.lambda_partner * 0.06 * max(context.pot, 1.0)
            scored.append((own_value + partner_value, action))
        action = max(scored, key=lambda row: (row[0], row[1].value))[1]
        if action is ActionType.RAISE:
            if partner_signal and self.team_reward:
                self.coordination_actions += 1
            return Decision(
                action=action,
                raise_scale=max(0.5, fallback.raise_scale),
                equity=equity,
                reasoning_depth=1,
                reasoning_ops=3,
                metadata={
                    "coalition_condition": self.condition_id,
                    "partner_signal": partner_signal,
                    "public_only": True,
                },
            )
        return Decision(
            action=action,
            equity=equity,
            reasoning_depth=1,
            reasoning_ops=3,
            metadata={
                "coalition_condition": self.condition_id,
                "partner_signal": partner_signal,
                "public_only": True,
            },
        )

    def act(self, context: DecisionContext) -> Decision:
        # ``_policy`` uses only the agent's own cards and public context. The
        # partner's cards are never present in this state or in the metadata.
        partner_signal = self._partner_raised_this_hand(context.hand_index)
        aggression_shift = 0.08 if self.team_reward else 0.0
        if self.reflection_enabled:
            aggression_shift += 0.12 * (self.partner_aggression - 0.5)
        fallback = self._policy(
            context,
            aggression_shift=aggression_shift,
            reasoning_depth=1 if self.reflection_enabled else 0,
            metadata={
                "coalition_condition": self.condition_id,
                "partner_signal": partner_signal,
                "public_only": True,
            },
        )
        if (
            self.team_reward
            and not self.joint_simulation_enabled
            and ActionType.RAISE in context.legal_actions
            and fallback.equity > 0.40
        ):
            return Decision(
                action=ActionType.RAISE,
                raise_scale=max(0.5, fallback.raise_scale),
                equity=fallback.equity,
                reasoning_depth=fallback.reasoning_depth,
                reasoning_ops=fallback.reasoning_ops,
                metadata={
                    "coalition_condition": self.condition_id,
                    "partner_signal": partner_signal,
                    "public_only": True,
                    "team_support_policy": True,
                },
            )
        if (
            self.reflection_enabled
            and partner_signal
            and ActionType.RAISE in context.legal_actions
            and fallback.equity > 0.30
        ):
            return Decision(
                action=ActionType.RAISE,
                raise_scale=max(0.5, fallback.raise_scale),
                equity=fallback.equity,
                reasoning_depth=fallback.reasoning_depth,
                reasoning_ops=fallback.reasoning_ops,
                metadata={
                    "coalition_condition": self.condition_id,
                    "partner_signal": partner_signal,
                    "public_only": True,
                    "reflection_response_policy": True,
                },
            )
        if self.joint_simulation_enabled:
            equity = fallback.equity
            return self._joint_action(context, equity, fallback)
        return fallback

    def on_hand_end(self, record: HandRecord) -> None:
        super().on_hand_end(record)
        if not self.reflection_enabled:
            return
        partner_actions = [event for event in record.actions if event.actor == self.partner_name]
        if partner_actions:
            raises = sum(event.action is ActionType.RAISE for event in partner_actions)
            observed = raises / len(partner_actions)
            self.partner_aggression = 0.8 * self.partner_aggression + 0.2 * observed
        self.reflection_count += 1

    def audit_state(self) -> dict[str, Any]:
        return {
            "agent": self.name,
            "partner": self.partner_name,
            "condition": self.condition_id,
            "public_history_events": len(self.public_history),
            "reflection_count": self.reflection_count,
            "simulation_calls": self.simulation_calls,
            "coordination_actions": self.coordination_actions,
            "private_information_accesses": self.private_information_accesses,
            "public_only": True,
        }


def _pair_positions(seat_mirror: int) -> tuple[int, int]:
    if seat_mirror == 0:
        return (0, 1)
    if seat_mirror == 1:
        return (1, 2)
    raise ValueError(f"unsupported seat_mirror {seat_mirror}; use 0 or 1")


def _make_environment(
    *, seed: int, seat_mirror: int, condition: CoalitionCondition, config: CoalitionConfig
) -> tuple[HoldemEnvironment, tuple[str, str]]:
    pair_positions = _pair_positions(seat_mirror)
    names = [f"seat_{index}_control" for index in range(6)]
    pair_names = ("coalition_a", "coalition_b")
    names[pair_positions[0]] = pair_names[0]
    names[pair_positions[1]] = pair_names[1]
    style = AgentStyle(
        aggression=0.47,
        risk_margin=0.055,
        belief_sensitivity=0.24,
        social_learning_rate=0.20,
        equity_samples=config.equity_samples,
    )
    agents_by_position: list[PokerAgent] = [None] * len(names)  # type: ignore[list-item]
    for position, name in enumerate(names):
        if position in pair_positions:
            partner = pair_names[1] if name == pair_names[0] else pair_names[0]
            agents_by_position[position] = CoalitionAgent(
                name,
                seed * 1009 + position + 1,
                style,
                partner_name=partner,
                team_reward=condition.team_reward,
                reflection=condition.reflection,
                joint_simulation=condition.joint_simulation,
                lambda_partner=config.lambda_partner,
            )
        else:
            control_type = BASELINE_TYPES[(position + seat_mirror) % len(BASELINE_TYPES)]
            agents_by_position[position] = make_tournament_agent(
                control_type,
                name,
                tuple(other for other in names if other != name),
                seed * 1009 + position + 1,
                equity_samples=config.equity_samples,
            )
    environment = HoldemEnvironment(
        agents_by_position,
        seed=seed,
        config=EnvironmentConfig(
            max_raises_per_street=None,
            regime_switch_hand=config.hands + 1,
        ),
    )
    return environment, pair_names


def _dependency(agent: CoalitionAgent) -> float:
    decisions = [row for row in agent.decision_log if "partner_signal" in row]
    signal = [row for row in decisions if row["partner_signal"]]
    no_signal = [row for row in decisions if not row["partner_signal"]]
    if not signal or not no_signal:
        return 0.0
    signal_raise = sum(row["action"] == ActionType.RAISE.value for row in signal) / len(signal)
    no_signal_raise = sum(row["action"] == ActionType.RAISE.value for row in no_signal) / len(
        no_signal
    )
    return signal_raise - no_signal_raise


def _hand_rows(
    records: list[HandRecord], *, seed: int, seat_mirror: int, condition: CoalitionCondition
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        pair_reward = record.rewards["coalition_a"] + record.rewards["coalition_b"]
        control_reward = sum(
            reward for name, reward in record.rewards.items() if name not in {"coalition_a", "coalition_b"}
        )
        rows.append(
            {
                "seed": seed,
                "seat_mirror": seat_mirror,
                "condition": condition.condition_id,
                "hand_index": record.hand_index,
                "pair_reward": pair_reward,
                "control_reward": control_reward,
                "showdown": record.showdown,
                "action_count": len(record.actions),
            }
        )
    return rows


def _seed_row(
    records: list[HandRecord],
    agents: list[PokerAgent],
    *,
    seed: int,
    seat_mirror: int,
    condition: CoalitionCondition,
    pair_names: tuple[str, str],
    hands: int,
) -> dict[str, Any]:
    pair_reward = sum(
        record.rewards[pair_names[0]] + record.rewards[pair_names[1]] for record in records
    )
    controls_reward = sum(
        reward
        for record in records
        for name, reward in record.rewards.items()
        if name not in pair_names
    )
    pair_agents = [agent for agent in agents if isinstance(agent, CoalitionAgent)]
    dependency = sum(_dependency(agent) for agent in pair_agents) / len(pair_agents)
    return {
        "seed": seed,
        "seat_mirror": seat_mirror,
        "condition": condition.condition_id,
        "team_reward": condition.team_reward,
        "reflection": condition.reflection,
        "joint_simulation": condition.joint_simulation,
        "hands": hands,
        "pair_reward": pair_reward,
        "control_reward": controls_reward,
        "pair_chips_per_100": 100.0 * pair_reward / hands,
        "control_chips_per_100": 100.0 * controls_reward / (4 * hands),
        "partner_action_dependency": dependency,
        "simulation_calls": sum(agent.simulation_calls for agent in pair_agents),
        "reflection_calls": sum(agent.reflection_count for agent in pair_agents),
        "coordination_actions": sum(agent.coordination_actions for agent in pair_agents),
        "private_information_accesses": sum(
            agent.private_information_accesses for agent in pair_agents
        ),
        "public_only": all(agent.audit_state()["public_only"] for agent in pair_agents),
    }


def _interaction(per_seed: pd.DataFrame) -> dict[str, float | None]:
    means = per_seed.groupby("condition")["pair_chips_per_100"].mean().to_dict()
    required = ("t1r1s1", "t1r1s0", "t1r0s1", "t1r0s0")
    if not all(condition in means for condition in required):
        return {"mean_pair_chips_per_100": None, "r_x_s_given_team": None}
    return {
        "mean_pair_chips_per_100": float(means["t1r1s1"]),
        "r_x_s_given_team": float(
            means["t1r1s1"] - means["t1r1s0"] - means["t1r0s1"] + means["t1r0s0"]
        ),
    }


def run_coalition_experiment(config: CoalitionConfig) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Run the paired 2x2x2 coalition mechanism pilot."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows: list[dict[str, Any]] = []
    hand_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for seed in config.seeds:
        for seat_mirror in config.seat_mirrors:
            for condition in condition_grid():
                environment, pair_names = _make_environment(
                    seed=seed, seat_mirror=seat_mirror, condition=condition, config=config
                )
                records = environment.play(config.hands)
                per_seed_rows.append(
                    _seed_row(
                        records,
                        environment.agents,
                        seed=seed,
                        seat_mirror=seat_mirror,
                        condition=condition,
                        pair_names=pair_names,
                        hands=config.hands,
                    )
                )
                hand_rows.extend(
                    _hand_rows(
                        records,
                        seed=seed,
                        seat_mirror=seat_mirror,
                        condition=condition,
                    )
                )
                audits.extend(
                    agent.audit_state()
                    for agent in environment.agents
                    if isinstance(agent, CoalitionAgent)
                )

    per_seed = pd.DataFrame(per_seed_rows).sort_values(["seed", "seat_mirror", "condition"])
    hand_level = pd.DataFrame(hand_rows).sort_values(["seed", "seat_mirror", "condition", "hand_index"])
    baseline = per_seed[per_seed["condition"] == "t0r0s0"][
        ["seed", "seat_mirror", "pair_chips_per_100"]
    ].rename(columns={"pair_chips_per_100": "baseline_pair_chips_per_100"})
    per_seed = per_seed.merge(baseline, on=["seed", "seat_mirror"], how="left")
    per_seed["coalition_surplus_chips_per_100"] = (
        per_seed["pair_chips_per_100"] - per_seed["baseline_pair_chips_per_100"]
    )
    summary = (
        per_seed.groupby("condition", as_index=False)
        .agg(
            blocks=("condition", "size"),
            mean_pair_chips_per_100=("pair_chips_per_100", "mean"),
            mean_coalition_surplus=("coalition_surplus_chips_per_100", "mean"),
            mean_partner_action_dependency=("partner_action_dependency", "mean"),
            mean_simulation_calls=("simulation_calls", "mean"),
            mean_reflection_calls=("reflection_calls", "mean"),
            private_information_accesses=("private_information_accesses", "sum"),
        )
        .sort_values("condition")
    )
    interaction = _interaction(per_seed)
    metadata = {
        "protocol": "coalition-pilot-v1",
        "claim_status": "mechanism_smoke_only",
        "formal_conclusion_allowed": False,
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "conditions": [asdict(condition) | {"condition_id": condition.condition_id} for condition in condition_grid()],
        "interaction": interaction,
        "information_boundary": {
            "private_cards_shared": False,
            "public_actions_only": True,
            "private_information_accesses": int(per_seed["private_information_accesses"].sum()),
        },
        "audits": audits,
    }
    per_seed.to_csv(config.output_dir / "per_seed.csv", index=False)
    hand_level.to_csv(config.output_dir / "per_hand.csv", index=False)
    summary.to_csv(config.output_dir / "summary.csv", index=False)
    (config.output_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (config.output_dir / "design.md").write_text(
        "# Coalition 2x2x2 mock pilot\n\n"
        "Two coalition agents play against four independent controls. The factors are "
        "team reward, public-action reflection, and bounded joint simulation. No private "
        "hole cards are shared and this smoke result is not formal payoff evidence.\n",
        encoding="utf-8",
    )
    return {"per_seed": per_seed, "per_hand": hand_level, "summary": summary, "metadata": metadata}
