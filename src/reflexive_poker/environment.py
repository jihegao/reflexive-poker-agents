from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from .agents import PokerAgent
from .cards import make_deck
from .evaluator import best_hand_rank
from .models import (
    ActionEvent,
    ActionType,
    DecisionContext,
    HandRecord,
    ResponseEvent,
    Street,
)


@dataclass
class EnvironmentConfig:
    starting_stack: float = 100.0
    small_blind: float = 0.5
    big_blind: float = 1.0
    max_raises_per_street: int | None = 2
    regime_switch_hand: int = 120


class HoldemEnvironment:
    """A reproducible multi-player Texas Hold'em environment.

    Cards, blinds, four streets, folding, side pots, and showdown are standard.
    Set ``max_raises_per_street=None`` for an uncapped no-limit betting experiment.
    """

    def __init__(
        self, agents: list[PokerAgent], seed: int, config: EnvironmentConfig | None = None
    ) -> None:
        if len(agents) < 2:
            raise ValueError("Texas Hold'em requires at least two agents")
        self.agents = agents
        self.agent_by_name = {agent.name: agent for agent in agents}
        if len(self.agent_by_name) != len(agents):
            raise ValueError("Agent names must be unique")
        self.rng = random.Random(seed)
        self.config = config or EnvironmentConfig()
        self.records: list[HandRecord] = []

    def regime_for_hand(self, hand_index: int) -> str:
        return "stable" if hand_index < self.config.regime_switch_hand else "shifted"

    def play(self, hands: int) -> list[HandRecord]:
        start_hand = len(self.records)
        for hand_index in range(start_hand, start_hand + hands):
            self.records.append(self.play_hand(hand_index))
        return self.records

    def play_hand(self, hand_index: int) -> HandRecord:
        n = len(self.agents)
        button = hand_index % n
        deck = make_deck()
        self.rng.shuffle(deck)
        cursor = 0
        holes: dict[str, tuple[int, int]] = {}
        for offset in range(n):
            seat = (button + 1 + offset) % n
            holes[self.agents[seat].name] = (deck[cursor], deck[cursor + 1])
            cursor += 2
        board = tuple(deck[cursor : cursor + 5])

        stacks = [self.config.starting_stack for _ in self.agents]
        total_contrib = [0.0 for _ in self.agents]
        active = [True for _ in self.agents]
        all_in = [False for _ in self.agents]
        actions: list[ActionEvent] = []

        sb = (button + 1) % n
        bb = (button + 2) % n
        self._pay(sb, self.config.small_blind, stacks, total_contrib)
        self._pay(bb, self.config.big_blind, stacks, total_contrib)

        streets: list[tuple[Street, tuple[int, ...], int]] = [
            (Street.PREFLOP, (), (bb + 1) % n),
            (Street.FLOP, board[:3], (button + 1) % n),
            (Street.TURN, board[:4], (button + 1) % n),
            (Street.RIVER, board[:5], (button + 1) % n),
        ]
        preflop_contrib = [0.0 for _ in self.agents]
        preflop_contrib[sb] = min(self.config.small_blind, self.config.starting_stack)
        preflop_contrib[bb] = min(self.config.big_blind, self.config.starting_stack)

        for street, visible_board, start in streets:
            if sum(active) <= 1:
                break
            street_contrib = (
                preflop_contrib[:] if street == Street.PREFLOP else [0.0 for _ in self.agents]
            )
            self._betting_round(
                hand_index=hand_index,
                street=street,
                visible_board=visible_board,
                start=start,
                button=button,
                holes=holes,
                stacks=stacks,
                total_contrib=total_contrib,
                street_contrib=street_contrib,
                active=active,
                all_in=all_in,
                actions=actions,
            )

        live_indices = [idx for idx, is_active in enumerate(active) if is_active]
        showdown = len(live_indices) > 1
        if len(live_indices) == 1:
            ranks: dict[int, tuple[int, ...]] = {}
        else:
            ranks = {
                idx: best_hand_rank((*holes[self.agents[idx].name], *board)) for idx in live_indices
            }
        winners = self._award_pots(stacks, total_contrib, live_indices, ranks)
        rewards = {
            agent.name: stacks[idx] - self.config.starting_stack
            for idx, agent in enumerate(self.agents)
        }
        winner_names = tuple(self.agents[idx].name for idx in winners)
        regime = self.regime_for_hand(hand_index)
        snapshots = {agent.name: agent.snapshot() for agent in self.agents}
        record = HandRecord(
            hand_index=hand_index,
            button=button,
            board=board,
            hole_cards=holes,
            actions=actions,
            rewards=rewards,
            winners=winner_names,
            showdown=showdown,
            regime=regime,
            snapshots=snapshots,
        )
        for agent in self.agents:
            agent.on_hand_end(record)
        return record

    @staticmethod
    def _award_pots(
        stacks: list[float],
        total_contrib: list[float],
        live_indices: list[int],
        ranks: dict[int, tuple[int, ...]],
    ) -> list[int]:
        """Award the main pot and any side pots from total hand contributions."""
        awarded: list[int] = []
        previous_level = 0.0
        for level in sorted({contribution for contribution in total_contrib if contribution > 0.0}):
            contributors = [
                idx for idx, contribution in enumerate(total_contrib) if contribution >= level
            ]
            pot = (level - previous_level) * len(contributors)
            eligible = [idx for idx in live_indices if total_contrib[idx] >= level]
            if not eligible:
                previous_level = level
                continue
            if len(eligible) == 1:
                winners = eligible
            else:
                best_rank = max(ranks[idx] for idx in eligible)
                winners = [idx for idx in eligible if ranks[idx] == best_rank]
            share = pot / len(winners)
            for idx in winners:
                stacks[idx] += share
            awarded.extend(winners)
            previous_level = level
        return sorted(set(awarded))

    def _betting_round(
        self,
        *,
        hand_index: int,
        street: Street,
        visible_board: tuple[int, ...],
        start: int,
        button: int,
        holes: dict[str, tuple[int, int]],
        stacks: list[float],
        total_contrib: list[float],
        street_contrib: list[float],
        active: list[bool],
        all_in: list[bool],
        actions: list[ActionEvent],
    ) -> None:
        n = len(self.agents)
        current_bet = max(street_contrib)
        pending = {idx for idx in range(n) if active[idx] and not all_in[idx]}
        actor = start
        raises = 0
        last_full_raise_increment = self.config.big_blind
        last_raiser: int | None = None
        guard = 0

        while pending and sum(active) > 1:
            guard += 1
            if guard > 100:
                raise RuntimeError("Betting round failed to converge")
            if actor not in pending or not active[actor] or all_in[actor]:
                actor = (actor + 1) % n
                continue

            to_call = max(0.0, current_bet - street_contrib[actor])
            legal: list[ActionType] = [ActionType.CHECK_CALL]
            if to_call > 1e-9:
                legal.insert(0, ActionType.FOLD)
            can_raise = (
                (
                    self.config.max_raises_per_street is None
                    or raises < self.config.max_raises_per_street
                )
                and stacks[actor] >= to_call + last_full_raise_increment
                and sum(active) > 1
            )
            if can_raise:
                legal.append(ActionType.RAISE)

            opponent_names = tuple(
                self.agents[idx].name for idx in range(n) if idx != actor and active[idx]
            )
            context = DecisionContext(
                hand_index=hand_index,
                street=street,
                player_name=self.agents[actor].name,
                hole_cards=holes[self.agents[actor].name],
                board=visible_board,
                pot=sum(total_contrib),
                to_call=to_call,
                stack=stacks[actor],
                current_bet=current_bet,
                legal_actions=tuple(legal),
                active_players=sum(active),
                opponents=opponent_names,
                last_raiser=self.agents[last_raiser].name if last_raiser is not None else None,
                raises_this_street=raises,
                button_distance=(actor - button) % n,
                environment_regime=self.regime_for_hand(hand_index),
            )
            decision = self.agents[actor].act(context)
            if decision.action not in legal:
                decision_action = ActionType.CHECK_CALL
            else:
                decision_action = decision.action

            pot_before = sum(total_contrib)
            faced_raise = to_call > 1e-9 and last_raiser is not None
            raiser_name = (
                self.agents[last_raiser].name if faced_raise and last_raiser is not None else None
            )
            paid = 0.0
            if decision_action == ActionType.FOLD:
                active[actor] = False
                pending.discard(actor)
            elif decision_action == ActionType.CHECK_CALL:
                paid = self._pay(actor, to_call, stacks, total_contrib)
                street_contrib[actor] += paid
                if stacks[actor] <= 1e-9:
                    all_in[actor] = True
                pending.discard(actor)
            else:
                call_paid = self._pay(actor, to_call, stacks, total_contrib)
                street_contrib[actor] += call_paid
                pot_after_call = sum(total_contrib)
                increment = (
                    stacks[actor]
                    if decision.raise_scale >= 1.20
                    else max(
                        last_full_raise_increment, decision.raise_scale * max(pot_after_call, 1.0)
                    )
                )
                raise_paid = self._pay(actor, increment, stacks, total_contrib)
                street_contrib[actor] += raise_paid
                paid = call_paid + raise_paid
                current_bet = street_contrib[actor]
                last_raiser = actor
                raises += 1
                last_full_raise_increment = raise_paid
                if stacks[actor] <= 1e-9:
                    all_in[actor] = True
                pending = {
                    idx for idx in range(n) if idx != actor and active[idx] and not all_in[idx]
                }

            event = ActionEvent(
                hand_index=hand_index,
                street=street,
                actor=self.agents[actor].name,
                action=decision_action,
                faced_raise=faced_raise,
                raiser=raiser_name,
                to_call=to_call,
                paid=paid,
                pot_before=pot_before,
                active_players=sum(active),
            )
            actions.append(event)
            for observer in self.agents:
                observer.observe_action(event)
            self.agents[actor].on_own_action(event)

            if faced_raise and raiser_name is not None:
                response = ResponseEvent(
                    hand_index=hand_index,
                    street=street,
                    raiser=raiser_name,
                    responder=self.agents[actor].name,
                    folded=decision_action == ActionType.FOLD,
                    reraised=decision_action == ActionType.RAISE,
                    pot_before=pot_before,
                )
                self.agent_by_name[raiser_name].observe_response(response)

            actor = (actor + 1) % n

    @staticmethod
    def _pay(player: int, amount: float, stacks: list[float], total_contrib: list[float]) -> float:
        paid = min(max(amount, 0.0), stacks[player])
        stacks[player] -= paid
        total_contrib[player] += paid
        return paid


def mean_ground_truth_image(agents: Iterable[PokerAgent], target_name: str) -> float:
    values = [
        agent.beliefs[target_name].aggression.mean for agent in agents if agent.name != target_name
    ]
    return sum(values) / len(values) if values else 0.5
