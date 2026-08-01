from __future__ import annotations

import random
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .cards import card_to_str, make_deck
from .evaluator import best_hand_rank
from .models import ActionEvent, ActionType, Decision, DecisionContext, Street
from .tournament_agents import make_tournament_agent

DEMO_OPPONENT_TYPES = ("rock", "tag", "lag", "calling_station", "myopic")
RAISE_SCALES = (0.25, 0.5, 0.75, 1.0, 1.25)
STREETS = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)
VISIBLE_BOARD_COUNT = (0, 3, 4, 5)


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 9200
    opponents: tuple[str, ...] = ("tag", "lag", "rock", "calling_station", "myopic")
    starting_stack: float = 100.0
    small_blind: float = 0.5
    big_blind: float = 1.0
    equity_samples: int = 32
    provider_mode: str = "mock"

    def __post_init__(self) -> None:
        if len(self.opponents) != 5:
            raise ValueError("The demo requires exactly five opponents")
        unknown = [value for value in self.opponents if value not in DEMO_OPPONENT_TYPES]
        if unknown:
            raise ValueError(f"Unsupported demo opponents: {unknown}")
        if self.provider_mode not in {"mock", "live_aliyun"}:
            raise ValueError("Unsupported provider mode")


@dataclass
class HandState:
    hand_index: int
    button: int
    board: list[int]
    holes: list[list[int]]
    stacks: list[float]
    total_contrib: list[float]
    street_contrib: list[float]
    active: list[bool]
    all_in: list[bool]
    street_index: int
    actor: int
    pending: list[int]
    current_bet: float
    raises: int
    last_full_raise_increment: float
    last_raiser: int | None
    actions: list[dict[str, Any]] = field(default_factory=list)
    complete: bool = False
    showdown: bool = False
    winners: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


def _round_bb(value: float) -> float:
    return round(float(value) + 1e-10, 2)


class DemoTable:
    """Serializable, step-wise six-max Hold'em table for the local web demo."""

    hero_seat = 0

    def __init__(
        self,
        config: DemoConfig,
        *,
        table_id: str | None = None,
        auto_start: bool = True,
    ) -> None:
        self.table_id = table_id or f"table_{uuid.uuid4().hex[:10]}"
        self.config = config
        self.version = 0
        self.controller = "human"
        self.opponent_controllers = ["rule_ai"] * 5
        self.controller_epoch = 0
        self.advice_enabled = False
        self.phase = "configuring"
        self.paused_reason: str | None = None
        self.hand: HandState | None = None
        self.completed_hands: list[dict[str, Any]] = []
        self.strategy_versions: list[dict[str, Any]] = [self._initial_strategy()]
        self.reflection_memory: list[dict[str, Any]] = []
        self.provider_usage = {
            "live_calls": 0,
            "mock_calls": 0,
            "failures": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
        }
        self.last_advice: dict[str, Any] | None = None
        self.ended = False
        self._emitted: list[dict[str, Any]] = []
        if auto_start:
            self.start_hand()

    @staticmethod
    def _initial_strategy() -> dict[str, Any]:
        return {
            "strategyId": "closed_loop_shaper",
            "version": 1,
            "aggressionBias": 0.0,
            "riskMarginDelta": 0.0,
            "preferredRaiseScale": 0.5,
            "bluffFrequencyCap": 0.08,
            "memoryHands": 6,
            "targeting": [],
            "notes": ["初始平衡策略"],
            "author": "system",
            "reason": "initial_strategy",
            "appliedAfterHand": None,
        }

    @property
    def strategy(self) -> dict[str, Any]:
        return self.strategy_versions[-1]

    @property
    def names(self) -> tuple[str, ...]:
        return ("hero", *(f"seat_{index}_{kind}" for index, kind in enumerate(self.config.opponents, 1)))

    def _emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self.version += 1
        self._emitted.append(
            {
                "seq": self.version,
                "type": event_type,
                "payload": payload or {},
            }
        )

    def drain_events(self) -> list[dict[str, Any]]:
        values, self._emitted = self._emitted, []
        return values

    def start_hand(self) -> None:
        if self.ended:
            raise ValueError("table_finished")
        if self.hand is not None and not self.hand.complete:
            raise ValueError("hand_in_progress")
        hand_index = len(self.completed_hands)
        n = 6
        button = hand_index % n
        deck = make_deck()
        random.Random(self.config.seed + hand_index * 104729).shuffle(deck)
        holes = [[0, 0] for _ in range(n)]
        cursor = 0
        for offset in range(n):
            seat = (button + 1 + offset) % n
            holes[seat] = [deck[cursor], deck[cursor + 1]]
            cursor += 2
        board = deck[cursor : cursor + 5]
        stacks = [self.config.starting_stack] * n
        total_contrib = [0.0] * n
        street_contrib = [0.0] * n
        active = [True] * n
        all_in = [False] * n
        sb = (button + 1) % n
        bb = (button + 2) % n
        self._pay(sb, self.config.small_blind, stacks, total_contrib)
        self._pay(bb, self.config.big_blind, stacks, total_contrib)
        street_contrib[sb] = min(self.config.small_blind, self.config.starting_stack)
        street_contrib[bb] = min(self.config.big_blind, self.config.starting_stack)
        self.hand = HandState(
            hand_index=hand_index,
            button=button,
            board=board,
            holes=holes,
            stacks=stacks,
            total_contrib=total_contrib,
            street_contrib=street_contrib,
            active=active,
            all_in=all_in,
            street_index=0,
            actor=(bb + 1) % n,
            pending=list(range(n)),
            current_bet=max(street_contrib),
            raises=0,
            last_full_raise_increment=self.config.big_blind,
            last_raiser=bb,
        )
        self.phase = "running"
        self.paused_reason = None
        self.last_advice = None
        self._emit(
            "hand.started",
            {"handIndex": hand_index, "button": button, "strategyVersion": self.strategy["version"]},
        )
        self.advance_until_blocked()

    def advance_until_blocked(self) -> None:
        guard = 0
        while not self.ended and self.hand is not None and not self.hand.complete:
            guard += 1
            if guard > 120:
                raise RuntimeError("demo table failed to converge")
            self._normalize_actor()
            if self.hand.complete:
                return
            actor = self.hand.actor
            controller = self.controller_for(actor)
            if controller in {"human", "llm_closed_loop"}:
                self.phase = "waiting_human" if controller == "human" else "waiting_llm"
                return
            self.phase = "running"
            decision = self._rule_decision(actor)
            self.apply_action(actor, decision.action.value, decision.raise_scale, advance=False)

    def _normalize_actor(self) -> None:
        assert self.hand is not None
        hand = self.hand
        for _ in range(10):
            if hand.complete:
                return
            if sum(hand.active) <= 1:
                self._finish_hand()
                return
            pending = set(hand.pending)
            if not pending:
                self._advance_street()
                continue
            for _ in range(6):
                if hand.actor in pending and hand.active[hand.actor] and not hand.all_in[hand.actor]:
                    return
                hand.actor = (hand.actor + 1) % 6
            self._advance_street()
        raise RuntimeError("demo table could not find the next actor")

    def _advance_street(self) -> None:
        assert self.hand is not None
        hand = self.hand
        if sum(hand.active) <= 1 or hand.street_index >= 3:
            self._finish_hand()
            return
        hand.street_index += 1
        hand.street_contrib = [0.0] * 6
        hand.current_bet = 0.0
        hand.raises = 0
        hand.last_full_raise_increment = self.config.big_blind
        hand.last_raiser = None
        hand.pending = [index for index in range(6) if hand.active[index] and not hand.all_in[index]]
        hand.actor = (hand.button + 1) % 6
        self._emit("street.started", {"street": STREETS[hand.street_index].value})
        if len(hand.pending) <= 1:
            self._advance_street()

    def _decision_context(self, actor: int) -> DecisionContext:
        assert self.hand is not None
        hand = self.hand
        to_call = max(0.0, hand.current_bet - hand.street_contrib[actor])
        legal = self.legal_actions(actor)
        return DecisionContext(
            hand_index=hand.hand_index,
            street=STREETS[hand.street_index],
            player_name=self.names[actor],
            hole_cards=tuple(hand.holes[actor]),
            board=tuple(hand.board[: VISIBLE_BOARD_COUNT[hand.street_index]]),
            pot=sum(hand.total_contrib),
            to_call=to_call,
            stack=hand.stacks[actor],
            current_bet=hand.current_bet,
            legal_actions=tuple(ActionType(value) for value in legal),
            active_players=sum(hand.active),
            opponents=tuple(
                self.names[index]
                for index in range(6)
                if index != actor and hand.active[index]
            ),
            last_raiser=self.names[hand.last_raiser] if hand.last_raiser is not None else None,
            raises_this_street=hand.raises,
            button_distance=(actor - hand.button) % 6,
            environment_regime="stable",
        )

    def decision_state(self, actor: int | None = None) -> dict[str, Any]:
        actor = self.hero_seat if actor is None else actor
        context = self._decision_context(actor)
        return {
            "task": "choose_poker_action",
            "hand_index": context.hand_index,
            "street": context.street.value,
            "hole_cards": [card_to_str(card) for card in context.hole_cards],
            "community_cards": [card_to_str(card) for card in context.board],
            "pot": context.pot,
            "to_call": context.to_call,
            "stack": context.stack,
            "legal_actions": [value.value for value in context.legal_actions],
            "active_players": context.active_players,
            "opponents": list(context.opponents),
            "strategy": self.strategy,
            "recent_reflections": self.reflection_memory[-int(self.strategy["memoryHands"]) :],
            "public_actions": list(self.hand.actions if self.hand else []),
            "controlled_seat": actor,
        }

    def _rule_decision(self, actor: int) -> Decision:
        assert self.hand is not None
        hand = self.hand
        player_type = self.config.opponents[actor - 1]
        action_index = len(hand.actions)
        seed = self.config.seed * 1009 + hand.hand_index * 9176 + actor * 131 + action_index
        agent = make_tournament_agent(
            player_type,
            self.names[actor],
            tuple(name for index, name in enumerate(self.names) if index != actor),
            seed,
            equity_samples=self.config.equity_samples,
        )
        for value in hand.actions:
            agent.observe_action(self._action_event(value))
        return agent.act(self._decision_context(actor))

    def legal_actions(self, actor: int | None = None) -> list[str]:
        assert self.hand is not None
        hand = self.hand
        actor = hand.actor if actor is None else actor
        if actor not in hand.pending or not hand.active[actor] or hand.all_in[actor]:
            return []
        to_call = max(0.0, hand.current_bet - hand.street_contrib[actor])
        legal = [ActionType.CHECK_CALL.value]
        if to_call > 1e-9:
            legal.insert(0, ActionType.FOLD.value)
        if (
            hand.stacks[actor] >= to_call + hand.last_full_raise_increment
            and sum(hand.active) > 1
        ):
            legal.append(ActionType.RAISE.value)
        return legal

    def apply_action(
        self,
        actor: int,
        action: str,
        raise_scale: float = 0.5,
        *,
        advance: bool = True,
    ) -> None:
        if self.ended or self.hand is None or self.hand.complete:
            raise ValueError("table_finished")
        hand = self.hand
        self._normalize_actor()
        if actor != hand.actor:
            raise ValueError("not_your_turn")
        legal = self.legal_actions(actor)
        if action not in legal:
            raise ValueError("illegal_action")
        if (
            actor == self.hero_seat
            and action == ActionType.RAISE.value
            and raise_scale not in RAISE_SCALES
        ):
            raise ValueError("invalid_raise_scale")

        to_call = max(0.0, hand.current_bet - hand.street_contrib[actor])
        pot_before = sum(hand.total_contrib)
        faced_raise = to_call > 1e-9 and hand.last_raiser is not None
        raiser_name = self.names[hand.last_raiser] if faced_raise and hand.last_raiser is not None else None
        paid = 0.0
        pending = set(hand.pending)
        if action == ActionType.FOLD.value:
            hand.active[actor] = False
            pending.discard(actor)
        elif action == ActionType.CHECK_CALL.value:
            paid = self._pay(actor, to_call, hand.stacks, hand.total_contrib)
            hand.street_contrib[actor] += paid
            if hand.stacks[actor] <= 1e-9:
                hand.all_in[actor] = True
            pending.discard(actor)
        else:
            call_paid = self._pay(actor, to_call, hand.stacks, hand.total_contrib)
            hand.street_contrib[actor] += call_paid
            pot_after_call = sum(hand.total_contrib)
            increment = (
                hand.stacks[actor]
                if raise_scale >= 1.20
                else max(hand.last_full_raise_increment, raise_scale * max(pot_after_call, 1.0))
            )
            raise_paid = self._pay(actor, increment, hand.stacks, hand.total_contrib)
            hand.street_contrib[actor] += raise_paid
            paid = call_paid + raise_paid
            hand.current_bet = hand.street_contrib[actor]
            hand.last_raiser = actor
            hand.raises += 1
            hand.last_full_raise_increment = raise_paid
            if hand.stacks[actor] <= 1e-9:
                hand.all_in[actor] = True
            pending = {
                index
                for index in range(6)
                if index != actor and hand.active[index] and not hand.all_in[index]
            }
        hand.pending = sorted(pending)
        event = {
            "hand_index": hand.hand_index,
            "street": STREETS[hand.street_index].value,
            "actor": self.names[actor],
            "seat": actor,
            "action": action,
            "raise_scale": raise_scale if action == ActionType.RAISE.value else 0.0,
            "faced_raise": faced_raise,
            "raiser": raiser_name,
            "to_call": _round_bb(to_call),
            "paid": _round_bb(paid),
            "pot_before": _round_bb(pot_before),
            "active_players": sum(hand.active),
            "strategy_version": self.strategy["version"] if actor == self.hero_seat else None,
            "controller": self.controller_for(actor),
        }
        hand.actions.append(event)
        self.last_advice = None
        self._emit("player.acted", event)
        hand.actor = (actor + 1) % 6
        self.phase = "running"
        if advance:
            self.advance_until_blocked()

    def _finish_hand(self) -> None:
        assert self.hand is not None
        hand = self.hand
        live = [index for index, active in enumerate(hand.active) if active]
        hand.showdown = len(live) > 1
        ranks = (
            {}
            if len(live) == 1
            else {
                index: best_hand_rank((*hand.holes[index], *hand.board)) for index in live
            }
        )
        winners = self._award_pots(hand.stacks, hand.total_contrib, live, ranks)
        hand.winners = winners
        hand.rewards = [
            _round_bb(hand.stacks[index] - self.config.starting_stack) for index in range(6)
        ]
        hand.complete = True
        self.phase = "hand_complete"
        summary = {
            "handIndex": hand.hand_index,
            "button": hand.button,
            "winners": [self.names[index] for index in winners],
            "showdown": hand.showdown,
            "rewards": {
                self.names[index]: hand.rewards[index] for index in range(6)
            },
            "strategyVersion": self.strategy["version"],
            "controller": self.controller,
            "actions": list(hand.actions),
        }
        self.completed_hands.append(summary)
        self._emit("hand.completed", summary)

    @staticmethod
    def _award_pots(
        stacks: list[float],
        total_contrib: list[float],
        live_indices: list[int],
        ranks: dict[int, tuple[int, ...]],
    ) -> list[int]:
        awarded: list[int] = []
        previous_level = 0.0
        for level in sorted({value for value in total_contrib if value > 0.0}):
            contributors = [index for index, value in enumerate(total_contrib) if value >= level]
            pot = (level - previous_level) * len(contributors)
            eligible = [index for index in live_indices if total_contrib[index] >= level]
            if not eligible:
                previous_level = level
                continue
            if len(eligible) == 1:
                winners = eligible
            else:
                best = max(ranks[index] for index in eligible)
                winners = [index for index in eligible if ranks[index] == best]
            share = pot / len(winners)
            for index in winners:
                stacks[index] += share
            awarded.extend(winners)
            previous_level = level
        return sorted(set(awarded))

    @staticmethod
    def _pay(player: int, amount: float, stacks: list[float], total: list[float]) -> float:
        paid = min(max(amount, 0.0), stacks[player])
        stacks[player] -= paid
        total[player] += paid
        return paid

    @staticmethod
    def _action_event(value: dict[str, Any]) -> ActionEvent:
        return ActionEvent(
            hand_index=int(value["hand_index"]),
            street=Street(value["street"]),
            actor=str(value["actor"]),
            action=ActionType(value["action"]),
            faced_raise=bool(value["faced_raise"]),
            raiser=value.get("raiser"),
            to_call=float(value["to_call"]),
            paid=float(value["paid"]),
            pot_before=float(value["pot_before"]),
            active_players=int(value["active_players"]),
        )

    def set_controller(self, controller: str) -> None:
        self.set_seat_controller(self.hero_seat, controller)

    def controller_for(self, seat: int) -> str:
        if seat == self.hero_seat:
            return self.controller
        if not 1 <= seat <= 5:
            raise ValueError("invalid_seat")
        return self.opponent_controllers[seat - 1]

    def set_seat_controller(self, seat: int, controller: str) -> None:
        allowed = {"human", "llm_closed_loop"} if seat == self.hero_seat else {
            "rule_ai",
            "llm_closed_loop",
        }
        if controller not in allowed:
            raise ValueError("invalid_controller")
        if self.controller_for(seat) == controller:
            self.paused_reason = None
            return
        if seat == self.hero_seat:
            self.controller = controller
        else:
            self.opponent_controllers[seat - 1] = controller
        self.controller_epoch += 1
        self.paused_reason = None
        self._emit(
            "hero.controller_changed" if seat == self.hero_seat else "player.controller_changed",
            {"seat": seat, "controller": controller, "controllerEpoch": self.controller_epoch},
        )
        if self.hand and not self.hand.complete and self.hand.actor == seat:
            if controller == "human":
                self.phase = "waiting_human"
            elif controller == "llm_closed_loop":
                self.phase = "waiting_llm"
            else:
                self.phase = "running"
                self.advance_until_blocked()

    def set_advice_enabled(self, enabled: bool) -> None:
        self.advice_enabled = bool(enabled)
        if not enabled:
            self.last_advice = None
        self._emit("hero.advice_changed", {"enabled": self.advice_enabled})

    def record_provider_call(self, response: Any, *, purpose: str) -> None:
        key = "live_calls" if self.config.provider_mode == "live_aliyun" else "mock_calls"
        self.provider_usage[key] += 1
        self.provider_usage["input_tokens"] += int(response.input_tokens or 0)
        self.provider_usage["output_tokens"] += int(response.output_tokens or 0)
        self.provider_usage["latency_ms"] = _round_bb(
            self.provider_usage["latency_ms"] + float(response.latency_ms)
        )
        self._emit(
            "llm.completed",
            {
                "purpose": purpose,
                "provider": response.provider,
                "model": response.model,
                "latencyMs": _round_bb(response.latency_ms),
                "totalTokens": response.total_tokens,
                "responseId": response.response_id,
            },
        )

    def record_advice(self, advice: dict[str, Any], *, actor: int = 0) -> None:
        if actor == self.hero_seat:
            self.last_advice = advice
        self._emit(
            "hero.advice_ready" if actor == self.hero_seat else "player.decision_ready",
            {"seat": actor, "advice": advice},
        )

    def record_reflection(self, reflection: dict[str, Any]) -> None:
        self.reflection_memory.append(reflection)
        maximum = max(12, int(self.strategy["memoryHands"]))
        self.reflection_memory = self.reflection_memory[-maximum:]
        self._emit(
            "hero.reflection_completed",
            {
                "handIndex": reflection.get("handIndex"),
                "summary": reflection.get("outcomeSummary"),
            },
        )

    def record_stale_llm_response(self, purpose: str) -> None:
        self._emit("llm.discarded", {"purpose": purpose, "reason": "state_changed"})

    def pause_for_provider_failure(self, reason: str) -> None:
        self.provider_usage["failures"] += 1
        current_controller = (
            self.controller_for(self.hand.actor)
            if self.hand and not self.hand.complete
            else self.controller
        )
        self.phase = (
            "hand_complete"
            if self.hand and self.hand.complete
            else "waiting_llm"
            if current_controller == "llm_closed_loop"
            else "waiting_human"
        )
        self.paused_reason = reason
        self._emit(
            "llm.failed",
            {
                "reason": reason,
                "controller": current_controller,
                "controllerEpoch": self.controller_epoch,
            },
        )

    def apply_strategy_patch(self, patch: dict[str, Any]) -> dict[str, Any]:
        if patch.get("author") != "llm_closed_loop":
            raise ValueError("invalid_strategy_patch_author")
        if int(patch.get("baseStrategyVersion", -1)) != int(self.strategy["version"]):
            raise ValueError("strategy_version_conflict")
        changes = patch.get("changes")
        if not isinstance(changes, dict) or not changes:
            raise ValueError("invalid_strategy_patch")
        allowed = {
            "aggressionBias",
            "riskMarginDelta",
            "preferredRaiseScale",
            "bluffFrequencyCap",
            "memoryHands",
            "targeting",
            "notes",
        }
        if set(changes) - allowed:
            raise ValueError("invalid_strategy_patch_field")
        updated = {**self.strategy, **changes}
        self._validate_strategy(updated)
        updated.update(
            {
                "strategyId": "closed_loop_shaper",
                "version": int(self.strategy["version"]) + 1,
                "author": "llm_closed_loop",
                "reason": str(patch.get("reason", "closed_loop_adjustment"))[:240],
                "appliedAfterHand": self.hand.hand_index if self.hand else None,
            }
        )
        self.strategy_versions.append(updated)
        self._emit(
            "hero.strategy_applied",
            {
                "patchId": patch.get("patchId"),
                "fromVersion": updated["version"] - 1,
                "toVersion": updated["version"],
                "changes": changes,
                "reason": updated["reason"],
            },
        )
        return updated

    def _validate_strategy(self, value: dict[str, Any]) -> None:
        bounds = {
            "aggressionBias": (-0.20, 0.20),
            "riskMarginDelta": (-0.10, 0.10),
            "preferredRaiseScale": (0.25, 1.25),
            "bluffFrequencyCap": (0.0, 0.25),
        }
        for field_name, (minimum, maximum) in bounds.items():
            number = float(value[field_name])
            if not minimum <= number <= maximum:
                raise ValueError(f"invalid_strategy_patch_{field_name}")
        if not 1 <= int(value["memoryHands"]) <= 12:
            raise ValueError("invalid_strategy_patch_memoryHands")
        notes = value.get("notes", [])
        if not isinstance(notes, list) or len(notes) > 4 or any(len(str(note)) > 120 for note in notes):
            raise ValueError("invalid_strategy_patch_notes")
        targeting = value.get("targeting", [])
        signals = {"folds_to_pressure", "raises_often", "calls_wide"}
        if not isinstance(targeting, list) or len(targeting) > 5:
            raise ValueError("invalid_strategy_patch_targeting")
        for target in targeting:
            if (
                not isinstance(target, dict)
                or target.get("opponent") not in self.names[1:]
                or target.get("signal") not in signals
                or not 0.0 <= float(target.get("weight", -1.0)) <= 0.5
            ):
                raise ValueError("invalid_strategy_patch_targeting")

    def finish_table(self) -> None:
        if self.hand is not None and not self.hand.complete:
            raise ValueError("hand_in_progress")
        self.ended = True
        self.phase = "finished"
        self._emit("table.finished", {"hands": len(self.completed_hands)})

    def snapshot(self, *, owner: bool = True) -> dict[str, Any]:
        hand = self.hand
        if hand is None:
            hand_data = None
        else:
            visible_count = 5 if hand.complete else VISIBLE_BOARD_COUNT[hand.street_index]
            seats = []
            for index, name in enumerate(self.names):
                show_cards = index == self.hero_seat or (
                    hand.complete and hand.showdown and hand.active[index]
                )
                seats.append(
                    {
                        "seat": index,
                        "name": name,
                        "strategy": "hero" if index == 0 else self.config.opponents[index - 1],
                        "stackBb": _round_bb(hand.stacks[index]),
                        "active": hand.active[index],
                        "allIn": hand.all_in[index],
                        "cards": [card_to_str(card) for card in hand.holes[index]] if show_cards else [],
                        "isButton": index == hand.button,
                        "isActor": not hand.complete and index == hand.actor,
                        "controller": self.controller_for(index),
                    }
                )
            hand_data = {
                "handIndex": hand.hand_index,
                "street": STREETS[hand.street_index].value,
                "board": [card_to_str(card) for card in hand.board[:visible_count]],
                "potBb": _round_bb(sum(hand.total_contrib)),
                "toCallBb": _round_bb(
                    max(0.0, hand.current_bet - hand.street_contrib[hand.actor])
                )
                if not hand.complete
                else 0.0,
                "currentBetBb": _round_bb(hand.current_bet),
                "seats": seats,
                "actions": list(hand.actions),
                "complete": hand.complete,
                "showdown": hand.showdown,
                "winners": [self.names[index] for index in hand.winners],
                "rewards": {
                    self.names[index]: hand.rewards[index]
                    for index in range(6)
                }
                if hand.complete
                else {},
            }
        can_act = bool(
            owner
            and hand
            and not hand.complete
            and hand.actor == self.hero_seat
            and self.controller == "human"
        )
        return {
            "tableId": self.table_id,
            "version": self.version,
            "phase": self.phase,
            "ended": self.ended,
            "controller": self.controller,
            "controllerEpoch": self.controller_epoch,
            "seatControllers": [self.controller_for(index) for index in range(6)],
            "adviceEnabled": self.advice_enabled,
            "pausedReason": self.paused_reason,
            "canAct": can_act,
            "legalActions": self.legal_actions(self.hero_seat) if can_act else [],
            "raiseScales": list(RAISE_SCALES),
            "strategy": self.strategy,
            "strategyVersions": list(self.strategy_versions),
            "lastAdvice": self.last_advice if owner else None,
            "providerUsage": self.provider_usage if owner else {},
            "providerMode": self.config.provider_mode,
            "model": "deepseek-v4-flash" if self.config.provider_mode == "live_aliyun" else "mock-narrative-v1",
            "completedHandCount": len(self.completed_hands),
            "hand": hand_data,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "config": asdict(self.config),
            "version": self.version,
            "controller": self.controller,
            "opponent_controllers": self.opponent_controllers,
            "controller_epoch": self.controller_epoch,
            "advice_enabled": self.advice_enabled,
            "phase": self.phase,
            "paused_reason": self.paused_reason,
            "hand": asdict(self.hand) if self.hand else None,
            "completed_hands": self.completed_hands,
            "strategy_versions": self.strategy_versions,
            "reflection_memory": self.reflection_memory,
            "provider_usage": self.provider_usage,
            "last_advice": self.last_advice,
            "ended": self.ended,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DemoTable:
        config_value = dict(value["config"])
        config_value["opponents"] = tuple(config_value["opponents"])
        table = cls(DemoConfig(**config_value), table_id=value["table_id"], auto_start=False)
        table.version = int(value["version"])
        table.controller = value["controller"]
        table.opponent_controllers = list(value.get("opponent_controllers", ["rule_ai"] * 5))
        table.controller_epoch = int(value["controller_epoch"])
        table.advice_enabled = bool(value["advice_enabled"])
        table.phase = value["phase"]
        table.paused_reason = value.get("paused_reason")
        table.hand = HandState(**value["hand"]) if value.get("hand") else None
        table.completed_hands = list(value.get("completed_hands", []))
        table.strategy_versions = list(value["strategy_versions"])
        table.reflection_memory = list(value.get("reflection_memory", []))
        table.provider_usage = dict(value["provider_usage"])
        table.last_advice = value.get("last_advice")
        table.ended = bool(value.get("ended", False))
        table._emitted = []
        return table
