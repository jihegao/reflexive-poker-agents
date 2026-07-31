from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK_CALL = "check_call"
    RAISE = "raise"


@dataclass(frozen=True)
class Decision:
    action: ActionType
    raise_scale: float = 0.5
    equity: float = 0.0
    predicted_all_fold: float = 0.0
    reasoning_depth: int = 0
    reasoning_ops: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionEvent:
    hand_index: int
    street: Street
    actor: str
    action: ActionType
    faced_raise: bool
    raiser: str | None
    to_call: float
    paid: float
    pot_before: float
    active_players: int


@dataclass(frozen=True)
class ResponseEvent:
    hand_index: int
    street: Street
    raiser: str
    responder: str
    folded: bool
    reraised: bool
    pot_before: float


@dataclass
class DecisionContext:
    hand_index: int
    street: Street
    player_name: str
    hole_cards: tuple[int, int]
    board: tuple[int, ...]
    pot: float
    to_call: float
    stack: float
    current_bet: float
    legal_actions: tuple[ActionType, ...]
    active_players: int
    opponents: tuple[str, ...]
    last_raiser: str | None
    raises_this_street: int
    button_distance: int
    environment_regime: str


@dataclass
class HandRecord:
    hand_index: int
    button: int
    board: tuple[int, ...]
    hole_cards: dict[str, tuple[int, int]]
    actions: list[ActionEvent]
    rewards: dict[str, float]
    winners: tuple[str, ...]
    showdown: bool
    regime: str
    snapshots: dict[str, dict[str, Any]]
