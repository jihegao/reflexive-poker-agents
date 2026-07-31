from __future__ import annotations

from collections.abc import Iterable

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"


def make_deck() -> list[int]:
    return list(range(52))


def rank(card: int) -> int:
    return 2 + card % 13


def suit(card: int) -> int:
    return card // 13


def card_to_str(card: int) -> str:
    if not 0 <= card < 52:
        raise ValueError(f"Invalid card: {card}")
    return f"{RANK_CHARS[card % 13]}{SUIT_CHARS[card // 13]}"


def cards_to_str(cards: Iterable[int]) -> str:
    return " ".join(card_to_str(card) for card in cards)
