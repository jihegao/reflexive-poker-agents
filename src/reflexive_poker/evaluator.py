from __future__ import annotations

from collections import Counter
from itertools import combinations

from .cards import rank, suit

HandRank = tuple[int, ...]


def _straight_high(ranks: list[int]) -> int | None:
    values = sorted(set(ranks), reverse=True)
    if 14 in values:
        values.append(1)
    for index in range(len(values) - 4):
        window = values[index : index + 5]
        if window == list(range(window[0], window[0] - 5, -1)):
            return window[0]
    return None


def five_card_rank(cards: tuple[int, ...] | list[int]) -> HandRank:
    ranks = [rank(card) for card in cards]
    groups = sorted(((count, value) for value, count in Counter(ranks).items()), reverse=True)
    straight = _straight_high(ranks)
    flush = len({suit(card) for card in cards}) == 1
    if flush and straight is not None:
        return 8, straight
    if groups[0][0] == 4:
        return 7, groups[0][1], max(value for value in ranks if value != groups[0][1])
    if groups[0][0] == 3 and groups[1][0] == 2:
        return 6, groups[0][1], groups[1][1]
    if flush:
        return 5, *sorted(ranks, reverse=True)
    if straight is not None:
        return 4, straight
    if groups[0][0] == 3:
        return 3, groups[0][1], *sorted((v for v in ranks if v != groups[0][1]), reverse=True)
    pairs = sorted((value for value, count in Counter(ranks).items() if count == 2), reverse=True)
    if len(pairs) >= 2:
        return 2, pairs[0], pairs[1], max(value for value in ranks if value not in pairs[:2])
    if pairs:
        return 1, pairs[0], *sorted((v for v in ranks if v != pairs[0]), reverse=True)
    return 0, *sorted(ranks, reverse=True)


def best_hand_rank(cards: tuple[int, ...] | list[int]) -> HandRank:
    return max(five_card_rank(combo) for combo in combinations(cards, 5))
