from __future__ import annotations

import random

from .cards import make_deck
from .evaluator import best_hand_rank


def estimate_equity(
    hole_cards: tuple[int, int],
    board: tuple[int, ...],
    opponents: int,
    rng: random.Random,
    samples: int = 20,
) -> float:
    if opponents <= 0:
        return 1.0
    remaining = [card for card in make_deck() if card not in {*hole_cards, *board}]
    missing_board = 5 - len(board)
    score = 0.0
    for _ in range(max(samples, 1)):
        draw = rng.sample(remaining, missing_board + 2 * opponents)
        complete_board = (*board, *draw[:missing_board])
        hero_rank = best_hand_rank((*hole_cards, *complete_board))
        rival_ranks = [
            best_hand_rank(
                (*draw[missing_board + 2 * index : missing_board + 2 * index + 2], *complete_board)
            )
            for index in range(opponents)
        ]
        best = max(hero_rank, *rival_ranks)
        if hero_rank == best:
            score += 1 / (1 + sum(value == best for value in rival_ranks))
    return score / max(samples, 1)
