from __future__ import annotations

import hashlib
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .environment import EnvironmentConfig, HoldemEnvironment
from .models import ActionType
from .tournament_agents import TYPE_NAMES, make_tournament_agent


@dataclass(frozen=True)
class TypeMatchupConfig:
    pairwise_hands: int = 200
    pairwise_seeds: tuple[int, ...] = tuple(range(5000, 5024))
    ecology_hands: int = 300
    ecology_seeds: tuple[int, ...] = tuple(range(7000, 7032))
    equity_samples: int = 2
    workers: int = 8
    output_dir: Path = Path("results/type_matchups")


def _agent_seed(seed: int, player_type: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{player_type}".encode("utf-8"),
        digest_size=4,
    ).digest()
    return int.from_bytes(digest, "big")


def _action_rates(agent) -> tuple[float, float, float]:
    actions = [entry["action"] for entry in agent.decision_log]
    total = max(len(actions), 1)
    return (
        sum(action == ActionType.RAISE.value for action in actions) / total,
        sum(action == ActionType.FOLD.value for action in actions) / total,
        sum(action == ActionType.CHECK_CALL.value for action in actions) / total,
    )


def _play_mirror(
    type_a: str,
    type_b: str,
    seed: int,
    hands: int,
    equity_samples: int,
    swap: bool,
) -> dict[str, float]:
    seat_types = (type_b, type_a) if swap else (type_a, type_b)
    names = ("seat_0", "seat_1")
    agents = []
    for index, player_type in enumerate(seat_types):
        agent = make_tournament_agent(
            player_type,
            names[index],
            (names[1 - index],),
            _agent_seed(seed, player_type),
            equity_samples=equity_samples,
        )
        agents.append(agent)
    env = HoldemEnvironment(
        agents,
        seed=seed,
        config=EnvironmentConfig(regime_switch_hand=hands + 1),
    )
    records = env.play(hands)
    type_to_agent = {player_type: agents[index] for index, player_type in enumerate(seat_types)}
    rewards = {
        player_type: sum(record.rewards[type_to_agent[player_type].name] for record in records)
        for player_type in (type_a, type_b)
    }
    result: dict[str, float] = {
        "a_reward": rewards[type_a],
        "b_reward": rewards[type_b],
        "a_showdown_rate": sum(record.showdown for record in records) / hands,
    }
    for prefix, player_type in (("a", type_a), ("b", type_b)):
        raise_rate, fold_rate, call_rate = _action_rates(type_to_agent[player_type])
        result[f"{prefix}_raise_rate"] = raise_rate
        result[f"{prefix}_fold_rate"] = fold_rate
        result[f"{prefix}_call_rate"] = call_rate
    return result


def _run_pair_seed(
    type_a: str,
    type_b: str,
    seed: int,
    hands: int,
    equity_samples: int,
) -> dict[str, float | int | str]:
    first = _play_mirror(type_a, type_b, seed, hands, equity_samples, swap=False)
    second = _play_mirror(type_a, type_b, seed, hands, equity_samples, swap=True)
    row: dict[str, float | int | str] = {
        "type_a": type_a,
        "type_b": type_b,
        "seed": seed,
        "hands_per_mirror": hands,
        "a_chips_per_100": 100.0 * (first["a_reward"] + second["a_reward"]) / (2.0 * hands),
        "b_chips_per_100": 100.0 * (first["b_reward"] + second["b_reward"]) / (2.0 * hands),
        "showdown_rate": 0.5 * (first["a_showdown_rate"] + second["a_showdown_rate"]),
    }
    for metric in ("raise_rate", "fold_rate", "call_rate"):
        row[f"a_{metric}"] = 0.5 * (first[f"a_{metric}"] + second[f"a_{metric}"])
        row[f"b_{metric}"] = 0.5 * (first[f"b_{metric}"] + second[f"b_{metric}"])
    return row


def _run_ecology_seed(seed: int, hands: int, equity_samples: int) -> list[dict[str, float | int | str]]:
    import random

    rng = random.Random(seed * 99991)
    seat_types = list(TYPE_NAMES)
    rng.shuffle(seat_types)
    names = tuple(f"seat_{index}" for index in range(len(seat_types)))
    agents = []
    for index, player_type in enumerate(seat_types):
        opponents = tuple(name for name in names if name != names[index])
        agents.append(
            make_tournament_agent(
                player_type,
                names[index],
                opponents,
                _agent_seed(seed, player_type),
                equity_samples=equity_samples,
            )
        )
    env = HoldemEnvironment(
        agents,
        seed=seed,
        config=EnvironmentConfig(regime_switch_hand=hands + 1),
    )
    records = env.play(hands)
    rows: list[dict[str, float | int | str]] = []
    for index, player_type in enumerate(seat_types):
        agent = agents[index]
        reward = sum(record.rewards[agent.name] for record in records)
        raise_rate, fold_rate, call_rate = _action_rates(agent)
        rows.append(
            {
                "seed": seed,
                "player_type": player_type,
                "seat": index,
                "hands": hands,
                "chips_per_100": 100.0 * reward / hands,
                "raise_rate": raise_rate,
                "fold_rate": fold_rate,
                "call_rate": call_rate,
                "showdown_rate": sum(record.showdown for record in records) / hands,
            }
        )
    return rows


def run_type_matchups(config: TypeMatchupConfig) -> dict[str, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pair_jobs = [
        (type_a, type_b, seed)
        for type_a, type_b in itertools.combinations(TYPE_NAMES, 2)
        for seed in config.pairwise_seeds
    ]
    pair_rows: list[dict[str, float | int | str]] = []
    ecology_rows: list[dict[str, float | int | str]] = []
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {
            executor.submit(
                _run_pair_seed,
                type_a,
                type_b,
                seed,
                config.pairwise_hands,
                config.equity_samples,
            ): (type_a, type_b, seed)
            for type_a, type_b, seed in pair_jobs
        }
        for future in as_completed(futures):
            pair_rows.append(future.result())
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {
            executor.submit(
                _run_ecology_seed,
                seed,
                config.ecology_hands,
                config.equity_samples,
            ): seed
            for seed in config.ecology_seeds
        }
        for future in as_completed(futures):
            ecology_rows.extend(future.result())
    pairwise = pd.DataFrame(pair_rows).sort_values(["type_a", "type_b", "seed"])
    ecology = pd.DataFrame(ecology_rows).sort_values(["seed", "player_type"])
    pairwise.to_csv(config.output_dir / "pairwise_per_seed.csv", index=False)
    ecology.to_csv(config.output_dir / "ecology_per_seed.csv", index=False)
    return {"pairwise": pairwise, "ecology": ecology}
