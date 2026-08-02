from __future__ import annotations

import copy
import gzip
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .agents import AgentStyle, PokerAgent
from .environment import EnvironmentConfig, HoldemEnvironment
from .llm_player import CodexProvider, DeterministicNarrativeProvider, OpenCodeGoProvider
from .models import ActionType, HandRecord
from .phase1_models import (
    TYPE_ACTION_LIKELIHOODS,
    AbstractMCCFRPolicy,
    ApproximateEquilibriumOpponent,
    Arena,
    BudgetedRetryProvider,
    ExperimentalOpponent,
    OpponentComposition,
    Phase1LLMHero,
    Phase1RuleHero,
    ProviderBudget,
    ProviderLedger,
    ReasoningTreatment,
    Stability,
)
from .phase1_protocol import canonical_checkpoint_id, mirror_assignment
from .phase1_statistics import inference_table, large_pot_sensitivity

DEFAULT_TREATMENTS = tuple(ReasoningTreatment)
DEPTH_TREATMENTS = (
    ReasoningTreatment.STATE_ONLY,
    ReasoningTreatment.ACTION_PREDICTION,
    ReasoningTreatment.BUDGET_MATCHED_D1,
    ReasoningTreatment.RECURSIVE_D2,
    ReasoningTreatment.RECURSIVE_D3,
)
PAPER_CLOSED_LOOP_TREATMENTS = (
    ReasoningTreatment.STATE_ONLY,
    ReasoningTreatment.BUDGET_MATCHED_D1,
    ReasoningTreatment.RECURSIVE_D2,
)
CLASSIC_SIX_MAX = ("tag", "lag", "rock", "calling_station", "myopic")
CONFIRMATION_MODELS = (
    ("opencode-go", "deepseek-v4-flash"),
    ("codex", "gpt-5.6-luna"),
)


@dataclass(frozen=True)
class Phase1ExperimentConfig:
    arena: Arena = Arena.HEADS_UP
    treatments: tuple[ReasoningTreatment, ...] = DEFAULT_TREATMENTS
    opponent_type: str = "tag"
    opponent_composition: OpponentComposition = OpponentComposition.HETEROGENEOUS_CLASSIC
    stability: Stability = Stability.FIXED
    epsilon: float = 0.05
    seeds: tuple[int, ...] = (9400,)
    horizon: int = 40
    formation_hands: int = 10
    switch_hand: int | None = None
    equity_samples: int = 8
    provider: str = "rule"
    model: str = "none"
    provider_budget: ProviderBudget = field(
        default_factory=lambda: ProviderBudget(max_calls=10_000, max_retries=400)
    )
    bootstrap_samples: int = 5_000
    permutation_samples: int = 20_000
    mccfr_iterations: int = 20_000
    output_dir: Path = Path("results/phase1/smoke")
    preregistered: bool = False
    # The resumable cross-model runner supplies one immutable protocol hash for
    # every serving system.  It identifies the provider-independent formation
    # checkpoint, rather than the per-provider manifest (which intentionally
    # differs because it records the serving system).
    formation_protocol_hash: str | None = None

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0 <= self.formation_hands < self.horizon:
            raise ValueError("formation_hands must be in [0, horizon)")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.treatments)) != len(self.treatments):
            raise ValueError("treatments must be unique")
        if self.arena is Arena.SIX_MAX and self.opponent_type == "approximate_equilibrium":
            raise ValueError("the approximate-equilibrium reference is Heads-Up only")


@dataclass(frozen=True)
class ConfirmationJob:
    name: str
    arena: Arena
    treatments: tuple[ReasoningTreatment, ...]
    call_budget: int
    opponent_type: str = "tag"
    opponent_composition: OpponentComposition = OpponentComposition.HETEROGENEOUS_CLASSIC
    stability: Stability = Stability.FIXED
    epsilon: float = 0.05


@dataclass(frozen=True)
class Phase1LLMConfirmationPlan:
    selected_depth: ReasoningTreatment
    models: tuple[tuple[str, str], ...] = CONFIRMATION_MODELS
    max_calls_per_model: int = 10_000
    offline_call_budget: int = 1_600
    preflight_retry_reserve: int = 400
    jobs: tuple[ConfirmationJob, ...] = ()

    def __post_init__(self) -> None:
        if self.selected_depth not in {
            ReasoningTreatment.ACTION_PREDICTION,
            ReasoningTreatment.RECURSIVE_D2,
            ReasoningTreatment.RECURSIVE_D3,
        }:
            raise ValueError("selected_depth must be D1, D2, or D3")
        experimental = sum(job.call_budget for job in self.jobs)
        if (
            experimental + self.offline_call_budget + self.preflight_retry_reserve
            != self.max_calls_per_model
        ):
            raise ValueError("confirmation job budgets must exactly match the per-model ceiling")


def build_llm_confirmation_plan(
    selected_depth: ReasoningTreatment,
) -> Phase1LLMConfirmationPlan:
    heads_up = tuple(
        ConfirmationJob(
            name=f"hu_{stability.value}_paper_contrast",
            arena=Arena.HEADS_UP,
            treatments=(
                ReasoningTreatment.STATE_ONLY,
                ReasoningTreatment.BUDGET_MATCHED_D1,
                selected_depth,
            ),
            call_budget=4_000,
            stability=stability,
        )
        for stability in (Stability.FIXED, Stability.ADAPTIVE)
    )
    return Phase1LLMConfirmationPlan(
        selected_depth=selected_depth,
        jobs=heads_up,
    )


def write_llm_confirmation_plan(
    output_dir: Path,
    selected_depth: ReasoningTreatment,
) -> dict[str, Any]:
    plan = build_llm_confirmation_plan(selected_depth)
    payload = _jsonable(asdict(plan))
    payload["execution_policy"] = {
        "serial_within_model": True,
        "pool_results_across_models": False,
        "silent_model_substitution": False,
        "start_requirement": "zero-failure small smoke and exact model identity",
        "incomplete_block_policy": "discard the incomplete paired block",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["plan_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "LLM_CONFIRMATION_PLAN.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _config_payload(config: Phase1ExperimentConfig) -> dict[str, Any]:
    payload = _jsonable(asdict(config))
    payload.pop("output_dir", None)
    return payload


def _manifest(config: Phase1ExperimentConfig) -> dict[str, Any]:
    configuration = _config_payload(config)
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return {
        "protocol": "reflexive-poker-phase1-v1",
        "configuration": configuration,
        "manifest_hash": hashlib.sha256(canonical.encode()).hexdigest(),
        "evidence_class": "preregistered" if config.preregistered else "exploratory_or_smoke",
        "claim_boundary": (
            "Results compare policies inside the repository simulator under frozen assumptions; "
            "they do not establish real-world poker optimality."
        ),
    }


def _provider(kind: str, model: str, seed: int):
    if kind == "mock":
        return DeterministicNarrativeProvider(seed=seed)
    if kind == "opencode-go":
        return OpenCodeGoProvider(model=model)
    if kind == "codex":
        return CodexProvider(model=model)
    raise ValueError(f"unknown Phase 1 provider: {kind}")


def _style(equity_samples: int) -> AgentStyle:
    return AgentStyle(
        aggression=0.47,
        risk_margin=0.055,
        belief_sensitivity=0.24,
        social_learning_rate=0.20,
        equity_samples=equity_samples,
    )


def _names(config: Phase1ExperimentConfig) -> tuple[str, ...]:
    if config.arena is Arena.HEADS_UP:
        return ("hero", f"seat_1_{config.opponent_type}")
    types = (
        ("tag",) * 5
        if config.opponent_composition is OpponentComposition.HOMOGENEOUS_TAG
        else CLASSIC_SIX_MAX
    )
    return ("hero", *(f"seat_{index}_{value}" for index, value in enumerate(types, start=1)))


def _lineup(config: Phase1ExperimentConfig) -> tuple[str, ...]:
    if config.arena is Arena.HEADS_UP:
        return (config.opponent_type,)
    if config.opponent_composition is OpponentComposition.HOMOGENEOUS_TAG:
        return ("tag",) * 5
    return CLASSIC_SIX_MAX


def _make_environment(
    config: Phase1ExperimentConfig,
    seed: int,
    treatment: ReasoningTreatment,
    provider: BudgetedRetryProvider | None,
    mccfr_policy: AbstractMCCFRPolicy | None,
) -> HoldemEnvironment:
    names = _names(config)
    opponents = names[1:]
    if provider is None:
        hero: PokerAgent = Phase1RuleHero(
            names[0], seed * 1009 + 1, treatment, opponents, _style(config.equity_samples)
        )
    else:
        hero = Phase1LLMHero(
            names[0],
            seed * 1009 + 1,
            provider,
            treatment,
            opponents,
            _style(config.equity_samples),
        )
    agents: list[PokerAgent] = [hero]
    switch_hand = config.switch_hand if config.switch_hand is not None else config.horizon // 2
    for index, (name, player_type) in enumerate(zip(opponents, _lineup(config), strict=True), 1):
        if player_type == "approximate_equilibrium":
            if mccfr_policy is None:
                raise RuntimeError("MCCFR policy was not trained")
            agents.append(
                ApproximateEquilibriumOpponent(
                    name,
                    seed * 1009 + index + 1,
                    mccfr_policy,
                    config.epsilon,
                    config.equity_samples,
                )
            )
            continue
        switch_type = "lag" if player_type != "lag" else "rock"
        agents.append(
            ExperimentalOpponent(
                name,
                seed * 1009 + index + 1,
                player_type,
                tuple(other for other in names if other != name),
                config.epsilon,
                config.stability,
                switch_hand,
                switch_type=switch_type,
                equity_samples=config.equity_samples,
            )
        )
    # A paired seed also determines Hero's actual physical seat.  The ordering
    # controls blinds and deal order in HoldemEnvironment, so this is a real
    # seat mirror rather than a reporting-only label.
    if mirror_assignment(seed):
        agents = [*agents[1:], agents[0]]
    return HoldemEnvironment(
        agents,
        seed=seed,
        config=EnvironmentConfig(
            starting_stack=100.0,
            max_raises_per_street=2,
            regime_switch_hand=switch_hand,
        ),
    )


def _hero(environment: HoldemEnvironment) -> Phase1RuleHero | Phase1LLMHero:
    hero = next((agent for agent in environment.agents if agent.name == "hero"), None)
    if not isinstance(hero, Phase1RuleHero | Phase1LLMHero):
        raise TypeError("Phase 1 environment must include exactly one Hero")
    return hero


def _install_llm_hero(
    environment: HoldemEnvironment,
    *,
    config: Phase1ExperimentConfig,
    seed: int,
    treatment: ReasoningTreatment,
    provider: BudgetedRetryProvider,
) -> Phase1LLMHero:
    """Replace the deterministic formation actor with a continuation actor.

    Formation must not depend on which serving system will later be evaluated.
    We therefore run it once with the deterministic Phase1RuleHero, deep-copy
    the resulting public state, then transplant only the public/belief and RNG
    state into a fresh LLM Hero for each treatment branch.  The deck/RNG and
    opponent state stay inside the copied environment unchanged.
    """
    formed_hero = _hero(environment)
    if not isinstance(formed_hero, Phase1RuleHero):
        raise TypeError("shared formation must be performed by Phase1RuleHero")
    continuation = Phase1LLMHero(
        formed_hero.name,
        seed * 1009 + 1,
        provider,
        treatment,
        tuple(agent.name for agent in environment.agents if agent.name != formed_hero.name),
        _style(config.equity_samples),
    )
    continuation.rng.setstate(formed_hero.rng.getstate())
    continuation.cumulative_reward = formed_hero.cumulative_reward
    continuation.recent_rewards = copy.deepcopy(formed_hero.recent_rewards)
    continuation.decision_log = copy.deepcopy(formed_hero.decision_log)
    continuation.belief_states = copy.deepcopy(formed_hero.belief_states)
    environment.agents[environment.agents.index(formed_hero)] = continuation
    environment.agent_by_name[continuation.name] = continuation
    return continuation


def environment_fork_signature(environment: HoldemEnvironment) -> str:
    agent_rows = []
    for agent in environment.agents:
        row: dict[str, Any] = {
            "name": agent.name,
            "class": type(agent).__name__,
            "rng": repr(agent.rng.getstate()),
            "cumulative_reward": agent.cumulative_reward,
            "recent_rewards": list(agent.recent_rewards),
            "decision_log_length": len(agent.decision_log),
            "beliefs": {
                name: [belief.aggression_total, belief.aggression_raises]
                for name, belief in sorted(agent.beliefs.items())
            },
        }
        if isinstance(agent, Phase1RuleHero | Phase1LLMHero):
            row["phase1_belief_hash"] = agent.belief_digest()
        if isinstance(agent, ExperimentalOpponent):
            row["hero_actions"] = list(agent.hero_actions)
        agent_rows.append(row)
    payload = {
        "environment_rng": repr(environment.rng.getstate()),
        "record_count": len(environment.records),
        "agents": agent_rows,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _record_rows(
    records: list[HandRecord],
    treatment: ReasoningTreatment,
    seed: int,
    fork_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        hero_name = "hero"
        hero_actions = [event for event in record.actions if event.actor == hero_name]
        largest_pot_action = max((event.pot_before for event in record.actions), default=0.0)
        opponent_images = [
            float(snapshot.get("belief_aggression", {}).get(hero_name, 0.5))
            for name, snapshot in record.snapshots.items()
            if name != hero_name
        ]
        rows.append(
            {
                "seed": seed,
                "treatment": treatment.value,
                "reasoning_depth": treatment.depth,
                "hand_index": record.hand_index,
                "reward": float(record.rewards[hero_name]),
                "decision_count": len(hero_actions),
                "raise_count": sum(event.action is ActionType.RAISE for event in hero_actions),
                "showdown": record.showdown,
                "largest_pot_before_action": largest_pot_action,
                "mean_opponent_image_of_hero": (
                    sum(opponent_images) / len(opponent_images) if opponent_images else 0.5
                ),
                "fork_hash": fork_hash,
            }
        )
    return rows


def _trace_rows(
    hero: Phase1RuleHero | Phase1LLMHero,
    seed: int,
    fork_hash: str,
) -> list[dict[str, Any]]:
    source = hero.phase1_traces if isinstance(hero, Phase1RuleHero) else hero.decision_traces
    return [{"seed": seed, "fork_hash": fork_hash, **trace} for trace in source]


def _mechanism_rows(
    hero: Phase1RuleHero | Phase1LLMHero,
    records: list[HandRecord],
    seed: int,
    treatment: ReasoningTreatment,
    true_types: dict[str, str],
) -> list[dict[str, Any]]:
    traces = hero.phase1_traces if isinstance(hero, Phase1RuleHero) else hero.decision_traces
    rows: list[dict[str, Any]] = []
    record_by_hand = {record.hand_index: record for record in records}
    for trace in traces:
        hand_index = int(trace["hand_index"])
        record = record_by_hand.get(hand_index)
        if record is None:
            continue
        if isinstance(hero, Phase1RuleHero):
            distribution = trace.get("action_prediction")
        else:
            distribution = trace.get("provider_output", {}).get("opponent_state", {}).get(
                "action_probabilities"
            )
        if not isinstance(distribution, dict):
            continue
        street = trace["street"]
        observed = next(
            (
                event.action.value
                for event in record.actions
                if event.street.value == street and event.actor != hero.name
            ),
            None,
        )
        if observed is None:
            continue
        probability = max(1e-9, float(distribution.get(observed, 0.0)))
        brier = sum(
            (float(distribution.get(action, 0.0)) - float(action == observed)) ** 2
            for action in ("fold", "check_call", "raise")
        )
        rows.append(
            {
                "seed": seed,
                "treatment": treatment.value,
                "metric": "action_prediction",
                "hand_index": hand_index,
                "log_loss": -math.log(probability),
                "brier": brier,
                "type_probability": float("nan"),
                "type_correct": float("nan"),
                "model_confidence": float("nan"),
                "type_brier": float("nan"),
                "decision_regret": float("nan"),
            }
        )
        true_type = next(iter(true_types.values()), None)
        state = trace.get("state", {})
        final_action = trace.get("final_decision", {}).get("action") or trace.get("final_action")
        legal_actions = state.get("legal_actions", list(ActionType))
        if true_type in TYPE_ACTION_LIKELIHOODS and final_action in legal_actions:
            strength = float(state.get("equity_estimate", 0.5))
            pot_odds = float(state.get("pot_odds", 0.0))
            values = {
                action: sum(
                    probability
                    * AbstractMCCFRPolicy._utility(
                        action, opponent_action, strength, pot_odds
                    )
                    for opponent_action, probability in TYPE_ACTION_LIKELIHOODS[
                        true_type
                    ].items()
                )
                for action in legal_actions
            }
            rows.append(
                {
                    "seed": seed,
                    "treatment": treatment.value,
                    "metric": "decision_regret",
                    "hand_index": hand_index,
                    "log_loss": float("nan"),
                    "brier": float("nan"),
                    "type_probability": float("nan"),
                    "type_correct": float("nan"),
                    "model_confidence": float("nan"),
                    "type_brier": float("nan"),
                    "decision_regret": max(values.values()) - values[final_action],
                }
            )
        if isinstance(hero, Phase1LLMHero) and true_type is not None:
            opponent_state = trace.get("provider_output", {}).get("opponent_state", {})
            type_probabilities = opponent_state.get("type_probabilities")
            if isinstance(type_probabilities, dict) and true_type in type_probabilities:
                rows.append(
                    {
                        "seed": seed,
                        "treatment": treatment.value,
                        "metric": "strategy_type",
                        "hand_index": hand_index,
                        "log_loss": float("nan"),
                        "brier": float("nan"),
                        "type_probability": float(type_probabilities[true_type]),
                        "type_correct": float(
                            max(type_probabilities, key=type_probabilities.get) == true_type
                        ),
                        "model_confidence": float(trace["provider_output"]["confidence"]),
                        "type_brier": sum(
                            (float(probability) - float(name == true_type)) ** 2
                            for name, probability in type_probabilities.items()
                        ),
                        "decision_regret": float("nan"),
                    }
                )
    if isinstance(hero, Phase1RuleHero):
        belief_items = hero.belief_states.items()
    else:
        belief_items = ()
    for opponent, state in belief_items:
        true_type = true_types.get(opponent)
        if true_type is None or true_type not in state.type_posterior:
            continue
        predicted_type = max(state.type_posterior, key=state.type_posterior.get)
        rows.append(
            {
                "seed": seed,
                "treatment": treatment.value,
                "metric": "strategy_type",
                "hand_index": records[-1].hand_index if records else -1,
                "log_loss": float("nan"),
                "brier": float("nan"),
                "type_probability": state.type_posterior[true_type],
                "type_correct": float(predicted_type == true_type),
                "model_confidence": state.confidence,
                "type_brier": sum(
                    (float(probability) - float(name == true_type)) ** 2
                    for name, probability in state.type_posterior.items()
                ),
                "decision_regret": float("nan"),
            }
        )
    return rows


def _per_seed(per_hand: pd.DataFrame, config: Phase1ExperimentConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, treatment), group in per_hand.groupby(["seed", "treatment"]):
        sensitivity = large_pot_sensitivity(group)
        reward = float(group["reward"].sum())
        calls = int(group["decision_count"].sum())
        rows.append(
            {
                "seed": seed,
                "treatment": treatment,
                "reasoning_depth": int(group["reasoning_depth"].iloc[0]),
                "arena": config.arena.value,
                "opponent_type": config.opponent_type,
                "opponent_composition": config.opponent_composition.value,
                "stability": config.stability.value,
                "epsilon": config.epsilon,
                "hands": len(group),
                "chips_per_100": 100.0 * reward / len(group),
                "positive_hand_rate": float((group["reward"] > 0).mean()),
                "decision_count": calls,
                "raise_rate": float(group["raise_count"].sum() / max(calls, 1)),
                "largest_abs_reward": sensitivity["largest_abs_reward"],
                "top_1pct_abs_share": sensitivity["top_1pct_abs_share"],
                "mean_decision_regret": float(group["decision_regret"].mean()),
                "trimmed_1pct_chips_per_100": 100.0
                * sensitivity["trimmed_1pct_reward"]
                / max(1, len(group) - max(1, math.ceil(len(group) * 0.01))),
            }
        )
    return pd.DataFrame(rows)


def _paired(per_seed: pd.DataFrame, treatments: tuple[ReasoningTreatment, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    contrasts = list(itertools.pairwise(treatments))
    core_contrasts = (
        (ReasoningTreatment.STATE_ONLY, ReasoningTreatment.RECURSIVE_D2),
        (ReasoningTreatment.ACTION_PREDICTION, ReasoningTreatment.RECURSIVE_D2),
        (ReasoningTreatment.BUDGET_MATCHED_D1, ReasoningTreatment.RECURSIVE_D2),
    )
    for core in core_contrasts:
        if all(treatment in treatments for treatment in core) and core not in contrasts:
            contrasts.append(core)
    for lower, upper in contrasts:
        left = per_seed[per_seed["treatment"] == lower.value]
        right = per_seed[per_seed["treatment"] == upper.value]
        pair = left.merge(right, on="seed", suffixes=("_lower", "_upper"))
        if pair.empty:
            continue
        pair["contrast"] = f"{upper.value}-{lower.value}"
        pair["lower_treatment"] = lower.value
        pair["upper_treatment"] = upper.value
        pair["chips_per_100_delta"] = (
            pair["chips_per_100_upper"] - pair["chips_per_100_lower"]
        )
        pair["trimmed_delta"] = (
            pair["trimmed_1pct_chips_per_100_upper"]
            - pair["trimmed_1pct_chips_per_100_lower"]
        )
        pair["decision_regret_reduction"] = (
            pair["mean_decision_regret_lower"] - pair["mean_decision_regret_upper"]
        )
        rows.append(
            pair[
                [
                    "seed",
                    "contrast",
                    "lower_treatment",
                    "upper_treatment",
                    "chips_per_100_lower",
                    "chips_per_100_upper",
                    "chips_per_100_delta",
                    "trimmed_delta",
                    "decision_regret_reduction",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "seed",
                "contrast",
                "lower_treatment",
                "upper_treatment",
                "chips_per_100_lower",
                "chips_per_100_upper",
                "chips_per_100_delta",
                "trimmed_delta",
                "decision_regret_reduction",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _paired_hand_deltas(
    per_hand: pd.DataFrame, treatments: tuple[ReasoningTreatment, ...]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    contrasts = list(itertools.pairwise(treatments))
    core_contrasts = (
        (ReasoningTreatment.STATE_ONLY, ReasoningTreatment.RECURSIVE_D2),
        (ReasoningTreatment.ACTION_PREDICTION, ReasoningTreatment.RECURSIVE_D2),
        (ReasoningTreatment.BUDGET_MATCHED_D1, ReasoningTreatment.RECURSIVE_D2),
    )
    for core in core_contrasts:
        if all(treatment in treatments for treatment in core) and core not in contrasts:
            contrasts.append(core)
    for lower, upper in contrasts:
        left = per_hand[per_hand["treatment"] == lower.value]
        right = per_hand[per_hand["treatment"] == upper.value]
        pair = left.merge(right, on=["seed", "hand_index"], suffixes=("_lower", "_upper"))
        if pair.empty:
            continue
        pair["contrast"] = f"{upper.value}-{lower.value}"
        pair["reward_delta"] = pair["reward_upper"] - pair["reward_lower"]
        pair = pair.sort_values(["seed", "hand_index"])
        pair["cumulative_delta"] = pair.groupby("seed")["reward_delta"].cumsum()
        pair["rolling_25_delta"] = pair.groupby("seed")["reward_delta"].transform(
            lambda values: values.rolling(25, min_periods=25).sum()
        )
        rows.append(
            pair[
                [
                    "seed",
                    "hand_index",
                    "contrast",
                    "reward_lower",
                    "reward_upper",
                    "reward_delta",
                    "cumulative_delta",
                    "rolling_25_delta",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _cost_metrics(per_hand: pd.DataFrame, traces: list[dict[str, Any]]) -> pd.DataFrame:
    if not traces:
        return pd.DataFrame(
            columns=[
                "treatment",
                "calls",
                "total_tokens",
                "latency_ms",
                "reported_cost_usd",
                "chips_per_100",
                "chips_per_1000_tokens",
                "cumulative_profit_crossing_hand",
            ]
        )
    trace_frame = pd.DataFrame(traces)
    trace_frame["treatment"] = trace_frame.get(
        "phase1_treatment", trace_frame.get("condition")
    )
    trace_frame["total_tokens"] = pd.to_numeric(trace_frame.get("total_tokens"), errors="coerce")
    trace_frame["latency_ms"] = pd.to_numeric(trace_frame.get("latency_ms"), errors="coerce")
    trace_frame["cost_usd"] = pd.to_numeric(trace_frame.get("cost_usd"), errors="coerce")
    rows: list[dict[str, Any]] = []
    for treatment, group in trace_frame.groupby("treatment"):
        hands = per_hand[per_hand["treatment"] == treatment].sort_values(["seed", "hand_index"])
        reward = float(hands["reward"].sum())
        tokens = float(group["total_tokens"].sum(min_count=1))
        cumulative = hands.groupby("seed")["reward"].cumsum()
        crossing = hands.loc[cumulative > 0, "hand_index"]
        rows.append(
            {
                "treatment": treatment,
                "calls": len(group),
                "total_tokens": tokens,
                "latency_ms": float(group["latency_ms"].sum(min_count=1)),
                "reported_cost_usd": float(group["cost_usd"].sum(min_count=1)),
                "chips_per_100": 100.0 * reward / max(len(hands), 1),
                "chips_per_1000_tokens": (
                    1000.0 * reward / tokens if math.isfinite(tokens) and tokens > 0 else float("nan")
                ),
                "cumulative_profit_crossing_hand": (
                    int(crossing.min()) if not crossing.empty else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")


def _write_optional_parquet(frame: pd.DataFrame, path: Path) -> bool:
    try:
        frame.to_parquet(path, index=False)
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def _plot(per_seed: pd.DataFrame, per_hand: pd.DataFrame, output_dir: Path) -> None:
    if per_seed.empty:
        return
    depth = per_seed.groupby("reasoning_depth", as_index=False)["chips_per_100"].agg(
        ["mean", "std"]
    )
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.errorbar(depth.index, depth["mean"], yerr=depth["std"].fillna(0), marker="o")
    axis.axhline(0, color="#666", linewidth=0.8)
    axis.set(xlabel="Auditable reasoning depth", ylabel="Hero chips/100")
    fig.tight_layout()
    fig.savefig(output_dir / "depth_payoff.png", dpi=160)
    plt.close(fig)

    ordered = per_hand.sort_values(["treatment", "seed", "hand_index"]).copy()
    ordered["cumulative_reward"] = ordered.groupby(["treatment", "seed"])["reward"].cumsum()
    horizon = ordered.groupby(["treatment", "hand_index"], as_index=False)[
        "cumulative_reward"
    ].mean()
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for treatment, group in horizon.groupby("treatment"):
        axis.plot(group["hand_index"], group["cumulative_reward"], label=treatment)
    axis.axhline(0, color="#666", linewidth=0.8)
    axis.set(xlabel="Hand", ylabel="Mean cumulative reward")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "horizon_payoff.png", dpi=160)
    plt.close(fig)


def _provider_gate(
    config: Phase1ExperimentConfig,
    ledger: ProviderLedger | None,
    heroes: list[Phase1RuleHero | Phase1LLMHero],
    traces: list[dict[str, Any]],
    forks_identical: bool,
) -> dict[str, Any]:
    if ledger is None:
        return {
            "applicable": False,
            "valid": forks_identical,
            "forks_identical": forks_identical,
            "provider": "rule",
        }
    fallbacks = sum(
        bool(trace.get("final_decision", {}).get("fallback_used")) for trace in traces
    )
    del heroes
    provider_failures = ledger.unresolved_failures
    identities = {
        (trace.get("provider"), trace.get("model"))
        for trace in traces
        if trace.get("provider") is not None
    }
    expected_identity = {
        "mock": ("deterministic_mock", "mock-narrative-v1"),
        "opencode-go": ("opencode_go", config.model),
        "codex": ("codex_exec", config.model),
    }[config.provider]
    actual_models = sorted(
        {str(trace["actual_model"]) for trace in traces if trace.get("actual_model")}
    )
    model_versions = sorted(
        {str(trace["model_version"]) for trace in traces if trace.get("model_version")}
    )
    model_traces = [trace for trace in traces if trace.get("phase1_treatment") != "shared_formation"]
    actual_identity_matches = config.provider == "mock" or actual_models == [config.model]
    allowed_identity_sources = {
        "opencode-go": {"provider_stream", "opencode_session_export"},
        "codex": {"provider_stream", "cli_selected_model"},
        "mock": {"unavailable"},
    }
    observed_identity_sources = sorted(
        {
            str(trace["model_identity_source"])
            for trace in model_traces
            if trace.get("model_identity_source")
        }
    )
    identity_source_valid = config.provider == "mock" or (
        bool(model_traces)
        and all(
            trace.get("model_identity_source")
            in allowed_identity_sources.get(config.provider, set())
            for trace in model_traces
        )
    )
    complete_model_version_attestation = config.provider == "mock" or (
        len(model_versions) == 1 and all(bool(trace.get("model_version")) for trace in model_traces)
    )
    complete_tokens = config.provider == "mock" or ledger.token_observed_calls == ledger.calls
    calls_by_treatment: dict[str, int] = {}
    for trace in traces:
        treatment = trace.get("phase1_treatment") or trace.get("condition")
        if treatment is not None:
            calls_by_treatment[str(treatment)] = calls_by_treatment.get(str(treatment), 0) + 1
    experimental_call_counts = [
        count for treatment, count in calls_by_treatment.items() if treatment != "shared_formation"
    ]
    call_count_balanced = len(set(experimental_call_counts)) <= 1
    paired_arms_complete = all(
        calls_by_treatment.get(treatment.value, 0) > 0 for treatment in config.treatments
    )
    input_tokens_by_treatment: dict[str, list[float]] = {}
    for trace in traces:
        treatment = trace.get("phase1_treatment") or trace.get("condition")
        input_tokens = trace.get("input_tokens")
        if treatment is not None and isinstance(input_tokens, int | float):
            input_tokens_by_treatment.setdefault(str(treatment), []).append(float(input_tokens))
    budget_match_ratio: float | None = None
    budget_match_valid = True
    if {
        ReasoningTreatment.BUDGET_MATCHED_D1,
        ReasoningTreatment.RECURSIVE_D2,
    }.issubset(config.treatments):
        lower = input_tokens_by_treatment.get(ReasoningTreatment.BUDGET_MATCHED_D1.value, [])
        upper = input_tokens_by_treatment.get(ReasoningTreatment.RECURSIVE_D2.value, [])
        if lower and upper:
            budget_match_ratio = (sum(lower) / len(lower)) / (sum(upper) / len(upper))
            budget_match_valid = 0.90 <= budget_match_ratio <= 1.10
        else:
            budget_match_valid = False
    valid = (
        forks_identical
        and ledger.unresolved_failures == 0
        and provider_failures == 0
        and fallbacks == 0
        and identities == {expected_identity}
        and actual_identity_matches
        and identity_source_valid
        and complete_tokens
        and paired_arms_complete
        and budget_match_valid
    )
    return {
        "applicable": True,
        "valid": valid,
        "forks_identical": forks_identical,
        "provider": config.provider,
        "model": config.model,
        "expected_identity": list(expected_identity),
        "observed_identities": [list(value) for value in sorted(identities)],
        "observed_actual_models": actual_models,
        "observed_model_versions": model_versions,
        "actual_identity_matches": actual_identity_matches,
        "observed_model_identity_sources": observed_identity_sources,
        "model_identity_source_valid": identity_source_valid,
        "complete_model_version_attestation": complete_model_version_attestation,
        "provider_failures": provider_failures,
        "fallbacks": fallbacks,
        "complete_token_accounting": complete_tokens,
        "calls_by_treatment": calls_by_treatment,
        "call_count_balanced": call_count_balanced,
        "paired_arms_complete": paired_arms_complete,
        "budget_match_ratio": budget_match_ratio,
        "budget_match_valid": budget_match_valid,
        "ledger": ledger.snapshot(),
    }


def _report(
    config: Phase1ExperimentConfig,
    manifest: dict[str, Any],
    inference: pd.DataFrame,
    gate: dict[str, Any],
    paired: pd.DataFrame,
) -> str:
    lines = [
        "# 第一阶段对手建模与递归推理实验报告",
        "",
        "## 证据状态",
        "",
        f"- 证据类别：`{manifest['evidence_class']}`",
        f"- Arena：`{config.arena.value}`；对手稳定性：`{config.stability.value}`；ε：`{config.epsilon}`",
        f"- 种子数：`{len(config.seeds)}`；形成期/总时域：`{config.formation_hands}/{config.horizon}` 手",
        f"- Provider gate：`{gate['valid']}`",
        "- 本报告只解释冻结模拟器与配置下的配对结果，不证明真实扑克最优性。",
        "",
        "## 配对闭环推断",
        "",
        "| 指标 | 对比 | 配对种子 | 均值差 | 95% CI | 正向种子率 | Holm p | 去最大种子后 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in inference.to_dict(orient="records"):
        lines.append(
            f"| {row['metric']} | {row['contrast']} | {int(row['pairs'])} | {row['mean_delta']:+.2f} | "
            f"[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}] | "
            f"{row['positive_seed_rate']:.1%} | {row['holm_p']:.4f} | "
            f"{row['leave_largest_out_mean']:+.2f} |"
        )
    if inference.empty:
        lines.append("| n/a | 无可估计对比 | 0 | n/a | n/a | n/a | n/a | n/a |")
    trimmed_consistent = bool((paired["trimmed_delta"] > 0).all()) if not paired.empty else False
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            f"- 所有已配对结果去除每个条件绝对收益最高 1% 手牌后仍同向：`{trimmed_consistent}`。",
            "- `strong_support` 不能由 smoke、单种子、provider gate 失败或未预注册运行产生。",
            "- 两个真实模型的正式结果必须分别完成后，才能判断跨模型方向一致性。",
        ]
    )
    return "\n".join(lines)


def run_phase1_experiment(config: Phase1ExperimentConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(config)
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    mccfr_policy = (
        AbstractMCCFRPolicy.train(config.mccfr_iterations, seed=min(config.seeds))
        if config.opponent_type == "approximate_equilibrium"
        else None
    )
    if mccfr_policy is not None:
        (config.output_dir / "mccfr_training.json").write_text(
            json.dumps(asdict(mccfr_policy.report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    ledger: ProviderLedger | None = None
    budgeted_provider: BudgetedRetryProvider | None = None
    if config.provider != "rule":
        ledger = ProviderLedger()
        budgeted_provider = BudgetedRetryProvider(
            _provider(config.provider, config.model, min(config.seeds)),
            config.provider_budget,
            ledger,
            checkpoint_path=config.output_dir / "live_provider_ledger.json",
            attempt_log_path=config.output_dir / "live_provider_attempts.jsonl",
        )

    per_hand_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    fork_rows: list[dict[str, Any]] = []
    completed_heroes: list[Phase1RuleHero | Phase1LLMHero] = []
    for seed in config.seeds:
        # The common formation fork deliberately has no provider.  Otherwise
        # each model would create a different public history before the paired
        # treatment contrast begins, invalidating a cross-model comparison.
        formation = _make_environment(
            config,
            seed,
            ReasoningTreatment.STATE_ONLY,
            None,
            mccfr_policy,
        )
        formation.play(config.formation_hands)
        branches = [copy.deepcopy(formation) for _ in config.treatments]
        signatures = [environment_fork_signature(branch) for branch in branches]
        identical = len(set(signatures)) == 1
        fork_rows.append(
            {
                "seed": seed,
                "signature": signatures[0],
                "checkpoint_id": canonical_checkpoint_id(
                    config.formation_protocol_hash or manifest["manifest_hash"], seed
                ),
                "identical": identical,
                "record_count": config.formation_hands,
            }
        )
        if not identical:
            raise RuntimeError(f"fork signature mismatch for seed {seed}")
        formation_hero = _hero(formation)
        for trace in _trace_rows(formation_hero, seed, signatures[0]):
            trace["phase1_treatment"] = "shared_formation"
            trace["condition"] = "shared_formation"
            trace_rows.append(trace)
        for treatment, branch in zip(config.treatments, branches, strict=True):
            hero = (
                _install_llm_hero(
                    branch,
                    config=config,
                    seed=seed,
                    treatment=treatment,
                    provider=budgeted_provider,
                )
                if budgeted_provider is not None
                else _hero(branch)
            )
            if budgeted_provider is None:
                hero.set_treatment(treatment)
            trace_start = len(hero.phase1_traces if isinstance(hero, Phase1RuleHero) else hero.decision_traces)
            branch.play(config.horizon - config.formation_hands)
            exploitation = branch.records[config.formation_hands :]
            per_hand_rows.extend(_record_rows(exploitation, treatment, seed, signatures[0]))
            traces = _trace_rows(hero, seed, signatures[0])
            trace_rows.extend(traces[trace_start:])
            true_types = {
                agent.name: (
                    "adaptive"
                    if isinstance(agent, ExperimentalOpponent)
                    and agent.stability is Stability.ADAPTIVE
                    else agent.active_type(exploitation[-1].hand_index)
                    if isinstance(agent, ExperimentalOpponent)
                    else "approximate_equilibrium"
                )
                for agent in branch.agents
                if agent.name != hero.name
            }
            mechanism_rows.extend(
                _mechanism_rows(hero, exploitation, seed, treatment, true_types)
            )
            completed_heroes.append(hero)

    mechanisms = pd.DataFrame(mechanism_rows)
    per_hand = pd.DataFrame(per_hand_rows)
    if not mechanisms.empty and "decision_regret" in mechanisms:
        regret = (
            mechanisms[mechanisms["metric"] == "decision_regret"]
            .groupby(["seed", "treatment", "hand_index"], as_index=False)["decision_regret"]
            .mean()
        )
        per_hand = per_hand.merge(
            regret,
            on=["seed", "treatment", "hand_index"],
            how="left",
        )
    if "decision_regret" not in per_hand:
        per_hand["decision_regret"] = float("nan")
    per_seed = _per_seed(per_hand, config)
    paired = _paired(per_seed, config.treatments)
    paired_hands = _paired_hand_deltas(per_hand, config.treatments)
    # Regret is the preregistered primary closed-loop outcome; return remains
    # secondary and must not silently stand in for it. Holm correction is
    # applied within each metric family.
    inference = pd.concat(
        [
            inference_table(
                paired,
                metric="decision_regret_reduction",
                bootstrap_samples=config.bootstrap_samples,
                permutation_samples=config.permutation_samples,
            ),
            inference_table(
                paired,
                metric="chips_per_100_delta",
                bootstrap_samples=config.bootstrap_samples,
                permutation_samples=config.permutation_samples,
            ),
        ],
        ignore_index=True,
    )
    forks = pd.DataFrame(fork_rows)
    if not mechanisms.empty:
        mechanism_summary = mechanisms.groupby(["treatment", "metric"], as_index=False).agg(
            observations=("hand_index", "size"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
            type_accuracy=("type_correct", "mean"),
            mean_type_probability=("type_probability", "mean"),
            mean_model_confidence=("model_confidence", "mean"),
            mean_type_brier=("type_brier", "mean"),
            mean_decision_regret=("decision_regret", "mean"),
        )
    else:
        mechanism_summary = pd.DataFrame(
            columns=[
                "treatment",
                "metric",
                "observations",
                "mean_log_loss",
                "mean_brier",
                "type_accuracy",
                "mean_type_probability",
                "mean_model_confidence",
                "mean_type_brier",
                "mean_decision_regret",
            ]
        )
    forks_identical = bool(forks["identical"].all()) if not forks.empty else False
    gate = _provider_gate(config, ledger, completed_heroes, trace_rows, forks_identical)
    cost_metrics = _cost_metrics(per_hand, trace_rows)

    per_hand.to_csv(config.output_dir / "per_hand.csv", index=False)
    per_seed.to_csv(config.output_dir / "per_seed.csv", index=False)
    paired.to_csv(config.output_dir / "paired.csv", index=False)
    paired_hands.to_csv(config.output_dir / "paired_hand_deltas.csv", index=False)
    inference.to_csv(config.output_dir / "inference.csv", index=False)
    forks.to_csv(config.output_dir / "forks.csv", index=False)
    mechanisms.to_csv(config.output_dir / "mechanism_rows.csv", index=False)
    mechanism_summary.to_csv(config.output_dir / "mechanism_metrics.csv", index=False)
    cost_metrics.to_csv(config.output_dir / "cost_metrics.csv", index=False)
    _write_jsonl(config.output_dir / "decision_traces.jsonl.gz", trace_rows)
    (config.output_dir / "provider_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    parquet_written = {
        "per_hand": _write_optional_parquet(per_hand, config.output_dir / "per_hand.parquet"),
        "per_seed": _write_optional_parquet(per_seed, config.output_dir / "per_seed.parquet"),
    }
    (config.output_dir / "artifact_formats.json").write_text(
        json.dumps({"csv": True, "parquet": parquet_written}, indent=2), encoding="utf-8"
    )
    _plot(per_seed, per_hand, config.output_dir)
    (config.output_dir / "REPORT.zh-CN.md").write_text(
        _report(config, manifest, inference, gate, paired), encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "per_hand": per_hand,
        "per_seed": per_seed,
        "paired": paired,
        "paired_hand_deltas": paired_hands,
        "inference": inference,
        "forks": forks,
        "mechanism_rows": mechanisms,
        "mechanism_metrics": mechanism_summary,
        "cost_metrics": cost_metrics,
        "provider_gate": gate,
    }


def phase1_simulation_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for opponent_type in ("rock", "tag", "lag", "calling_station", "myopic"):
        for epsilon in (0.05, 0.35):
            for stability in (Stability.FIXED, Stability.MIDPOINT_SHIFT):
                rows.append(
                    {
                        "arena": Arena.HEADS_UP,
                        "treatments": DEFAULT_TREATMENTS,
                        "opponent_type": opponent_type,
                        "epsilon": epsilon,
                        "stability": stability,
                    }
                )
            rows.append(
                {
                    "arena": Arena.HEADS_UP,
                    "treatments": DEFAULT_TREATMENTS,
                    "opponent_type": opponent_type,
                    "epsilon": epsilon,
                    "stability": Stability.ADAPTIVE,
                }
            )
    for epsilon in (0.05, 0.35):
        rows.append(
            {
                "arena": Arena.HEADS_UP,
                "treatments": DEFAULT_TREATMENTS,
                "opponent_type": "approximate_equilibrium",
                "epsilon": epsilon,
                "stability": Stability.FIXED,
            }
        )
    for composition in OpponentComposition:
        for epsilon in (0.05, 0.35):
            for stability in (Stability.FIXED, Stability.ADAPTIVE):
                rows.append(
                    {
                        "arena": Arena.SIX_MAX,
                        "treatments": DEPTH_TREATMENTS,
                        "opponent_composition": composition,
                        "epsilon": epsilon,
                        "stability": stability,
                    }
                )
    return rows


def run_phase1_matrix_smoke(output_dir: Path, seed: int = 9499) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, overrides in enumerate(phase1_simulation_matrix()):
        cell_output = output_dir / f"cell_{index:03d}"
        config = Phase1ExperimentConfig(
            **overrides,
            seeds=(seed,),
            horizon=3,
            formation_hands=1,
            equity_samples=1,
            bootstrap_samples=20,
            permutation_samples=20,
            mccfr_iterations=100,
            output_dir=cell_output,
        )
        result = run_phase1_experiment(config)
        rows.append(
            {
                "cell": index,
                "manifest_hash": result["manifest"]["manifest_hash"],
                "forks_valid": bool(result["forks"]["identical"].all()),
                "rows": len(result["per_hand"]),
            }
        )
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "matrix_smoke_summary.csv", index=False)
    return frame
