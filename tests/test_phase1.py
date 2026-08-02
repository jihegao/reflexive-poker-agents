import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reflexive_poker.environment import EnvironmentConfig, HoldemEnvironment
from reflexive_poker.llm_player import ProviderResponse
from reflexive_poker.models import ActionType, DecisionContext, Street
from reflexive_poker.phase1_experiment import (
    Phase1ExperimentConfig,
    _make_environment,
    build_llm_confirmation_plan,
    environment_fork_signature,
    phase1_simulation_matrix,
    run_phase1_experiment,
)
from reflexive_poker.phase1_models import (
    AbstractMCCFRPolicy,
    Arena,
    BudgetedRetryProvider,
    ExperimentalOpponent,
    OpponentBeliefState,
    Phase1LLMHero,
    Phase1RuleHero,
    ProviderBudget,
    ProviderBudgetExceeded,
    ProviderLedger,
    ReasoningTreatment,
    Stability,
    validate_phase1_decision,
)
from reflexive_poker.phase1_protocol import validate_closed_loop_completion
from reflexive_poker.phase1_resumable import (
    FullSimulationRunConfig,
    LLMConfirmationRunConfig,
    run_full_simulation_matrix,
    run_llm_confirmation_resumable,
)
from reflexive_poker.phase1_statistics import (
    classify_core_hypothesis,
    holm_adjust,
    inference_table,
    large_pot_sensitivity,
    paired_bootstrap_interval,
)


def test_belief_state_updates_and_normalizes() -> None:
    state = OpponentBeliefState("villain", window_size=3)
    for hand, action in enumerate(
        (ActionType.RAISE, ActionType.RAISE, ActionType.FOLD, ActionType.CHECK_CALL)
    ):
        state.observe_opponent_action(action, hand)
    state.observe_hero_action(ActionType.RAISE, 4)
    state.observe_response(ActionType.FOLD, 4)
    assert len(state.recent_actions) == 3
    assert sum(state.action_distribution.values()) == pytest.approx(1.0)
    assert sum(state.type_posterior.values()) == pytest.approx(1.0)
    assert sum(state.conditional_response_model.values()) == pytest.approx(1.0)
    assert state.hero_public_image > 0.5
    assert state.confidence > 0
    assert len(state.digest()) == 64


def test_treatment_masks_and_depths_are_isolated() -> None:
    hero = Phase1RuleHero(
        "hero", 1, ReasoningTreatment.STATE_ONLY, ("villain",)
    )
    state_only = hero.treatment_features(("villain",))
    assert set(state_only.values()) == {"__MASKED__"}
    hero.set_treatment(ReasoningTreatment.ACTION_PREDICTION)
    d1 = hero.treatment_features(("villain",))
    assert isinstance(d1["action_prediction"], dict)
    assert d1["opponent_view_of_hero"] == "__MASKED__"
    hero.set_treatment(ReasoningTreatment.RECURSIVE_D2)
    d2 = hero.treatment_features(("villain",))
    assert isinstance(d2["opponent_view_of_hero"], float)
    assert d2["anticipated_adjustment"] == "__MASKED__"
    hero.set_treatment(ReasoningTreatment.RECURSIVE_D3)
    assert isinstance(hero.treatment_features(("villain",))["anticipated_adjustment"], float)
    assert ReasoningTreatment.RECURSIVE_D3.depth == 3


def test_experimental_opponent_switches_and_tracks_hero() -> None:
    opponent = ExperimentalOpponent(
        "villain",
        3,
        "tag",
        ("hero",),
        epsilon=0.05,
        stability=Stability.MIDPOINT_SHIFT,
        switch_hand=2,
        switch_type="lag",
        equity_samples=1,
    )
    assert opponent.active_type(1) == "tag"
    assert opponent.active_type(2) == "lag"
    hero = Phase1RuleHero("hero", 2, ReasoningTreatment.STATE_ONLY, ("villain",))
    HoldemEnvironment(
        [hero, opponent], seed=4, config=EnvironmentConfig(max_raises_per_street=1)
    ).play(3)
    assert opponent.hero_actions


def test_abstract_mccfr_policy_is_frozen_and_legal() -> None:
    first = AbstractMCCFRPolicy.train(iterations=200, seed=7)
    second = AbstractMCCFRPolicy.train(iterations=200, seed=7)
    assert first.report.policy_hash == second.report.policy_hash
    assert first.report.infosets == 40
    distribution = first.distribution(0.7, 0.2, True)
    assert set(distribution) == {"fold", "check_call", "raise"}
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert first.report.label.startswith("abstract_external_sampling")
    short = AbstractMCCFRPolicy.train(iterations=1, seed=9)
    assert all(sum(values.values()) == pytest.approx(1.0) for values in short.policy.values())


class _FlakyProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, _state):
        self.calls += 1
        payload = (
            {"action": "check_call"}
            if self.calls == 1
            else {
                "action": "check_call",
                "raise_scale": 0.5,
                "confidence": 0.7,
                "situation_summary": "test",
                "rationale": "test",
                "self_model": "test",
                "opponent_model": "test",
                "risk_flags": [],
                "next_step": "test",
            }
        )
        return ProviderResponse(
            payload=payload,
            provider=self.name,
            model=self.model,
            latency_ms=1.0,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )

    def reflect(self, _state):
        raise AssertionError("reflection must not be called")


class _ProbabilityRepairProvider:
    name = "repair-fake"
    model = "repair-fake-model"

    def __init__(self) -> None:
        self.instructions: list[str] = []

    def structured(self, *, instructions, state, schema_name, schema):
        del state, schema_name, schema
        self.instructions.append(instructions)
        invalid = len(self.instructions) == 1
        payload = {
            "action": "check_call",
            "raise_scale": 0.5,
            "confidence": 0.7,
            "situation_summary": "test",
            "rationale": "test",
            "self_model": "test",
            "opponent_model": "test",
            "risk_flags": [],
            "next_step": "test",
            "opponent_state": {
                "type_probabilities": {
                    "rock": 0.2 if invalid else 1 / 6,
                    "tag": 0.2 if invalid else 1 / 6,
                    "lag": 0.2 if invalid else 1 / 6,
                    "calling_station": 0.2 if invalid else 1 / 6,
                    "myopic": 0.2 if invalid else 1 / 6,
                    "adaptive": 0.2 if invalid else 1 / 6,
                },
                "action_probabilities": {
                    "fold": 1 / 3,
                    "check_call": 1 / 3,
                    "raise": 1 / 3,
                },
                "hero_image_aggression": 0.5,
                "adaptation_probability": 0.5,
                "switch_detected": False,
            },
        }
        return ProviderResponse(
            payload=payload,
            provider=self.name,
            model=self.model,
            latency_ms=1.0,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )


def test_budgeted_provider_counts_repair_and_fails_closed() -> None:
    ledger = ProviderLedger()
    provider = BudgetedRetryProvider(
        _FlakyProvider(), ProviderBudget(max_calls=2, max_retries=1), ledger
    )
    response = provider.decide({})
    assert response.payload["action"] == "check_call"
    assert ledger.calls == 2
    assert ledger.retries == 1
    assert ledger.raw_failures == 1
    assert ledger.unresolved_failures == 0
    with pytest.raises(ProviderBudgetExceeded):
        provider.decide({})


def test_structured_retry_includes_probability_repair_constraint() -> None:
    raw_provider = _ProbabilityRepairProvider()
    ledger = ProviderLedger()
    provider = BudgetedRetryProvider(
        raw_provider, ProviderBudget(max_calls=2, max_retries=1), ledger
    )
    response = provider.structured(
        instructions="return probabilities",
        state={},
        schema_name="phase1_closed_loop_decision",
        schema={},
        validator=validate_phase1_decision,
    )
    assert response.payload["opponent_state"]["type_probabilities"]["rock"] == pytest.approx(1 / 6)
    assert ledger.calls == 2
    assert ledger.retries == 1
    assert ledger.raw_failures == 1
    assert ledger.unresolved_failures == 0
    assert "REPAIR REQUIRED" in raw_provider.instructions[1]
    assert "type_probabilities" in raw_provider.instructions[1]


def test_phase1_llm_hero_propagates_exhausted_provider_budget() -> None:
    hero = Phase1LLMHero(
        "hero",
        1,
        BudgetedRetryProvider(
            _FlakyProvider(), ProviderBudget(max_calls=1), ProviderLedger(calls=1)
        ),
        ReasoningTreatment.STATE_ONLY,
        ("villain",),
    )
    context = DecisionContext(
        hand_index=0,
        street=Street.PREFLOP,
        player_name="hero",
        hole_cards=(1, 2),
        board=(),
        pot=1.0,
        to_call=0.0,
        stack=99.0,
        current_bet=1.0,
        legal_actions=(ActionType.CHECK_CALL, ActionType.RAISE),
        active_players=2,
        opponents=("villain",),
        last_raiser=None,
        raises_this_street=0,
        button_distance=0,
        environment_regime="fixed",
    )
    with pytest.raises(ProviderBudgetExceeded):
        hero.act(context)


def test_provider_ledger_is_checkpointed_before_and_after_calls(tmp_path: Path) -> None:
    checkpoint = tmp_path / "live_provider_ledger.json"
    attempts = tmp_path / "live_provider_attempts.jsonl"
    provider = BudgetedRetryProvider(
        _FlakyProvider(),
        ProviderBudget(max_calls=2, max_primary_calls=1, max_retries=1),
        ProviderLedger(),
        checkpoint_path=checkpoint,
        attempt_log_path=attempts,
    )
    provider.decide({})
    payload = json.loads(checkpoint.read_text())
    assert payload["ledger"]["calls"] == 2
    assert payload["ledger"]["retries"] == 1
    assert payload["ledger"]["raw_failures"] == 1
    records = [json.loads(line) for line in attempts.read_text().splitlines()]
    assert [(row["outcome"], row["retry"]) for row in records] == [
        ("failed", False),
        ("succeeded", True),
    ]


def test_shared_formation_fork_signature_is_identical() -> None:
    config = Phase1ExperimentConfig(
        treatments=(ReasoningTreatment.STATE_ONLY, ReasoningTreatment.RECURSIVE_D2),
        seeds=(12,),
        horizon=4,
        formation_hands=2,
        equity_samples=1,
    )
    environment = _make_environment(
        config, 12, ReasoningTreatment.STATE_ONLY, None, None
    )
    environment.play(2)
    import copy

    assert environment_fork_signature(copy.deepcopy(environment)) == environment_fork_signature(
        copy.deepcopy(environment)
    )


def test_paired_seed_uses_a_real_hero_seat_mirror() -> None:
    config = Phase1ExperimentConfig(
        treatments=(ReasoningTreatment.STATE_ONLY,),
        seeds=(12,),
        horizon=4,
        formation_hands=1,
        equity_samples=1,
    )
    first = _make_environment(config, 12, ReasoningTreatment.STATE_ONLY, None, None)
    mirrored = _make_environment(config, 13, ReasoningTreatment.STATE_ONLY, None, None)
    assert first.agents[0].name == "hero"
    assert mirrored.agents[1].name == "hero"


def test_closed_loop_completion_validator_rejects_any_missing_paired_arm() -> None:
    providers = ("deepseek", "codex")
    treatments = ("state_only", "d1_budget_matched", "recursive_d2")
    regimes = ("fixed", "adaptive")
    rows = [
        {
            "seed": seed,
            "provider": provider,
            "treatment": treatment,
            "regime": regime,
            "checkpoint_id": f"checkpoint-{seed}",
            "valid": True,
        }
        for seed in (1, 2)
        for provider in providers
        for treatment in treatments
        for regime in regimes
    ]
    complete = validate_closed_loop_completion(
        pd.DataFrame(rows),
        providers=providers,
        treatments=treatments,
        regimes=regimes,
        target_seeds=(1, 2),
    )
    assert complete["formal_completion_valid"] is True
    incomplete = validate_closed_loop_completion(
        pd.DataFrame(rows[:-1]),
        providers=providers,
        treatments=treatments,
        regimes=regimes,
        target_seeds=(1, 2),
    )
    assert incomplete["formal_completion_valid"] is False
    assert incomplete["valid_paired_blocks"] == 1


def test_rule_phase1_smoke_writes_auditable_artifacts(tmp_path: Path) -> None:
    result = run_phase1_experiment(
        Phase1ExperimentConfig(
            treatments=(
                ReasoningTreatment.STATE_ONLY,
                ReasoningTreatment.ACTION_PREDICTION,
                ReasoningTreatment.RECURSIVE_D2,
            ),
            seeds=(21, 22),
            horizon=6,
            formation_hands=2,
            equity_samples=1,
            bootstrap_samples=40,
            permutation_samples=40,
            output_dir=tmp_path,
        )
    )
    assert result["forks"]["identical"].all()
    assert len(result["per_hand"]) == 2 * 3 * 4
    assert len(result["paired"]) == 6
    assert result["provider_gate"]["applicable"] is False
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "mechanism_metrics.csv").exists()
    assert (tmp_path / "paired_hand_deltas.csv").exists()
    assert (tmp_path / "depth_payoff.png").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["evidence_class"] == (
        "exploratory_or_smoke"
    )


def test_mock_llm_smoke_uses_no_reflection_and_passes_gate(tmp_path: Path) -> None:
    result = run_phase1_experiment(
        Phase1ExperimentConfig(
            treatments=(ReasoningTreatment.STATE_ONLY, ReasoningTreatment.RECURSIVE_D2),
            seeds=(31,),
            horizon=4,
            formation_hands=1,
            equity_samples=1,
            provider="mock",
            model="mock",
            provider_budget=ProviderBudget(max_calls=100, max_retries=5),
            bootstrap_samples=20,
            permutation_samples=20,
            output_dir=tmp_path,
        )
    )
    assert result["provider_gate"]["valid"] is True
    ledger = result["provider_gate"]["ledger"]
    assert ledger["calls"] == ledger["token_observed_calls"]
    assert ledger["unresolved_failures"] == 0
    assert not (tmp_path / "reflection_traces.jsonl.gz").exists()
    assert {"state_only", "recursive_d2"}.issubset(
        set(result["cost_metrics"]["treatment"])
    )
    assert "decision_regret" in result["per_hand"].columns
    assert result["mechanism_rows"]["metric"].isin(
        {"action_prediction", "strategy_type", "decision_regret"}
    ).all()
    with gzip.open(tmp_path / "decision_traces.jsonl.gz", "rt", encoding="utf-8") as handle:
        traces = [json.loads(line) for line in handle]
    model_traces = [row for row in traces if row.get("phase1_treatment") != "shared_formation"]
    assert model_traces
    assert all("opponent_state" in row["provider_output"] for row in model_traces)


def test_phase1_matrix_has_all_frozen_cells() -> None:
    matrix = phase1_simulation_matrix()
    assert len(matrix) == 40
    assert sum(cell["arena"] is Arena.HEADS_UP for cell in matrix) == 32
    assert sum(cell["arena"] is Arena.SIX_MAX for cell in matrix) == 8


def test_llm_confirmation_plan_locks_models_and_budget() -> None:
    plan = build_llm_confirmation_plan(ReasoningTreatment.RECURSIVE_D2)
    assert plan.models == (
        ("opencode-go", "deepseek-v4-flash"),
        ("codex", "gpt-5.6-luna"),
    )
    assert sum(job.call_budget for job in plan.jobs) == 8_000
    assert plan.offline_call_budget == 1_600
    assert plan.preflight_retry_reserve == 400
    assert len(plan.jobs) == 2
    assert plan.jobs[0].stability is Stability.FIXED
    assert plan.jobs[-1].stability is Stability.ADAPTIVE
    assert plan.jobs[-1].treatments == (
        ReasoningTreatment.STATE_ONLY,
        ReasoningTreatment.BUDGET_MATCHED_D1,
        ReasoningTreatment.RECURSIVE_D2,
    )


def test_statistics_cover_pairing_holm_and_large_pots() -> None:
    low, high = paired_bootstrap_interval(np.array([1.0, 2.0, 3.0]), samples=100, seed=1)
    assert low <= 2.0 <= high
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    paired = pd.DataFrame(
        {
            "contrast": ["d2-d1"] * 4,
            "chips_per_100_delta": [1.0, 2.0, 3.0, 4.0],
        }
    )
    inference = inference_table(
        paired,
        metric="chips_per_100_delta",
        bootstrap_samples=100,
        permutation_samples=100,
    )
    assert inference.loc[0, "metric"] == "chips_per_100_delta"
    assert inference.loc[0, "positive_seed_rate"] == 1.0
    sensitivity = large_pot_sensitivity(pd.DataFrame({"reward": [1.0, 2.0, 100.0]}))
    assert sensitivity["largest_abs_reward"] == 100.0
    assert sensitivity["trimmed_1pct_reward"] == 3.0


def test_core_hypothesis_classifier_requires_two_valid_models() -> None:
    simulation = pd.DataFrame(
        {
            "contrast": ["recursive_d2-state_only", "recursive_d2-action_prediction"],
            "ci95_low": [1.0, 0.5],
        }
    )
    paired = pd.DataFrame(
        {
            "contrast": ["recursive_d2-state_only", "recursive_d2-action_prediction"],
            "trimmed_delta": [2.0, 1.0],
        }
    )
    llm = {
        model: {
            "inference": pd.DataFrame(
                {
                    "contrast": ["recursive_d2-action_prediction"],
                    "mean_delta": [2.0],
                    "ci95_low": [0.2],
                }
            ),
            "provider_gate": {"valid": True},
            "cost_metrics": pd.DataFrame(
                {"treatment": ["recursive_d2"], "chips_per_1000_tokens": [0.1]}
            ),
        }
        for model in ("deepseek", "codex")
    }
    result = classify_core_hypothesis(simulation, paired, llm)
    assert result["classification"] == "strong_support"


def test_full_simulation_runner_resumes_completed_seed_blocks(tmp_path: Path) -> None:
    first = run_full_simulation_matrix(
        FullSimulationRunConfig(
            output_dir=tmp_path,
            seeds=(501, 502),
            horizon=4,
            formation_hands=1,
            equity_samples=1,
            mccfr_iterations=20,
            bootstrap_samples=20,
            permutation_samples=20,
            max_cells=1,
            max_seed_blocks=1,
            allow_dirty_worktree=True,
        )
    )
    assert first.loc[0, "completed_seeds"] == 1
    marker = tmp_path / "cell_000" / "seeds" / "seed_501" / "COMPLETED.json"
    original = marker.read_text()
    interrupted = tmp_path / "cell_000" / "seeds" / "seed_502.running"
    interrupted.mkdir(parents=True)
    (interrupted / "partial.txt").write_text("interrupted")
    second = run_full_simulation_matrix(
        FullSimulationRunConfig(
            output_dir=tmp_path,
            seeds=(501, 502),
            horizon=4,
            formation_hands=1,
            equity_samples=1,
            mccfr_iterations=20,
            bootstrap_samples=20,
            permutation_samples=20,
            max_cells=1,
            allow_dirty_worktree=True,
        )
    )
    assert second.loc[0, "completed_seeds"] == 2
    assert second.loc[0, "complete"]
    assert marker.read_text() == original
    assert list((interrupted.parent / "interrupted").glob("seed_502.running__*/partial.txt"))
    assert (tmp_path / "cell_000" / "aggregate" / "inference.csv").exists()


def test_mock_llm_confirmation_resumes_without_repeating_attempts(tmp_path: Path) -> None:
    config = LLMConfirmationRunConfig(
        output_dir=tmp_path,
        models=(("mock", "mock"),),
        seeds=(601, 602),
        horizon=4,
        formation_hands=1,
        equity_samples=1,
        minimum_primary_calls_to_start_block=1,
        max_primary_calls_per_paired_block=120,
        max_blocks=1,
        allow_dirty_worktree=True,
    )
    first = run_llm_confirmation_resumable(config)
    assert first["attempted_blocks"].sum() == 1
    assert set(first["block_primary_call_cap"].dropna()) == {120}
    assert json.loads(
        (tmp_path / "models" / "mock__mock" / "preflight" / "ATTEMPT.json").read_text()
    )["valid"]
    source_provenance = json.loads((tmp_path / "SOURCE_PROVENANCE.json").read_text())
    repository = Path(__file__).resolve().parents[1]
    actual_worktree_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert source_provenance["worktree_dirty"] is actual_worktree_dirty
    assert source_provenance["protocol_semantics_id"] == "prbench-cross-model-v1"
    assert (tmp_path / source_provenance["source_snapshot"]).exists()
    second = run_llm_confirmation_resumable(config)
    assert second["attempted_blocks"].sum() == 2
    attempts = list(tmp_path.glob("models/mock__mock/jobs/*/blocks/seed_*/ATTEMPT.json"))
    assert len(attempts) == 2
    seeds = {json.loads(path.read_text())["seed"] for path in attempts}
    assert seeds == {601, 602}


def test_llm_confirmation_recovers_running_directory_and_counts_usage(tmp_path: Path) -> None:
    config = LLMConfirmationRunConfig(
        output_dir=tmp_path,
        models=(("mock", "mock"),),
        seeds=(701,),
        horizon=4,
        formation_hands=1,
        equity_samples=1,
        minimum_primary_calls_to_start_block=1,
        max_primary_calls_per_paired_block=120,
        max_blocks=0,
        allow_dirty_worktree=True,
    )
    # Establish the immutable plan, then emulate a process killed during a provider call.
    run_llm_confirmation_resumable(config)
    job = "hu_fixed_paper_contrast"
    running = tmp_path / "models" / "mock__mock" / "jobs" / job / "blocks" / "seed_701.running"
    running.mkdir(parents=True)
    (running / "live_provider_ledger.json").write_text(
        json.dumps({"ledger": {"calls": 3, "retries": 1}}), encoding="utf-8"
    )
    result = run_llm_confirmation_resumable(config)
    interrupted = list(
        (running.parent / "interrupted").glob("seed_701.running__*/ATTEMPT.json")
    )
    assert len(interrupted) == 1
    usage = json.loads(interrupted[0].read_text())
    assert usage["calls"] == 3
    assert usage["retries"] == 1
    assert result.loc[result["job"] == job, "attempted_blocks"].iloc[0] == 1


def test_llm_confirmation_rejects_insufficient_block_budget_before_preflight(tmp_path: Path) -> None:
    config = LLMConfirmationRunConfig(
        output_dir=tmp_path,
        models=(("mock", "mock"),),
        seeds=(9700,),
        horizon=20,
        formation_hands=5,
        max_primary_calls_per_paired_block=100,
        allow_dirty_worktree=True,
    )
    with pytest.raises(ValueError, match="required_upper_bound=600"):
        run_llm_confirmation_resumable(config)
    assert not (tmp_path / "models").exists()


def test_llm_confirmation_rejects_job_budget_below_frozen_seed_coverage(tmp_path: Path) -> None:
    config = LLMConfirmationRunConfig(
        output_dir=tmp_path,
        models=(("mock", "mock"),),
        seeds=tuple(range(9700, 9707)),
        horizon=20,
        formation_hands=5,
        max_primary_calls_per_paired_block=600,
        max_blocks=0,
        allow_dirty_worktree=True,
    )
    with pytest.raises(ValueError, match="job_budget=4000, required_upper_bound=4200"):
        run_llm_confirmation_resumable(config)
    assert not (tmp_path / "models").exists()
