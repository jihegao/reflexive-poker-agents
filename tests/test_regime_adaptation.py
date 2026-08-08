from __future__ import annotations

from pathlib import Path

from reflexive_poker.models import ActionType
from reflexive_poker.regime_adaptation import (
    DEFAULT_WORLD,
    ConditionalDistributionDetector,
    HeuristicHypothesisGenerator,
    OpponentObservation,
    OpponentWorld,
    ProviderHypothesisGenerator,
    RegimeExperimentConfig,
    SignedConditionalEProcessDetector,
    SurpriseDetector,
    WorldSimulator,
    empirical_world,
    paired_regime_effects,
    run_regime_switch_experiment,
    summarize_paired_regime_effects,
    summarize_regime_experiment,
)
from reflexive_poker.regime_simulation import SimulationResult
from reflexive_poker.stage2a1_calibration import (
    load_stage2a1_config,
    select_threshold,
)


def test_surprise_detector_is_quiet_under_likely_actions() -> None:
    detector = SurpriseDetector(window_size=12, min_observations=6, threshold=0.40)
    updates = [detector.update(0.58) for _ in range(20)]
    assert not any(update.change_detected for update in updates)


def test_surprise_detector_triggers_on_persistent_improbability() -> None:
    detector = SurpriseDetector(
        window_size=10,
        min_observations=6,
        threshold=0.45,
        cooldown_observations=10,
    )
    updates = [detector.update(0.01) for _ in range(10)]
    assert any(update.change_detected for update in updates)


def test_conditional_detector_detects_likely_action_becoming_more_frequent() -> None:
    detector = ConditionalDistributionDetector(
        reference_size=40,
        recent_size=24,
        min_recent_observations=16,
        likelihood_ratio_threshold=3.0,
        min_probability_delta=0.12,
        required_streak=1,
        evaluation_stride=1,
    )
    reference = [
        *[OpponentObservation(ActionType.FOLD, True) for _ in range(8)],
        *[OpponentObservation(ActionType.CHECK_CALL, True) for _ in range(12)],
        *[OpponentObservation(ActionType.RAISE, False) for _ in range(4)],
        *[OpponentObservation(ActionType.CHECK_CALL, False) for _ in range(16)],
    ]
    detector.fit_reference(reference)
    recent = [
        *[OpponentObservation(ActionType.FOLD, True) for _ in range(10)],
        *[OpponentObservation(ActionType.CHECK_CALL, True) for _ in range(2)],
        *[OpponentObservation(ActionType.RAISE, False) for _ in range(8)],
        *[OpponentObservation(ActionType.CHECK_CALL, False) for _ in range(4)],
    ]
    updates = [detector.update(observation) for observation in recent]
    assert any(update.change_detected for update in updates)
    assert updates[-1].probability_deltas["facing_bet:fold"] > 0.0


def test_conditional_detector_is_two_sided_and_quiet_for_same_distribution() -> None:
    reference = [
        *[OpponentObservation(ActionType.FOLD, True) for _ in range(16)],
        *[OpponentObservation(ActionType.CHECK_CALL, True) for _ in range(4)],
        *[OpponentObservation(ActionType.RAISE, False) for _ in range(12)],
        *[OpponentObservation(ActionType.CHECK_CALL, False) for _ in range(8)],
    ]
    detector = ConditionalDistributionDetector(
        reference_size=40,
        recent_size=20,
        min_recent_observations=20,
        likelihood_ratio_threshold=3.0,
        min_probability_delta=0.12,
        required_streak=1,
        evaluation_stride=1,
    )
    detector.fit_reference(reference)
    stable = [
        *reference[:8],
        *reference[16:18],
        *reference[20:26],
        *reference[32:36],
    ]
    assert not any(detector.update(item).change_detected for item in stable)

    detector.fit_reference(reference)
    changed = [
        *[OpponentObservation(ActionType.FOLD, True) for _ in range(2)],
        *[OpponentObservation(ActionType.CHECK_CALL, True) for _ in range(8)],
        *[OpponentObservation(ActionType.RAISE, False) for _ in range(2)],
        *[OpponentObservation(ActionType.CHECK_CALL, False) for _ in range(8)],
    ]
    updates = [detector.update(item) for item in changed]
    assert updates[-1].change_detected
    assert updates[-1].probability_deltas["facing_bet:fold"] < 0.0


def test_signed_e_process_detects_joint_upward_shift_without_stable_alarm() -> None:
    reference = [
        *[OpponentObservation(ActionType.RAISE, False) for _ in range(9)],
        *[OpponentObservation(ActionType.CHECK_CALL, False) for _ in range(39)],
        *[OpponentObservation(ActionType.FOLD, True) for _ in range(20)],
        *[OpponentObservation(ActionType.CHECK_CALL, True) for _ in range(28)],
    ]
    stable_block = [
        *[OpponentObservation(ActionType.RAISE, False) for _ in range(2)],
        *[OpponentObservation(ActionType.CHECK_CALL, False) for _ in range(6)],
        *[OpponentObservation(ActionType.FOLD, True) for _ in range(3)],
        *[OpponentObservation(ActionType.CHECK_CALL, True) for _ in range(5)],
    ]
    detector = SignedConditionalEProcessDetector(
        reference_size=96,
        block_size=16,
        e_value_threshold=3.0,
        maximum_blocks=32,
        minimum_direction_delta=0.05,
    )
    detector.fit_reference(reference)
    stable_updates = [detector.update(item) for _ in range(8) for item in stable_block]
    assert not any(update.change_detected for update in stable_updates)

    shifted_block = [
        *[OpponentObservation(ActionType.RAISE, False) for _ in range(3)],
        *[OpponentObservation(ActionType.CHECK_CALL, False) for _ in range(5)],
        *[OpponentObservation(ActionType.FOLD, True) for _ in range(5)],
        *[OpponentObservation(ActionType.CHECK_CALL, True) for _ in range(3)],
    ]
    detector.fit_reference(reference)
    shifted_updates = [detector.update(item) for _ in range(6) for item in shifted_block]
    completed = [update for update in shifted_updates if update.block_complete]
    assert any(update.direction == "up" for update in completed)
    assert completed[-1].e_value_up > completed[-1].e_value_down


def test_signed_e_process_tracks_downward_shift_separately() -> None:
    reference = [
        *[OpponentObservation(ActionType.RAISE, False) for _ in range(9)],
        *[OpponentObservation(ActionType.CHECK_CALL, False) for _ in range(39)],
        *[OpponentObservation(ActionType.FOLD, True) for _ in range(20)],
        *[OpponentObservation(ActionType.CHECK_CALL, True) for _ in range(28)],
    ]
    downward_block = [
        *[OpponentObservation(ActionType.CHECK_CALL, False) for _ in range(8)],
        *[OpponentObservation(ActionType.FOLD, True) for _ in range(2)],
        *[OpponentObservation(ActionType.CHECK_CALL, True) for _ in range(6)],
    ]
    detector = SignedConditionalEProcessDetector(
        reference_size=96,
        block_size=16,
        e_value_threshold=3.0,
        maximum_blocks=32,
        minimum_direction_delta=0.05,
    )
    detector.fit_reference(reference)
    updates = [detector.update(item) for _ in range(4) for item in downward_block]
    completed = [update for update in updates if update.block_complete]
    assert any(update.direction == "down" for update in completed)
    assert completed[-1].e_value_down > completed[-1].e_value_up


def test_empirical_world_separates_opening_and_facing_bet_contexts() -> None:
    observations = [
        *[OpponentObservation(ActionType.RAISE, False) for _ in range(7)],
        *[OpponentObservation(ActionType.CHECK_CALL, False) for _ in range(3)],
        *[OpponentObservation(ActionType.FOLD, True) for _ in range(6)],
        *[OpponentObservation(ActionType.CHECK_CALL, True) for _ in range(3)],
        OpponentObservation(ActionType.RAISE, True),
    ]
    world = empirical_world(observations)
    assert world.open_raise_probability > 0.60
    assert world.fold_vs_bet_probability > 0.50
    assert world.reraise_probability < 0.40


def test_heuristic_generator_returns_conditional_worlds() -> None:
    generator = HeuristicHypothesisGenerator()
    observations = [
        *[OpponentObservation(ActionType.RAISE, False) for _ in range(8)],
        *[OpponentObservation(ActionType.RAISE, True) for _ in range(2)],
    ]
    worlds = generator.generate(observations, [DEFAULT_WORLD])
    assert generator.calls == 1
    assert len(worlds) == 4
    empirical = next(world for world in worlds if world.name == "empirical_shift")
    assert empirical.open_raise_probability > 0.70


class _FakeProviderResponse:
    def __init__(self) -> None:
        self.payload = {
            "worlds": [
                {
                    "name": "raise_heavy",
                    "open_raise_probability": 0.60,
                    "fold_vs_bet_probability": 0.15,
                    "reraise_probability": 0.30,
                    "prior": 1.0,
                    "rationale": "Observed initiative increased.",
                },
                {
                    "name": "call_heavy",
                    "open_raise_probability": 0.10,
                    "fold_vs_bet_probability": 0.15,
                    "reraise_probability": 0.05,
                    "prior": 0.5,
                    "rationale": "Alternative passive explanation.",
                },
            ]
        }


class _FakeProvider:
    def structured(self, **kwargs: object) -> _FakeProviderResponse:
        assert kwargs["schema_name"] == "opponent_regime_hypotheses"
        return _FakeProviderResponse()


def test_provider_generator_uses_conditional_world_contract() -> None:
    generator = ProviderHypothesisGenerator(_FakeProvider())
    worlds = generator.generate(
        [OpponentObservation(ActionType.RAISE, False) for _ in range(4)],
        [DEFAULT_WORLD],
    )
    assert generator.calls == 1
    assert [world.name for world in worlds] == ["raise_heavy", "call_heavy"]
    assert worlds[0].open_raise_probability == 0.60


def test_world_simulator_runs_paired_complete_hands() -> None:
    simulator = WorldSimulator(rollouts=6, seed=4, equity_samples=1)
    aggressive = OpponentWorld(
        "aggressive",
        open_raise_probability=0.55,
        fold_vs_bet_probability=0.15,
        reraise_probability=0.35,
    )
    observations = [
        *[OpponentObservation(ActionType.RAISE, False) for _ in range(8)],
        *[OpponentObservation(ActionType.RAISE, True) for _ in range(4)],
    ]
    result = simulator.evaluate([aggressive], observations)[0]
    assert result.simulation_unit == "full_hand"
    assert result.rollout_hands == 6
    assert set(result.response_values) == {"pressure", "balanced", "bluff_catch"}
    assert simulator.simulated_hands == 18
    assert len(set(result.response_values.values())) > 1


def test_robust_response_selector_requires_positive_paired_lower_bound() -> None:
    safe = SimulationResult(
        world_name="world",
        posterior=1.0,
        best_response="pressure",
        expected_bb_per_decision=0.3,
        response_values={"pressure": 0.3, "balanced": 0.0, "bluff_catch": -0.2},
        rollout_hands=6,
        response_samples={
            "pressure": (0.4, 0.3, 0.5, 0.2, 0.4, 0.3),
            "balanced": (0.0, 0.1, 0.0, 0.0, 0.1, 0.0),
            "bluff_catch": (-0.2, -0.1, -0.3, -0.2, -0.1, -0.3),
        },
    )
    response, _, lower = WorldSimulator.choose_response_robust([safe])
    assert response == "pressure"
    assert lower > 0.0

    noisy = SimulationResult(
        world_name="world",
        posterior=1.0,
        best_response="pressure",
        expected_bb_per_decision=0.1,
        response_values={"pressure": 0.1, "balanced": 0.0, "bluff_catch": -0.1},
        rollout_hands=4,
        response_samples={
            "pressure": (2.0, -1.8, 2.0, -1.8),
            "balanced": (0.0, 0.0, 0.0, 0.0),
            "bluff_catch": (-0.1, -0.1, -0.1, -0.1),
        },
    )
    response, _, lower = WorldSimulator.choose_response_robust([noisy])
    assert response == "balanced"
    assert lower == 0.0


def test_stage2a1_config_uses_disjoint_tuning_and_validation_seeds() -> None:
    config = load_stage2a1_config(Path("configs/regime_stage2a1.yaml"))
    assert not set(config.tuning_seeds) & set(config.validation_seeds)
    assert config.threshold_candidates == (3.0, 5.0, 10.0, 20.0, 50.0)


def test_stage2a1_threshold_selection_prefers_smallest_passing_candidate() -> None:
    selected = select_threshold(
        [
            {"threshold": 3.0, "gate_pass": False, "selection_penalty": 0.4},
            {"threshold": 10.0, "gate_pass": True, "selection_penalty": 0.0},
            {"threshold": 5.0, "gate_pass": True, "selection_penalty": 0.0},
        ]
    )
    assert selected["threshold"] == 5.0


def test_regime_experiment_writes_paired_outputs(tmp_path: Path) -> None:
    rows = run_regime_switch_experiment(
        RegimeExperimentConfig(
            seeds=(17,),
            hands=72,
            switch_hand=36,
            equity_samples=2,
            recovery_window=8,
            simulation_rollout_hands=4,
            simulation_equity_samples=1,
            formation_observations=8,
            calibration_observations=4,
            output_dir=tmp_path,
        )
    )
    assert len(rows) == 6
    assert {row.condition for row in rows} == {
        "baseline",
        "reflection",
        "reflection_simulation",
    }
    summary = summarize_regime_experiment(rows)
    effects = paired_regime_effects(rows)
    paired_summary = summarize_paired_regime_effects(effects)
    assert len(summary) == 3
    assert len(effects) == 2
    assert paired_summary["post_switch_bb100_delta"]["n"] == 2
    assert (tmp_path / "matches.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "paired_effects.csv").exists()
    assert (tmp_path / "paired_summary.json").exists()
