from __future__ import annotations

from pathlib import Path

from reflexive_poker.models import ActionType
from reflexive_poker.regime_adaptation import (
    DEFAULT_WORLD,
    HeuristicHypothesisGenerator,
    OpponentObservation,
    OpponentWorld,
    ProviderHypothesisGenerator,
    RegimeExperimentConfig,
    SurpriseDetector,
    WorldSimulator,
    empirical_world,
    paired_regime_effects,
    run_regime_switch_experiment,
    summarize_paired_regime_effects,
    summarize_regime_experiment,
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
