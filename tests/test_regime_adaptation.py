from __future__ import annotations

from pathlib import Path

from reflexive_poker.models import ActionType
from reflexive_poker.regime_adaptation import (
    DEFAULT_WORLD,
    HeuristicHypothesisGenerator,
    OpponentWorld,
    ProviderHypothesisGenerator,
    RegimeExperimentConfig,
    SurpriseDetector,
    WorldSimulator,
    run_regime_switch_experiment,
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


def test_heuristic_generator_returns_valid_worlds() -> None:
    generator = HeuristicHypothesisGenerator()
    worlds = generator.generate(
        [ActionType.RAISE] * 8 + [ActionType.CHECK_CALL] * 2,
        [DEFAULT_WORLD],
    )
    assert generator.calls == 1
    assert len(worlds) >= 3
    empirical = next(world for world in worlds if world.name == "empirical_shift")
    assert empirical.raise_probability > empirical.fold_probability


class _FakeProviderResponse:
    def __init__(self) -> None:
        self.payload = {
            "worlds": [
                {
                    "name": "raise_heavy",
                    "fold_probability": 0.1,
                    "call_probability": 0.3,
                    "raise_probability": 0.6,
                    "prior": 1.0,
                    "rationale": "Observed raise frequency increased.",
                },
                {
                    "name": "call_heavy",
                    "fold_probability": 0.1,
                    "call_probability": 0.8,
                    "raise_probability": 0.1,
                    "prior": 0.5,
                    "rationale": "Alternative passive explanation.",
                },
            ]
        }


class _FakeProvider:
    def structured(self, **kwargs: object) -> _FakeProviderResponse:
        assert kwargs["schema_name"] == "opponent_regime_hypotheses"
        return _FakeProviderResponse()


def test_provider_generator_uses_structured_world_contract() -> None:
    generator = ProviderHypothesisGenerator(_FakeProvider())
    worlds = generator.generate([ActionType.RAISE] * 4, [DEFAULT_WORLD])
    assert generator.calls == 1
    assert [world.name for world in worlds] == ["raise_heavy", "call_heavy"]
    assert worlds[0].raise_probability == 0.6


def test_world_simulator_selects_expected_counter_strategy() -> None:
    simulator = WorldSimulator(rollouts=20_000, seed=4)
    aggressive = OpponentWorld("aggressive", 0.10, 0.35, 0.55)
    passive = OpponentWorld("passive", 0.55, 0.38, 0.07)
    aggressive_result = simulator.evaluate([aggressive], [ActionType.RAISE] * 8)
    passive_result = simulator.evaluate([passive], [ActionType.FOLD] * 8)
    assert aggressive_result[0].best_response == "bluff_catch"
    assert passive_result[0].best_response == "pressure"


def test_regime_experiment_smoke_writes_outputs(tmp_path: Path) -> None:
    rows = run_regime_switch_experiment(
        RegimeExperimentConfig(
            seeds=(17,),
            hands=36,
            switch_hand=18,
            equity_samples=1,
            recovery_window=6,
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
    assert len(summary) == 3
    assert (tmp_path / "matches.csv").exists()
    assert (tmp_path / "summary.json").exists()
