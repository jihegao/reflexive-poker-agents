import gzip
import json
from pathlib import Path

from reflexive_poker.agents import PokerAgent
from reflexive_poker.environment import EnvironmentConfig, HoldemEnvironment
from reflexive_poker.models import ActionType, Decision, DecisionContext
from reflexive_poker.second_order_experiment import (
    SecondOrderConfig,
    run_second_order_experiment,
)
from reflexive_poker.shared_history_experiment import (
    SharedHistoryConfig,
    run_shared_history_experiment,
)
from reflexive_poker.simulation import (
    CONFIRMATORY_CONDITIONS,
    IMAGE_SHAPING_CONDITIONS,
    run_study,
    summarize,
)
from reflexive_poker.six_max_experiment import SixMaxConfig, run_six_max_experiment


class _AlwaysRaiseAgent(PokerAgent):
    def act(self, context: DecisionContext) -> Decision:
        action = (
            ActionType.RAISE if ActionType.RAISE in context.legal_actions else ActionType.CHECK_CALL
        )
        return Decision(action=action, raise_scale=0.25)


def test_confirmatory_demo_runs():
    data = run_study(CONFIRMATORY_CONDITIONS, seeds=[1], hands=12, hidden_shift=True)
    assert set(data["condition"]) == {c.name for c in CONFIRMATORY_CONDITIONS}
    assert data.groupby("condition").size().eq(12).all()
    summary = summarize(data)
    assert {"chips_per_100", "image_mae", "avg_reasoning_ops"}.issubset(summary.columns)


def test_closed_loop_can_stop_signaling_early():
    data = run_study(IMAGE_SHAPING_CONDITIONS, seeds=[7, 8], hands=96)
    stops = summarize(data).set_index("condition")["signal_stop_hand"]
    assert stops["closed_loop_shaping"] <= 96
    assert stops["open_loop_shaping"] <= 30


def test_no_limit_environment_removes_raise_cap_and_preserves_chips():
    agents = [_AlwaysRaiseAgent(f"player_{index}", index) for index in range(3)]
    record = HoldemEnvironment(
        agents,
        seed=5,
        config=EnvironmentConfig(max_raises_per_street=None),
    ).play_hand(0)
    assert sum(event.action is ActionType.RAISE for event in record.actions) > 2
    assert abs(sum(record.rewards.values())) < 1e-9


def test_environment_play_continues_hand_indices():
    agents = [_AlwaysRaiseAgent(f"player_{index}", index) for index in range(3)]
    environment = HoldemEnvironment(
        agents,
        seed=6,
        config=EnvironmentConfig(max_raises_per_street=1),
    )
    environment.play(2)
    records = environment.play(2)
    assert [record.hand_index for record in records] == [0, 1, 2, 3]


def test_six_max_mock_pilot_writes_all_player_results(tmp_path: Path):
    result = run_six_max_experiment(
        SixMaxConfig(
            provider="mock",
            seeds=(41,),
            hands=2,
            equity_samples=1,
            output_dir=tmp_path,
        )
    )
    assert len(result["per_seed"]) == 6
    assert set(result["summary"]["player_type"]) == {
        "llm",
        "tag",
        "lag",
        "rock",
        "calling_station",
        "myopic",
    }
    assert (tmp_path / "llm_decision_traces.jsonl.gz").exists()
    assert (tmp_path / "design.md").exists()


def test_reflexive_off_removes_second_order_decision_fields(tmp_path: Path):
    run_six_max_experiment(
        SixMaxConfig(
            provider="mock",
            seeds=(42,),
            hands=1,
            equity_samples=2,
            condition="reflexive_off",
            reflexive_enabled=False,
            output_dir=tmp_path,
        )
    )
    with gzip.open(tmp_path / "llm_decision_traces.jsonl.gz", "rt", encoding="utf-8") as handle:
        state = json.loads(next(handle))["state"]
    assert state["reasoning_mode"] == "first_order"
    assert "reflexive_tools" not in state
    assert "recent_reflections" not in state
    assert "self_image_estimate" not in state


def test_second_order_mock_experiment_pairs_conditions(tmp_path: Path):
    result = run_second_order_experiment(
        SecondOrderConfig(
            model_specs=(("mock", "mock"),),
            seeds=(43,),
            hands=2,
            equity_samples=2,
            bootstrap_samples=20,
            output_dir=tmp_path,
        )
    )
    assert len(result["paired"]) == 1
    assert set(result["per_seed"]["condition"]) == {"reflexive_off", "reflexive_on"}
    assert result["paired_summary"].loc[0, "provider_failures"] == 0
    assert (tmp_path / "EXPERIMENT.md").exists()
    assert (tmp_path / "paired_summary.csv").exists()


def test_shared_history_experiment_forks_identical_state(tmp_path: Path):
    result = run_shared_history_experiment(
        SharedHistoryConfig(
            provider="mock",
            model="mock",
            seed=44,
            formation_hands=2,
            exploitation_hands=2,
            equity_samples=2,
            memory_hands=2,
            branch_workers=1,
            output_dir=tmp_path,
        )
    )
    assert result["fork"]["identical"] is True
    assert result["fork"]["record_count"] == 2
    assert set(result["summary"]["condition"]) == {
        "reflexive_off",
        "reflexive_on",
    }
    assert set(result["per_hand"].query("phase == 'exploitation'")["hand_index"]) == {2, 3}
    assert len(result["image_trajectory"]) == 6
    assert (tmp_path / "fork_summary.json").exists()
    assert (tmp_path / "decision_traces.jsonl.gz").exists()
