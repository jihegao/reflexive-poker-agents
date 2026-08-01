from __future__ import annotations

from reflexive_poker.demo_engine import DemoConfig, DemoTable


def _play_human_hand(table: DemoTable) -> None:
    for _ in range(40):
        if table.hand and table.hand.complete:
            return
        state = table.snapshot()
        assert state["canAct"]
        table.apply_action(table.hero_seat, "check_call")
    raise AssertionError("hand did not complete")


def test_demo_table_plays_and_round_trips() -> None:
    table = DemoTable(DemoConfig(seed=9200, equity_samples=4))
    assert table.snapshot()["phase"] == "waiting_human"
    assert table.snapshot()["hand"]["seats"][0]["cards"]

    _play_human_hand(table)
    snapshot = table.snapshot()
    assert snapshot["phase"] == "hand_complete"
    assert snapshot["completedHandCount"] == 1
    assert sum(snapshot["hand"]["rewards"].values()) == 0.0

    restored = DemoTable.from_dict(table.to_dict())
    assert restored.snapshot() == snapshot
    restored.start_hand()
    assert restored.snapshot()["hand"]["handIndex"] == 1
    assert all(seat["stackBb"] <= 100.0 for seat in restored.snapshot()["hand"]["seats"])


def test_showdown_reveals_only_live_players() -> None:
    showdown = None
    for seed in range(9200, 9240):
        table = DemoTable(DemoConfig(seed=seed, equity_samples=2))
        _play_human_hand(table)
        if table.hand and table.hand.showdown:
            showdown = table.snapshot()["hand"]
            break
    assert showdown is not None
    for seat in showdown["seats"]:
        if seat["seat"] == 0 or seat["active"]:
            assert len(seat["cards"]) == 2
        else:
            assert seat["cards"] == []


def test_only_llm_authored_bounded_strategy_patch_is_accepted() -> None:
    table = DemoTable(DemoConfig(equity_samples=2))
    _play_human_hand(table)
    updated = table.apply_strategy_patch(
        {
            "patchId": "patch_test",
            "baseStrategyVersion": 1,
            "author": "llm_closed_loop",
            "reason": "test",
            "changes": {"aggressionBias": 0.05, "notes": ["bounded"]},
        }
    )
    assert updated["version"] == 2
    assert updated["aggressionBias"] == 0.05

    try:
        table.apply_strategy_patch(
            {
                "baseStrategyVersion": 2,
                "author": "player",
                "changes": {"aggressionBias": 0.0},
            }
        )
    except ValueError as exc:
        assert str(exc) == "invalid_strategy_patch_author"
    else:
        raise AssertionError("player-authored patch should be rejected")


def test_opponent_controller_can_switch_to_llm_and_blocks_on_its_turn() -> None:
    table = DemoTable(DemoConfig(seed=9200, equity_samples=2))
    table.set_seat_controller(1, "llm_closed_loop")

    table.apply_action(table.hero_seat, "check_call")

    assert table.hand is not None
    assert table.hand.actor == 1
    assert table.phase == "waiting_llm"
    assert table.snapshot()["hand"]["seats"][1]["controller"] == "llm_closed_loop"


def test_provider_failure_keeps_selected_llm_controller() -> None:
    table = DemoTable(DemoConfig(equity_samples=2))
    table.set_controller("llm_closed_loop")

    table.pause_for_provider_failure("action_timeout_mock")

    assert table.controller == "llm_closed_loop"
    assert table.phase == "waiting_llm"
    assert table.paused_reason == "action_timeout_mock"


def test_switching_current_opponent_back_to_rule_ai_resumes_table() -> None:
    table = DemoTable(DemoConfig(seed=9200, equity_samples=2))
    table.set_seat_controller(1, "llm_closed_loop")
    table.apply_action(table.hero_seat, "check_call")
    assert table.hand is not None and table.hand.actor == 1

    table.set_seat_controller(1, "rule_ai")

    assert table.controller_for(1) == "rule_ai"
    assert not (table.hand.actor == 1 and table.phase == "waiting_llm")


def test_each_seat_model_is_persisted_and_exposed_in_snapshots() -> None:
    table = DemoTable(DemoConfig(equity_samples=2))
    table.set_seat_model(0, "qwen3.7-plus")
    table.set_seat_model(3, "opencode-go/kimi-k2.7-code")

    restored = DemoTable.from_dict(table.to_dict())
    seats = restored.snapshot()["hand"]["seats"]

    assert restored.model_for(0) == "qwen3.7-plus"
    assert restored.model_for(3) == "kimi-k2.7-code"
    assert seats[0]["model"] == "qwen3.7-plus"
    assert seats[3]["model"] == "kimi-k2.7-code"


def test_each_seat_has_an_independent_persona_strategy_and_memory() -> None:
    table = DemoTable(DemoConfig(equity_samples=2))

    assert [table.strategy_for(seat)["basePersona"] for seat in range(6)] == [
        "closed_loop_shaper",
        "tag",
        "lag",
        "rock",
        "calling_station",
        "myopic",
    ]
    hero_before = dict(table.strategy_for(0))
    seat_one = table.apply_strategy_patch(
        {
            "patchId": "patch_seat_one",
            "baseStrategyVersion": 1,
            "author": "llm_closed_loop",
            "reason": "tighten the TAG response",
            "changes": {"aggressionBias": 0.08, "notes": ["seat one only"]},
        },
        actor=1,
    )
    table.record_reflection(
        {"handIndex": 0, "outcomeSummary": "seat one review"}, actor=1
    )

    assert seat_one["version"] == 2
    assert seat_one["basePersona"] == "tag"
    assert table.strategy_for(0) == hero_before
    assert table.reflection_memory_for(0) == []
    assert table.reflection_memory_for(1)[0]["seat"] == 1

    restored = DemoTable.from_dict(table.to_dict())
    assert restored.strategy_for(1) == seat_one
    assert restored.reflection_memory_for(1) == table.reflection_memory_for(1)
    assert restored.snapshot()["hand"]["seats"][1]["strategyProfile"] == seat_one


def test_old_table_payload_gets_default_opponent_strategy_profiles() -> None:
    table = DemoTable(DemoConfig(equity_samples=2))
    payload = table.to_dict()
    payload.pop("opponent_strategy_versions")
    payload.pop("opponent_reflection_memories")
    for strategy in payload["strategy_versions"]:
        strategy.pop("strategyId", None)
        strategy.pop("basePersona", None)

    restored = DemoTable.from_dict(payload)

    assert restored.strategy_for(0)["basePersona"] == "closed_loop_shaper"
    assert restored.strategy_for(1)["basePersona"] == "tag"
    assert restored.strategy_for(5)["basePersona"] == "myopic"
    assert restored.reflection_memory_for(1) == []


def test_completed_hand_queues_reflection_for_each_llm_seat_that_acted() -> None:
    table = DemoTable(DemoConfig(seed=9200, equity_samples=2))
    table.set_seat_controller(0, "llm_closed_loop")
    table.set_seat_controller(1, "llm_closed_loop")
    assert table.hand is not None
    table.hand.actions.extend(
        [
            {"seat": 1, "controller": "llm_closed_loop"},
            {"seat": 0, "controller": "llm_closed_loop"},
            {"seat": 2, "controller": "rule_ai"},
        ]
    )
    table.hand.complete = True

    assert table.pending_reflection_seats() == [0, 1]
    table.record_reflection({"handIndex": 0, "outcomeSummary": "hero"}, actor=0)
    assert table.pending_reflection_seats() == [1]
