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
