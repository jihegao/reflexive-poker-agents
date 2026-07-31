from reflexive_poker.simulation import (
    CONFIRMATORY_CONDITIONS,
    IMAGE_SHAPING_CONDITIONS,
    run_study,
    summarize,
)


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
