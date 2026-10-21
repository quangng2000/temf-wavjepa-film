"""Rates describe observed progress and never mix runs or invent durations."""
from tools.dashboard.pace import Pace


def test_waits_for_one_minute_and_includes_pauses(tmp_path):
    pace = Pace(tmp_path / "pace.json")
    assert pace.observe("run-a", 100, 10)["seconds_per_update"] is None
    assert pace.observe("run-a", 130, 20)["seconds_per_update"] is None
    result = pace.observe("run-a", 160, 20)
    assert result["seconds_per_update"] == 6 and result["rate_sample_count"] == 10
    result = pace.observe("run-a", 190, 30)
    assert result["rate_history"][-1]["seconds"] == 6


def test_run_change_regression_and_long_gaps_reset_history(tmp_path):
    pace = Pace(tmp_path / "pace.json")
    for identity, now, step in [("a", 100, 100), ("a", 160, 120), ("b", 170, 2),
                                ("b", 180, 1), ("b", 300, 20)]:
        result = pace.observe(identity, now, step)
    assert result["seconds_per_update"] is None
    assert result["rate_history"] == [] and result["observation_seconds"] == 0


def test_same_run_reopens_local_observations_without_reset(tmp_path):
    path = tmp_path / "pace.json"
    pace = Pace(path)
    pace.observe("run", 100, 10)
    result = Pace(path).observe("run", 160, 30)
    assert result["seconds_per_update"] == 3
