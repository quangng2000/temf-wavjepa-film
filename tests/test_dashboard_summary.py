"""Run status distinguishes pauses, failed metrics and process loss."""
from tools.dashboard.pace import Pace
from tools.dashboard.summary import summarize


def payload():
    return dict(rows=[dict(step=19, loss=.05, gradient_norm=.78, clipped=0)],
                workers=dict(trainer=[6153]), remote_time=100, metrics_modified=99)


def config():
    return dict(pod_id="pod", run_path="/opt/run", run_id="new", target=200000, region="US-MD-1")


def test_actual_completed_step_and_gradient_values(tmp_path):
    result = summarize(payload(), config(), Pace(tmp_path / "pace.json"))
    assert result["step"] == 19 and result["target"] == 200000
    assert result["gradient_median"] == .78 and result["clipping_fraction"] == 0
    assert result["status"] == "training" and result["seconds_per_update"] is None


def test_absent_process_and_stale_log_are_not_the_same_failure(tmp_path):
    raw = payload()
    raw["remote_time"] = 250
    assert summarize(raw, config(), Pace(tmp_path / "a"))["status"] == "waiting"
    raw["workers"]["trainer"] = []
    assert summarize(raw, config(), Pace(tmp_path / "b"))["status"] == "inactive"
    raw["failure"] = "Nonfinite gradient rejected"
    assert summarize(raw, config(), Pace(tmp_path / "c"))["status"] == "failed"
