"""The exact concatenated remote program must serialize the API contract."""
import json
import subprocess
import sys

from tools.dashboard.transport import build_script


def test_snapshot_program_runs_as_standalone_python_and_returns_log_lines(tmp_path):
    (tmp_path / "console.log").write_text("training log\nsecond line\n")
    code = build_script(dict(run_path=str(tmp_path)))
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, check=True)
    snapshot = json.loads(result.stdout)
    assert snapshot["log"] == ["training log", "second line"]
    assert snapshot["rows"] == [] and snapshot["metrics_modified"] is None
    assert snapshot["metrics_path"] is None
    assert snapshot["failure"] is None and snapshot["failure_modified"] is None
    assert snapshot["checkpoint"] is None


def test_shell_metacharacters_in_run_path_are_python_data(tmp_path):
    path = tmp_path / "$(touch should-not-exist);quoted'run"
    path.mkdir()
    code = build_script(dict(run_path=str(path)))
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["run_path"] == str(path)
    assert not (tmp_path / "should-not-exist").exists()
