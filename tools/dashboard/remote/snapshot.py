"""Read-only snapshot entry point, after concatenating this directory's modules."""
# ruff: noqa: F821 -- helpers are concatenated before this module for SSH stdin.
import time
from pathlib import Path


def snapshot(run_path):
    path = Path(run_path)
    if not path.is_absolute():
        raise ValueError("Run path must be absolute")
    telemetry = snapshot_gpu()
    failure_path = path / "failure.txt"
    failure = read_text_tail(failure_path, limit=8192)
    try:
        failure_modified = failure_path.stat().st_mtime if failure else None
    except OSError:
        failure_modified = None
    result = {"remote_time": time.time(), "run_path": str(path),
              "workers": snapshot_workers(str(path)),
              "gpu": next(iter(telemetry["devices"]), None), "gpu_error": telemetry["error"],
              "checkpoint": snapshot_checkpoint(path),
              "log": (read_text_tail(path / "console.log") or "").splitlines()[-25:],
              "failure": failure, "failure_modified": failure_modified}
    result.update(snapshot_metrics(path))
    return result
