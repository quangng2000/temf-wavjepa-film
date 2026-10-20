"""Read-only snapshot entry point, after concatenating this directory's modules."""
# ruff: noqa: F821 -- helpers are concatenated before this module for SSH stdin.
import time
from pathlib import Path


def snapshot(run_path):
    path = Path(run_path)
    if not path.is_absolute():
        raise ValueError("Run path must be absolute")
    telemetry = snapshot_gpu()
    result = {"remote_time": time.time(), "run_path": str(path),
              "workers": snapshot_workers(str(path)),
              "gpu": next(iter(telemetry["devices"]), None), "gpu_error": telemetry["error"],
              "checkpoint": snapshot_checkpoint(path),
              "log": (read_text_tail(path / "console.log") or "").splitlines()[-25:],
              "failure": read_text_tail(path / "failure.txt", limit=8192)}
    result.update(snapshot_metrics(path))
    return result
