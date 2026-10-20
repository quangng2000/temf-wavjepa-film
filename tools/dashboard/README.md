# Live training dashboard

Run from the repository root:

```sh
.venv/bin/python -m tools.dashboard.server \
  --config runs/live-dashboard-200k/config.json --port 8765
```

Open <http://127.0.0.1:8765/>. The local configuration references the run's
activation record and existing strict SSH configuration; credentials stay out
of the browser. The server binds only to localhost and exposes no training or
pod controls.

The viewer reads bounded metric and log tails, process identity, GPU telemetry,
and checkpoint file metadata. It refreshes every ten seconds while visible.
It never loads checkpoints, runs inference, or writes to the pod.

Charts show weighted/raw TEMF loss and gradient norms before clipping. CSV
step indices are converted to completed optimizer updates; validation-only
rows do not advance progress. Recent stability summaries use the last 100
logged training updates.

Update time is observed wall-clock time divided by completed updates, because
this trainer does not log individual update durations. The headline needs at
least one minute of observations; chart windows are at least thirty seconds.
Timing includes data waits, validation, and checkpoint writes. Remaining time
is extrapolated from recent pace and can change with the TEMF schedule.

Observation history is stored locally in the configured observations file.
Changing the run or resetting its step counter resets the pace window.
Connection failures preserve the last successful display and mark it stale.

Focused checks:

```sh
.venv/bin/python -m pytest tests/test_dashboard_*.py
.venv/bin/ruff check tools/dashboard tests/test_dashboard_*.py
```
