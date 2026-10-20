"""Describe only the current run and distinguish staleness from failure."""
import statistics
import time


def run_status(raw, step, target, age):
    if raw.get("failure_active") or raw.get("nonfinite_metrics"):
        return "failed"
    if step >= target:
        return "complete"
    if not raw["workers"]["trainer"]:
        return "inactive"
    if age is None:
        return "starting"
    return "waiting" if age > 120 else "training"


def summarize(raw, config, pace):
    rows = raw["rows"]
    step = rows[-1]["step"] if rows else 0
    modified = raw.get("metrics_modified")
    age = max(0, raw["remote_time"] - modified) if modified else None
    workers = raw["workers"]["trainer"]
    failure = raw.get("failure")
    failure_modified = raw.get("failure_modified")
    recovered = bool(failure and workers and modified and failure_modified
                     and modified > failure_modified)
    failure_active = bool(failure and not recovered)
    identity = "|".join([config["pod_id"], config["run_path"],
                         raw.get("metrics_path") or "no-metrics"])
    timing = pace.observe(identity, raw["remote_time"], step)
    recent = rows[-100:]
    gradients = [r["gradient_norm"] for r in recent if r.get("gradient_norm") is not None]
    clipped = [r["clipped"] for r in recent if r.get("clipped") is not None]
    normalized = dict(raw, failure_active=failure_active,
                      previous_failure=failure if recovered else None)
    if recovered:
        normalized["failure"] = None
    return dict(normalized, **timing, step=step, target=config["target"],
                run_id=config["run_id"],
                status=run_status(normalized, step, config["target"], age), update_age=age,
                gradient_median=statistics.median(gradients) if gradients else None,
                clipping_fraction=statistics.mean(clipped) if clipped else None,
                region=config["region"], pod_id=config["pod_id"], fetched_at=time.time(),
                read_only=True, connection_error=None)
