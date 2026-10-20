"""Bounded, read-only Lightning CSV parsing; safe to concatenate over SSH."""
import csv
import math
from pathlib import Path

METRIC_BYTES = 1024 * 1024
METRIC_NAMES = ("loss", "raw_loss", "gradient_norm", "clipped", "target_rms",
                "correction_rms", "boundary_actual", "boundary_ratio", "delta")


def read_metric_tail(path):
    with Path(path).open("rb") as source:
        header = source.readline(8192)
        if not header.endswith(b"\n"):
            raise ValueError("CSV header is incomplete or exceeds 8192 bytes")
        end = source.seek(0, 2)
        offset = max(len(header), end - (METRIC_BYTES - len(header)))
        source.seek(offset)
        content = source.read(METRIC_BYTES - len(header))
    if offset > len(header):
        content = content.partition(b"\n")[2]
    lines = content.splitlines(keepends=True)
    complete = [line for line in lines if line.endswith(b"\n")][-1000:]
    return (header + b"".join(complete)).decode("utf-8", errors="replace")


def parse_metric_values(record, step, nonfinite):
    result = {}
    for name in METRIC_NAMES:
        value = record.get("train/" + name)
        if not value:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[name] = number
        else:
            result[name] = None
            nonfinite.append({"step": step + 1, "metric": name, "value": value})
    return result


def parse_metric_csv(content):
    merged, nonfinite = {}, []
    for record in csv.DictReader(content.splitlines()):
        try:
            step = int(record.get("step", ""))
        except (TypeError, ValueError):
            continue
        if step < 0 or None in record:
            continue
        values = parse_metric_values(record, step, nonfinite)
        if values:
            merged.setdefault(step, {"step": step + 1, "csv_step": step}).update(values)
    rows = [row for _, row in sorted(merged.items()) if "loss" in row]
    return rows[-1000:], nonfinite[-30:]


def snapshot_metrics(run_path):
    path = Path(run_path) / "metrics/version_0/metrics.csv"
    try:
        content = read_metric_tail(path)
        rows, nonfinite = parse_metric_csv(content)
        return {"rows": rows, "nonfinite_metrics": nonfinite,
                "metrics_modified": path.stat().st_mtime, "metrics_error": None}
    except FileNotFoundError:
        return {"rows": [], "nonfinite_metrics": [], "metrics_modified": None,
                "metrics_error": None}
    except (OSError, ValueError, csv.Error) as error:
        return {"rows": [], "nonfinite_metrics": [], "metrics_modified": None,
                "metrics_error": str(error)}
