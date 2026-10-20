"""Read-only dashboard parser behavior without GPU or production access."""
from pathlib import Path

REMOTE = Path(__file__).parents[1] / "tools/dashboard/remote"
NAMESPACE = {}
for module in sorted(REMOTE.glob("*.py")):
    exec(compile(module.read_text(), str(module), "exec"), NAMESPACE)


def test_csv_joins_train_fields_and_excludes_validation():
    content = ("step,train/loss,train/gradient_norm,val/loss\n"
               "0,0.2,,\n0,,0.4,\n1,,,0.3\n2,0.1,0.5,\n")
    rows, nonfinite = NAMESPACE["parse_metric_csv"](content)
    assert rows == [{"step": 1, "csv_step": 0, "loss": .2, "gradient_norm": .4},
                    {"step": 3, "csv_step": 2, "loss": .1, "gradient_norm": .5}]
    assert nonfinite == []


def test_tail_skips_partial_writes_and_bounds_history(tmp_path):
    path = tmp_path / "metrics.csv"
    header = "step,train/loss\n"
    path.write_text(header + "".join(f"{step},0.2\n" for step in range(150000)) + "150000,9")
    content = NAMESPACE["read_metric_tail"](path)
    rows, _ = NAMESPACE["parse_metric_csv"](content)
    assert len(content.encode()) <= 1024 * 1024
    assert len(rows) == 1000 and rows[-1]["step"] == 150000


def test_nonfinite_values_are_json_safe_and_visible():
    content = "step,train/loss,train/gradient_norm\n0,nan,inf\n1,0.1,0.4\n"
    rows, nonfinite = NAMESPACE["parse_metric_csv"](content)
    assert rows[0]["loss"] is None and rows[0]["gradient_norm"] is None
    assert [item["metric"] for item in nonfinite] == ["loss", "gradient_norm"]


def test_process_match_requires_exact_module_and_run():
    match = NAMESPACE["is_run_trainer"]
    arguments = ["python", "-m", "src.train", "output_dir=/run/current"]
    assert match(arguments, "/run/current")
    assert not match(arguments, "/run")
    assert not match(["python", "dashboard.py", " ".join(arguments)], "/run/current")
    assert not match(["python", "-m", "src.training", arguments[-1]], "/run/current")


def test_checkpoint_inspection_never_loads_contents(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "update-001000.ckpt").write_bytes(b"invalid-checkpoint-by-design")
    result = NAMESPACE["snapshot_checkpoint"](tmp_path)
    assert result["step"] == 1000
    assert result["verified"] is False and result["bytes"] == 28


def test_absent_metrics_are_normal_during_startup(tmp_path):
    result = NAMESPACE["snapshot_metrics"](tmp_path)
    assert result["rows"] == [] and result["metrics_error"] is None


def test_gpu_parser_never_emits_nonfinite_json():
    parse = NAMESPACE["parse_gpu_csv"]
    assert parse("0, NVIDIA A100, nan, 100, 80000") == []
    assert parse("0, NVIDIA A100, 99, 100, 80000")[0]["utilization"] == 99
