"""Inspect only exact production trainer arguments and NVIDIA telemetry."""
import csv
import math
import subprocess
from pathlib import Path


def is_run_trainer(arguments, run_path):
    if not arguments or not Path(arguments[0]).name.startswith("python"):
        return False
    module = any(arguments[index:index + 2] == ["-m", "src.train"]
                 for index in range(max(0, len(arguments) - 1)))
    return module and f"output_dir={run_path}" in arguments


def snapshot_workers(run_path, proc_path="/proc"):
    trainers = []
    for entry in Path(proc_path).glob("[0-9]*"):
        try:
            with (entry / "cmdline").open("rb") as source:
                arguments = source.read(65536).decode("utf-8", "replace").split("\0")
            if is_run_trainer(arguments, run_path):
                trainers.append(int(entry.name))
        except (OSError, ValueError):
            continue  # A process may exit while /proc is inspected.
    return {"trainer": sorted(trainers)}


def parse_gpu_csv(content):
    devices = []
    for fields in csv.reader(content.splitlines()):
        if len(fields) != 5:
            continue
        index, name, utilization, used, total = [item.strip() for item in fields]
        try:
            numbers = [float(value) for value in (utilization, used, total)]
            if not all(math.isfinite(value) for value in numbers):
                continue
            devices.append({"index": int(index), "name": name,
                            "utilization": numbers[0],
                            "memory_used_mib": numbers[1], "memory_total_mib": numbers[2]})
        except ValueError:
            continue
    return devices


def snapshot_gpu():
    command = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        return {"devices": parse_gpu_csv(result.stdout), "error": None}
    except (OSError, subprocess.SubprocessError) as error:
        return {"devices": [], "error": str(error)}
