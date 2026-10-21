"""Read explicit run identity and pinned SSH configuration without exposing credentials."""
import json
from pathlib import Path, PurePosixPath


def load_config(path):
    settings = json.loads(Path(path).read_text())
    activation = json.loads(Path(settings["activation_file"]).read_text())
    ssh = json.loads(Path(settings["ssh_file"]).read_text())
    if activation["pod_id"] != ssh["pod_id"] or not ssh["strict_host_key_checking"]:
        raise ValueError("Run and pinned SSH identity disagree")
    output = PurePosixPath(activation["output"])
    if not output.is_absolute() or ".." in output.parts:
        raise ValueError("Run path must be absolute")
    command = activation["command"]
    target = int(next(arg.split("=", 1)[1] for arg in command if arg.startswith("total_updates=")))
    return dict(settings, ssh=ssh, run_path=str(output), pod_id=activation["pod_id"],
                run_id=activation["run_id"], target=target)
