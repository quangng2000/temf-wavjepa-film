"""Direct script invocation must discover the project from another working directory."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry,overrides", [
    ("train", ["experiment=smoke"]),
    ("eval", ["experiment=smoke", "ckpt_path=/tmp/unused.ckpt"]),
    ("sample", ["checkpoint=/tmp/unused.ckpt", "input=/tmp/unused.wav"]),
])
def test_direct_entrypoint_resolves_config_from_other_directory(tmp_path, entry, overrides):
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    env.pop("PROJECT_ROOT", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / f"src/{entry}.py"), "--cfg", "job", "--resolve", *overrides],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60, check=True,
    )
    config = yaml.safe_load(result.stdout)
    assert config["paths"]["root_dir"] == str(ROOT)
    assert config["task_name"] == entry
    assert not list(tmp_path.glob("**/*.ckpt"))
