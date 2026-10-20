"""Composition checks for the Lightning-Hydra entry points and optional groups."""

from pathlib import Path
from shutil import copytree

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, read_write
from omegaconf.errors import MissingMandatoryValue

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def project_root(monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(ROOT))


def config(name="train", overrides=(), directory=None):
    with initialize_config_dir(config_dir=str(directory or ROOT / "configs"), version_base="1.3"):
        return compose(config_name=name, overrides=list(overrides))


def test_defaults_preserve_model_recipe_and_training_limits():
    cfg = config()
    assert (cfg.model.net.nf, cfg.total_updates, cfg.seed) == (256, 300000, 42)
    assert cfg.model.net.normalize_time and not cfg.fixture
    assert cfg.trainer.precision == "32-true" and cfg.trainer.devices == 1
    assert cfg.callbacks.checkpoint.every_n_train_steps == 1000
    assert cfg.trainer.val_check_interval == 5000 and cfg.optimized_metric is None


def test_fixture_and_legacy_output_override_resolve_without_hydra_runtime(tmp_path, monkeypatch):
    monkeypatch.delenv("PWD", raising=False)
    cfg = config(overrides=["experiment=smoke", f"output_dir={tmp_path}"])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["fixture"] and resolved["model"]["net"]["nf"] == 8
    assert resolved["paths"]["output_dir"] == str(tmp_path)
    assert resolved["paths"]["work_dir"] == str(ROOT)
    assert cfg.logger.csv.save_dir == str(tmp_path)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_trainer_presets_keep_fp32_single_device(device):
    cfg = config(overrides=[f"trainer={device}"])
    assert cfg.trainer.accelerator == device
    assert (cfg.trainer.devices, cfg.trainer.precision) == (1, "32-true")


@pytest.mark.parametrize("preset", ["default", "fast_dev_run", "overfit"])
def test_debug_presets_keep_validation_reachable(preset):
    cfg = config(overrides=["experiment=smoke", f"debug={preset}", "logger=[]"])
    assert cfg.logger is None and cfg.fixture
    assert cfg.trainer.val_check_interval <= cfg.trainer.max_steps <= cfg.total_updates


def test_eval_requires_checkpoint_and_preserves_fixture_overrides():
    cfg = config("eval", ["experiment=smoke", "logger=[]"])
    assert cfg.task_name == "eval" and not cfg.train and cfg.weights == "raw"
    assert cfg.fixture and cfg.model.net.nf == 8 and cfg.logger is None
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.ckpt_path
    normal = config("eval", ["ckpt_path=/tmp/model.ckpt", "data.val_manifest=/tmp/val.jsonl"])
    assert normal.ckpt_path == "/tmp/model.ckpt" and "train_manifest" not in normal.data


def test_optional_local_overrides_are_composed_when_present(tmp_path):
    directory = copytree(ROOT / "configs", tmp_path / "configs")
    (directory / "local").mkdir(exist_ok=True)
    (directory / "local/default.yaml").write_text("# @package _global_\nseed: 79\n")
    assert config(directory=directory).seed == 79


def test_hydra_multirun_has_distinct_job_directories(tmp_path):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["hparams_search=lr"],
                      return_hydra_config=True)
    previous = HydraConfig.instance().cfg
    try:
        for index in (0, 1):
            with read_write(cfg.hydra):
                cfg.hydra.job.num = index
                cfg.hydra.runtime.output_dir = str(tmp_path / str(index))
            HydraConfig.instance().set_config(cfg)
            assert cfg.hydra.sweep.subdir == str(index)
            assert cfg.paths.output_dir == str(tmp_path / str(index))
        assert "/train/multiruns/" in cfg.hydra.sweep.dir
        assert cfg.optimized_metric == "val/coefficient_mse"
    finally:
        HydraConfig.instance().cfg = previous


def test_sampling_preserves_ema_and_single_step_defaults():
    cfg = config("sample", ["checkpoint=/tmp/model.ckpt", "input=/tmp/input.wav"])
    assert cfg.weights == "ema" and cfg.nfe == 1 and cfg.device == "cpu"
    assert cfg.task_name == "sample" and cfg.paths.root_dir == str(ROOT)


def test_trigflow_experiment_is_explicit_and_keeps_temf_schedule():
    baseline = config(overrides=["experiment=temf_325m_accum8"])
    trigflow = config(overrides=["experiment=temf_325m_trigflow_accum8"])
    assert baseline.model.objective.flow_path == "linear"
    assert trigflow.model.objective.flow_path == "trigflow"
    assert trigflow.total_updates == baseline.total_updates == 200000
    assert trigflow.model.objective.jvp_backend == baseline.model.objective.jvp_backend
    assert trigflow.data.batch_size == baseline.data.batch_size == 8
