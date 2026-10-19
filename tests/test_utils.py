import pytest
import torch
from omegaconf import OmegaConf

from src.utils import extras, get_metric_value, instantiate_loggers


def test_logger_groups_support_disabled_multiple_and_legacy_entries(tmp_path):
    logger = dict(_target_="lightning.pytorch.loggers.CSVLogger", save_dir=str(tmp_path))
    group = OmegaConf.create(dict(first=logger, second=logger, disabled=None))
    assert len(instantiate_loggers(group)) == 2
    assert len(instantiate_loggers(OmegaConf.create(logger))) == 1
    assert instantiate_loggers(None) == []
    with pytest.raises(ValueError, match="needs a _target_"):
        instantiate_loggers(OmegaConf.create(dict(csv="invalid")))


def test_optimized_metric_is_explicit_and_missing_metric_fails():
    metrics = {"val/coefficient_mse": torch.tensor(0.25)}
    assert get_metric_value(metrics, "val/coefficient_mse") == 0.25
    assert get_metric_value(metrics, None) is None
    with pytest.raises(ValueError, match="was not logged"):
        get_metric_value(metrics, "val/missing")


def test_required_tags_fail_without_prompt_and_config_prints(capsys):
    cfg = OmegaConf.create(dict(tags=[], extras=dict(enforce_tags=True, print_config=True)))
    with pytest.raises(ValueError, match="Supply tags"):
        extras(cfg)
    cfg.tags = ["test"]
    extras(cfg)
    assert "test" in capsys.readouterr().out
