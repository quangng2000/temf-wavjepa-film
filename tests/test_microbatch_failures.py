"""An unsafe accumulated update must not advance parameters or optimizer state."""
import pytest
import torch

from src.utils.setup import build_training_objects
from tests.test_microbatch import microbatch_config


def test_nonfinite_accumulated_gradient_rejects_entire_effective_update(tmp_path):
    objects = build_training_objects(microbatch_config(tmp_path))
    trainer, module = objects["trainer"], objects["model"]
    parameter = next(module.net.parameters())
    original = parameter.detach().clone()
    with parameter.register_hook(lambda gradient: gradient * float("nan")):
        with pytest.raises(FloatingPointError, match="rejected"):
            trainer.fit(module, datamodule=objects["datamodule"])
    torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    assert trainer.global_step == 0
    assert not trainer.optimizers[0].state
    ema = next(callback for callback in trainer.callbacks if hasattr(callback, "averaged"))
    assert ema.averaged.n_averaged.item() == 1
    assert "Nonfinite gradient" in (tmp_path / "failure.txt").read_text()
