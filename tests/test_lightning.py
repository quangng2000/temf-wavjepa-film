from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.callbacks.stability import StabilityGuard
from src.sample import load_model, restore
from src.train import train


def smoke_config(output):
    directory = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=directory, version_base="1.3"):
        return compose(config_name="train", overrides=["experiment=smoke", f"output_dir={output}"])


def test_optimizer_checkpoint_resume_ema_and_sampling(tmp_path):
    cfg = smoke_config(tmp_path)
    trainer, module = train(cfg)
    assert trainer.global_step == 2
    checkpoint = trainer.checkpoint_callback.last_model_path
    weights = torch.load(checkpoint, weights_only=False)
    assert weights["callbacks"]["restoration_ema"]["last_step"] == 2
    cfg.ckpt_path, cfg.trainer.max_steps = checkpoint, 3
    resumed, _ = train(cfg)
    assert resumed.global_step == 3
    model, step = load_model(resumed.checkpoint_callback.last_model_path, "ema", "cpu")
    output = restore(model, torch.zeros(1, 1024), 1, 7)
    assert step == 3 and output.shape == (1, 1024) and torch.isfinite(output).all()
    module.recipe["objective"] = "different"
    with pytest.raises(ValueError, match="recipe mismatch"):
        module.on_load_checkpoint(weights)


def test_nonfinite_gradient_is_rejected_before_optimizer_advances():
    network = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(network.parameters())
    network(torch.ones(1, 1)).sum().backward()
    network.weight.grad.fill_(float("inf"))
    module = SimpleNamespace(net=network)
    with pytest.raises(FloatingPointError, match="rejected"):
        StabilityGuard().on_before_optimizer_step(None, module, optimizer)
    assert not optimizer.state
