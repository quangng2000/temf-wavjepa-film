"""Lightning integration for optimizer, schedule, clipping, EMA, and resumption."""
import torch
from omegaconf import open_dict

import src.flow.accumulation as accumulation
from src.callbacks.stability import StabilityGuard
from src.models.temf_module import RestorationModule
from src.train import train
from tests.test_lightning import smoke_config


def microbatch_config(path):
    cfg = smoke_config(path)
    with open_dict(cfg.model):
        cfg.model._target_ = "src.models.microbatch.MicrobatchRestorationModule"
        cfg.model.microbatch_size = 2
        cfg.model.effective_batch_size = 8
        cfg.model.microbatch_gradient_clip = .001
    cfg.data.batch_size = 8
    cfg.trainer.gradient_clip_val = None
    return cfg


def observe_sampling(monkeypatch, draws, encodes):
    original_prepare = accumulation.prepare_batch
    original_encode = RestorationModule.encode_batch

    def prepare(objective, reference, step):
        draws.append((len(reference), step))
        return original_prepare(objective, reference, step)

    def encode(module, batch):
        if module.training:
            encodes.append(len(batch["clean"]))
        return original_encode(module, batch)

    monkeypatch.setattr(accumulation, "prepare_batch", prepare)
    monkeypatch.setattr(RestorationModule, "encode_batch", encode)


def observe_raw_gradients(monkeypatch, values):
    original_guard = StabilityGuard.on_before_optimizer_step

    def guard(callback, trainer, module, optimizer):
        gradients = [value.grad for value in module.net.parameters() if value.grad is not None]
        values.append(torch.nn.utils.get_total_norm(gradients).detach())
        return original_guard(callback, trainer, module, optimizer)

    monkeypatch.setattr(StabilityGuard, "on_before_optimizer_step", guard)


def assert_checkpoint(path, step):
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["global_step"] == step
    ema = checkpoint["callbacks"]["restoration_ema"]
    assert ema["last_step"] == step
    assert ema["averaged"]["n_averaged"].item() == step + 1
    states = checkpoint["optimizer_states"][0]["state"].values()
    assert states and all(value["step"].item() == step for value in states)


def assert_accumulated_gradients_were_clipped(module):
    gradients = [value.grad for value in module.net.parameters() if value.grad is not None]
    assert torch.nn.utils.get_total_norm(gradients) <= .001001


def test_effective_batches_advance_optimizer_and_ema_once_then_resume(tmp_path, monkeypatch):
    draws, encodes, gradients = [], [], []
    observe_sampling(monkeypatch, draws, encodes)
    observe_raw_gradients(monkeypatch, gradients)
    cfg = microbatch_config(tmp_path)
    metrics, objects = train(cfg)
    checkpoint = objects["trainer"].checkpoint_callback.last_model_path
    assert_checkpoint(checkpoint, 2)
    assert draws == [(8, 0), (8, 1)] and encodes == [2] * 8
    assert len(gradients) == 2 and gradients[-1] > .001
    torch.testing.assert_close(metrics["train/gradient_norm"], gradients[-1])
    assert metrics["train/clipped"].item() == 1
    assert_accumulated_gradients_were_clipped(objects["model"])
    cfg.ckpt_path, cfg.trainer.max_steps = checkpoint, 3
    _, objects = train(cfg)
    assert_checkpoint(objects["trainer"].checkpoint_callback.last_model_path, 3)
    assert draws == [(8, 0), (8, 1), (8, 2)] and encodes == [2] * 12
    assert len(gradients) == 3
    assert_accumulated_gradients_were_clipped(objects["model"])
