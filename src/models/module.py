"""Lightning training hooks for direct-velocity TEMF restoration."""

import torch
from lightning import LightningModule

from src.flow.solver import solve


class RestorationModule(LightningModule):
    def __init__(self, net, spectrum, conditioner, objective, optimizer):
        super().__init__()
        self.net, self.spectrum, self.conditioner = net, spectrum, conditioner
        self.objective, self.optimizer_factory = objective, optimizer
        self.run_config = {}
        self.recipe = dict(network=net.recipe, spectrum=spectrum.recipe,
                           conditioner=conditioner.recipe, total_updates=objective.total_updates,
                           objective="temf-continuous-pinned-v1")

    def setup(self, stage=None):
        if self.spectrum.recipe["sample_rate"] != 16000:
            raise ValueError("This Nat/data contract requires 16 kHz")
        self.conditioner.load(self.device)
        if self.trainer.precision != "32-true":
            raise ValueError("Only FP32 has been validated for the full JVP scaffold")

    def configure_optimizers(self):
        return self.optimizer_factory(params=self.net.parameters())

    def encode_batch(self, batch):
        clean = self.spectrum.encode(batch["clean"])
        observed = self.spectrum.encode(batch["observed"])
        positions = self.spectrum.frame_times(observed.shape[-1], self.device)
        context = self.conditioner(batch["observed"], positions)
        return clean, observed, context

    def training_step(self, batch, batch_idx):
        encoded = self.encode_batch(batch)
        loss, metrics = self.objective(self.net, encoded, self.global_step)
        if not all(torch.isfinite(value).all() for value in (loss, *metrics.values())):
            raise FloatingPointError("Nonfinite TEMF loss/target/JVP; optimizer update rejected")
        metrics["loss"] = loss.detach()
        self.log_dict({f"train/{key}": value for key, value in metrics.items()},
                      on_step=True, on_epoch=False, batch_size=len(batch["clean"]))
        return loss

    def validation_step(self, batch, batch_idx):
        clean, observed, context = self.encode_batch(batch)
        restored = solve(self.net, (observed, context), nfe=1, seed=1000 + batch_idx)
        self.log("val/coefficient_mse", (restored - clean).square().mean(),
                 batch_size=len(clean), sync_dist=True)
        self.log("val/input_coefficient_mse", (observed - clean).square().mean(),
                 batch_size=len(clean), sync_dist=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["lab_recipe"] = self.recipe
        checkpoint["run_config"] = self.run_config

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("lab_recipe") != self.recipe:
            raise ValueError("Checkpoint recipe mismatch; explicit migration is required")
