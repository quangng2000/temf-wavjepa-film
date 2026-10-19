"""Checkpointed restoration EMA; frozen Nat weights are not averaged."""

import torch
from lightning import Callback


class EMA(Callback):
    def __init__(self, decay=0.999):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay, self.shadow, self.saved = decay, {}, None
        self.last_step = 0

    @property
    def state_key(self):
        return "restoration_ema"

    def on_fit_start(self, trainer, module):
        if not self.shadow:
            self.shadow = {key: value.detach().clone()
                           for key, value in module.net.state_dict().items()}
        self.shadow = {key: value.to(module.device) for key, value in self.shadow.items()}
        self.last_step = trainer.global_step

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step == self.last_step:
            return
        for key, value in module.net.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value, 1 - self.decay)
            else:
                self.shadow[key].copy_(value)
        self.last_step = trainer.global_step

    def state_dict(self):
        return dict(decay=self.decay, shadow=self.shadow, last_step=self.last_step)

    def load_state_dict(self, state_dict):
        if self.decay != state_dict["decay"]:
            raise ValueError("EMA decay changed on resume")
        self.shadow, self.last_step = state_dict["shadow"], state_dict["last_step"]
