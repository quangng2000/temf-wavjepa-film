"""Opt-in Lightning path: one global TEMF batch and one update per four chunks."""
import torch

from src.flow.accumulation import accumulate_gradients
from src.models.temf_module import RestorationModule


class MicrobatchRestorationModule(RestorationModule):
    def __init__(self, microbatch_size=2, effective_batch_size=8,
                 microbatch_gradient_clip=1.0, **kwargs):
        super().__init__(**kwargs)
        if not 0 < microbatch_size <= effective_batch_size or microbatch_gradient_clip <= 0:
            raise ValueError("Invalid microbatch size, effective batch, or clipping threshold")
        self.microbatch_size, self.effective_batch_size = microbatch_size, effective_batch_size
        self.microbatch_gradient_clip = microbatch_gradient_clip
        self.automatic_optimization = False
        self.recipe["batching"] = dict(effective=effective_batch_size, microbatch=microbatch_size,
                                      sampling="global-times-noise-weights-v1",
                                      gradient_clip=microbatch_gradient_clip)

    def setup(self, stage=None):
        super().setup(stage)
        if self.trainer.world_size != 1 or self.trainer.accumulate_grad_batches != 1:
            raise ValueError("Global TEMF microbatching currently requires one device and batch")
        if self.trainer.gradient_clip_val:
            raise ValueError("Set trainer.gradient_clip_val=null; module clips once per update")

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)
        metrics = accumulate_gradients(self, batch, self.global_step, self.manual_backward)
        optimizer.step()
        self.log_dict({f"train/{key}": value for key, value in metrics.items()},
                      on_step=True, on_epoch=False, batch_size=self.effective_batch_size)
        return metrics["loss"]

    def on_before_optimizer_step(self, optimizer):
        # Lightning runs StabilityGuard first, so it observes the accumulated raw norm.
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.microbatch_gradient_clip,
                                      error_if_nonfinite=True)
