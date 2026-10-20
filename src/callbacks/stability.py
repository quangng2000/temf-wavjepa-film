"""Observe pre-clipping gradients and reject nonfinite optimizer updates."""

import torch
from lightning import Callback


class StabilityGuard(Callback):
    def on_before_optimizer_step(self, trainer, module, optimizer):
        gradients = [parameter.grad.detach() for parameter in module.net.parameters()
                     if parameter.grad is not None]
        if not gradients:
            raise FloatingPointError("Missing gradients; optimizer update rejected")
        try:
            norm = torch.nn.utils.get_total_norm(gradients, error_if_nonfinite=True)
        except RuntimeError as error:
            message = "Nonfinite gradient norm; optimizer update rejected"
            raise FloatingPointError(message) from error
        module.log("train/gradient_norm", norm, on_step=True, on_epoch=False)
        threshold = getattr(module, "microbatch_gradient_clip", trainer.gradient_clip_val)
        threshold = threshold or float("inf")
        module.log("train/clipped", (norm > threshold).float(), on_step=True, on_epoch=False)

    def on_exception(self, trainer, module, exception):
        # No cloud, process, or resource lifecycle actions belong in this callback.
        if trainer.is_global_zero:
            from pathlib import Path

            path = Path(trainer.default_root_dir) / "failure.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            message = f"{type(exception).__name__}: {exception}"
            path.write_text(f"update={trainer.global_step}\n{message}\n")
