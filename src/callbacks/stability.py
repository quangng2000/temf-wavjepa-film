"""Observe pre-clipping gradients and reject nonfinite optimizer updates."""

import torch
from lightning import Callback


class StabilityGuard(Callback):
    def on_before_optimizer_step(self, trainer, module, optimizer):
        gradients = [parameter.grad.detach() for parameter in module.net.parameters()
                     if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("Missing/nonfinite gradients; optimizer update rejected")
        norm = torch.linalg.vector_norm(torch.stack([gradient.norm() for gradient in gradients]))
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm; optimizer update rejected")
        module.log("train/gradient_norm", norm, on_step=True, on_epoch=False)
        threshold = trainer.gradient_clip_val or float("inf")
        module.log("train/clipped", (norm > threshold).float(), on_step=True, on_epoch=False)

    def on_exception(self, trainer, module, exception):
        # No cloud, process, or resource lifecycle actions belong in this callback.
        if trainer.is_global_zero:
            from pathlib import Path

            path = Path(trainer.default_root_dir) / "failure.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            message = f"{type(exception).__name__}: {exception}"
            path.write_text(f"update={trainer.global_step}\n{message}\n")
