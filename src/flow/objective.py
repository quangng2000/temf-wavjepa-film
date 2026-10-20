"""Small audio adapter around the pinned authors' TEMFLoss.forward."""
import torch

from src.flow.interface import AudioCondition, AudioVelocity
from src.flow.telemetry import TargetObserver
from src.upstream.prepared import load_prepared_forward
from src.upstream.source import COMMIT
from src.upstream.temf import create_temf


class TEMFObjective:
    recipe = f"authors-temf-continuous-{COMMIT}"

    def __init__(self, total_updates=300000, jvp_backend="reverse"):
        if total_updates < 1:
            raise ValueError("total_updates must be positive")
        self.total_updates = total_updates
        self.upstream = create_temf(total_updates, jvp_backend=jvp_backend)
        if jvp_backend != "reverse":
            self.recipe = f"{self.recipe}:jvp-{jvp_backend}-v1"
        self.observer = TargetObserver(self.upstream)

    def __call__(self, net, batch, step, prepared=None):
        clean, observed, context = batch
        arguments = (AudioVelocity(net), clean.unsqueeze(1), AudioCondition(observed, context))
        if prepared is None:
            result = self.upstream(*arguments, cur_step=step)
        else:
            prepared.validate(clean.unsqueeze(1), step)
            result = load_prepared_forward()(self.upstream, *arguments,
                                             cur_step=step, prepared=prepared)
        return result["total"].mean(), self.summarize(result)

    def summarize(self, result):
        metrics = {key: value.detach().mean() for key, value in result.items()}
        metrics.update(self.observer.metrics)
        metrics["raw_loss"] = (result["total"] / result["temporal_weight"]).detach().mean()
        metrics["boundary_actual"] = torch.isclose(result["delta"],
                                                  torch.zeros_like(result["delta"])).float().mean()
        return metrics
