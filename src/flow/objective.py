"""Small audio adapter around the pinned authors' TEMFLoss.forward."""
import torch

from src.flow.interface import AudioCondition, AudioVelocity
from src.flow.telemetry import TargetObserver
from src.upstream.source import COMMIT
from src.upstream.temf import create_temf


class TEMFObjective:
    recipe = f"authors-temf-continuous-{COMMIT}"

    def __init__(self, total_updates=300000):
        if total_updates < 1:
            raise ValueError("total_updates must be positive")
        self.total_updates = total_updates
        self.upstream = create_temf(total_updates)
        self.observer = TargetObserver(self.upstream)

    def __call__(self, net, batch, step):
        clean, observed, context = batch
        result = self.upstream(AudioVelocity(net), clean.unsqueeze(1),
                               AudioCondition(observed, context), cur_step=step)
        metrics = {key: value.detach().mean() for key, value in result.items()}
        metrics.update(self.observer.metrics)
        metrics["raw_loss"] = (result["total"] / result["temporal_weight"]).detach().mean()
        metrics["boundary_actual"] = torch.isclose(result["delta"],
                                                  torch.zeros_like(result["delta"])).float().mean()
        return result["total"].mean(), metrics
