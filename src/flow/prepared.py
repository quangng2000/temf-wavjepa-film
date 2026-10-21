"""One effective-batch draw, sliced without re-sampling at microbatch boundaries."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreparedBatch:
    trajectory: tuple
    noise: torch.Tensor
    temporal_weight: torch.Tensor
    step: int

    def slice(self, selection):
        current, destination, delta, alpha = self.trajectory
        return PreparedBatch((current[selection], destination[selection], delta[selection], alpha),
                             self.noise[selection], self.temporal_weight[selection], self.step)

    def validate(self, reference, step):
        expected = (len(reference), 1, 1, 1, 1)
        if step != self.step or reference.shape != self.noise.shape:
            raise ValueError("Prepared batch shape or optimizer step mismatch")
        tensors = (*self.trajectory[:3], self.noise, self.temporal_weight)
        if any(value.device != reference.device for value in tensors):
            raise ValueError("Prepared batch device mismatch")
        if any(value.shape != expected for value in self.trajectory[:3]):
            raise ValueError("Prepared time-pair shape mismatch")
        if self.temporal_weight.shape != (len(reference),) or self.noise.dtype != reference.dtype:
            raise ValueError("Prepared weights or noise dtype mismatch")
        if self.trajectory[3] != 0 or torch.count_nonzero(self.trajectory[2]):
            raise ValueError("Prepared batches support the continuous TEMF recipe only")


def prepare_batch(upstream, reference, step):
    if reference.ndim != 5 or len(reference) < 1:
        raise ValueError("Expected a nonempty five-dimensional effective batch")
    if upstream.cfg.model.label_dropout != 0:
        raise ValueError("Prepared audio batches require deterministic conditioning")
    trajectory = upstream.sample_traj_params(len(reference), step, reference.device)
    noise = torch.randn_like(reference)
    weights = upstream.compute_temporal_equilibrium_weight(*trajectory[:2])
    result = PreparedBatch(trajectory, noise, weights, step)
    result.validate(reference, step)
    return result
