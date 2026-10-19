"""Velocity-space TEMF loss with unit branch weights, no CFG or auxiliary losses."""

import torch

from tests.reference.legacy_schedule import expand_time, sample_times, schedule, temporal_weight
from tests.reference.legacy_target import corrected_target, reference_velocity


class TEMFObjective:
    def __init__(self, total_updates=300000):
        if total_updates < 1:
            raise ValueError("total_updates must be positive")
        self.total_updates = total_updates

    def __call__(self, net, batch, step, generator=None):
        clean, observed, context = batch
        boundary, mix = schedule(step, self.total_updates)
        times = sample_times(clean, boundary, generator)
        noise = torch.randn(clean.shape, device=clean.device,
                            dtype=clean.dtype, generator=generator)
        current = expand_time(times[0], clean)
        state = (1 - current) * clean + current * noise
        condition = observed, context
        reference = reference_velocity(net, (state, noise - clean), times, condition, mix)
        target, derivative, correction = corrected_target(net, state, times, condition, reference)
        prediction = net(state, *times, condition)
        error = (prediction - target).square().flatten(1).mean(1)
        weight = temporal_weight(*times)
        metrics = diagnostics(error, target, derivative, correction)
        metrics.update(weight=weight.mean(), interval=(times[0] - times[1]).mean())
        metrics.update(boundary_requested=clean.new_tensor(boundary))
        metrics.update(boundary_actual=torch.isclose(*times).float().mean())
        return (weight * error).mean(), metrics


def diagnostics(error, target, derivative, correction):
    return {
        "raw_loss": error.mean().detach(),
        "target_rms": target.square().mean().sqrt(),
        "jvp_rms": derivative.square().mean().sqrt(),
        "correction_rms": correction.square().mean().sqrt(),
    }
