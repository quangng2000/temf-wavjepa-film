"""Continuous TEMF defaults at commit 78c531cca978a23113b262cd0b6bede2197b1916."""

import math

import torch


def schedule(step, total):
    if total <= 0 or step < 0:
        raise ValueError("Expected positive total updates and nonnegative step")
    if step > total:
        return 0.1, 0.1
    progress = step / total
    boundary = 0.9 - 0.8 / (1 + math.exp(-8 * (progress - 0.6)))
    return boundary, 0.3 - 0.2 * progress


def sample_times(reference, boundary, generator=None):
    size = len(reference)
    diagonal_count = int(size * boundary)
    options = dict(device=reference.device, dtype=reference.dtype, generator=generator)
    diagonal = (torch.randn(diagonal_count, **options) - 0.4).sigmoid()
    first = (torch.randn(size - diagonal_count, **options) - 0.4).sigmoid()
    second = (torch.randn(size - diagonal_count, **options) - 0.4).sigmoid()
    return (torch.cat((diagonal, torch.maximum(first, second))),
            torch.cat((diagonal, torch.minimum(first, second))))


def temporal_weight(current, destination):
    delta = (current - destination).clamp_min(0.001)
    return (1 / (1 + delta)) * torch.rsqrt(1 + delta.square())


def expand_time(time, state):
    return time.reshape(len(state), *([1] * (state.ndim - 1)))
