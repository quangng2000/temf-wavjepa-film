"""Reverse integration of direct mean velocity; no JVP during inference."""

import torch


@torch.no_grad()
def solve(net, condition, nfe=1, seed=0):
    if nfe < 1:
        raise ValueError("NFE must be positive")
    observed, _ = condition
    generator = torch.Generator(device=observed.device).manual_seed(seed)
    state = torch.randn(observed.shape, device=observed.device,
                        dtype=observed.dtype, generator=generator)
    times = torch.linspace(1, 0, nfe + 1, device=state.device, dtype=state.dtype)
    for current, destination in zip(times[:-1], times[1:]):
        start, end = current.expand(len(state)), destination.expand(len(state))
        state = state - (current - destination) * net(state, start, end, condition)
        if not torch.isfinite(state).all():
            raise FloatingPointError("Nonfinite restoration state")
    return state
