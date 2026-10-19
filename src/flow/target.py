"""Full state/time JVP; target parameter gradients are intentionally detached."""

import torch
from torch.autograd.functional import jvp

from src.flow.schedule import expand_time


def reference_velocity(net, path, times, condition, mix):
    state, velocity = path
    current, _ = times
    with torch.no_grad():
        diagonal = net(state, current, current, condition)
        interior = expand_time((current > 0) & (current < 1), state)
        return torch.where(interior, mix * velocity + (1 - mix) * diagonal, velocity)


def corrected_target(net, state, times, condition, reference):
    current, destination = times
    interval = ~torch.isclose(destination, current)
    derivative = torch.zeros_like(state)
    if interval.any():
        selected = tuple(value[interval] for value in condition)

        def velocity(inputs, start, end):
            return net(inputs, start, end, selected)

        primals = state[interval], current[interval], destination[interval]
        tangents = reference[interval], torch.ones_like(primals[1]), torch.zeros_like(primals[2])
        derivative[interval] = jvp(velocity, primals, tangents, create_graph=False)[1]
    correction = expand_time(current - destination, state) * derivative
    return (reference - correction).detach(), derivative.detach(), correction.detach()
