import torch
from torch import nn

from src.flow.target import corrected_target, reference_velocity


class AnalyticVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))

    def forward(self, state, current, destination, condition):
        return (self.scale * state.square() + current[:, None, None, None].square()
                + 3 * destination[:, None, None, None] + condition[0] * 0.1)


def test_jvp_includes_state_and_time_but_holds_destination_fixed():
    net = AnalyticVelocity()
    state = torch.randn(3, 2, 2, 2, dtype=torch.float64)
    current = torch.tensor([0.8, 0.6, 0.4], dtype=torch.float64)
    destination = torch.tensor([0.1, 0.6, 0.4 - 1e-8], dtype=torch.float64)
    reference = torch.ones_like(state) * 0.7
    target, derivative, _ = corrected_target(net, state, (current, destination),
                                             (state, state), reference)
    expected = torch.zeros_like(state)
    expected[0] = 2 * 0.2 * state[0] * 0.7 + 2 * current[0]
    torch.testing.assert_close(derivative, expected, atol=1e-12, rtol=1e-12)
    assert not target.requires_grad
    loss = (net(state, current, destination, (state, state)) - target).square().mean()
    loss.backward()
    assert net.scale.grad is not None and torch.isfinite(net.scale.grad)


def test_reference_uses_mixture_interior_and_path_velocity_at_endpoints():
    net = AnalyticVelocity()
    state = torch.ones(3, 2, 2, 2, dtype=torch.float64)
    current = torch.tensor([0., 0.5, 1.], dtype=torch.float64)
    velocity = torch.ones_like(state) * 2
    result = reference_velocity(net, (state, velocity), (current, current), (state, state), 0.3)
    torch.testing.assert_close(result[[0, 2]], velocity[[0, 2]])
    diagonal = net(state, current, current, (state, state))
    torch.testing.assert_close(result[1], 0.3 * velocity[1] + 0.7 * diagonal[1])
    assert not result.requires_grad
