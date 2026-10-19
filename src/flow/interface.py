"""Translate audio tensors to TEMF's five-dimensional call contract."""
from types import SimpleNamespace

from torch import nn


class AudioCondition:
    def __init__(self, observed, context):
        self.observed, self.context = observed, context

    def __getitem__(self, selection):
        return AudioCondition(self.observed[selection], self.context[selection])


class AudioVelocity(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cfg = SimpleNamespace(is_lgm=False)

    def forward(self, state, sigma_next, sigma, cond, **options):
        result = self.net(state.squeeze(1), sigma.flatten(), sigma_next.flatten(),
                          (cond.observed, cond.context)).unsqueeze(1)
        return (result, {}) if options.get("return_extra_output", False) else result
