"""Training paths parameterized by the TEMF schedule coordinate."""
import math

import torch


class LinearPath:
    name = "linear"

    def __init__(self, noise_scale=1.0):
        self.noise_scale = float(noise_scale)

    def state(self, clean, noise, time):
        return (1 - time) * clean + time * self.noise_scale * noise

    def velocity(self, clean, noise, time):
        del time
        return self.noise_scale * noise - clean


class NormalizedTrigFlowPath:
    """Variance-preserving trigonometric path over a unit TEMF time interval."""

    name = "trigflow"

    def __init__(self, noise_scale=1.0):
        self.noise_scale = float(noise_scale)

    def state(self, clean, noise, time):
        angle = time * (math.pi / 2)
        return torch.cos(angle) * clean + torch.sin(angle) * self.noise_scale * noise

    def velocity(self, clean, noise, time):
        angle = time * (math.pi / 2)
        scale = math.pi / 2
        return scale * (-torch.sin(angle) * clean
                        + torch.cos(angle) * self.noise_scale * noise)


def create_flow_path(name, noise_scale=1.0):
    paths = {"linear": LinearPath, "trigflow": NormalizedTrigFlowPath}
    try:
        return paths[name](noise_scale)
    except KeyError as error:
        raise ValueError(f"Unsupported flow path: {name}") from error
