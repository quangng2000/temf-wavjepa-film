"""Synthetic conditioner for offline smoke tests only; NOT pretrained Nat."""

import torch
from torch import nn
from torch.nn import functional as F


class FixtureConditioner(nn.Module):
    def __init__(self, features=16, local_files_only=None):
        super().__init__()
        if local_files_only is not None:
            raise ValueError("Synthetic conditioning has no model files")
        self.features = features
        self.recipe = dict(model="synthetic_fixture", features=features)

    def load(self, device):
        return None

    @torch.no_grad()
    def forward(self, waveform, positions):
        pooled = F.interpolate(waveform[:, None], size=len(positions), mode="linear",
                               align_corners=False).transpose(1, 2)
        scales = torch.linspace(0.1, 1, self.features, device=waveform.device)
        return torch.sin(pooled * scales)
