"""Positional flow-time embeddings and frame-local adaptive double normalization."""

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, width=256, frequencies=32):
        super().__init__()
        self.register_buffer("frequencies", torch.logspace(0, -4, frequencies))
        self.mlp = nn.Sequential(nn.Linear(4 * frequencies, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU())

    def forward(self, current, destination):
        angles = torch.stack((current, destination), dim=1)[..., None] * self.frequencies
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(1))


def pixel_norm(value, epsilon=1e-4):
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + epsilon)


def frame_norm(value, groups):
    batch, channels, frequencies, frames = value.shape
    frames_first = value.permute(0, 3, 1, 2).reshape(batch * frames, channels, frequencies)
    normalized = F.group_norm(frames_first, groups)
    return normalized.reshape(batch, frames, channels, frequencies).permute(0, 2, 3, 1)
