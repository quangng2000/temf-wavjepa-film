"""Residual speech blocks; FiLM modifies channels separately at each audio frame."""

import math

from torch import nn
from torch.nn import functional as F

from src.models.conditioning import frame_norm, pixel_norm


class ResidualBlock(nn.Module):
    def __init__(self, channels, time_width, context_width, groups=8):
        super().__init__()
        if channels % groups:
            raise ValueError("Channels must be divisible by normalization groups")
        self.groups = groups
        self.time = nn.Linear(time_width, 2 * channels)
        self.adapter = nn.Sequential(nn.Linear(context_width, channels), nn.SiLU())
        self.film = nn.Linear(channels, 2 * channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.first = nn.Conv2d(channels, channels, 3, padding=1)
        self.second = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value, time, context):
        scale, shift = self.time(time).chunk(2, dim=-1)
        normalized = frame_norm(value, self.groups)
        normalized = (pixel_norm(scale)[:, :, None, None] * normalized
                      + pixel_norm(shift)[:, :, None, None])
        gamma, beta = self.film(self.adapter(context)).transpose(1, 2).chunk(2, dim=1)
        normalized = (1 + gamma[:, :, None]) * normalized + beta[:, :, None]
        hidden = self.first(F.silu(normalized))
        hidden = self.second(F.silu(frame_norm(hidden, self.groups)))
        return (value + hidden) / math.sqrt(2)


class Stage(nn.Module):
    def __init__(self, channels, count, time_width, context_width, groups=8):
        super().__init__()
        self.blocks = nn.ModuleList(ResidualBlock(channels, time_width, context_width, groups)
                                    for _ in range(count))

    def forward(self, value, time, context):
        for block in self.blocks:
            value = block(value, time, context)
        return value
