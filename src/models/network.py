"""Configurable frequency-downsampling U-Net with a direct mean-velocity head."""

import torch
from torch import nn
from torch.nn import functional as F

from src.models.blocks import Stage
from src.models.conditioning import TimeEmbedding


class VelocityUNet(nn.Module):
    def __init__(self, channels=(128, 256, 512, 1024), blocks=(2, 2, 4, 5),
                 context_width=1536, time_width=256, groups=8):
        super().__init__()
        if len(channels) != len(blocks) or not channels or min(blocks) < 1:
            raise ValueError("Specify positive block counts for each channel level")
        self.recipe = dict(channels=list(channels), blocks=list(blocks),
                           context_width=context_width, time_width=time_width, groups=groups)
        self.time = TimeEmbedding(time_width)
        self.stem = nn.Conv2d(4, channels[0], 3, padding=1)
        self.down = nn.ModuleList(nn.Conv2d(a, b, 3, stride=(2, 1), padding=1)
                                 for a, b in zip(channels[:-1], channels[1:]))
        self.up = nn.ModuleList(nn.Conv2d(b, a, 3, padding=1)
                               for a, b in zip(channels[:-1], channels[1:]))
        self.encoders = self._stages(channels, blocks)
        self.decoders = self._stages(channels, blocks)
        self.head = nn.Conv2d(channels[0], 2, 3, padding=1)

    def _stages(self, channels, blocks):
        config = self.recipe
        return nn.ModuleList(Stage(c, n, config["time_width"], config["context_width"],
                                    config["groups"]) for c, n in zip(channels, blocks))

    def forward(self, state, current, destination, condition):
        observed, context = condition
        if state.shape != observed.shape or context.shape[1] != state.shape[-1]:
            raise ValueError("State, observation, and frame-aligned context must agree")
        time = self.time(current, destination)
        value = self.stem(torch.cat((state, observed), dim=1))
        skips = []
        for index, stage in enumerate(self.encoders):
            value = stage(value, time, context)
            skips.append(value)
            if index < len(self.down):
                value = self.down[index](value)
        for index in reversed(range(len(self.decoders))):
            if index < len(self.up):
                value = F.interpolate(value, size=skips[index].shape[-2:], mode="nearest")
                value = (self.up[index](value) + skips[index]) / 2**0.5
            value = self.decoders[index](value, time, context)
        return self.head(value)
