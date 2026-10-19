"""Nat FiLM adapter around an unchanged upstream residual block."""
from torch import nn
from torch.nn import functional as F


class NatFiLM(nn.Module):
    def __init__(self, channels, context_width):
        super().__init__()
        self.head = nn.Linear(context_width, 2 * channels)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, value, context):
        context = F.interpolate(context.transpose(1, 2), size=value.shape[-1],
                                mode="linear", align_corners=False).transpose(1, 2)
        scale, shift = self.head(context).transpose(1, 2).chunk(2, dim=1)
        return (1 + scale[:, :, None]) * value + shift[:, :, None]


class ConditionedBlock(nn.Module):
    def __init__(self, block, context_width, normalize_time):
        super().__init__()
        self.block = block
        channels = block.GroupNorm_1.num_channels
        self.film = NatFiLM(channels, context_width)
        if normalize_time:
            block.Dense_0 = nn.Sequential(block.Dense_0,
                nn.RMSNorm(channels, eps=1e-4, elementwise_affine=False))

    def forward(self, value, controls):
        time, context = controls
        def condition(module, inputs, output):
            return self.film(output, context)
        # Local hook lifetime: no global condition state or backward recomputation.
        with self.block.GroupNorm_1.register_forward_hook(condition):
            return self.block(value, time)
