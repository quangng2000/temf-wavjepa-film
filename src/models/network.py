"""SGMSE NCSN++ with a two-time input adapter and frame-aligned Nat FiLM."""
import torch
from torch import nn
from torch.nn import functional as F

from src.models.film import ConditionedBlock
from src.vendor.sgmse.ncsnpp_utils.layers import get_timestep_embedding
from src.vendor.sgmse.ncsnpp_utils.layerspp import ResnetBlockBigGANpp
from src.vendor.sgmse.ncsnpp_v2 import NCSNpp_v2

SGMSE_COMMIT = "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e"


class NCSNppVelocity(nn.Module):
    def __init__(self, nf=256, channel_mult=(1, 1, 2, 2, 2, 2, 2), blocks=2,
                 context_width=1536, normalize_time=True):
        super().__init__()
        if nf < 8 or nf % 8 or not channel_mult or blocks < 1:
            raise ValueError("Use nf >= 8 divisible by 8 and positive block counts")
        self.recipe = dict(backbone="sgmse-ncsnpp-v2", commit=SGMSE_COMMIT, nf=nf,
                           channel_mult=list(channel_mult), blocks=blocks,
                           context_width=context_width, normalize_time=normalize_time,
                           adapter="two-time-positional-nat-film-v1", fir="upstream-native")
        self.multiple = 2 ** (len(channel_mult) - 1)
        self.backbone = NCSNpp_v2(nf=nf, ch_mult=channel_mult, num_res_blocks=blocks,
            attn_resolutions=(), embedding_type="positional", dropout=0.0)
        self._install_adapters(context_width, normalize_time)

    def _install_adapters(self, context_width, normalize_time):
        modules = self.backbone.all_modules
        for index, block in enumerate(modules):
            if isinstance(block, ResnetBlockBigGANpp):
                modules[index] = ConditionedBlock(block, context_width, normalize_time)
        if normalize_time:
            modules[1] = nn.Sequential(modules[1],
                nn.RMSNorm(self.backbone.nf * 4, eps=1e-4, elementwise_affine=False))

    def forward(self, state, current, destination, condition):
        observed, context = condition
        if state.shape != observed.shape or context.shape[1] != state.shape[-1]:
            raise ValueError("State, observation, and aligned context must agree")
        padding = ((-state.shape[-1]) % self.multiple, (-state.shape[-2]) % self.multiple)
        padded = [F.pad(value, (0, padding[0], 0, padding[1])) for value in (state, observed)]
        complex_pair = [torch.complex(value[:, 0], value[:, 1])[:, None] for value in padded]
        context = F.pad(context.transpose(1, 2), (0, padding[0]), mode="replicate").transpose(1, 2)
        embedding = torch.cat([get_timestep_embedding(time, self.backbone.nf // 2)
                               for time in (current, destination)], dim=1)
        output = self.backbone(*complex_pair, current, embedding=embedding, context=context)
        result = torch.cat((output.real, output.imag), dim=1)
        return result[..., :state.shape[-2], :state.shape[-1]]
