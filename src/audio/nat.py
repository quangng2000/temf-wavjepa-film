"""Pinned, frozen Nat encoder; loading weights is deferred until setup."""

import torch
from torch import nn

from src.audio.alignment import align_features

MODEL_ID = "labhamlet/wavjepa-nat-base"
REVISION = "15d95ff67fa98117b17e83a1653bbca97877ff6f"


class FrozenNat(nn.Module):
    def __init__(self, local_files_only=True):
        super().__init__()
        self.local_files_only = local_files_only
        self.encoder = None
        self.processor = None
        self.recipe = dict(model=MODEL_ID, revision=REVISION, channel_policy="dual_mono",
                           features="channel_concat_1536",
                           alignment="released_timestamps_linear_v1")

    def load(self, device):
        if self.encoder is None:
            from transformers import AutoFeatureExtractor, AutoModel

            options = dict(revision=REVISION, trust_remote_code=True,
                           local_files_only=self.local_files_only)
            self.processor = AutoFeatureExtractor.from_pretrained(MODEL_ID, **options)
            self.encoder = AutoModel.from_pretrained(MODEL_ID, **options).requires_grad_(False)
        self.encoder.to(device).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, waveform, positions):
        if self.encoder is None:
            raise RuntimeError("Call conditioner.load(device) before extracting features")
        if waveform.ndim != 2 or not torch.isfinite(waveform).all():
            raise ValueError("Expected finite, equal-length mono waveforms at 16 kHz")
        dual = waveform.detach().cpu().float().unsqueeze(1).repeat(1, 2, 1)
        inputs = self.processor([clip.numpy() for clip in dual], sampling_rate=16000,
                                return_tensors="pt")["input_values"].to(waveform.device)
        # Exit upstream inference_mode before cloning into ordinary tensors for adapters.
        features, timestamps = self.encoder(inputs)
        if features.ndim != 4 or features.shape[1] != 2 or features.shape[-1] != 768:
            raise ValueError("Unexpected Nat feature shape")
        context = torch.cat((features[:, 0], features[:, 1]), dim=-1).detach().clone()
        result = align_features(context, timestamps.detach().clone() / 1000, positions)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite Nat conditioning")
        return result
