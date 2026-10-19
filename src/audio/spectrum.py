"""Invertible compressed complex STFT, shared by training and inference."""

import torch
from torch import nn


class Spectrum(nn.Module):
    def __init__(self, n_fft=512, hop=160, exponent=0.3, sample_rate=16000):
        super().__init__()
        if n_fft < 4 or not 0 < hop < n_fft or not 0 < exponent <= 1:
            raise ValueError("Invalid STFT/compression settings")
        self.recipe = dict(n_fft=n_fft, hop=hop, exponent=exponent, sample_rate=sample_rate)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def encode(self, waveform):
        if waveform.ndim != 2 or waveform.shape[-1] < 1:
            raise ValueError("Expected nonempty [batch, samples] waveform")
        config = self.recipe
        value = torch.stft(waveform, config["n_fft"], config["hop"], window=self.window,
                           center=True, pad_mode="constant", normalized=True, return_complex=True)
        compressed = torch.polar(value.abs().pow(config["exponent"]), value.angle())
        return torch.stack((compressed.real, compressed.imag), dim=1)

    def decode(self, coefficients, length):
        value = torch.complex(coefficients[:, 0], coefficients[:, 1])
        config = self.recipe
        spectrum = torch.polar(value.abs().pow(1 / config["exponent"]), value.angle())
        return torch.istft(spectrum, config["n_fft"], config["hop"], window=self.window,
                            center=True, normalized=True, length=length)

    def frame_times(self, frames, device):
        seconds_per_frame = self.recipe["hop"] / self.recipe["sample_rate"]
        return torch.arange(frames, device=device) * seconds_per_frame
