"""Actual waveform transforms for the small explanatory model, not speech data."""
import torch

RATE, FFT, HOP, LENGTH, POWER = 16000, 512, 256, 5888, .3


def window():
    return torch.hann_window(FFT)


def encode(waveform):
    spectrum = torch.stft(waveform, FFT, HOP, window=window(), normalized=True,
                          return_complex=True, center=True, pad_mode='constant')
    magnitude = spectrum.abs()
    compressed = spectrum * magnitude.clamp_min(1e-12).pow(POWER - 1)
    return torch.view_as_real(compressed).permute(2, 0, 1)[None]


def uncompress(coefficients):
    value = torch.view_as_complex(coefficients[0].permute(1, 2, 0).contiguous())
    return value * value.abs().pow(1 / POWER - 1)


def decode(coefficients):
    return torch.istft(uncompress(coefficients), FFT, HOP, window=window(),
                       normalized=True, center=True, length=LENGTH)


def inputs():
    time = torch.arange(LENGTH) / RATE
    envelope = .12 + .1 * torch.sin(2 * torch.pi * 4 * time).square()
    phase = 2 * torch.pi * (170 * time + 60 * time.square())
    clean_wave = envelope * (phase.sin() + .4 * (2 * phase).sin() + .2 * (3 * phase).sin())
    damaged_wave = .72 * clean_wave + .025 * torch.randn_like(clean_wave)
    frames = torch.nn.functional.pad(damaged_wave, (FFT // 2, FFT // 2)).unfold(0, FFT, HOP)
    projection = torch.randn(4, FFT) / FFT ** .5
    context = torch.tanh(torch.einsum('cf,tf->ct', projection, frames))[None]
    return encode(clean_wave), encode(damaged_wave), context, clean_wave, damaged_wave


def reconstruction(coefficients):
    spectrum = uncompress(coefficients)
    frames = torch.fft.irfft(spectrum, n=FFT, dim=0, norm='ortho')
    contributions = frames * window()[:, None]
    numerator = torch.zeros((frames.shape[1] - 1) * HOP + FFT)
    denominator = torch.zeros_like(numerator)
    for frame in range(frames.shape[1]):
        start = frame * HOP
        numerator[start:start+FFT] += contributions[:, frame]
        denominator[start:start+FFT] += window().square()
    crop = slice(FFT // 2, FFT // 2 + LENGTH)
    ola = numerator[crop] / denominator[crop]
    waveform = decode(coefficients)
    assert (ola - waveform).abs().max() < 1e-10
    return dict(output_wave=waveform, spectrum_real=spectrum.real[:12],
                spectrum_imag=spectrum.imag[:12], frame_ids=[9, 10, 11],
                contributions=contributions[:, 9:12].T,
                overlap_add_max_error=(ola - waveform).abs().max())


def display_planes(value):
    """Keep all data for calculation; serialize only the displayed frequency slice."""
    if isinstance(value, dict):
        return {key: display_planes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [display_planes(item) for item in value]
    if torch.is_tensor(value) and value.ndim == 4 and value.shape[-2:] == (257, 24):
        return value[:, :, :12, :]
    return value
