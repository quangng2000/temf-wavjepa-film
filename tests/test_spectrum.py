import pytest
import torch

from src.audio.spectrum import Spectrum


@pytest.mark.parametrize("length", [19, 1024, 2049])
@pytest.mark.parametrize("silent", [False, True])
def test_round_trip_including_short_and_silent_audio(length, silent):
    transform = Spectrum(n_fft=64, hop=16).double()
    waveform = torch.randn(2, length, dtype=torch.float64) * (0 if silent else 0.05)
    encoded = transform.encode(waveform)
    reconstructed = transform.decode(encoded, length)
    torch.testing.assert_close(reconstructed, waveform, atol=1e-12, rtol=1e-10)
    assert encoded.shape[1] == 2


def test_stft_frame_positions_are_audio_seconds():
    transform = Spectrum(n_fft=512, hop=160)
    torch.testing.assert_close(transform.frame_times(3, "cpu"), torch.tensor([0, 0.01, 0.02]))
