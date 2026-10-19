import random

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import correlate, correlation_lags

from src.augmentation.pipeline import Degradation


def pipeline(tmp_path, effects=(), noise_probability=0.0):
    noise_path = tmp_path / "noise.wav"
    noise = np.random.default_rng(1).normal(0, 0.1, 64000).astype(np.float32)
    sf.write(noise_path, noise, 16000, subtype="FLOAT")
    config = dict(room_probability=0, noise=dict(min_snr_db=17, max_snr_db=17,
                  p=noise_probability), effects=list(effects))
    return Degradation([str(noise_path)], [], config)


def speech_fixture():
    rng = np.random.default_rng(9)
    return rng.normal(0, 0.05, 64000).astype(np.float32)


def test_noise_has_requested_snr_and_preserves_clean_target(tmp_path):
    audio = speech_fixture()
    clean, observed = pipeline(tmp_path, noise_probability=1)(audio, seed=7)
    np.testing.assert_array_equal(clean, audio)
    measured = 10 * np.log10(np.mean(clean**2) / np.mean((observed - clean)**2))
    assert measured == pytest.approx(17, abs=0.01)


def test_seed_repeats_effects_without_modifying_input_or_global_rng(tmp_path):
    effect = dict(_target_="audiomentations.TimeMask", p=1)
    degrade, audio = pipeline(tmp_path, [effect], 1), speech_fixture()
    before, python_state, numpy_state = audio.copy(), random.getstate(), np.random.get_state()
    first, second = degrade(audio, 19), degrade(audio, 19)
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    np.testing.assert_array_equal(audio, before)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(first[1], degrade(audio, 20)[1])


def test_mp3_roundtrip_preserves_timing_and_length(tmp_path):
    effect = dict(_target_="audiomentations.Mp3Compression", min_bitrate=64, max_bitrate=64,
                  backend="fast-mp3-augment", preserve_delay=False, p=1)
    audio = speech_fixture()
    clean, observed = pipeline(tmp_path, [effect])(audio, 3)
    correlation = correlate(observed, clean, mode="full", method="fft")
    lag = correlation_lags(len(observed), len(clean))[np.argmax(correlation)]
    assert observed.shape == audio.shape and lag == 0
    np.testing.assert_array_equal(clean, audio)
    assert np.isfinite(observed).all() and not np.array_equal(observed, clean)


def test_silence_is_finite_and_shared_headroom_preserves_gain_relation(tmp_path):
    effect = dict(_target_="audiomentations.Gain", min_gain_db=6, max_gain_db=6, p=1)
    degrade = pipeline(tmp_path, [effect])
    assert all(not np.any(x) for x in degrade(np.zeros(64000, dtype=np.float32), 1))
    clean, observed = degrade(np.full(64000, 0.8, dtype=np.float32), 1)
    assert np.max(np.abs(observed)) <= 0.950001
    np.testing.assert_allclose(observed, clean * 10**0.3, rtol=1e-6)
