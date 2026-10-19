import json

import numpy as np
import pytest

from src.augmentation.prepare_rooms import generate_bank
from src.augmentation.rooms import apply_room, simulate_room


def test_generated_direct_and_full_paths_share_arrival_time(tmp_path):
    room = simulate_room(71)
    path = tmp_path / "room.npz"
    np.savez(path, **room)
    impulse = np.zeros(8000, dtype=np.float32)
    impulse[4000] = 0.1
    clean, observed = apply_room(impulse, str(path))
    assert clean.shape == observed.shape == impulse.shape
    assert np.argmax(np.abs(clean)) == np.argmax(np.abs(observed)) == 4000
    assert np.sum(observed[4100:]**2) > np.sum(clean[4100:]**2)
    assert np.isfinite(observed).all()


def test_reflection_peak_does_not_control_alignment(tmp_path):
    direct, full = np.zeros(16, np.float32), np.zeros(64, np.float32)
    direct[7], full[7], full[40] = 1, 1, 3
    path = tmp_path / "reflection.npz"
    np.savez(path, direct=direct, full=full, offset=7, sample_rate=16000, version=1)
    impulse = np.zeros(256, np.float32)
    impulse[100] = 1
    clean, observed = apply_room(impulse, str(path))
    assert clean[100] == pytest.approx(1) and observed[100] == pytest.approx(1)
    assert np.argmax(observed) == 133


def test_zero_reflection_room_matches_target_exactly(tmp_path):
    path = tmp_path / "direct.npz"
    np.savez(path, **simulate_room(13, max_order=0))
    audio = np.random.default_rng(3).normal(0, 0.1, 4000).astype(np.float32)
    clean, observed = apply_room(audio, str(path))
    np.testing.assert_array_equal(clean, observed)


def test_room_bank_is_explicit_reproducible_and_never_overwrites(tmp_path):
    left = generate_bank(tmp_path / "first", count=1, seed=99)
    right = generate_bank(tmp_path / "second", count=1, seed=99)
    row = json.loads(left.read_text())
    with np.load(left.parent / row["path"]) as a, np.load(right.parent / row["path"]) as b:
        np.testing.assert_array_equal(a["full"], b["full"])
    with pytest.raises(FileExistsError):
        generate_bank(left.parent, count=1, seed=100)
