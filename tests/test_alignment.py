import pytest
import torch

from src.audio.alignment import align_features


def test_alignment_preserves_channel_features_and_interpolates_time():
    features = torch.tensor([[[0., 100.], [10., 110.], [20., 120.]]])
    times = torch.tensor([[0., 1., 2.]])
    positions = torch.tensor([-1., 0.5, 1.5, 3.])
    expected = torch.tensor([[[0., 100.], [5., 105.], [15., 115.], [20., 120.]]])
    torch.testing.assert_close(align_features(features, times, positions), expected)


def test_duplicate_timestamps_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        align_features(torch.ones(1, 2, 3), torch.zeros(1, 2), torch.zeros(2))
