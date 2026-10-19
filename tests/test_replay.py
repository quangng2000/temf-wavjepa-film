import copy

import numpy as np
import pytest
import soundfile as sf
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from src.data.augmented import AugmentedAudio
from src.data.pairs import PairedAudio
from tests.test_augmentation import pipeline, speech_fixture
from tests.test_pairs import make_pair


def build_loader(dataset, workers):
    return StatefulDataLoader(dataset, batch_size=2, shuffle=True, num_workers=workers,
                              generator=torch.Generator().manual_seed(9))


def assert_batches_equal(expected, actual):
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual):
        for key in left:
            if isinstance(left[key], torch.Tensor):
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            else:
                assert left[key] == right[key]


@pytest.mark.parametrize("workers", [0, 2])
def test_prepared_pairs_resume_shuffle_and_random_crop(tmp_path, workers):
    entries = [make_pair(tmp_path, f"clip-{i}", f"speaker-{i}") for i in range(10)]
    first = build_loader(PairedAudio(entries, samples=32), workers)
    iterator = iter(first)
    next(iterator)
    state = copy.deepcopy(first.state_dict())
    expected = list(iterator)
    resumed = build_loader(PairedAudio(entries, samples=32), workers)
    resumed.load_state_dict(state)
    assert_batches_equal(expected, list(resumed))


@pytest.mark.parametrize("workers", [0, 2])
def test_generated_pairs_resume_crops_and_library_randomness(tmp_path, workers):
    paths = []
    for index in range(10):
        path = tmp_path / f"speech-{index}.wav"
        sf.write(path, np.tile(speech_fixture(), 2), 16000, subtype="FLOAT")
        paths.append(dict(id=str(index), group=str(index), clean=str(path)))
    recipe = pipeline(tmp_path, [dict(_target_="audiomentations.TimeMask", p=1)], 1).config
    assets = dict(noise=[dict(path=str(tmp_path / "noise.wav"))], rooms=[])
    config = dict(samples=16000, seed=33, recipe=recipe)
    first = build_loader(AugmentedAudio(paths, assets, config), workers)
    iterator = iter(first)
    next(iterator)
    state = copy.deepcopy(first.state_dict())
    expected = list(iterator)
    resumed = build_loader(AugmentedAudio(paths, assets, config), workers)
    resumed.load_state_dict(state)
    assert_batches_equal(expected, list(resumed))


def test_changed_data_recipe_is_rejected(tmp_path):
    entries = [make_pair(tmp_path)]
    saved = PairedAudio(entries, samples=32).state_dict()
    with pytest.raises(ValueError, match="recipe changed"):
        PairedAudio(entries, samples=64).load_state_dict(saved)
