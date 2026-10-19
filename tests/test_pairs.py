import json

import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.manifest import check_disjoint, read_manifest
from src.data.pairs import PairedAudio


def make_pair(tmp_path, name="one", group="speaker1"):
    signal = np.linspace(-0.5, 0.5, 200, dtype=np.float32)
    paths = [tmp_path / f"{name}-{kind}.wav" for kind in ("clean", "observed")]
    for path, audio in zip(paths, (signal, signal / 2)):
        sf.write(path, audio, 16000, subtype="FLOAT")
    return dict(id=name, group=group, clean=str(paths[0]), observed=str(paths[1]))


def test_training_crop_is_aligned(tmp_path):
    dataset = PairedAudio([make_pair(tmp_path)], samples=32)
    pair = dataset[0]
    assert pair["clean"].shape == (32,)
    torch.testing.assert_close(pair["observed"], pair["clean"] / 2)


def test_manifest_resolution_and_split_leakage(tmp_path):
    row = make_pair(tmp_path)
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    assert read_manifest(manifest) == [row]
    with pytest.raises(ValueError, match="overlap"):
        check_disjoint([row], [dict(row, id="two")])


def test_mismatched_audio_lengths_fail(tmp_path):
    row = make_pair(tmp_path)
    sf.write(row["observed"], np.zeros(180), 16000)
    with pytest.raises(ValueError, match="equal lengths"):
        PairedAudio([row], samples=32)[0]
