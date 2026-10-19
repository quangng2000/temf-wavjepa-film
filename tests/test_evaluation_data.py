"""Standalone evaluation requires only fixed held-out speech pairs."""

import json

import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.evaluation_datamodule import EvaluationDataModule


@pytest.fixture
def evaluation_data(tmp_path):
    clean = np.linspace(-0.8, 0.8, 100, dtype=np.float32)
    observed = clean * 0.5
    sf.write(tmp_path / "clean.wav", clean, 16000, subtype="FLOAT")
    sf.write(tmp_path / "observed.wav", observed, 16000, subtype="FLOAT")
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(json.dumps(dict(id="heldout", group="speaker",
                         clean="clean.wav", observed="observed.wav")) + "\n")
    data = EvaluationDataModule(manifest, batch_size=1, samples=20, workers=0)
    return data, clean, observed


def test_validation_only_pairs_use_the_same_fixed_center_crop(evaluation_data):
    data, clean, observed = evaluation_data
    data.setup("validate")
    assert data.hparams.train_manifest is None
    assert not hasattr(data, "training")
    batch = next(iter(data.val_dataloader()))
    torch.testing.assert_close(batch["clean"][0], torch.from_numpy(clean[40:60]))
    torch.testing.assert_close(batch["observed"][0], torch.from_numpy(observed[40:60]))
    repeated = next(iter(data.val_dataloader()))
    assert torch.equal(batch["clean"], repeated["clean"])
    assert torch.equal(batch["observed"], repeated["observed"])


@pytest.mark.parametrize("stage", ["fit", "test", "predict"])
def test_evaluation_data_rejects_other_stages(evaluation_data, stage):
    data, _, _ = evaluation_data
    with pytest.raises(ValueError, match="validation only"):
        data.setup(stage)


def test_evaluation_data_cannot_create_a_training_loader(evaluation_data):
    data, _, _ = evaluation_data
    with pytest.raises(ValueError, match="validation only"):
        data.train_dataloader()
