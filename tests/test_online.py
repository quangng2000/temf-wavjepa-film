import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.augmentation.prepare_rooms import generate_bank
from tests.test_augmentation import speech_fixture


def make_online_config(tmp_path):
    audio = np.tile(speech_fixture(), 2)
    rows = []
    for index in range(10):
        path = tmp_path / f"speech-{index}.wav"
        sf.write(path, audio * (1 + index / 20), 16000, subtype="FLOAT")
        rows.append(dict(id=str(index), group=str(index), clean=str(path)))
    training = tmp_path / "train.jsonl"
    training.write_text("".join(json.dumps(row) + "\n" for row in rows[:8]))
    validation = tmp_path / "val.jsonl"
    validation.write_text(json.dumps(dict(rows[8], observed=rows[9]["clean"])) + "\n")
    noise = tmp_path / "noise.wav"
    sf.write(noise, speech_fixture(), 16000, subtype="FLOAT")
    noises = tmp_path / "noise.jsonl"
    noises.write_text(json.dumps(dict(id="noise", group="noise", path=str(noise))) + "\n")
    return configure_online(training, validation, noises, tmp_path)


def configure_online(training, validation, noises, tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs/data/augmented.yaml"
    cfg = OmegaConf.load(path)
    cfg.train_manifest, cfg.val_manifest, cfg.seed = str(training), str(validation), 12
    cfg.augmentation.noise_manifest = str(noises)
    cfg.augmentation.rooms_manifest = str(generate_bank(tmp_path / "rooms", count=1))
    cfg.batch_size = 2
    return cfg


def test_config_runs_real_library_effects_and_keeps_validation_fixed(tmp_path):
    cfg = make_online_config(tmp_path)
    cfg.augmentation.recipe.room_probability = 1
    cfg.augmentation.recipe.noise.p = 1
    for effect in cfg.augmentation.recipe.effects:
        effect.p = 1
    module = instantiate(cfg)
    module.setup()
    pair = module.training[0]
    assert pair["clean"].shape == pair["observed"].shape == (64000,)
    assert torch.isfinite(pair["observed"]).all()
    assert not torch.equal(pair["clean"], pair["observed"])
    first, second = module.validation[0], module.validation[0]
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_noise_cannot_use_validation_recording(tmp_path):
    cfg = make_online_config(tmp_path)
    row = json.loads(Path(cfg.val_manifest).read_text())
    Path(cfg.augmentation.noise_manifest).write_text(json.dumps(dict(
        id="noise", group="noise", path=row["clean"])))
    with pytest.raises(ValueError, match="reuse speech/validation"):
        instantiate(cfg).setup()
