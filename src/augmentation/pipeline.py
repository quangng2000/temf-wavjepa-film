"""A paired-audio coordinator; effect implementations come from audiomentations."""

import random
from importlib.metadata import version

import numpy as np
from audiomentations import AddBackgroundNoise
from hydra.utils import instantiate

from src.augmentation.randomness import seeded_transforms
from src.augmentation.rooms import apply_room


def shared_headroom(clean, observed):
    if clean.shape != observed.shape or not all(np.isfinite(x).all() for x in (clean, observed)):
        raise ValueError("Degradation must preserve length and produce finite audio")
    gain = min(1.0, 0.95 / max(float(np.abs(clean).max()), float(np.abs(observed).max()), 1e-8))
    return clean * gain, observed * gain


class Degradation:
    def __init__(self, noise_paths, room_paths, config):
        self.room_paths, self.config = room_paths, config
        if not 0 <= config["room_probability"] <= 1:
            raise ValueError("room_probability must be within [0, 1]")
        if config["room_probability"] and not room_paths:
            raise ValueError("Reverberation requires an explicit paired room bank")
        self.noise = AddBackgroundNoise(sounds_path=noise_paths, **config["noise"])
        self.effects = [instantiate(effect) for effect in config["effects"]]
        self.recipe = dict(config=config, versions={name: version(name) for name in
                           ("audiomentations", "pyroomacoustics", "fast-mp3-augment",
                            "numpy", "scipy", "torch", "torchdata")})

    def __call__(self, samples, seed, crop=slice(None)):
        with seeded_transforms(seed):
            clean, observed = self.render_room(samples)
            clean, observed = clean[crop].copy(), observed[crop].copy()
            observed = self.noise(observed, sample_rate=16000)
            clean, observed = shared_headroom(clean, observed)
            for effect in self.effects:
                observed = effect(observed, sample_rate=16000)
                clean, observed = shared_headroom(clean, observed)
        return clean.astype(np.float32), observed.astype(np.float32)

    def render_room(self, samples):
        if random.random() < self.config["room_probability"]:
            return apply_room(samples, random.choice(self.room_paths))
        return samples.copy(), samples.copy()
