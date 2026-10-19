"""Load aligned mono pairs and apply exactly the same crop to both waveforms."""

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.data.replay import ReplayRandom, fingerprint


def read_audio(path, start=0, frames=-1):
    audio, rate = sf.read(path, start=start, frames=frames, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all() or not len(audio):
        raise ValueError(f"Expected nonempty finite 16 kHz mono audio: {path}")
    return torch.from_numpy(audio)


class PairedAudio(Dataset):
    def __init__(self, entries, samples=64000, training=True, seed=42):
        if samples < 1:
            raise ValueError("Crop length must be positive")
        self.entries, self.samples, self.training = entries, samples, training
        self.random = ReplayRandom(seed, fingerprint(entries, [samples, training, seed]))

    def state_dict(self):
        return self.random.state_dict()

    def load_state_dict(self, state):
        self.random.load_state_dict(state)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        clean, observed = sf.info(entry["clean"]), sf.info(entry["observed"])
        if clean.frames != observed.frames or clean.frames < self.samples:
            raise ValueError(f"Pair requires equal lengths at least crop size: {entry['id']}")
        available = clean.frames - self.samples
        start = self.random.integer(available + 1) if self.training else available // 2
        return dict(clean=read_audio(entry["clean"], start, self.samples),
                    observed=read_audio(entry["observed"], start, self.samples))
