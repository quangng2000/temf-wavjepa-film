"""On-the-fly observed audio from clean speech, with replayable per-example seeds."""

import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.augmentation.pipeline import Degradation
from src.augmentation.rooms import load_room
from src.data.components.pairs import read_audio
from src.data.components.replay import ReplayRandom, fingerprint


class AugmentedAudio(Dataset):
    def __init__(self, speech, assets, config):
        self.entries, self.samples = speech, config["samples"]
        self.context = 16000  # Crop after reverberation using up to 1 s of real context.
        noise_paths = [row["path"] for row in assets["noise"]]
        rooms = [row["path"] for row in assets["rooms"]]
        self.validate_assets(noise_paths, rooms)
        self.pipeline = Degradation(noise_paths, rooms, config["recipe"])
        identity = fingerprint(speech + assets["noise"] + assets["rooms"],
                               [config, self.pipeline.recipe])
        self.random = ReplayRandom(config["seed"], identity)

    def validate_assets(self, noise_paths, rooms):
        for row in self.entries + [dict(clean=path) for path in noise_paths]:
            info = sf.info(row["clean"])
            if info.samplerate != 16000 or info.channels != 1 or info.frames < self.samples:
                raise ValueError("Speech/noise must be 16 kHz mono and at least crop length")
        for path in rooms:
            if max(len(value) for value in load_room(path)[:2]) > self.context:
                raise ValueError("Room impulse exceeds the 1-second context window")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row = self.entries[index]
        start = self.random.integer(sf.info(row["clean"]).frames - self.samples + 1)
        read_start = max(0, start - self.context)
        frames = self.samples + start - read_start + self.context
        audio = read_audio(row["clean"], read_start, frames)
        seed = self.random.integer(2**32)
        crop = slice(start - read_start, start - read_start + self.samples)
        clean, observed = self.pipeline(audio.numpy(), seed, crop)
        return dict(clean=torch.from_numpy(clean), observed=torch.from_numpy(observed),
                    id=row["id"], augmentation_seed=seed, crop_start=start)

    def state_dict(self):
        return self.random.state_dict()

    def load_state_dict(self, state):
        self.random.load_state_dict(state)
