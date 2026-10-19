"""Dataset randomness isolated from model RNG and saved by StatefulDataLoader."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import get_worker_info


def fingerprint(entries, recipe):
    files = sorted({row[key] for row in entries for key in ("clean", "observed", "path")
                    if key in row})
    stamps = [(path, Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in files]
    payload = json.dumps([entries, stamps, recipe], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class ReplayRandom:
    def __init__(self, seed, identity):
        self.seed, self.identity, self.generator = seed, identity, None

    def initialize(self):
        if self.generator is None:
            worker = get_worker_info()
            offset = 0 if worker is None else worker.seed + 1000003 * (worker.id + 1)
            self.generator = torch.Generator().manual_seed((self.seed + offset) % 2**63)

    def integer(self, maximum):
        self.initialize()
        return torch.randint(maximum, (), generator=self.generator).item()

    def state_dict(self):
        state = None if self.generator is None else self.generator.get_state()
        return dict(identity=self.identity, rng=state)

    def load_state_dict(self, state):
        if state["identity"] != self.identity:
            raise ValueError("Dataset/augmentation recipe changed; cannot replay checkpoint")
        self.generator = None
        if state["rng"] is not None:
            self.generator = torch.Generator()
            self.generator.set_state(state["rng"])
