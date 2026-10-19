"""Deterministic synthetic pairs for offline smoke tests."""

import torch
from torch.utils.data import Dataset


class SyntheticPairs(Dataset):
    def __init__(self, count=8, samples=1024, seed=17):
        self.count, self.samples, self.seed = count, samples, seed

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        time = torch.arange(self.samples) / 16000
        clean = 0.1 * torch.sin(2 * torch.pi * (200 + index * 17) * time)
        noise = torch.randn(self.samples, generator=generator) * 0.025
        return dict(clean=clean, observed=0.7 * clean + noise)
