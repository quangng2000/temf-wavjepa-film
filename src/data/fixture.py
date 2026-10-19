"""Small deterministic synthetic pairs for offline pipeline checks, not speech quality."""

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


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


class SyntheticDataModule(LightningDataModule):
    def __init__(self, batch_size=2, samples=1024):
        super().__init__()
        self.batch_size, self.samples = batch_size, samples

    def train_dataloader(self):
        return DataLoader(SyntheticPairs(samples=self.samples), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(SyntheticPairs(2, self.samples, seed=100), batch_size=self.batch_size)
