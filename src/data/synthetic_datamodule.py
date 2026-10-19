"""Tiny synthetic LightningDataModule; not a speech-quality experiment."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.synthetic import SyntheticPairs


class SyntheticDataModule(LightningDataModule):
    def __init__(self, batch_size=2, samples=1024):
        super().__init__()
        self.batch_size, self.samples = batch_size, samples

    def train_dataloader(self):
        return DataLoader(SyntheticPairs(samples=self.samples), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(SyntheticPairs(2, self.samples, seed=100), batch_size=self.batch_size)
