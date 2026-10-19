"""LightningDataModule for prepared, aligned speech pairs."""

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from src.data.manifest import check_disjoint, read_manifest
from src.data.pairs import PairedAudio


class PairedDataModule(LightningDataModule):
    def __init__(self, train_manifest, val_manifest, **options):
        super().__init__()
        settings = dict(batch_size=4, samples=64000, workers=0, seed=42)
        if options.keys() - settings.keys():
            raise ValueError(f"Unknown data options: {options.keys() - settings.keys()}")
        settings.update(options)
        if settings["samples"] < 1 or settings["batch_size"] < 1 or settings["workers"] < 0:
            raise ValueError("Positive crop/batch sizes and nonnegative workers are required")
        self.save_hyperparameters(dict(train_manifest=train_manifest, val_manifest=val_manifest,
                                      **settings))

    def setup(self, stage=None):
        config = self.hparams
        training = read_manifest(config.train_manifest)
        validation = read_manifest(config.val_manifest)
        check_disjoint(training, validation)
        self.training = PairedAudio(training, config.samples, training=True, seed=config.seed)
        self.validation = PairedAudio(validation, config.samples, training=False)
        if len(self.training) < config.batch_size:
            raise ValueError("Training manifest must contain at least one complete batch")

    def train_dataloader(self):
        return StatefulDataLoader(self.training, batch_size=self.hparams.batch_size, shuffle=True,
                                  num_workers=self.hparams.workers, drop_last=True,
                                  generator=torch.Generator().manual_seed(self.hparams.seed))

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.workers)
