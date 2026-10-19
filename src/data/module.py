"""LightningDataModule for prepared, aligned speech pairs."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.manifest import check_disjoint, read_manifest
from src.data.pairs import PairedAudio


class PairedDataModule(LightningDataModule):
    def __init__(self, train_manifest, val_manifest, batch_size=4, samples=64000, workers=0):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        config = self.hparams
        training = read_manifest(config.train_manifest)
        validation = read_manifest(config.val_manifest)
        check_disjoint(training, validation)
        self.training = PairedAudio(training, config.samples, training=True)
        self.validation = PairedAudio(validation, config.samples, training=False)
        if len(self.training) < config.batch_size:
            raise ValueError("Training manifest must contain at least one complete batch")

    def train_dataloader(self):
        return DataLoader(self.training, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.workers)
