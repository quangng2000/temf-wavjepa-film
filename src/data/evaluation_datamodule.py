"""Validation-only paired data without a training manifest or training loader."""

from src.data.components.manifest import read_manifest
from src.data.components.pairs import PairedAudio
from src.data.paired_datamodule import PairedDataModule


class EvaluationDataModule(PairedDataModule):
    def __init__(self, val_manifest, **options):
        super().__init__(train_manifest=None, val_manifest=val_manifest, **options)

    def setup(self, stage=None):
        if stage not in (None, "validate"):
            raise ValueError("EvaluationDataModule supports validation only")
        entries = read_manifest(self.hparams.val_manifest)
        self.validation = PairedAudio(entries, self.hparams.samples, training=False)

    def train_dataloader(self):
        raise ValueError("EvaluationDataModule supports validation only")
