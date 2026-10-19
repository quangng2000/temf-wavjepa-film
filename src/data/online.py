"""Opt-in generated training pairs; validation always uses fixed held-out pairs."""

from omegaconf import OmegaConf

from src.data.augmented import AugmentedAudio
from src.data.manifest import check_disjoint, read_manifest
from src.data.module import PairedDataModule
from src.data.pairs import PairedAudio


class OnlineDataModule(PairedDataModule):
    def __init__(self, train_manifest, val_manifest, augmentation, **options):
        super().__init__(train_manifest, val_manifest, **options)
        self.augmentation = OmegaConf.to_container(OmegaConf.create(augmentation), resolve=True)

    def setup(self, stage=None):
        config, augmentation = self.hparams, self.augmentation
        speech = read_manifest(config.train_manifest, fields=("clean",))
        validation = read_manifest(config.val_manifest)
        check_disjoint(speech, validation)
        assets = {name: read_manifest(augmentation[f"{name}_manifest"], fields=("path",))
                  for name in ("noise", "rooms")}
        self.reject_shared_assets(assets, speech + validation)
        recipe = dict(samples=config.samples, seed=config.seed, recipe=augmentation["recipe"])
        self.training = AugmentedAudio(speech, assets, recipe)
        self.validation = PairedAudio(validation, config.samples, training=False)
        if len(speech) < config.batch_size:
            raise ValueError("Training manifest must contain at least one complete batch")

    @staticmethod
    def reject_shared_assets(assets, speech):
        speech_paths = {row[key] for row in speech for key in ("clean", "observed") if key in row}
        if any(row["path"] in speech_paths for rows in assets.values() for row in rows):
            raise ValueError("Noise/room assets must not reuse speech/validation audio paths")
