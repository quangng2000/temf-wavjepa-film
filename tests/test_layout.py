"""Checkpoint/import compatibility for the template layout migration."""

import hashlib
from importlib import import_module

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.models.components.network import NCSNppVelocity

ALIASES = [
    ("models.module", "models.temf_module", "RestorationModule"),
    ("models.network", "models.components.network", "NCSNppVelocity"),
    ("models.film", "models.components.film", "ConditionedBlock"),
    ("models.film", "models.components.film", "NatFiLM"),
    ("data.module", "data.paired_datamodule", "PairedDataModule"),
    ("data.online", "data.augmented_datamodule", "OnlineDataModule"),
    ("data.fixture", "data.synthetic_datamodule", "SyntheticDataModule"),
    ("data.fixture", "data.components.synthetic", "SyntheticPairs"),
    ("data.pairs", "data.components.pairs", "PairedAudio"),
    ("data.augmented", "data.components.augmented", "AugmentedAudio"),
    ("data.replay", "data.components.replay", "ReplayRandom"),
    ("data.manifest", "data.components.manifest", "read_manifest"),
]


@pytest.mark.parametrize("old,new,name", ALIASES)
def test_legacy_imports_resolve_to_the_same_objects(old, new, name):
    assert getattr(import_module(f"src.{old}"), name) is getattr(import_module(f"src.{new}"), name)


def test_network_checkpoint_schema_and_recipe_are_unchanged():
    model = NCSNppVelocity(nf=8, channel_mult=(1, 2), blocks=1, context_width=16)
    # Captured from the pre-migration implementation, before relocating any code.
    signature = hashlib.sha256("\n".join(model.state_dict()).encode()).hexdigest()
    assert signature == "92d80d467b1a95bec1c9ace60226eb542eb2b05f0c0c25066bd025a3daaae2b1"
    assert model.recipe == dict(backbone="sgmse-ncsnpp-v2", nf=8, channel_mult=[1, 2],
        commit="1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e", blocks=1, context_width=16,
        normalize_time=True, adapter="two-time-positional-nat-film-v1", fir="upstream-native")


def test_saved_legacy_hydra_network_target_loads_canonical_weights():
    saved = OmegaConf.create(dict(_target_="src.models.network.NCSNppVelocity",
                                  nf=8, channel_mult=[1, 2], blocks=1, context_width=16))
    legacy = instantiate(saved)
    canonical = NCSNppVelocity(nf=8, channel_mult=(1, 2), blocks=1, context_width=16)
    canonical.load_state_dict(legacy.state_dict(), strict=True)
    assert canonical.recipe == legacy.recipe


def test_datamodule_checkpoint_identity_and_hyperparameters_are_unchanged():
    data = instantiate(OmegaConf.create(dict(_target_="src.data.module.PairedDataModule",
                       train_manifest="train.jsonl", val_manifest="val.jsonl")))
    assert type(data).__qualname__ == "PairedDataModule"
    assert dict(data.hparams) == dict(train_manifest="train.jsonl", val_manifest="val.jsonl",
                                    batch_size=4, samples=64000, workers=0, seed=42)
    assert data.state_dict() == {}
