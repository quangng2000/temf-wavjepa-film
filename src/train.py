"""Lightning/Hydra entry point following ashleve/lightning-hydra-template's layout."""

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def validate_config(cfg):
    if cfg.fixture != ("Synthetic" in cfg.data._target_):
        raise ValueError("Synthetic data must be explicitly marked fixture=true")
    synthetic_encoder = "Fixture" in cfg.model.conditioner._target_
    if cfg.fixture != synthetic_encoder:
        raise ValueError("Fixture conditioning is allowed only in a synthetic smoke run")
    if cfg.trainer.max_steps > cfg.total_updates or cfg.trainer.max_steps < 1:
        raise ValueError("max_steps must be positive and no greater than total_updates")


def train(cfg):
    validate_config(cfg)
    L.seed_everything(cfg.seed, workers=True)
    torch.set_num_threads(cfg.cpu_threads)
    data = instantiate(cfg.data)
    data.setup("fit")  # Validate manifests before model allocation/download.
    model = instantiate(cfg.model)
    model.run_config = OmegaConf.to_container(cfg, resolve=True)
    callbacks = [instantiate(value) for value in cfg.callbacks.values() if value is not None]
    logger = instantiate(cfg.logger) if cfg.logger else False
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=data, ckpt_path=cfg.ckpt_path)
    return trainer, model


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
