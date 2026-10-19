"""Shared experiment construction; keep data validation before model allocation."""

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.upstream.source import HASHES, read_source
from src.utils import instantiate_callbacks, instantiate_loggers


def validate_config(cfg):
    if cfg.fixture != ("Synthetic" in cfg.data._target_):
        raise ValueError("Synthetic data must be explicitly marked fixture=true")
    if cfg.fixture != ("Fixture" in cfg.model.conditioner._target_):
        raise ValueError("Fixture conditioning is allowed only in a synthetic smoke run")
    if cfg.trainer.max_steps > cfg.total_updates or cfg.trainer.max_steps < 1:
        raise ValueError("max_steps must be positive and no greater than total_updates")


def prepare_data(cfg, stage):
    validate_config(cfg)
    if cfg.seed is not None:
        L.seed_everything(cfg.seed, workers=True)
    torch.set_num_threads(cfg.cpu_threads)
    data = instantiate(cfg.data)
    data.setup(stage)
    for name in HASHES:
        read_source(name)
    return data


def build_objects(cfg, data, model, training=True):
    callbacks = instantiate_callbacks(cfg.callbacks) if training else []
    loggers = instantiate_loggers(cfg.get("logger"))
    options = dict(callbacks=callbacks, logger=loggers or False)
    if not training:
        options["enable_checkpointing"] = False
    trainer = instantiate(cfg.trainer, **options)
    return dict(cfg=cfg, datamodule=data, model=model, callbacks=callbacks,
                logger=loggers, trainer=trainer)


def build_training_objects(cfg):
    data = prepare_data(cfg, "fit")
    model = instantiate(cfg.model)
    model.run_config = OmegaConf.to_container(cfg, resolve=True)
    return build_objects(cfg, data, model)
