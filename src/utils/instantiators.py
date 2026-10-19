"""Instantiate optional Hydra callback and logger collections."""

from hydra.utils import instantiate
from omegaconf import DictConfig


def _instantiate_collection(config):
    if config is None:
        return []
    if not isinstance(config, DictConfig):
        raise TypeError("Expected a Hydra config mapping or null")
    # Existing saved configurations used one logger directly, without a group key.
    if "_target_" in config:
        return [instantiate(config)]
    instances = []
    for name, entry in config.items():
        if entry is None:
            continue
        if not isinstance(entry, DictConfig) or "_target_" not in entry:
            raise ValueError(f"Config entry {name!r} needs a _target_ or null")
        instances.append(instantiate(entry))
    return instances


def instantiate_callbacks(config):
    """Build callbacks from a named config group, omitting disabled entries."""
    return _instantiate_collection(config)


def instantiate_loggers(config):
    """Build one or more loggers; accept the earlier single-logger config too."""
    return _instantiate_collection(config)
