"""Train a configured experiment with the Lightning-Hydra template task contract."""
# Root discovery must precede project imports for `python /path/to/src/train.py`.
# ruff: noqa: E402

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import extras, get_metric_value, log_hyperparameters, task_wrapper
from src.utils.setup import build_training_objects


@task_wrapper
def train(cfg: DictConfig):
    """Return logged metrics and instantiated objects for tests and Hydra sweeps."""
    objects = build_training_objects(cfg)
    log_hyperparameters(objects)
    trainer = objects["trainer"]
    if cfg.get("train", True):
        trainer.fit(objects["model"], datamodule=objects["datamodule"], ckpt_path=cfg.ckpt_path)
    return dict(trainer.callback_metrics), objects


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    extras(cfg)
    metrics, _ = train(cfg)
    return get_metric_value(metrics, cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
