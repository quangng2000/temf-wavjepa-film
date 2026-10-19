"""Evaluate a trusted checkpoint on the configured validation split, without fitting."""
# Root discovery must precede project imports for the direct-script entry point.
# ruff: noqa: E402

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.sample import load_model
from src.utils import extras, log_hyperparameters, task_wrapper
from src.utils.setup import build_objects, prepare_data


@task_wrapper
def evaluate(cfg: DictConfig):
    """Return validation metrics for explicit raw/EMA weights; never run an optimizer."""
    if not cfg.ckpt_path:
        raise ValueError("Evaluation requires an explicit ckpt_path")
    data = prepare_data(cfg, "validate")
    model, _ = load_model(cfg.ckpt_path, cfg.weights, "cpu")
    is_fixture = model.conditioner.recipe["model"] == "synthetic_fixture"
    if cfg.fixture != is_fixture:
        raise ValueError("Checkpoint conditioning and the data fixture flag disagree")
    objects = build_objects(cfg, data, model, training=False)
    log_hyperparameters(objects)
    trainer = objects["trainer"]
    trainer.validate(model, datamodule=data)
    return dict(trainer.callback_metrics), objects


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
