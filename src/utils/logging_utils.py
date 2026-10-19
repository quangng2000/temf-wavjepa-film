"""Record the resolved experiment config and parameter counts in each logger."""

from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf


@rank_zero_only
def log_hyperparameters(objects):
    """Log experiment metadata once, before entering the Lightning loop."""
    if not objects["logger"]:
        return
    model = objects["model"]
    metadata = OmegaConf.to_container(objects["cfg"], resolve=True)
    # Evaluation instantiates the saved model recipe, not the training defaults.
    metadata["model"] = model.run_config["model"]
    metadata["parameters/restoration"] = sum(p.numel() for p in model.net.parameters())
    metadata["parameters/restoration_trainable"] = sum(
        p.numel() for p in model.net.parameters() if p.requires_grad
    )
    metadata["checkpoint_recipe"] = model.recipe
    for logger in objects["logger"]:
        logger.log_hyperparams(metadata)
