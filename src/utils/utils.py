"""Small task, configuration, and optimization helpers for Hydra entry points."""

import logging
import warnings
from functools import wraps
from pathlib import Path
from traceback import format_exc

from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


@rank_zero_only
def _record_failure(cfg):
    log.exception("Experiment failed")
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "error.log").write_text(format_exc())


def task_wrapper(task):
    """Record a failed task without suppressing its exception or retrying it."""
    @wraps(task)
    def wrapped(cfg):
        try:
            return task(cfg)
        except Exception:
            try:
                _record_failure(cfg)
            except Exception:
                log.exception("Could not write the task failure record")
            raise
        finally:
            log.info("Experiment finished; see the configured output directory")
    return wrapped


@rank_zero_only
def extras(cfg):
    """Apply requested extras; never prompt during unattended jobs."""
    options = cfg.get("extras")
    if not options:
        return
    if options.get("ignore_warnings", False):
        warnings.filterwarnings("ignore")
    if options.get("enforce_tags", False) and not cfg.get("tags"):
        raise ValueError("Supply tags=[name] when extras.enforce_tags=true")
    if options.get("print_config", False):
        print(OmegaConf.to_yaml(cfg, resolve=True))


def get_metric_value(metrics, name):
    """Return an explicitly requested sweep metric, failing if it was not logged."""
    if not name:
        return None
    if name not in metrics:
        raise ValueError(f"Optimized metric {name!r} was not logged")
    value = metrics[name]
    return float(value.item() if hasattr(value, "item") else value)
