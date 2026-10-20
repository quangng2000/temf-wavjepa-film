"""Load original TEMFLoss forward, schedules, targets and weights."""
import copy
import math
from functools import lru_cache
from types import SimpleNamespace

import einops
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.upstream.derivatives import resolve_jvp
from src.upstream.selection import METHODS, execute_definitions, select_definitions
from src.upstream.source import ROOT, read_source


def load_temf_class(jvp_backend="reverse"):
    return _load_temf_class(jvp_backend)


@lru_cache(maxsize=2)
def _load_temf_class(jvp_backend):
    namespace = dict(torch=torch, math=math, einops=einops, copy=copy,
                     DictConfig=DictConfig, ListConfig=ListConfig,
                     BaseLoss=torch.nn.Module, LossPhase=SimpleNamespace(Gen="gen"),
                     jvp=resolve_jvp(jvp_backend))
    nodes = select_definitions("src/structs.py", {"EasyDict": None})
    execute_definitions(nodes, namespace, str(ROOT / "src/structs.py"))
    helpers = {key: None for key in ("randn", "unwrap_module", "filter_nones")}
    nodes = select_definitions("src/utils/misc.py", helpers)
    execute_definitions(nodes, namespace, str(ROOT / "src/utils/misc.py"))
    namespace["misc"] = SimpleNamespace(**{key: namespace[key] for key in helpers})
    functions = {"sample_logit_norm_time": None, "maybe_augment_videos": None}
    nodes = select_definitions("src/training/loss.py", METHODS | functions)
    execute_definitions(nodes, namespace, str(ROOT / "src/training/loss.py"))
    return namespace["TEMFLoss"]


def create_temf(total_updates, jvp_backend="reverse"):
    cfg = OmegaConf.create(read_source("configs/loss/temf.yaml"))
    del cfg["defaults"]
    cfg.training = dict(max_steps=total_updates)
    cfg.model = dict(is_lgm=False, label_dropout=0.0, sampling=dict(sigma_noise=1.0))
    cfg.use_single_step_denoising = False
    OmegaConf.resolve(cfg)
    return load_temf_class(jvp_backend)(cfg)
