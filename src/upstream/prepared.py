"""Inject explicit random draws into the pinned forward; retain its loss algebra."""
import ast
import copy
from functools import lru_cache

from src.upstream.selection import execute_definitions, select_definitions
from src.upstream.source import ROOT
from src.upstream.temf import load_temf_class

REPLACEMENTS = {
    "self.sample_traj_params(x_aug.shape[0], cur_step, x_aug.device)": "prepared.trajectory",
    "torch.randn_like(x_aug)": "prepared.noise",
    "self.compute_temporal_equilibrium_weight(t, t_next)": "prepared.temporal_weight",
}


def replace_draws(method):
    for original, replacement in REPLACEMENTS.items():
        expected = ast.dump(ast.parse(original, mode="eval").body)
        matches = [node for node in ast.walk(method) if isinstance(node, ast.Assign)
                   and ast.dump(node.value) == expected]
        if len(matches) != 1:
            raise ValueError(f"Pinned TEMF injection site changed: {original}")
        matches[0].value = ast.parse(replacement, mode="eval").body
    return method


@lru_cache(maxsize=1)
def load_prepared_forward():
    definition = select_definitions("src/training/loss.py", {"TEMFLoss": ["forward"]})[0]
    method = replace_draws(copy.deepcopy(definition.body[0]))
    method.name = "forward_prepared"
    method.args.kwonlyargs.append(ast.arg(arg="prepared"))
    method.args.kw_defaults.append(None)
    namespace = dict(load_temf_class().forward.__globals__)
    execute_definitions([method], namespace, str(ROOT / "src/training/loss.py"))
    return namespace[method.name]
