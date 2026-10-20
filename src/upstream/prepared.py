"""Adapt checked sites in the pinned forward without changing its loss algebra."""
import ast
import copy
from functools import lru_cache

from src.upstream.selection import execute_definitions, select_definitions
from src.upstream.source import ROOT
from src.upstream.temf import load_temf_class

DRAW_REPLACEMENTS = {
    "self.sample_traj_params(x_aug.shape[0], cur_step, x_aug.device)": "prepared.trajectory",
    "torch.randn_like(x_aug)": "prepared.noise",
    "self.compute_temporal_equilibrium_weight(t, t_next)": "prepared.temporal_weight",
}

PATH_REPLACEMENTS = {
    "self.apply_noise(x_aug, noise_scaled, t, net)":
        "self.flow_path.state(x_aug, noise_unscaled, t)",
    "noise_unscaled - x_aug": "self.flow_path.velocity(x_aug, noise_unscaled, t)",
}


def replace_assignments(method, replacements):
    for original, replacement in replacements.items():
        expected = ast.dump(ast.parse(original, mode="eval").body)
        matches = [node for node in ast.walk(method) if isinstance(node, ast.Assign)
                   and ast.dump(node.value) == expected]
        if len(matches) != 1:
            raise ValueError(f"Pinned TEMF injection site changed: {original}")
        matches[0].value = ast.parse(replacement, mode="eval").body
    return method


@lru_cache(maxsize=None)
def load_adapted_forward(jvp_backend="reverse", prepared=True, flow_path="linear"):
    if flow_path not in {"linear", "trigflow"}:
        raise ValueError(f"Unsupported flow path: {flow_path}")
    definition = select_definitions("src/training/loss.py", {"TEMFLoss": ["forward"]})[0]
    method = copy.deepcopy(definition.body[0])
    suffix = []
    if prepared:
        method = replace_assignments(method, DRAW_REPLACEMENTS)
        method.args.kwonlyargs.append(ast.arg(arg="prepared"))
        method.args.kw_defaults.append(None)
        suffix.append("prepared")
    if flow_path != "linear":
        method = replace_assignments(method, PATH_REPLACEMENTS)
        suffix.append(flow_path)
    method.name = "forward_" + "_".join(suffix)
    namespace = dict(load_temf_class(jvp_backend).forward.__globals__)
    execute_definitions([method], namespace, str(ROOT / "src/training/loss.py"))
    return namespace[method.name]


def load_prepared_forward(jvp_backend="reverse", flow_path="linear"):
    return load_adapted_forward(jvp_backend, prepared=True, flow_path=flow_path)


def load_path_forward(jvp_backend="reverse", flow_path="trigflow"):
    if flow_path == "linear":
        return load_temf_class(jvp_backend).forward
    return load_adapted_forward(jvp_backend, prepared=False, flow_path=flow_path)
