"""Select unchanged method bodies without importing the upstream image stack."""
import ast

from src.upstream.source import read_source

METHODS = {
    "DiffusionLoss": ["__init__", "_maybe_encode"],
    "RecFlowLoss": ["apply_noise"],
    "AlphaFlowLoss": ["sample_timestep", "get_ratio", "sample_timesteps_mf",
                      "_get_teacher_prediction_weight", "_compute_teacher_prediction_loss",
                      "_compute_velocity_cfg", "_compute_mean_velocity_c",
                      "_compute_mean_velocity_d"],
    "TEMFLoss": None,
}


def select_definitions(name, selection):
    nodes = []
    for node in ast.parse(read_source(name)).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selection:
            if isinstance(node, ast.ClassDef) and selection[node.name] is not None:
                node.body = [child for child in node.body
                             if getattr(child, "name", None) in selection[node.name]]
            nodes.append(node)
    if {node.name for node in nodes} != set(selection):
        raise ValueError("Pinned upstream definitions are missing")
    return nodes


def execute_definitions(nodes, namespace, filename):
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = [item for item in node.decorator_list
                                       if not isinstance(item, ast.Name) or item.id != "beartype"]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, filename, "exec"), namespace)
