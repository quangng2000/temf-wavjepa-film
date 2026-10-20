"""Path geometry and opt-in TrigFlow integration checks."""
import copy
import math

import pytest
import torch

from src.flow.objective import TEMFObjective
from src.flow.path import LinearPath, NormalizedTrigFlowPath, create_flow_path
from src.flow.prepared import prepare_batch
from tests.test_target import AnalyticVelocity


def make_values():
    generator = torch.Generator().manual_seed(93)
    clean = torch.randn(4, 1, 2, 3, 5, generator=generator, dtype=torch.float64)
    noise = torch.randn(4, 1, 2, 3, 5, generator=generator, dtype=torch.float64)
    time = torch.linspace(0, 1, 4, dtype=torch.float64).view(4, 1, 1, 1, 1)
    return clean, noise, time


@pytest.mark.parametrize("path", [LinearPath(), NormalizedTrigFlowPath()])
def test_path_velocity_is_exact_time_derivative(path):
    clean, noise, time = make_values()
    _, tangent = torch.func.jvp(lambda value: path.state(clean, noise, value),
                                (time,), (torch.ones_like(time),))
    torch.testing.assert_close(tangent, path.velocity(clean, noise, time),
                               rtol=1e-12, atol=1e-12)


def test_trigflow_preserves_endpoints_and_unit_coefficient_energy():
    clean, noise, _ = make_values()
    path = NormalizedTrigFlowPath()
    zero, one = (torch.full((4, 1, 1, 1, 1), value, dtype=torch.float64)
                 for value in (0, 1))
    torch.testing.assert_close(path.state(clean, noise, zero), clean)
    torch.testing.assert_close(path.state(clean, noise, one), noise)
    time = torch.linspace(0, 1, 101, dtype=torch.float64)
    angle = time * math.pi / 2
    torch.testing.assert_close(angle.cos().square() + angle.sin().square(),
                               torch.ones_like(time), rtol=1e-12, atol=1e-12)


def make_batch():
    generator = torch.Generator().manual_seed(23)
    clean = torch.randn(8, 2, 2, 3, generator=generator, dtype=torch.float64) * .1
    context = torch.randn(8, 3, 16, generator=generator, dtype=torch.float64)
    return clean, clean * .7, context


@pytest.mark.parametrize("backend", ["reverse", "forward_ad"])
def test_prepared_trigflow_matches_unprepared_loss_and_gradient(backend):
    objective, batch = TEMFObjective(jvp_backend=backend, flow_path="trigflow"), make_batch()
    original, prepared_net = AnalyticVelocity(), AnalyticVelocity()
    torch.manual_seed(404)
    expected, expected_metrics = objective(original, batch, 150000)
    torch.manual_seed(404)
    prepared = prepare_batch(objective.upstream, batch[0][:, None], 150000)
    actual, metrics = objective(prepared_net, batch, 150000, prepared=prepared)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(prepared_net.scale.grad, original.scale.grad,
                               rtol=1e-12, atol=1e-12)
    for key in expected_metrics:
        torch.testing.assert_close(metrics[key], expected_metrics[key], rtol=1e-12, atol=1e-12)


def test_trigflow_changes_recipe_and_rejects_unknown_path():
    linear = TEMFObjective()
    trigflow = TEMFObjective(flow_path="trigflow")
    assert "path-trigflow-q-v1" not in linear.recipe
    assert "path-trigflow-q-v1" in trigflow.recipe
    assert copy.copy(trigflow.upstream.flow_path).name == "trigflow"
    with pytest.raises(ValueError, match="Unsupported flow path"):
        create_flow_path("unknown")
