import copy

import pytest
import torch

from src.flow.interface import AudioCondition, AudioVelocity
from src.flow.objective import TEMFObjective
from src.upstream import source
from src.upstream.temf import create_temf
from tests.reference.legacy_objective import TEMFObjective as LegacyObjective
from tests.reference.legacy_target import corrected_target
from tests.test_target import AnalyticVelocity


@pytest.mark.parametrize("step", [0, 50, 100])
def test_original_forward_matches_legacy_loss_and_parameter_gradients(step):
    net = AnalyticVelocity().float()
    baseline = copy.deepcopy(net)
    clean = torch.randn(4, 2, 2, 3) * .1
    batch = (clean, clean * .7, torch.randn(4, 3, 16))
    objective, reference = TEMFObjective(100), LegacyObjective(100)
    torch.manual_seed(777)
    loss, _ = objective(net, batch, step)
    torch.manual_seed(777)
    expected, _ = reference(baseline, batch, step)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(loss, expected, rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(net.scale.grad, baseline.scale.grad, rtol=2e-6, atol=1e-6)


def test_original_target_matches_analytic_reference_at_diagonal_and_endpoints():
    net, objective = AnalyticVelocity(), create_temf(100)
    state = torch.randn(4, 2, 2, 2, dtype=torch.float64)
    current = torch.tensor([.8, .6, .4, 1.], dtype=torch.float64)
    destination = torch.tensor([.1, .6, .4 - 1e-8, 0.], dtype=torch.float64)
    reference = torch.full_like(state, .7)
    expected, _, _ = corrected_target(net, state, (current, destination),
                                      (state, state), reference)
    actual = objective._compute_mean_velocity_c(state[:, None], destination, current,
        reference[:, None], AudioCondition(state, state), None, AudioVelocity(net))
    torch.testing.assert_close(actual.squeeze(1), expected, rtol=1e-14, atol=1e-14)
    assert not actual.requires_grad


def test_cached_source_tampering_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "ROOT", tmp_path)
    path = tmp_path / "src/training/loss.py"
    path.parent.mkdir(parents=True)
    path.write_text("raise RuntimeError('must never execute')")
    with pytest.raises(ValueError, match="checksum"):
        source.read_source("src/training/loss.py")
