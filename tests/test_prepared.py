"""Effective-batch preparation preserves the pinned TEMF forward and gradients."""
import pytest
import torch

from src.flow.objective import TEMFObjective
from src.flow.prepared import prepare_batch
from tests.test_target import AnalyticVelocity


def make_batch():
    generator = torch.Generator().manual_seed(51)
    clean = torch.randn(8, 2, 2, 3, generator=generator, dtype=torch.float64) * .1
    context = torch.randn(8, 3, 16, generator=generator, dtype=torch.float64)
    return clean, clean * .7, context


@pytest.mark.parametrize("step", [0, 150000, 299999])
def test_prepared_forward_matches_original_authors_forward(step):
    objective, batch = TEMFObjective(), make_batch()
    original, prepared_net = AnalyticVelocity(), AnalyticVelocity()
    torch.manual_seed(777)
    expected, expected_metrics = objective(original, batch, step)
    torch.manual_seed(777)
    prepared = prepare_batch(objective.upstream, batch[0][:, None], step)
    actual, metrics = objective(prepared_net, batch, step, prepared=prepared)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(prepared_net.scale.grad, original.scale.grad,
                               rtol=1e-12, atol=1e-12)
    for key in expected_metrics:
        torch.testing.assert_close(metrics[key], expected_metrics[key], rtol=1e-12, atol=1e-12)


def accumulate_chunks(objective, net, batch, prepared, step):
    loss_sum = torch.tensor(0., dtype=torch.float64)
    for start in range(0, 8, 2):
        selection = slice(start, start + 2)
        chunk = tuple(value[selection] for value in batch)
        loss, _ = objective(net, chunk, step, prepared=prepared.slice(selection))
        (loss / 4).backward()
        loss_sum += loss.detach() / 4
    return loss_sum


@pytest.mark.parametrize("step", [0, 150000, 299999])
@pytest.mark.parametrize("normalize", [False, True])
def test_four_microbatches_match_full_loss_gradient_and_adamw_update(step, normalize):
    objective, batch = TEMFObjective(), make_batch()
    objective.upstream.cfg.temporal_equilibrium.normalize = normalize
    whole, chunks = AnalyticVelocity(), AnalyticVelocity()
    torch.manual_seed(999)
    prepared = prepare_batch(objective.upstream, batch[0][:, None], step)
    expected, _ = objective(whole, batch, step, prepared=prepared)
    expected.backward()
    actual = accumulate_chunks(objective, chunks, batch, prepared, step)
    torch.testing.assert_close(actual, expected.detach(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(chunks.scale.grad, whole.scale.grad, rtol=1e-12, atol=1e-12)
    for net in (whole, chunks):
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        torch.optim.AdamW(net.parameters(), lr=1e-4).step()
    torch.testing.assert_close(chunks.scale, whole.scale, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("step,count", [(0, 7), (299999, 1)])
def test_full_eight_sampler_keeps_boundary_count_and_exact_noise_slices(step, count):
    objective, batch = TEMFObjective(), make_batch()
    prepared = prepare_batch(objective.upstream, batch[0][:, None], step)
    current, destination, _, _ = prepared.trajectory
    assert torch.isclose(current, destination).sum().item() == count
    chunks = [prepared.slice(slice(start, start + 2)) for start in range(0, 8, 2)]
    torch.testing.assert_close(torch.cat([part.noise for part in chunks]), prepared.noise)
    torch.testing.assert_close(torch.cat([part.temporal_weight for part in chunks]),
                               prepared.temporal_weight)
    assert all(part.step == step for part in chunks)


def test_prepared_forward_rejects_another_schedule_step():
    objective, batch = TEMFObjective(), make_batch()
    prepared = prepare_batch(objective.upstream, batch[0][:, None], 0)
    with pytest.raises(ValueError):
        objective(AnalyticVelocity(), batch, 1, prepared=prepared)
