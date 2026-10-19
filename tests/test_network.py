import copy

import torch
from torch.overrides import TorchFunctionMode

from src.flow.objective import TEMFObjective
from src.models.film import ConditionedBlock, NatFiLM
from src.models.network import NCSNppVelocity
from src.vendor.sgmse.ncsnpp_utils.layers import get_timestep_embedding


def tiny_network(normalize=True):
    return NCSNppVelocity(nf=8, channel_mult=(1, 2), blocks=1,
                          context_width=16, normalize_time=normalize)


def test_full_temf_backpropagates_through_upstream_network_and_film():
    model = tiny_network()
    clean = torch.randn(2, 2, 17, 5) * 0.1
    observed, context = clean * 0.8, torch.randn(2, 5, 16)
    loss, metrics = TEMFObjective(100)(model, (clean, observed, context), 50)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    blocks = [block for block in model.modules() if isinstance(block, ConditionedBlock)]
    assert any(block.film.head.weight.grad.abs().sum() > 0 for block in blocks)
    assert model.backbone.all_modules[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_zero_film_is_identity_and_adaptation_preserves_upstream_forward():
    model = tiny_network(normalize=False)
    upstream = copy.deepcopy(model.backbone)
    upstream.all_modules = torch.nn.ModuleList(
        block.block if isinstance(block, ConditionedBlock) else block
        for block in upstream.all_modules)
    state = torch.randn(2, 2, 18, 6)
    context = torch.randn(2, 6, 16)
    current, destination = torch.tensor([.8, .6]), torch.tensor([.2, .1])
    embedding = torch.cat([get_timestep_embedding(time, 4)
                           for time in (current, destination)], dim=1)
    complex_state = torch.complex(state[:, 0], state[:, 1])[:, None]
    expected = upstream(complex_state, complex_state, current, embedding=embedding)
    actual = model(state, current, destination, (state, context))
    torch.testing.assert_close(actual, torch.cat((expected.real, expected.imag), dim=1))
    film = NatFiLM(8, 16)
    value = torch.randn(2, 8, 9, 3)
    torch.testing.assert_close(film(value, context), value)


def test_padding_and_both_flow_times_are_differentiable():
    model = tiny_network()
    state = torch.randn(2, 2, 17, 5, requires_grad=True)
    current = torch.tensor([.8, .6], requires_grad=True)
    destination = torch.tensor([.2, .1], requires_grad=True)
    output = model(state, current, destination, (state.detach(), torch.randn(2, 5, 16)))
    output.square().sum().backward()
    assert output.shape == state.shape
    assert current.grad.abs().sum() > 0 and destination.grad.abs().sum() > 0
    assert torch.isfinite(state.grad).all()


class MetaFactories(TorchFunctionMode):
    def __torch_function__(self, function, types, args=(), kwargs=None):
        options = dict(kwargs or {})
        if "device" in options:
            options["device"] = "meta"
        return function(*args, **options)


def test_large_configuration_parameter_count_without_allocating_weights():
    with torch.device("meta"), MetaFactories():
        model = NCSNppVelocity()
    assert sum(parameter.numel() for parameter in model.parameters()) == 325497126
