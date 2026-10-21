"""Backend isolation, full derivative semantics, and unchanged TEMF gradients."""
import copy

import pytest
import torch

from src.flow.interface import AudioCondition, AudioVelocity
from src.flow.prepared import prepare_batch
from src.models.components.network import NCSNppVelocity
from src.upstream.derivatives import forward_ad_jvp, resolve_jvp
from src.upstream.prepared import load_prepared_forward
from src.upstream.temf import create_temf, load_temf_class
from tests.test_target import AnalyticVelocity


def test_backend_namespaces_are_cached_separately_without_mutating_authors_default():
    reverse, forward = load_temf_class(), load_temf_class("forward_ad")
    assert reverse is load_temf_class("reverse") and reverse is not forward
    assert forward is load_temf_class("forward_ad")
    methods = [model._compute_mean_velocity_c.__wrapped__ for model in (reverse, forward)]
    assert methods[0].__globals__ is not methods[1].__globals__
    assert methods[0].__globals__["jvp"] is torch.autograd.functional.jvp
    assert methods[1].__globals__["jvp"] is forward_ad_jvp
    with pytest.raises(ValueError, match="Unsupported TEMF JVP backend"):
        create_temf(300000, "unknown")


def test_forward_derivative_keeps_state_and_time_terms_and_detaches_parameters():
    scale = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))
    state = torch.tensor([.4, .8], dtype=torch.float64, requires_grad=True)
    current, destination, velocity = state + .1, state * .1, state + .3
    def function(x, r, t):
        return scale * x.square() + t.square() + 3 * r
    inputs = state, destination, current
    tangents = velocity, torch.zeros_like(destination), torch.ones_like(current)
    expected = resolve_jvp("reverse")(function, inputs, tangents)
    actual = forward_ad_jvp(function, inputs, tangents)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=1e-12, rtol=1e-12)
        assert not value.requires_grad and value.grad_fn is None
    torch.testing.assert_close(actual[1], .4 * state * velocity + 2 * current)
    assert scale.grad is None and state.grad is None


def test_independent_output_returns_zero_tangent():
    value = torch.ones(2, requires_grad=True)
    primal, tangent = forward_ad_jvp(lambda x: value, (value,), (value,))
    torch.testing.assert_close(primal, value)
    torch.testing.assert_close(tangent, torch.zeros_like(value))
    assert not primal.requires_grad and not tangent.requires_grad


def make_case(kind):
    torch.manual_seed(120)
    dtype = torch.float64 if kind == "analytic" else torch.float32
    model = AnalyticVelocity() if kind == "analytic" else NCSNppVelocity(
        nf=8, channel_mult=(1, 2), blocks=1, context_width=16)
    if kind == "cnn":
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * .015)
    clean = torch.randn(8, 2, 17, 5, dtype=dtype) * .1
    return model, (clean, clean * .7, torch.randn(8, 5, 16, dtype=dtype))


def evaluate(model, batch, prepared, backend):
    objective, targets = create_temf(300000, backend), []
    original = objective._compute_mean_velocity_c
    def capture(*args):
        target = original(*args)
        assert not target.requires_grad and target.grad_fn is None
        targets.append(target)
        return target
    objective._compute_mean_velocity_c = capture
    result = load_prepared_forward(backend)(objective, AudioVelocity(model), batch[0][:, None],
        AudioCondition(*batch[1:]), cur_step=prepared.step, prepared=prepared)
    loss = result["total"].mean()
    loss.backward()
    gradients = torch.cat([value.grad.flatten() for value in model.parameters()
                           if value.grad is not None])
    return loss.detach(), targets[0], gradients


@pytest.mark.parametrize("kind", ["analytic", "cnn"])
@pytest.mark.parametrize("step", [0, 150000, 299999])
def test_prepared_full_temf_target_loss_and_gradient_match_reverse(kind, step):
    model, batch = make_case(kind)
    prepared = prepare_batch(create_temf(300000), batch[0][:, None], step)
    expected = evaluate(copy.deepcopy(model), batch, prepared, "reverse")
    actual = evaluate(copy.deepcopy(model), batch, prepared, "forward_ad")
    tolerances = dict(atol=2e-5, rtol=5e-4) if kind == "cnn" else dict(atol=1e-10, rtol=1e-8)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, **tolerances)


def test_prepared_forward_uses_requested_jvp_namespace():
    reverse = load_prepared_forward("reverse").__globals__["jvp"]
    forward = load_prepared_forward("forward_ad").__globals__["jvp"]
    assert reverse is torch.autograd.functional.jvp
    assert forward is forward_ad_jvp
