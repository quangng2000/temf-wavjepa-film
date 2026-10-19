"""Compute one real toy update and solver trace for the explanatory animation."""
import json
from pathlib import Path
import torch
from toy_model import ToyVelocity, sample


def values(item):
    if torch.is_tensor(item):
        return item.detach().flatten().tolist() if item.numel() > 1 else item.item()
    if isinstance(item, dict):
        return {key: values(value) for key, value in item.items()}
    if isinstance(item, list):
        return [values(value) for value in item]
    return item


def build():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(17)
    frequency, frame = torch.meshgrid(torch.arange(12), torch.arange(24), indexing='ij')
    clean = (torch.exp(-frequency * .11) * (.15 + .8 * torch.sin(frame * .36).square())
             * torch.sin(frame * .55 + frequency * .43))[None, None]
    observed = .72 * clean + .12 * torch.randn_like(clean)
    prior = .6 * torch.randn_like(clean)
    frozen_projection = torch.randn(4, 12) / 12 ** .5
    context = torch.tanh(torch.einsum('cf,bft->bct', frozen_projection, observed[:, 0]))
    model = ToyVelocity().double()
    t, r = clean.new_tensor(.8), clean.new_tensor(.2)
    state, analytic = (1 - t) * clean + t * prior, prior - clean
    before = model.inspect(state, observed, t, r, context)
    fn = lambda x, a, b: model(x, observed, a, b, context)
    diagonal = fn(state, t, t)
    mixed = (.2 * analytic + .8 * diagonal).detach()
    args, direction = (state, t, r), (mixed, t.new_tensor(1), r.new_tensor(0))
    prediction, jvp = torch.func.jvp(fn, args, direction)
    _, jx = torch.func.jvp(fn, args, (mixed, t.new_tensor(0), r.new_tensor(0)))
    _, jt = torch.func.jvp(fn, args, (torch.zeros_like(state), t.new_tensor(1), r.new_tensor(0)))
    epsilon = 1e-4
    nudged = fn(state + epsilon * mixed, t + epsilon, r)
    difference = (nudged - fn(state - epsilon * mixed, t - epsilon, r)) / (2 * epsilon)
    target = (mixed - (t - r) * jvp).detach()
    weight = 1 / ((1 + (t - r)) * (1 + (t - r).square()).sqrt())
    error_before = weight * (prediction - target).square()
    loss = error_before.mean()
    snapshots = {name: p.detach().clone() for name, p in model.named_parameters()}
    loss.backward()
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(parameter.grad, alpha=-.003)
    after = model.inspect(state, observed, t, r, context)
    error_after = weight * (after['velocity'] - target).square()
    states, steps = sample(model, observed, context, prior)
    checks = dict(jvp_fd_max_error=(jvp-difference).abs().max().item(),
                  jvp_decomposition_error=(jvp-jx-jt).abs().max().item(),
                  fixed_target_loss_before=loss.item(), fixed_target_loss_after=error_after.mean().item(),
                  sampler_clean_target_argument=False, finite=all(torch.isfinite(x).all().item() for x in states))
    assert checks['jvp_fd_max_error'] < 1e-6 and checks['jvp_decomposition_error'] < 1e-10
    assert checks['fixed_target_loss_after'] < checks['fixed_target_loss_before'] and checks['finite']
    for index, step in enumerate(steps):
        assert torch.equal(states[index] + step['delta'], states[index+1])
    return values(dict(shape=[12,24], tracked=[5,10], clean=clean, observed=observed,
                       prior=prior, context=context, t=t, r=r, state=state, before=before, after=after,
                       diagonal=diagonal, mixed=mixed, jvp=jvp, jx=jx, jt=jt, nudged=nudged,
                       epsilon=epsilon, target=target, weight=weight, error_before=error_before,
                       error_after=error_after, weights_before=snapshots, gradients=gradients,
                       weights_after={name:p for name,p in model.named_parameters()},
                       sampler_states=states, sampler_steps=steps, checks=checks))


if __name__ == '__main__':
    trace = build()
    Path(__file__).with_name('trace.json').write_text(json.dumps(trace, separators=(',', ':')) + '\n')
    print(json.dumps(trace['checks'], indent=2))
