"""Exact TEMF target derivatives with an opt-in forward-mode implementation."""
import torch


@torch.no_grad()
def forward_ad_jvp(function, inputs, tangents):
    """Compute the full directional derivative and return stop-gradient targets."""
    with torch.autograd.forward_ad.dual_level():
        duals = tuple(torch.autograd.forward_ad.make_dual(value, tangent)
                      for value, tangent in zip(inputs, tangents, strict=True))
        output = function(*duals)
        primal, derivative = torch.autograd.forward_ad.unpack_dual(output)
    if derivative is None:
        derivative = torch.zeros_like(primal)
    return primal.detach(), derivative.detach()


def resolve_jvp(backend):
    """Select an explicit backend; never silently change the derivative method."""
    choices = {"reverse": torch.autograd.functional.jvp, "forward_ad": forward_ad_jvp}
    if backend not in choices:
        raise ValueError(f"Unsupported TEMF JVP backend: {backend}")
    return choices[backend]
