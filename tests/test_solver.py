import pytest
import torch

from src.flow.solver import solve


@pytest.mark.parametrize("nfe", [1, 5, 20])
def test_oracle_solver_reaches_clean_endpoint(nfe):
    clean = torch.randn(2, 2, 3, 5)

    def velocity(state, current, destination, condition):
        return (state - clean) / current[:, None, None, None]

    restored = solve(velocity, (torch.zeros_like(clean), torch.zeros(2, 5, 1)), nfe=nfe)
    torch.testing.assert_close(restored, clean)


def test_seed_reproduces_identical_initial_state():
    observed = torch.zeros(1, 2, 3, 4)
    def zero(state, *args):
        return torch.zeros_like(state)

    first = solve(zero, (observed, observed), seed=17)
    torch.testing.assert_close(first, solve(zero, (observed, observed), seed=17))
    assert not torch.equal(first, solve(zero, (observed, observed), seed=18))
