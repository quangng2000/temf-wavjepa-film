import torch

from src.flow.objective import TEMFObjective
from src.models.blocks import ResidualBlock
from src.models.network import VelocityUNet


def test_full_meanflow_backpropagates_through_network_and_film():
    model = VelocityUNet((8, 16), (1, 1), 16, 32, 4)
    clean = torch.randn(2, 2, 17, 5) * 0.1
    observed, context = clean * 0.8, torch.randn(2, 5, 16)
    loss, metrics = TEMFObjective(100)(model, (clean, observed, context), 50)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert model.encoders[0].blocks[0].film.weight.grad.abs().sum() > 0
    assert model.time.mlp[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_zero_film_heads_start_with_identity_modulation():
    block = ResidualBlock(8, 32, 16, 4)
    controls = block.film(block.adapter(torch.randn(2, 5, 16)))
    torch.testing.assert_close(controls, torch.zeros_like(controls))


def test_default_model_is_approximately_300m_without_allocating_weights():
    with torch.device("meta"):
        model = VelocityUNet()
    parameters = sum(parameter.numel() for parameter in model.parameters()
                     if parameter.requires_grad)
    assert 290_000_000 <= parameters <= 310_000_000
