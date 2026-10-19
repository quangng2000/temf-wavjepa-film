import numpy as np
import torch
from torch import nn

from src.audio.nat import FrozenNat


class MockNat(nn.Module):
    @torch.inference_mode()
    def forward(self, audio):
        assert audio.shape == (2, 2, 1000)
        torch.testing.assert_close(audio[:, 0], audio[:, 1])
        features = torch.zeros(2, 2, 3, 768)
        features[:, 0] = 1
        features[:, 1] = 2
        return features, torch.tensor([[0., 10., 20.]]).repeat(2, 1)


def test_nat_channel_contract_and_trainable_adapter_gradients():
    model = FrozenNat()
    model.encoder = MockNat()
    model.processor = lambda clips, **_: {"input_values": torch.tensor(np.array(clips))}
    audio = torch.randn(2, 1000)
    original = audio.clone()
    context = model(audio, torch.tensor([0., 0.005, 0.02]))
    assert context.shape == (2, 3, 1536) and not torch.is_inference(context)
    torch.testing.assert_close(context[..., :768], torch.ones(2, 3, 768))
    torch.testing.assert_close(context[..., 768:], torch.full((2, 3, 768), 2.))
    adapter = nn.Linear(1536, 4)
    adapter(context).square().mean().backward()
    assert adapter.weight.grad.abs().sum() > 0 and not context.requires_grad
    torch.testing.assert_close(audio, original)
    model.train()
    assert not model.training and not model.encoder.training
