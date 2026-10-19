import torch
from pystoi import stoi

from src.audio.metrics import SpeechMetrics


def test_torchmetrics_matches_pystoi_for_restored_and_input():
    generator = torch.Generator().manual_seed(81)
    clean = torch.randn(2, 16000, generator=generator) * .1
    observed = clean + torch.randn(2, 16000, generator=generator) * .1
    metrics = SpeechMetrics()
    metrics.update(clean, observed, clean)
    expected = sum(stoi(c.numpy(), o.numpy(), 16000)
                   for c, o in zip(clean, observed)) / 2
    torch.testing.assert_close(metrics.restored.compute(), torch.tensor(1.), atol=1e-6, rtol=0)
    torch.testing.assert_close(metrics.observed.compute(), torch.tensor(expected).float())
    metrics.restored.reset()
    metrics.observed.reset()
    assert metrics.observed.total.item() == 0
