"""Speech intelligibility measured by TorchMetrics/pystoi, not a custom formula."""
from torch import nn
from torchmetrics.audio import ShortTimeObjectiveIntelligibility


class SpeechMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.restored = ShortTimeObjectiveIntelligibility(fs=16000)
        self.observed = ShortTimeObjectiveIntelligibility(fs=16000)

    def update(self, restored, observed, clean):
        self.restored.update(restored, clean)
        self.observed.update(observed, clean)
