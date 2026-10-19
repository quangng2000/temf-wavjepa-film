"""Linear interpolation on encoder timestamps, clamped at the available boundaries."""

import torch


def align_features(features, timestamps, positions):
    timestamps = timestamps.to(features).contiguous()
    if features.ndim != 3 or timestamps.shape != features.shape[:2] or features.shape[1] < 1:
        raise ValueError("Feature/timestamp shape mismatch or empty feature sequence")
    if not torch.isfinite(timestamps).all() or (timestamps.diff(dim=1) <= 0).any():
        raise ValueError("Timestamps must be finite and strictly increasing")
    positions = positions.to(features).expand(len(features), -1).contiguous()
    right = torch.searchsorted(timestamps, positions).clamp(0, features.shape[1] - 1)
    left = (right - 1).clamp_min(0)
    start, end = timestamps.gather(1, left), timestamps.gather(1, right)
    weight = ((positions - start) / (end - start).clamp_min(1e-12)).clamp(0, 1)
    def gather(indices):
        return features.gather(1, indices[..., None].expand(-1, -1, features.shape[-1]))

    return torch.lerp(gather(left), gather(right), weight[..., None])
