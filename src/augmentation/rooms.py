"""Paired RIRs from the same pyroomacoustics geometry; SciPy performs convolution."""

from functools import lru_cache

import numpy as np
from scipy.signal import fftconvolve


def simulate_room(seed, max_order=6):
    import pyroomacoustics as pra

    rng = np.random.default_rng(seed)
    dimensions = rng.uniform([3, 3, 2.4], [7, 6, 3.5])
    source, microphone = [rng.uniform(0.5, dimensions - 0.5) for _ in range(2)]
    absorption = float(rng.uniform(0.3, 0.7))
    scene = dict(dimensions=dimensions, source=source, microphone=microphone,
                 absorption=absorption)
    filters = [room_filter(pra, scene, order) for order in (0, max_order)]
    scale = np.linalg.norm(filters[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Room has no finite direct path")
    return dict(direct=filters[0] / scale, full=filters[1] / scale,
                offset=int(np.argmax(np.abs(filters[0]))), sample_rate=16000,
                version=1, seed=seed, max_order=max_order, **scene)


def room_filter(pra, scene, order):
    room = pra.ShoeBox(scene["dimensions"], fs=16000, max_order=order,
                       materials=pra.Material(scene["absorption"]), use_rand_ism=False)
    room.add_source(scene["source"])
    room.add_microphone(scene["microphone"])
    room.compute_rir()
    return np.asarray(room.rir[0][0], dtype=np.float32)


@lru_cache(maxsize=16)
def load_room(path):
    with np.load(path, allow_pickle=False) as saved:
        direct, full, offset = saved["direct"], saved["full"], int(saved["offset"])
        if int(saved["version"]) != 1 or int(saved["sample_rate"]) != 16000:
            raise ValueError("Expected version-1 16 kHz room bank")
    for impulse in (direct, full):
        if impulse.ndim != 1 or not len(impulse) or not np.isfinite(impulse).all():
            raise ValueError("Room filters must be finite nonempty vectors")
    if not 0 <= offset < len(direct) or not np.any(direct) or not np.any(full):
        raise ValueError("Room filters have invalid direct-path offset or zero energy")
    return direct, full, offset


def apply_room(samples, path):
    direct, full, offset = load_room(path)
    # One common delay and gain for both branches; never align to the reflection peak.
    return tuple(fftconvolve(samples, impulse)[offset:offset + len(samples)].astype(np.float32)
                 for impulse in (direct, full))
