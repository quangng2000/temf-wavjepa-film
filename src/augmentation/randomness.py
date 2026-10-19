"""Seed third-party NumPy/Python transforms without changing caller RNG state."""

import random
from contextlib import contextmanager

import numpy as np


@contextmanager
def seeded_transforms(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
