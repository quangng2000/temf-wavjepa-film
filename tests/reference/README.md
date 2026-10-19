# Numeric TEMF reference

`temf.json` was generated on 2026-10-19 by executing the isolated function bodies
from the pinned authors' `src/training/loss.py` (URL and SHA-256 are in the file):

- `TEMFLoss.compute_temporal_equilibrium_weight`, using the released defaults.
- `AlphaFlowLoss.get_ratio`, using boundary sigmoid and velocity-mix linear schedules
  from `configs/loss/temf.yaml`, with total updates set to 100.
- `AlphaFlowLoss._compute_mean_velocity_c`, including its `isclose` boundary mask
  and `torch.autograd.functional.jvp` call.

The runtime type-check decorator was removed; function bodies were unchanged.
Only these methods were executed, with minimal configuration objects, not imports
of the entire TEMF training stack. Computation used PyTorch 2.8.0, float64 on CPU.

The target oracle used `0.2*x² + t² + 3*r + 0.1*condition`, `condition=x` held fixed
within the JVP, and reference velocity 0.7. The 32 state values are evenly spaced
from -1 to 1. Upstream shape `[4,1,2,2,2]` maps to audio shape `[4,2,2,2]` by removing
one singleton axis. Cases cover intervals, exact diagonal and near diagonal.

This checks the selected continuous, no-CFG branch. It does not establish parity
for other TEMF settings, distributed training, image architectures or quality.
