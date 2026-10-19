"""Read-only target diagnostics around the authors' unchanged target method."""


class TargetObserver:
    def __init__(self, objective):
        self.original = objective._compute_mean_velocity_c
        self.metrics = {}
        objective._compute_mean_velocity_c = self

    def __call__(self, *args, **kwargs):
        target = self.original(*args, **kwargs)
        reference = args[3]
        self.metrics = dict(target_rms=target.square().mean().sqrt(),
                            correction_rms=(reference - target).square().mean().sqrt())
        return target
