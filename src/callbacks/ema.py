"""Lightning lifecycle adapter; PyTorch owns all averaging and EMA state."""
from lightning import Callback
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class EMA(Callback):
    def __init__(self, decay=0.999):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay, self.averaged, self.pending = decay, None, None
        self.last_step = 0

    @property
    def state_key(self):
        return "restoration_ema"

    def on_fit_start(self, trainer, module):
        self.averaged = AveragedModel(module.net,
            multi_avg_fn=get_ema_multi_avg_fn(self.decay), use_buffers=False)
        if self.pending is not None:
            self.averaged.load_state_dict(self.pending, strict=True)
            self.pending = None
        else:
            # Seed the initial model so the first optimizer update uses decay too.
            self.averaged.update_parameters(module.net)
        self.last_step = trainer.global_step

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step != self.last_step:
            self.averaged.update_parameters(module.net)
            self.last_step = trainer.global_step

    def state_dict(self):
        return dict(format="torch-averaged-model-v1", decay=self.decay,
                    averaged=self.averaged.state_dict(), last_step=self.last_step)

    def load_state_dict(self, state_dict):
        if state_dict.get("format") != "torch-averaged-model-v1":
            raise ValueError("Old EMA format requires explicit migration")
        if self.decay != state_dict["decay"]:
            raise ValueError("EMA decay changed on resume")
        self.pending, self.last_step = state_dict["averaged"], state_dict["last_step"]
