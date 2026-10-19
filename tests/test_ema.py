from types import SimpleNamespace

import torch

from src.callbacks.ema import EMA


def test_library_ema_preserves_decay_initialization_and_resume():
    network = torch.nn.Linear(1, 1, bias=False)
    network.weight.data.fill_(2.)
    trainer, module = SimpleNamespace(global_step=0), SimpleNamespace(net=network)
    ema = EMA(decay=.9)
    ema.on_fit_start(trainer, module)
    network.weight.data.fill_(4.)
    trainer.global_step = 1
    ema.on_train_batch_end(trainer, module, None, None, 0)
    torch.testing.assert_close(ema.averaged.module.weight, torch.tensor([[2.2]]))
    resumed = EMA(decay=.9)
    resumed.load_state_dict(ema.state_dict())
    resumed.on_fit_start(trainer, module)
    trainer.global_step = 2
    resumed.on_train_batch_end(trainer, module, None, None, 1)
    torch.testing.assert_close(resumed.averaged.module.weight, torch.tensor([[2.38]]))
    assert resumed.averaged.n_averaged.item() == 3
