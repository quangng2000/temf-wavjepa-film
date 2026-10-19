import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint

from tests.test_online import make_online_config


class ProbeModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.seen = []

    def training_step(self, batch, index):
        self.seen.append((batch["augmentation_seed"].clone(), batch["observed"].clone()))
        return (self.weight * batch["observed"] - batch["clean"]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def trainer(path, steps):
    checkpoint = ModelCheckpoint(dirpath=path, every_n_train_steps=1, save_last=True)
    return L.Trainer(default_root_dir=path, max_steps=steps, max_epochs=-1,
                     accelerator="cpu", devices=1, logger=False, callbacks=[checkpoint],
                     enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                     limit_val_batches=0)


def test_lightning_checkpoint_restores_library_loader_and_augmentations(tmp_path):
    cfg = make_online_config(tmp_path)
    baseline = ProbeModule()
    trainer(tmp_path / "full", 3).fit(baseline, datamodule=instantiate(cfg))
    partial, runner = ProbeModule(), trainer(tmp_path / "partial", 1)
    runner.fit(partial, datamodule=instantiate(cfg))
    checkpoint = runner.checkpoint_callback.last_model_path
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["loops"]["fit_loop"]["state_dict"]["combined_loader"]
    resumed = ProbeModule()
    runner = trainer(tmp_path / "resumed", 3)
    runner.fit(resumed, datamodule=instantiate(cfg), ckpt_path=checkpoint)
    compare_continuation(baseline, resumed)


def compare_continuation(baseline, resumed):
    assert len(resumed.seen) == 2
    for actual, expected in zip(resumed.seen, baseline.seen[1:]):
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(resumed.weight, baseline.weight, rtol=0, atol=0)
