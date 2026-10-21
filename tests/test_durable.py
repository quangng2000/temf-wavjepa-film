"""Local atomic saves precede backup, and backup failure stops training."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import open_dict

from src.callbacks.durable import DurableCheckpoint
from src.checkpoints.retention import owned_receipts
from src.train import train
from tests.test_lightning import smoke_config


def callback(tmp_path, **kwargs):
    return DurableCheckpoint(dirpath=str(tmp_path / "local"),
        backup_dir=str(tmp_path / "persistent"), run_id="new-run", **kwargs)


def test_backup_failure_propagates_and_preserves_local_checkpoint(tmp_path, monkeypatch):
    target = tmp_path / "local.ckpt"
    trainer = SimpleNamespace(global_step=1000, is_global_zero=True, loggers=[],
        save_checkpoint=lambda path, weights: Path(path).write_bytes(b"complete"))
    def fail(*args):
        assert target.read_bytes() == b"complete"
        raise OSError("persistent storage unavailable")
    monkeypatch.setattr("src.callbacks.durable.publish_checkpoint", fail)
    with pytest.raises(OSError, match="persistent storage unavailable"):
        callback(tmp_path)._save_checkpoint(trainer, str(target))
    assert target.read_bytes() == b"complete"


def test_resume_rejects_a_different_owner_or_destination(tmp_path):
    original = callback(tmp_path)
    state = original.state_dict()
    callback(tmp_path).load_state_dict(state)
    with pytest.raises(ValueError, match="owner changed"):
        original.load_state_dict(state | {"durable_run_id": "old-run"})
    with pytest.raises(ValueError, match="destination changed"):
        original.load_state_dict(state | {"durable_backup_dir": "/different"})


def test_checkpoint_backup_retention_runs_in_lightning(tmp_path):
    cfg = smoke_config(tmp_path / "training")
    cfg.total_updates = cfg.trainer.max_steps = 4
    with open_dict(cfg.callbacks.checkpoint):
        cfg.callbacks.checkpoint._target_ = "src.callbacks.durable.DurableCheckpoint"
        cfg.callbacks.checkpoint.backup_dir = str(tmp_path / "persistent")
        cfg.callbacks.checkpoint.run_id = "new-run"
        cfg.callbacks.checkpoint.retention = dict(latest=2, every=2, prune=True)
        cfg.callbacks.checkpoint.monitor, cfg.callbacks.checkpoint.mode = "step", "max"
    cfg.callbacks.checkpoint.save_top_k = 2
    cfg.callbacks.checkpoint.save_last = "link"
    _, objects = train(cfg)
    saved = objects["trainer"].checkpoint_callback
    assert objects["trainer"].global_step == 4 and len(saved.best_k_models) == 2
    assert Path(saved.last_model_path).exists()
    receipts = owned_receipts(tmp_path / "persistent", "new-run")
    assert {record["step"] for _, record in receipts} == {2, 3, 4}


def test_default_never_prunes_persistent_checkpoints(tmp_path):
    assert callback(tmp_path).prune_backups is False


@pytest.mark.parametrize("options", [{"prune": "true"}, {"every": 0}, {"unexpected": 1}])
def test_reject_invalid_retention(tmp_path, options):
    with pytest.raises(ValueError):
        callback(tmp_path, retention=options)
