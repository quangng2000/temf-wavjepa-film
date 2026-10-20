"""Save locally, verify an immutable persistent copy, then apply owned retention."""
import re
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

from src.checkpoints.retention import prune_checkpoints
from src.checkpoints.storage import publish_checkpoint


class DurableCheckpoint(ModelCheckpoint):
    def __init__(self, backup_dir, run_id, retention=None, **kwargs):
        policy = dict(latest=2, every=20000, prune=False) | dict(retention or {})
        if policy.keys() != {"latest", "every", "prune"} or not isinstance(policy["prune"], bool):
            raise ValueError("Retention accepts latest, every and boolean prune only")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
            raise ValueError("Checkpoint run_id must be a simple explicit identifier")
        if not Path(backup_dir).is_absolute() or policy["latest"] < 1 or policy["every"] < 1:
            raise ValueError("Absolute backup path and positive retention counts required")
        self.backup_dir, self.run_id = str(Path(backup_dir)), run_id
        self.retain_latest, self.retain_every = policy["latest"], policy["every"]
        self.prune_backups = policy["prune"]
        self.last_verified_receipt = None
        super().__init__(**kwargs)

    def setup(self, trainer, pl_module, stage):
        super().setup(trainer, pl_module, stage)
        if trainer.world_size != 1:
            raise ValueError("Durable checkpoints currently require single-device training")
        local, backup = Path(self.dirpath).resolve(), Path(self.backup_dir).resolve()
        if local == backup or backup in local.parents or local in backup.parents:
            raise ValueError("Local and persistent checkpoint directories must be separate")

    def _save_checkpoint(self, trainer, filepath):
        # Lightning atomically commits its local TorchCheckpointIO write first.
        super()._save_checkpoint(trainer, filepath)
        receipt = publish_checkpoint(filepath, self.backup_dir, self.run_id, trainer.global_step)
        self.last_verified_receipt = str(receipt)
        if self.prune_backups:
            prune_checkpoints(self.backup_dir, self.run_id, receipt,
                              self.retain_latest, self.retain_every)

    def state_dict(self):
        return dict(super().state_dict(), durable_backup_dir=self.backup_dir,
                    durable_run_id=self.run_id, last_verified_receipt=self.last_verified_receipt)

    def load_state_dict(self, state_dict):
        if state_dict.get("durable_run_id") != self.run_id:
            raise ValueError("Persistent checkpoint owner changed on resume")
        if state_dict.get("durable_backup_dir") != self.backup_dir:
            raise ValueError("Persistent checkpoint destination changed on resume")
        super().load_state_dict(state_dict)
        self.last_verified_receipt = state_dict.get("last_verified_receipt")
