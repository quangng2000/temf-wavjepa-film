"""Retention is restricted to verified receipt-owned files in the chosen run."""
import json

import pytest

from src.checkpoints.retention import owned_receipts, prune_checkpoints
from src.checkpoints.storage import publish_checkpoint


def publish(tmp_path, step, owner="new-run"):
    source = tmp_path / "local.ckpt"
    source.write_bytes(str(step).encode())
    return publish_checkpoint(source, tmp_path / "persistent", owner, step)


def test_keep_latest_two_and_twenty_thousand_milestones(tmp_path):
    root = tmp_path / "persistent"
    for step in (1000, 19000, 20000, 21000, 39000, 40000, 41000, 42000):
        latest = publish(tmp_path, step)
        prune_checkpoints(root, "new-run", latest)
    records = owned_receipts(root, "new-run")
    assert {record["step"] for _, record in records} == {20000, 40000, 41000, 42000}
    assert len(list(root.glob("*.ckpt"))) == 4


def test_retention_preserves_other_owners_unknown_files_and_parent_data(tmp_path):
    old = publish(tmp_path, 1000, owner="old-run")
    old_file = old.parent / json.loads(old.read_text())["filename"]
    unknown, outside = old.parent / "unreceipted.ckpt", tmp_path / "old-volume.ckpt"
    unknown.write_bytes(b"unknown")
    outside.write_bytes(b"must remain")
    (old.parent / "broken.complete.json").write_text("not JSON")
    for step in (1000, 2000, 3000):
        latest = publish(tmp_path, step)
    prune_checkpoints(old.parent, "new-run", latest)
    assert old.exists() and old_file.exists() and unknown.exists() and outside.exists()


def test_missing_new_receipt_blocks_all_cleanup(tmp_path):
    first = publish(tmp_path, 1000)
    publish(tmp_path, 2000)
    publish(tmp_path, 3000)
    with pytest.raises(RuntimeError, match="refusing checkpoint cleanup"):
        prune_checkpoints(first.parent, "new-run", first.parent / "absent.json")
    assert len(owned_receipts(first.parent, "new-run")) == 3


def test_duplicate_milestone_keeps_only_newest_verified_copy(tmp_path):
    first = publish(tmp_path, 20000)
    latest = publish(tmp_path, 20000)
    prune_checkpoints(first.parent, "new-run", latest)
    assert latest.exists() and not first.exists()


def test_receipt_path_traversal_is_not_owned(tmp_path):
    latest = publish(tmp_path, 1000)
    malicious = json.loads(latest.read_text()) | {"filename": "../protected.ckpt"}
    (latest.parent / "forged.complete.json").write_text(json.dumps(malicious))
    assert len(owned_receipts(latest.parent, "new-run")) == 1
