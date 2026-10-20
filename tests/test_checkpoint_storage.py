"""Completed-file verification precedes persistent completion receipts."""
import hashlib
import json

import pytest

from src.checkpoints import storage


def test_publish_verifies_bytes_and_marks_the_limits_of_mounted_readback(tmp_path):
    source, root = tmp_path / "local.ckpt", tmp_path / "persistent"
    payload = b"saved optimizer, model, EMA and loader state" * 1000
    source.write_bytes(payload)
    receipt_path = storage.publish_checkpoint(source, root, "new-run", 1000)
    receipt = json.loads(receipt_path.read_text())
    assert (root / receipt["filename"]).read_bytes() == payload
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["bytes"] == len(payload) and receipt["step"] == 1000
    assert receipt["verification"] == "fsync_and_mounted_readback_sha256"
    assert receipt["backend_durability_independently_verified"] is False


def test_readback_failure_never_creates_a_completion_receipt(tmp_path, monkeypatch):
    source, root = tmp_path / "local.ckpt", tmp_path / "persistent"
    source.write_bytes(b"checkpoint")
    monkeypatch.setattr(storage, "file_digest", lambda path: "mismatched-digest")
    with pytest.raises(RuntimeError, match="readback verification failed"):
        storage.publish_checkpoint(source, root, "new-run", 1000)
    assert len(list(root.glob("*.ckpt"))) == 1
    assert not list(root.glob("*.complete.json"))


def test_fsync_failure_never_creates_a_completion_receipt(tmp_path, monkeypatch):
    source, root = tmp_path / "local.ckpt", tmp_path / "persistent"
    source.write_bytes(b"checkpoint")
    def fail(descriptor):
        raise OSError("flush failed")
    monkeypatch.setattr(storage.os, "fsync", fail)
    with pytest.raises(OSError, match="flush failed"):
        storage.publish_checkpoint(source, root, "new-run", 1000)
    assert not list(root.glob("*.complete.json"))


def test_publication_is_immutable_even_for_repeated_steps(tmp_path):
    source, root = tmp_path / "local.ckpt", tmp_path / "persistent"
    source.write_bytes(b"first")
    first = storage.publish_checkpoint(source, root, "new-run", 1000)
    source.write_bytes(b"second")
    second = storage.publish_checkpoint(source, root, "new-run", 1000)
    assert first != second
    assert (root / json.loads(first.read_text())["filename"]).read_bytes() == b"first"
