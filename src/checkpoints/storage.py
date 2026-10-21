"""Publish completed local checkpoints without relying on object-mount rename."""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

FORMAT = "temf-durable-checkpoint-v1"
CHUNK_BYTES = 8 * 1024 * 1024


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_completed(source, destination):
    before = source.stat()
    digest, copied = hashlib.sha256(), 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for chunk in iter(lambda: reader.read(CHUNK_BYTES), b""):
            writer.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino) or copied != before.st_size:
        raise RuntimeError("Local checkpoint changed during publication")
    if destination.stat().st_size != copied or file_digest(destination) != digest.hexdigest():
        raise RuntimeError("Persistent checkpoint readback verification failed")
    return copied, digest.hexdigest()


def write_receipt(path, receipt):
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != payload:
        raise RuntimeError("Persistent checkpoint receipt verification failed")


def publish_checkpoint(source, backup_dir, run_id, step):
    source, root = Path(source), Path(backup_dir)
    root.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint-{step:09d}-{uuid.uuid4().hex}.ckpt"
    size, digest = copy_completed(source, root / name)
    receipt = dict(format=FORMAT, run_id=run_id, step=step, filename=name,
                   bytes=size, sha256=digest, created_ns=time.time_ns(),
                   verification="fsync_and_mounted_readback_sha256",
                   backend_durability_independently_verified=False)
    receipt_path = root / f"{name}.complete.json"
    write_receipt(receipt_path, receipt)
    return receipt_path
