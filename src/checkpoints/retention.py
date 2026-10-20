"""Delete only receipt-owned checkpoints after a verified newer publication."""
import json
import re
from pathlib import Path

from src.checkpoints.storage import FORMAT

FILENAME = re.compile(r"checkpoint-[0-9]{9,}-[0-9a-f]{32}\.ckpt")


def owned_receipts(root, run_id):
    records = []
    for path in root.glob("*.complete.json"):
        try:
            record = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if valid_record(record, path, run_id):
            records.append((path, record))
    return records


def valid_record(record, path, run_id):
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        return False
    name = record.get("filename", "")
    return (record.get("run_id") == run_id and isinstance(name, str)
            and FILENAME.fullmatch(name) is not None
            and path.name == f"{name}.complete.json"
            and isinstance(record.get("step"), int) and record["step"] >= 0
            and name.startswith(f"checkpoint-{record['step']:09d}-")
            and isinstance(record.get("created_ns"), int)
            and isinstance(record.get("bytes"), int) and record["bytes"] >= 0
            and isinstance(record.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None
            and record.get("verification") == "fsync_and_mounted_readback_sha256")


def retained_paths(records, latest, every):
    newest = {}
    for path, record in sorted(records, key=lambda item: item[1]["created_ns"]):
        newest[record["step"]] = path
    steps = set(sorted(newest)[-latest:])
    steps.update(step for step in newest if step > 0 and step % every == 0)
    return {newest[step] for step in steps}


def prune_checkpoints(backup_dir, run_id, completed_receipt, latest=2, every=20000):
    if latest < 1 or every < 1:
        raise ValueError("Retention needs positive latest and milestone counts")
    root = Path(backup_dir)
    records = owned_receipts(root, run_id)
    completed_receipt = Path(completed_receipt)
    if completed_receipt not in {path for path, _ in records}:
        raise RuntimeError("New verified receipt is absent; refusing checkpoint cleanup")
    keep = retained_paths(records, latest, every) | {completed_receipt}
    for receipt, record in records:
        if receipt not in keep:
            (root / record["filename"]).unlink(missing_ok=True)
            receipt.unlink()
