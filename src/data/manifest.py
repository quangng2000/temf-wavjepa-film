"""Explicit, paired JSONL manifests. No corpus scan or implicit augmentation."""

import json
from pathlib import Path


def read_manifest(path):
    path = Path(path).resolve(strict=True)
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError("Manifest is empty")
    for entry in entries:
        if not all(isinstance(entry.get(key), str) and entry[key]
                   for key in ("id", "group", "clean", "observed")):
            raise ValueError("Each pair requires nonempty id, group, clean, and observed strings")
        for key in ("clean", "observed"):
            entry[key] = str((path.parent / entry[key]).resolve(strict=True))
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate pair id within manifest")
    return entries


def check_disjoint(training, validation):
    for key in ("id", "group"):
        if {row[key] for row in training} & {row[key] for row in validation}:
            raise ValueError(f"Training/validation overlap in {key}")
    train_files = {row[key] for row in training for key in ("clean", "observed")}
    val_files = {row[key] for row in validation for key in ("clean", "observed")}
    if train_files & val_files:
        raise ValueError("Training/validation share an audio path")
