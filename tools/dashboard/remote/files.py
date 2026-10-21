"""Small text tails and checkpoint metadata; never deserialize model weights."""
import re
from pathlib import Path


def read_text_tail(path, limit=16384):
    try:
        with Path(path).open("rb") as source:
            source.seek(max(0, source.seek(0, 2) - limit))
            return source.read(limit).decode("utf-8", "replace")
    except FileNotFoundError:
        return None
    except OSError as error:
        return f"Unable to read {Path(path).name}: {error}"


def checkpoint_metadata(path):
    try:
        stat = path.stat()
        match = re.fullmatch(r"update-(\d+)\.ckpt", path.name)
        return {"name": path.name, "bytes": stat.st_size, "modified": stat.st_mtime,
                "step": int(match.group(1)) if match else None, "verified": False}
    except OSError:
        return None


def snapshot_checkpoint(run_path):
    directory = Path(run_path) / "checkpoints"
    candidates = [checkpoint_metadata(path) for path in directory.glob("update-*.ckpt")]
    candidates = [item for item in candidates if item and item["step"] is not None]
    latest = max(candidates, key=lambda item: item["step"], default=None)
    return latest or checkpoint_metadata(directory / "last.ckpt")
