"""Verify cached upstream source before executing selected definitions."""
import hashlib
from pathlib import Path

COMMIT = "78c531cca978a23113b262cd0b6bede2197b1916"
ROOT = Path(__file__).resolve().parents[2] / ".cache/upstream/temf" / COMMIT
HASHES = {
    "src/training/loss.py": "17af2d22abeee6912fe4eedd3386a00cb636df30751181cc1ed7d6e06e7feb17",
    "configs/loss/temf.yaml": "4eaee39e1024e0ae34e1a5c02d262452ceb12852233190a14ee68a26e0cf2d7e",
    "src/structs.py": "ae91294a63d75f03072830c3686a91c5403b7bf08c5316d4b4c249548a4f3f59",
    "src/utils/misc.py": "141457945c51e6bb40a133b1ed7afc5ba208396d9606349322da90045937d424",
}


def read_source(name):
    path = ROOT / name
    if not path.is_file():
        raise FileNotFoundError("Cache TEMF first: python -m src.upstream.fetch")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != HASHES[name]:
        raise ValueError(f"Upstream source checksum mismatch: {name}")
    return data.decode()
