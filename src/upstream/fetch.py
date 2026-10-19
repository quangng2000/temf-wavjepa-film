"""Explicitly fetch pinned TEMF into an ignored local cache."""
import hashlib
import ssl
import urllib.request

import certifi

from src.upstream.source import COMMIT, HASHES, ROOT, read_source


def fetch_source(name, expected):
    target = ROOT / name
    if target.exists():
        read_source(name)
        return
    url = f"https://raw.githubusercontent.com/yuanpengtu/TEMF/{COMMIT}/{name}"
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context, timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Downloaded source checksum mismatch: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


if __name__ == "__main__":
    for name, expected in HASHES.items():
        fetch_source(name, expected)
    print(f"Verified TEMF {COMMIT} in {ROOT}")
