"""Generate a small reusable RIR bank explicitly, outside the training loop."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.augmentation.rooms import simulate_room


def generate_bank(output, count=32, seed=42):
    if count < 1:
        raise ValueError("Room count must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index in range(count):
        name = f"room-{seed + index}"
        np.savez(output / f"{name}.npz", **simulate_room(seed + index))
        rows.append(dict(id=name, group=name, path=f"{name}.npz"))
    (output / "rooms.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return output / "rooms.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(generate_bank(args.output, args.count, args.seed))


if __name__ == "__main__":
    main()
