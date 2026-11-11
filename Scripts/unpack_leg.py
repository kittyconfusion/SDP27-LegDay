#!/usr/bin/env python3
"""Extract 120-byte frames from the current recorder's padded .leg blocks.

Usage: python3 Scripts/unpack_leg.py input.leg output.Q6B
This preserves measurements and flags; it does not resample, fuse, or label data.
Old 40-byte recordings do not use this format and must not be passed here.
"""

import argparse
from pathlib import Path

BLOCK_BYTES = 8192
FRAME_BYTES = 120


def unpack(data: bytes) -> bytes:
    if len(data) % BLOCK_BYTES:
        raise ValueError("Truncated .leg file: expected whole 8192-byte blocks")
    payload = bytearray()
    for offset in range(0, len(data), BLOCK_BYTES):
        block = data[offset : offset + BLOCK_BYTES]
        used = int.from_bytes(block[:2], "big")
        if not 2 <= used <= BLOCK_BYTES or (used - 2) % FRAME_BYTES:
            raise ValueError(f"Invalid 120-byte-frame block at byte {offset}: used={used}")
        if any(block[used:]):
            raise ValueError(f"Nonzero padding at byte {offset}; check recorder format")
        payload.extend(block[2:used])
    return bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        payload = unpack(args.input.read_bytes())
        with args.output.open("xb") as output:
            output.write(payload)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Extracted {len(payload) // FRAME_BYTES} frames into {args.output}")


if __name__ == "__main__":
    main()
