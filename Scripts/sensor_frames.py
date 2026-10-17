#!/usr/bin/env python3
"""Generate or decode LegDay's fixed 72-byte sensor frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import struct


IMU_COUNT = 4
IMU_CHANNEL_NAMES = ("ax", "ay", "az", "gx", "gy", "gz")
PRESSURE_NAMES = ("left_heel", "left_ball", "right_heel", "right_ball")
SAMPLE_RATE_HZ = 200
FRAME = struct.Struct("<QIBBH24h4H")


def column_names() -> list[str]:
    names = [
        "timestamp_us",
        "sequence",
        "valid_imu_mask",
        "valid_pressure_mask",
        "status_flags",
    ]
    for imu in range(IMU_COUNT):
        names.extend(f"imu{imu}_{channel}" for channel in IMU_CHANNEL_NAMES)
    names.extend(f"pressure_{name}_adc" for name in PRESSURE_NAMES)
    return names


def generate(path: Path, samples: int) -> None:
    with path.open("wb") as output:
        for sequence in range(samples):
            phase = sequence % 40
            imu_values: list[int] = []
            for imu in range(IMU_COUNT):
                imu_values.extend((phase + imu, phase - 20, 1_000, phase * 2, -phase, imu))
            pressures = (800, 1_600, 900, 1_700)
            output.write(
                FRAME.pack(
                    sequence * 1_000_000 // SAMPLE_RATE_HZ,
                    sequence,
                    0b1111,
                    0b1111,
                    0,
                    *imu_values,
                    *pressures,
                )
            )
    print(f"wrote {samples} frames ({samples * FRAME.size} bytes) to {path}")


def decode(input_path: Path, output_path: Path) -> None:
    byte_count = input_path.stat().st_size
    if byte_count % FRAME.size:
        raise SystemExit(
            f"{input_path} is {byte_count} bytes; expected a multiple of {FRAME.size}"
        )

    frame_count = byte_count // FRAME.size
    with input_path.open("rb") as source, output_path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(column_names())
        for _ in range(frame_count):
            encoded = source.read(FRAME.size)
            writer.writerow(FRAME.unpack(encoded))
    print(f"decoded {frame_count} frames to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="write synthetic frames")
    generate_parser.add_argument("output", type=Path)
    generate_parser.add_argument("--samples", type=int, default=400)

    decode_parser = subparsers.add_parser("decode", help="convert frames to CSV")
    decode_parser.add_argument("input", type=Path)
    decode_parser.add_argument("output", type=Path)

    args = parser.parse_args()
    if args.command == "generate":
        generate(args.output, args.samples)
    else:
        decode(args.input, args.output)


if __name__ == "__main__":
    main()
