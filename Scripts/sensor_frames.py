#!/usr/bin/env python3
"""Generate, decode, or summarise LegDay's fixed-size binary sensor frames.

Two frame layouts exist:

* 72-byte raw ``SensorFrame`` (``.BIN`` files from ``sd-card-test``).
* 120-byte ``FusedFrame`` (``.Q6B`` files from ``imu-sensor-test``): the same
  72 raw bytes followed by four DMP-style Quat6 vectors (x, y, z as signed
  Q30 ``int32``; ``w`` is reconstructed as ``sqrt(1 - x^2 - y^2 - z^2)``).

The layout is picked from the file extension and can be forced with
``--frame-bytes 72`` or ``--frame-bytes 120``.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import struct
from pathlib import Path


IMU_COUNT = 4
IMU_CHANNEL_NAMES = ("ax", "ay", "az", "gx", "gy", "gz")
PRESSURE_NAMES = ("left_heel", "left_ball", "right_heel", "right_ball")
SAMPLE_RATE_HZ = 200
RAW_FRAME = struct.Struct("<QIBBH24h4H")
FUSED_FRAME = struct.Struct("<QIBBH24h4H12i")
Q30 = float(1 << 30)
assert RAW_FRAME.size == 72 and FUSED_FRAME.size == 120

STATUS_ORIENTATION_MASK = 0x0003
STATUS_ORIENTATION_NAMES = {0: "none", 1: "dmp", 2: "software", 3: "?"}
STATUS_FIFO_OVERFLOW = 0x0004
STATUS_TIMESTAMP_RECONSTRUCTED = 0x0008
STATUS_SYNTHETIC = 0x8000


def frame_struct(path: Path, frame_bytes: int | None) -> struct.Struct:
    if frame_bytes == 72:
        return RAW_FRAME
    if frame_bytes == 120:
        return FUSED_FRAME
    if frame_bytes is not None:
        raise SystemExit("--frame-bytes must be 72 or 120")
    return FUSED_FRAME if path.suffix.upper() == ".Q6B" else RAW_FRAME


def column_names(fused: bool) -> list[str]:
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
    if fused:
        for imu in range(IMU_COUNT):
            names.extend(f"imu{imu}_q6_{axis}_q30" for axis in ("x", "y", "z"))
        names.append("orientation_source")
        for imu in range(IMU_COUNT):
            names.extend(f"imu{imu}_q{axis}" for axis in ("w", "x", "y", "z"))
    return names


def reconstruct_w(x: float, y: float, z: float) -> float:
    return math.sqrt(max(0.0, 1.0 - (x * x + y * y + z * z)))


def generate(path: Path, samples: int) -> None:
    with path.open("wb") as output:
        for sequence in range(samples):
            phase = sequence % 40
            imu_values: list[int] = []
            for imu in range(IMU_COUNT):
                imu_values.extend((phase + imu, phase - 20, 1_000, phase * 2, -phase, imu))
            pressures = (800, 1_600, 900, 1_700)
            output.write(
                RAW_FRAME.pack(
                    sequence * 1_000_000 // SAMPLE_RATE_HZ,
                    sequence,
                    0b1111,
                    0b1111,
                    0,
                    *imu_values,
                    *pressures,
                )
            )
    print(f"wrote {samples} frames ({samples * RAW_FRAME.size} bytes) to {path}")


def iter_frames(input_path: Path, frame: struct.Struct):
    byte_count = input_path.stat().st_size
    if byte_count % frame.size:
        raise SystemExit(
            f"{input_path} is {byte_count} bytes; expected a multiple of {frame.size}"
        )
    with input_path.open("rb") as source:
        for _ in range(byte_count // frame.size):
            yield frame.unpack(source.read(frame.size))


def decode(input_path: Path, output_path: Path, frame: struct.Struct) -> None:
    fused = frame is FUSED_FRAME
    count = 0
    with output_path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(column_names(fused))
        for values in iter_frames(input_path, frame):
            row = list(values)
            if fused:
                quats = values[33:45]
                row.append(STATUS_ORIENTATION_NAMES[values[4] & STATUS_ORIENTATION_MASK])
                for imu in range(IMU_COUNT):
                    x, y, z = (component / Q30 for component in quats[imu * 3 : imu * 3 + 3])
                    row.extend(f"{component:.6f}" for component in (reconstruct_w(x, y, z), x, y, z))
            writer.writerow(row)
            count += 1
    print(f"decoded {count} {'fused' if fused else 'raw'} frames to {output_path}")


def summarize(input_path: Path, frame: struct.Struct) -> None:
    fused = frame is FUSED_FRAME
    frames = list(iter_frames(input_path, frame))
    if not frames:
        raise SystemExit("file holds no frames")
    timestamps = [values[0] for values in frames]
    sequences = [values[1] for values in frames]
    flags = [values[4] for values in frames]
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    gaps = sum(1 for a, b in zip(sequences, sequences[1:]) if (b - a) & 0xFFFFFFFF != 1)
    overflow = sum(1 for f in flags if f & STATUS_FIFO_OVERFLOW)
    sources = {}
    for f in flags:
        sources[STATUS_ORIENTATION_NAMES[f & STATUS_ORIENTATION_MASK]] = (
            sources.get(STATUS_ORIENTATION_NAMES[f & STATUS_ORIENTATION_MASK], 0) + 1
        )
    duration_s = (timestamps[-1] - timestamps[0]) / 1e6 if len(timestamps) > 1 else 0.0

    print(f"file:            {input_path} ({'fused 120-byte' if fused else 'raw 72-byte'})")
    print(f"frames:          {len(frames)}")
    print(f"duration:        {duration_s:.3f} s")
    if deltas:
        mean_dt = statistics.mean(deltas)
        print(f"mean rate:       {1e6 / mean_dt:.2f} Hz (mean dt {mean_dt:.1f} us)")
        print(f"dt min/max:      {min(deltas)} / {max(deltas)} us")
        if len(deltas) > 1:
            print(f"dt stdev:        {statistics.pstdev(deltas):.1f} us")
    print(f"sequence gaps:   {gaps}")
    print(f"overflow flags:  {overflow}")
    print(f"synthetic:       {sum(1 for f in flags if f & STATUS_SYNTHETIC)}")
    print(f"orientation src: {sources}")
    for imu in range(IMU_COUNT):
        if not any(values[2] & (1 << imu) for values in frames):
            continue
        base = 5 + imu * 6
        for name, index in zip(IMU_CHANNEL_NAMES, range(base, base + 6)):
            channel = [values[index] for values in frames]
            print(
                f"imu{imu}_{name:<2} min {min(channel):>7} max {max(channel):>7} "
                f"mean {statistics.mean(channel):>9.1f}"
            )
        if fused:
            quats = [values[33 + imu * 3 : 36 + imu * 3] for values in frames]
            last = quats[-1]
            x, y, z = (component / Q30 for component in last)
            print(
                f"imu{imu} last quat w={reconstruct_w(x, y, z):+.4f} x={x:+.4f} y={y:+.4f} z={z:+.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame-bytes", type=int, choices=(72, 120), help="override layout detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="write synthetic 72-byte frames")
    generate_parser.add_argument("output", type=Path)
    generate_parser.add_argument("--samples", type=int, default=400)

    decode_parser = subparsers.add_parser("decode", help="convert frames to CSV")
    decode_parser.add_argument("input", type=Path)
    decode_parser.add_argument("output", type=Path)

    summary_parser = subparsers.add_parser("summary", help="print rate, gaps, and channel ranges")
    summary_parser.add_argument("input", type=Path)

    args = parser.parse_args()
    if args.command == "generate":
        generate(args.output, args.samples)
    elif args.command == "decode":
        decode(args.input, args.output, frame_struct(args.input, args.frame_bytes))
    else:
        summarize(args.input, frame_struct(args.input, args.frame_bytes))


if __name__ == "__main__":
    main()
