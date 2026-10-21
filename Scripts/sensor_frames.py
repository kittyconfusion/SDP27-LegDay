#!/usr/bin/env python3
"""Generate, decode, or summarise LegDay's fixed-size binary sensor frames.

Two frame layouts exist:

* 72-byte raw ``SensorFrame`` (``.BIN`` files from ``sd-card-test``).
* 120-byte ``FusedFrame`` (``.Q6B`` files from ``imu-sensor-test``): the same
  72 raw bytes followed by four DMP-style Quat6 vectors (x, y, z as signed
  Q30 ``int32``; ``w`` is reconstructed as ``sqrt(1 - x^2 - y^2 - z^2)``).

The layout is picked from the file extension and can be forced with
``--frame-bytes 72`` or ``--frame-bytes 120``.

Commands:

* ``generate``  write synthetic 72-byte frames (for exercising the SD test)
* ``decode``    binary -> CSV, one row per frame, quaternions also as floats
* ``summary``   rate, timing jitter, sequence gaps, overflow flags, ranges
* ``plot``      PNG of accel / gyro / quaternion traces (needs matplotlib)
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



def plot(input_path: Path, output_path: Path, frame: struct.Struct, max_seconds: float | None) -> None:
    """Draw accel, gyro, quaternion, and a gravity cross-check as a PNG.

    matplotlib is imported here rather than at the top so that generate,
    decode, and summary keep working on a machine without it.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: write a file, never open a window
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("plot needs matplotlib: python3 -m pip install matplotlib")

    fused = frame is FUSED_FRAME
    frames = list(iter_frames(input_path, frame))
    if not frames:
        raise SystemExit("file holds no frames")

    t0 = frames[0][0]
    seconds = [(values[0] - t0) / 1e6 for values in frames]
    if max_seconds is not None:
        keep = sum(1 for s in seconds if s <= max_seconds)
        frames, seconds = frames[:keep], seconds[:keep]

    # IMU 0 raw counts converted to physical units for readability.
    accel_g = [[values[5 + axis] / 8192.0 for values in frames] for axis in range(3)]
    gyro_dps = [[values[8 + axis] / 16.4 for values in frames] for axis in range(3)]

    rows = 4 if fused else 2
    figure, axes = plt.subplots(rows, 1, figsize=(12, 2.6 * rows), sharex=True)
    axes[0].set_title(f"{input_path.name}: {len(frames)} frames, IMU 0")
    for axis, label in enumerate("xyz"):
        axes[0].plot(seconds, accel_g[axis], label=f"a{label}", linewidth=0.8)
        axes[1].plot(seconds, gyro_dps[axis], label=f"g{label}", linewidth=0.8)
    axes[0].set_ylabel("accel [g]")
    axes[1].set_ylabel("gyro [deg/s]")

    if fused:
        # Quaternion components, with w rebuilt from the unit-length rule.
        quats = []
        for values in frames:
            x, y, z = (component / Q30 for component in values[33:36])
            quats.append((reconstruct_w(x, y, z), x, y, z))
        for axis, label in enumerate("wxyz"):
            axes[2].plot(seconds, [q[axis] for q in quats], label=f"q{label}", linewidth=0.8)
        axes[2].set_ylabel("quat6")

        # Cross-check: the quaternion's idea of "down" in the sensor frame
        # should match the accelerometer direction whenever the sensor is
        # not accelerating. Large gaps mean motion, not error.
        down = [
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
            for (w, x, y, z) in quats
        ]
        norms = [
            math.sqrt(sum(accel_g[axis][index] ** 2 for axis in range(3))) or 1.0
            for index in range(len(frames))
        ]
        for axis, label in enumerate("xyz"):
            axes[3].plot(seconds, [d[axis] for d in down], label=f"down {label} (quat)", linewidth=0.8)
            axes[3].plot(
                seconds,
                [accel_g[axis][index] / norms[index] for index in range(len(frames))],
                label=f"accel {label} / |a|",
                linewidth=0.6,
                linestyle="--",
            )
        axes[3].set_ylabel("gravity dir")

    for axis in axes:
        axis.legend(loc="upper right", fontsize="x-small", ncol=3)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("seconds")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    print(f"plotted {len(frames)} frames to {output_path}")

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

    plot_parser = subparsers.add_parser("plot", help="save a PNG of accel, gyro, and quaternion traces")
    plot_parser.add_argument("input", type=Path)
    plot_parser.add_argument("output", type=Path)
    plot_parser.add_argument("--max-seconds", type=float, help="only plot the first N seconds")

    args = parser.parse_args()
    if args.command == "generate":
        generate(args.output, args.samples)
    elif args.command == "decode":
        decode(args.input, args.output, frame_struct(args.input, args.frame_bytes))
    elif args.command == "plot":
        plot(args.input, args.output, frame_struct(args.input, args.frame_bytes), args.max_seconds)
    else:
        summarize(args.input, frame_struct(args.input, args.frame_bytes))


if __name__ == "__main__":
    main()
