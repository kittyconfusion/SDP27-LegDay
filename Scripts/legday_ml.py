#!/usr/bin/env python3
"""Shared machine-learning code for the LegDay motion classifier.

This module is the Python half of a contract with the Rust crate
``crates/legday-classifier``. Both sides must turn a recording into exactly the
same numbers, or a model trained here will misbehave on the board:

* ``frame_channels`` / ``preprocess.rs``: one 120-byte frame -> 40 channels.
* ``window_input`` / ``preprocess.rs``: 225 frames -> a (40, 75) model input.
* ``export_model`` / ``cnn.rs``: the trained network -> a ``.ldm`` file.

A test in the Rust crate (``demo_model_matches_pytorch``) feeds a window
exported from here through the Rust code and checks it reproduces the
PyTorch outputs, so a mismatch is caught immediately.

Channel layout (40 per frame)
-----------------------------
====== ======================================================== =========
index  content                                                   unit
====== ======================================================== =========
0-23   IMU 0..3 x (ax, ay, az, gx, gy, gz)                       g, deg/s
24-27  pressure 0..3                                             0..1 of ADC
28-39  IMU 0..3 x gravity direction "down" (x, y, z)             unit vector
====== ======================================================== =========

The gravity direction comes from each IMU's quaternion. It is used instead of
the raw quaternion because it has neither of the quaternion's ambiguities:
``q`` and ``-q`` give the same "down", and so does any heading (yaw), which
Quat6 cannot measure and which differs arbitrarily between IMUs.

IMUs or pressure sensors missing from a frame (their valid bit clear) give
zeros, so a model can be trained on whatever sensors exist.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants shared with the Rust crate (keep in sync with preprocess.rs)
# ---------------------------------------------------------------------------

IMU_COUNT = 4
PRESSURE_COUNT = 4
CHANNELS = IMU_COUNT * 6 + PRESSURE_COUNT + IMU_COUNT * 3  # 40

# The firmware keeps one second of frames at 225 Hz. The classifier averages
# groups of 3 frames, so it sees 75 steps at 75 Hz: plenty for gait (whose
# useful content is below ~20 Hz) and a third of the compute.
WINDOW_FRAMES = 225
DECIMATION = 3
STEPS = WINDOW_FRAMES // DECIMATION  # 75

Q30 = float(1 << 30)
ADC_FULL_SCALE = 4095.0

STATUS_ORIENTATION_MASK = 0x0003
STATUS_RANGES_RECORDED = 0x1000
ACCEL_LSB_PER_G = np.array([16_384.0, 8_192.0, 4_096.0, 2_048.0], dtype=np.float32)
GYRO_LSB_PER_DPS = np.array([131.0, 65.5, 32.8, 16.4], dtype=np.float32)

# One 120-byte FusedFrame, field for field (see legday-core fused.rs).
FRAME_DTYPE = np.dtype(
    [
        ("timestamp_us", "<u8"),
        ("sequence", "<u4"),
        ("imu_mask", "u1"),
        ("pressure_mask", "u1"),
        ("flags", "<u2"),
        ("imu", "<i2", (IMU_COUNT, 6)),
        ("pressure", "<u2", (PRESSURE_COUNT,)),
        ("quat", "<i4", (IMU_COUNT, 3)),
    ]
)
assert FRAME_DTYPE.itemsize == 120

# The older 72-byte frame (no quaternions).
RAW_FRAME_DTYPE = np.dtype(FRAME_DTYPE.descr[:-1])
assert RAW_FRAME_DTYPE.itemsize == 72

# Motion categories, in the firmware's folder names and order (recording.rs).
CATEGORIES = ("WALK", "RUN", "STAIRS", "SIT", "STAND", "TRANS")

# Layer kinds in a .ldm model file (match cnn.rs).
LAYER_CONV_RELU = 1
LAYER_GLOBAL_AVG_POOL = 2
LAYER_DENSE = 3
LAYER_SELF_ATTENTION = 4


# ---------------------------------------------------------------------------
# Reading recordings
# ---------------------------------------------------------------------------


def load_recording(path: Path) -> np.ndarray:
    """Read a .Q6B (120-byte) or .BIN (72-byte) file as FRAME_DTYPE records."""
    if path.suffix.upper() == ".BIN":
        raw = np.fromfile(path, dtype=RAW_FRAME_DTYPE)
        frames = np.zeros(len(raw), dtype=FRAME_DTYPE)
        for name in RAW_FRAME_DTYPE.names:
            frames[name] = raw[name]
        return frames
    return np.fromfile(path, dtype=FRAME_DTYPE)


def find_recordings(root: Path) -> list[tuple[str, Path]]:
    """Return (category, file) for every recording under root/<CATEGORY>/.

    The folder a file sits in is its label, exactly as the CATEGORY button
    chose it on the board.
    """
    found = []
    for folder in sorted(root.iterdir()):
        if folder.is_dir():
            for file in sorted(folder.iterdir()):
                if file.suffix.upper() in (".Q6B", ".BIN"):
                    found.append((folder.name.upper(), file))
    return found


# ---------------------------------------------------------------------------
# Preprocessing (mirrors preprocess.rs; float32 throughout, like the S3)
# ---------------------------------------------------------------------------


def frame_channels(frames: np.ndarray) -> np.ndarray:
    """Convert N frames to an (N, 40) float32 array of channels."""
    flags = frames["flags"].astype(np.int64)

    # Sensor ranges are recorded per frame; older files were ±4 g / ±2000 dps.
    ranged = (flags & STATUS_RANGES_RECORDED) != 0
    lsb_g = np.where(ranged, ACCEL_LSB_PER_G[(flags >> 8) & 3], ACCEL_LSB_PER_G[1])
    lsb_dps = np.where(ranged, GYRO_LSB_PER_DPS[(flags >> 10) & 3], GYRO_LSB_PER_DPS[3])

    imu_valid = ((frames["imu_mask"][:, None] >> np.arange(IMU_COUNT)) & 1).astype(np.float32)
    pressure_valid = ((frames["pressure_mask"][:, None] >> np.arange(PRESSURE_COUNT)) & 1).astype(np.float32)
    has_orientation = ((flags & STATUS_ORIENTATION_MASK) != 0).astype(np.float32)

    imu = frames["imu"].astype(np.float32)
    accel = imu[:, :, 0:3] / lsb_g[:, None, None].astype(np.float32)
    gyro = imu[:, :, 3:6] / lsb_dps[:, None, None].astype(np.float32)
    raw = np.concatenate([accel, gyro], axis=2) * imu_valid[:, :, None]

    pressure = frames["pressure"].astype(np.float32) / np.float32(ADC_FULL_SCALE) * pressure_valid

    # Quaternion vector part from Q30; w >= 0 rebuilt from unit length.
    quat = frames["quat"].astype(np.float32) / np.float32(Q30)
    x, y, z = quat[:, :, 0], quat[:, :, 1], quat[:, :, 2]
    w = np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - (x * x + y * y + z * z)))

    # Gravity direction in the sensor frame (same formula as fusion.rs).
    down = np.stack(
        [
            np.float32(2.0) * (x * z - w * y),
            np.float32(2.0) * (y * z + w * x),
            np.float32(1.0) - np.float32(2.0) * (x * x + y * y),
        ],
        axis=2,
    )
    down = down * (imu_valid * has_orientation[:, None])[:, :, None]

    return np.concatenate(
        [raw.reshape(len(frames), -1), pressure, down.reshape(len(frames), -1)], axis=1
    ).astype(np.float32)


def decimate(channels: np.ndarray) -> np.ndarray:
    """Average consecutive groups of DECIMATION frames: (225, 40) -> (75, 40).

    Written as an explicit sum so the arithmetic order matches the Rust code.
    """
    usable = (len(channels) // DECIMATION) * DECIMATION
    groups = channels[:usable].reshape(-1, DECIMATION, channels.shape[1])
    total = groups[:, 0]
    for index in range(1, DECIMATION):
        total = total + groups[:, index]
    return (total / np.float32(DECIMATION)).astype(np.float32)


def windows(channels: np.ndarray, hop_frames: int) -> np.ndarray:
    """Cut a recording's (N, 40) channels into (count, 40, 75) model inputs.

    Each window is the most recent WINDOW_FRAMES frames, decimated, with
    channels first: the same layout the S3 builds from its RAM window.
    """
    starts = range(0, len(channels) - WINDOW_FRAMES + 1, hop_frames)
    return np.stack([decimate(channels[s : s + WINDOW_FRAMES]).T for s in starts]).astype(np.float32)


# ---------------------------------------------------------------------------
# The model (PyTorch)
# ---------------------------------------------------------------------------


def build_model(classes: int, width: int, attention: bool):
    """A 1-D CNN over (40 channels x 75 steps).

    Four convolutions slide along time. Each sees a short stretch of motion;
    stacking them (with two stride-2 steps) lets the last layer respond to
    patterns spanning most of the second, about one gait cycle:

        conv 40 -> w,   kernel 7          75 -> 69 steps
        conv w  -> w,   kernel 5, stride 2 69 -> 33
        conv w  -> 2w,  kernel 5           33 -> 29
        conv 2w -> 2w,  kernel 3, stride 2 29 -> 14
        [optional self-attention over the 14 steps]
        average over time -> 2w numbers (the "embedding")
        dense 2w -> classes (logits; softmax turns them into probabilities)

    ``width`` sets the size: 16 ~ 12k parameters, 32 ~ 37k, 80 ~ 200k,
    190 ~ 1M.
    """
    import torch
    from torch import nn

    class SelfAttention(nn.Module):
        """Single-head self-attention across time steps, with a residual.

        Every time step looks at every other one and mixes in what is
        relevant, so a moment of the window can be interpreted in the
        context of the rest, which is the "awareness" a transformer adds.
        Written out by hand (not nn.MultiheadAttention) so the Rust version
        can follow it line by line.
        """

        def __init__(self, dim: int):
            super().__init__()
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.output = nn.Linear(dim, dim)
            self.scale = 1.0 / math.sqrt(dim)

        def forward(self, x):  # x: (batch, channels, steps)
            t = x.transpose(1, 2)  # (batch, steps, channels)
            scores = self.query(t) @ self.key(t).transpose(1, 2) * self.scale
            weights = torch.softmax(scores, dim=-1)
            mixed = self.output(weights @ self.value(t))
            return (t + mixed).transpose(1, 2)

    class LegDayCnn(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(CHANNELS, width, 7),
                    nn.Conv1d(width, width, 5, stride=2),
                    nn.Conv1d(width, 2 * width, 5),
                    nn.Conv1d(2 * width, 2 * width, 3, stride=2),
                ]
            )
            self.attention = SelfAttention(2 * width) if attention else None
            self.dropout = nn.Dropout(0.2)  # training only; no effect when exported
            self.head = nn.Linear(2 * width, classes)

        def embed(self, x):
            for conv in self.convs:
                x = torch.relu(conv(x))
            if self.attention is not None:
                x = self.attention(x)
            return x.mean(dim=2)

        def forward(self, x):
            return self.head(self.dropout(self.embed(x)))

    return LegDayCnn()


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_macs(model) -> int:
    """Multiply-accumulates for one inference (what the S3 has to compute)."""
    steps = STEPS
    macs = 0
    for conv in model.convs:
        steps = (steps - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
        macs += steps * conv.out_channels * conv.in_channels * conv.kernel_size[0]
    if model.attention is not None:
        dim = model.convs[-1].out_channels
        macs += 4 * steps * dim * dim + 2 * steps * steps * dim
    macs += model.head.in_features * model.head.out_features
    return macs


# ---------------------------------------------------------------------------
# Exporting to the .ldm format read by cnn.rs
# ---------------------------------------------------------------------------


def export_model(path: Path, model, mean: np.ndarray, std: np.ndarray, class_names: list[str]) -> None:
    """Write the model as a .ldm file.

    Layout (all little-endian; every field a multiple of 4 bytes so the S3
    can read the weights in place from flash):

        "LDM1", version, channels, steps, decimation, classes, layer count  (u32)
        class names: 8 ASCII bytes each
        input mean[channels], input std[channels]                           (f32)
        per layer: kind, 5 parameters (u32), then its weights and biases    (f32)
    """
    layers = []

    def tensor(t) -> bytes:
        return t.detach().cpu().numpy().astype("<f4").tobytes()

    for conv in model.convs:
        header = struct.pack(
            "<6I", LAYER_CONV_RELU, conv.in_channels, conv.out_channels,
            conv.kernel_size[0], conv.stride[0], conv.dilation[0],
        )
        layers.append(header + tensor(conv.weight) + tensor(conv.bias))

    if model.attention is not None:
        dim = model.convs[-1].out_channels
        body = b"".join(
            tensor(linear.weight) + tensor(linear.bias)
            for linear in (model.attention.query, model.attention.key, model.attention.value, model.attention.output)
        )
        layers.append(struct.pack("<6I", LAYER_SELF_ATTENTION, dim, 0, 0, 0, 0) + body)

    layers.append(struct.pack("<6I", LAYER_GLOBAL_AVG_POOL, 0, 0, 0, 0, 0))
    layers.append(
        struct.pack("<6I", LAYER_DENSE, model.head.in_features, model.head.out_features, 0, 0, 0)
        + tensor(model.head.weight) + tensor(model.head.bias)
    )

    names = b"".join(name.encode("ascii")[:8].ljust(8, b"\0") for name in class_names)
    header = struct.pack("<4s6I", b"LDM1", 1, CHANNELS, STEPS, DECIMATION, len(class_names), len(layers))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        header + names + mean.astype("<f4").tobytes() + std.astype("<f4").tobytes() + b"".join(layers)
    )


def export_golden(path: Path, frames: np.ndarray, logits: np.ndarray) -> None:
    """Write one window of raw frames and the logits PyTorch produced for it.

    The Rust test replays the frames through its own preprocessing and
    inference and must land on the same logits.
    """
    path.write_bytes(
        struct.pack("<I", len(frames)) + frames.tobytes()
        + struct.pack("<I", len(logits)) + logits.astype("<f4").tobytes()
    )


# ---------------------------------------------------------------------------
# Synthetic recordings (to exercise the pipeline before real data exists)
# ---------------------------------------------------------------------------

SAMPLE_RATE_HZ = 228.5  # the bench IMU's measured rate


def synthetic_recording(category: str, seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Make a plausible two-IMU, one-leg recording for one motion category.

    Model: a leg moving in its own plane. IMU 0 sits on the thigh, IMU 1 on
    the shank; both rotate about their x axis. Thigh angle and knee flexion
    follow simple patterns per category (gait cycles, a flexed hip while
    sitting, sit-to-stand ramps), each recording with its own random speed,
    size, sensor mounting offset and noise. The accelerometer sees gravity
    rotated into the sensor plus the segment's own acceleration and heel
    strike impacts; the gyro sees the rate of rotation; pressure 0/1 are heel
    and ball of the foot.

    This is physics-shaped, not physiology-accurate: it proves the training
    and deployment plumbing, not real-world accuracy.
    """
    n = int(seconds * SAMPLE_RATE_HZ)
    t = np.arange(n) / SAMPLE_RATE_HZ
    speed = rng.uniform(0.85, 1.15)
    size = rng.uniform(0.8, 1.2)
    phase = 2 * np.pi * rng.uniform()

    def cycle(frequency):
        return 2 * np.pi * frequency * speed * t + phase

    impact = np.zeros(n)
    heel = np.zeros(n)
    ball = np.zeros(n)

    if category == "STAND":
        thigh = 2.0 * np.sin(2 * np.pi * 0.3 * t + phase)
        knee = 3.0 + 1.0 * np.sin(2 * np.pi * 0.2 * t)
        heel[:] = 0.5 + 0.1 * np.sin(2 * np.pi * 0.3 * t + phase)
        ball[:] = 0.5 - 0.1 * np.sin(2 * np.pi * 0.3 * t + phase)
    elif category == "SIT":
        thigh = 88.0 + 3.0 * np.sin(2 * np.pi * 0.1 * t + phase)
        knee = 85.0 + 5.0 * np.sin(2 * np.pi * 0.07 * t)
        heel[:] = 0.08
        ball[:] = 0.05
    elif category in ("WALK", "RUN", "STAIRS"):
        frequency, thigh_amp, thigh_mid, knee_amp, knee_mid, hit = {
            "WALK": (0.9, 20.0, 5.0, 30.0, 30.0, 0.6),
            "RUN": (1.4, 35.0, 10.0, 50.0, 50.0, 2.5),
            "STAIRS": (0.7, 25.0, 30.0, 40.0, 45.0, 0.4),
        }[category]
        angle = cycle(frequency)
        thigh = thigh_mid + size * thigh_amp * np.sin(angle)
        knee = np.maximum(0.0, knee_mid + size * knee_amp * np.sin(angle - 1.2))
        stance = np.sin(angle) > 0
        heel = np.where(stance, 0.9 * np.clip(np.cos(angle), 0, 1), 0.0)
        ball = np.where(stance, 0.9 * np.clip(np.sin(angle), 0, 1), 0.0)
        strikes = np.diff(np.floor(angle / (2 * np.pi)), prepend=0) > 0
        impact = np.convolve(strikes.astype(float), np.hanning(15), mode="same") * hit
        if category == "STAIRS":
            heel *= 0.3  # mostly the ball of the foot on stairs
    elif category == "TRANS":
        # Repeated stand -> sit -> stand: a smooth 0..1 ramp of "how seated".
        seated = 0.5 - 0.5 * np.cos(cycle(0.25))
        thigh = 88.0 * seated
        knee = 3.0 + 82.0 * seated
        heel = 0.5 * (1 - seated) + 0.08
        ball = 0.5 * (1 - seated) + 0.05
    else:
        raise ValueError(f"unknown category {category}")

    mount = rng.uniform(-5, 5, size=2)
    segment_deg = [thigh + mount[0], thigh - knee + mount[1]]  # thigh, shank pitch

    frames = np.zeros(n, dtype=FRAME_DTYPE)
    frames["timestamp_us"] = (t * 1e6).astype(np.uint64)
    frames["sequence"] = np.arange(n, dtype=np.uint32)
    frames["imu_mask"] = 0b0011
    frames["pressure_mask"] = 0b0011
    # Software orientation, timestamps reconstructed, gyro level 3, ±16 g, ±2000 dps.
    frames["flags"] = 0x0002 | 0x0008 | (3 << 6) | (3 << 8) | (3 << 10) | STATUS_RANGES_RECORDED

    for imu, degrees in enumerate(segment_deg):
        theta = np.radians(degrees)
        rate = np.gradient(theta, t)
        angular_accel = np.gradient(rate, t)

        # Specific force in g: gravity seen in the tilted sensor, plus the
        # segment's tangential acceleration (0.4 m lever) and impacts.
        tangential = 0.4 * angular_accel / 9.81
        ax = rng.normal(0, 0.02, n)
        ay = np.sin(theta) + tangential + rng.normal(0, 0.02, n)
        az = np.cos(theta) + impact + rng.normal(0, 0.02, n)
        gx = np.degrees(rate) + rng.normal(0, 0.3, n)
        gy = rng.normal(0, 0.3, n)
        gz = rng.normal(0, 0.3, n)

        counts = np.stack([ax * 2048, ay * 2048, az * 2048, gx * 16.4, gy * 16.4, gz * 16.4], axis=1)
        frames["imu"][:, imu, :] = np.clip(np.round(counts), -32768, 32767).astype(np.int16)

        # Rotation about x by theta, stored as the Q30 vector part (w >= 0).
        frames["quat"][:, imu, 0] = np.round(np.sin(theta / 2) * np.sign(np.cos(theta / 2)) * Q30).astype(np.int32)

    frames["pressure"][:, 0] = np.clip((heel + rng.normal(0, 0.01, n)) * ADC_FULL_SCALE, 0, 4095).astype(np.uint16)
    frames["pressure"][:, 1] = np.clip((ball + rng.normal(0, 0.01, n)) * ADC_FULL_SCALE, 0, 4095).astype(np.uint16)
    return frames
