# IMU bring-up branch: `tim/testing-memory-and-sensors`

This branch answers one question with real hardware: can one ICM-20948 deliver
raw accelerometer + gyroscope data **and** an on-chip quaternion at roughly
200 Hz into a static one-second RAM window on the ESP32-S3, and from there onto
the SD card? It adds code alongside the existing structure rather than moving
anything.

## What was added

| Path | Purpose |
| --- | --- |
| `crates/legday-icm20948/` | Portable ICM-20948 driver over `embedded-hal` I2C: raw reads, hardware FIFO streaming, and a port of SparkFun's DMP sequence (firmware load, Quat6 + raw accel/gyro packets). |
| `crates/legday-core/src/fused.rs` | `FusedFrame` = the existing 72-byte `SensorFrame` + 4 x 12-byte Quat6 = **120 bytes**; `FusedWindowBuffer`, `FusedFrameBatch`, status-flag bit definitions. |
| `crates/legday-classifier/src/fusion.rs` | Madgwick 6-axis filter in `f32` with a software inverse square root (no `libm`), used as the fallback when the DMP is unavailable and as a live cross-check. |
| `crates/main_app/bin/imu-sensor-test.rs` | The test firmware: DMP first, software fusion fallback, 225-frame window, RECORD/CATEGORY buttons writing `.Q6B` files. |
| `Scripts/sensor_frames.py` | Now decodes 72- and 120-byte files and has a `summary` command (rate, gaps, overflow flags, channel ranges, last quaternion). |
| `Scripts/serial_capture.py` | Stdlib-only serial logger for capturing board output to a file without an interactive monitor. |
| `crates/scripts/firmware.sh` | New `imu` and `imu-build` actions. |

## Wiring (Arduino Nano ESP32 labels)

| Signal | Nano pin | GPIO |
| --- | --- | --- |
| IMU SDA (Qwiic blue) | A4 | 11 |
| IMU SCL (Qwiic yellow) | A5 | 12 |
| SD CLK / MISO / MOSI / CS | D13 / D12 / D11 / D10 | 48 / 47 / 38 / 21 |
| RECORD button to GND | D7 | 10 |
| CATEGORY button to GND | D8 | 17 |

The Adafruit ICM-20948 breakout answers at 0x69 by default; the firmware probes
0x69 then 0x68 so a SparkFun board with the ADR jumper closed also works.

## Running it

```sh
bash crates/scripts/firmware.sh imu /dev/cu.usbmodem2143101   # build, flash, monitor
bash crates/scripts/firmware.sh imu-build                     # build only
python3 Scripts/serial_capture.py /dev/cu.usbmodem2143101 30 imu.log   # log 30 s
python3 Scripts/sensor_frames.py summary /Volumes/SDCARD/WALK/S0000000.Q6B
python3 Scripts/sensor_frames.py decode  /Volumes/SDCARD/WALK/S0000000.Q6B walk0.csv
```

The port name changes between machines and reconnects; `ls /dev/cu.usbmodem*`.

Once per second the firmware prints frames per second, FIFO high-water mark,
largest burst drained in one loop pass, overflow/resync/I2C-error counters, the
latest raw accel/gyro, the DMP quaternion, the Madgwick quaternion, and each
one's predicted gravity ("down") direction in the sensor frame next to the
normalised accelerometer. At rest the two down vectors should agree with the
accelerometer; that is the quickest sanity check for the quaternion convention.

## Frame layout (120 bytes, little-endian)

| Bytes | Content |
| --- | --- |
| 0..72 | Exactly the existing `SensorFrame` encoding (timestamp, sequence, masks, flags, 4 x accel/gyro, 4 x pressure). |
| 72..84 | IMU 0 Quat6 `x, y, z` as signed Q30 `i32` (`w = sqrt(1 - x^2 - y^2 - z^2) >= 0`). |
| 84..120 | IMU 1, 2, 3 Quat6, same layout (zero until those IMUs exist). |

So a raw-only classifier takes bytes `0..72` of every 120-byte chunk; the Rust
helpers are `FusedFrame::raw_bytes`, `FusedWindowBuffer::raw_frames`, and
`FusedWindowBuffer::copy_raw_into`. Only IMU index 0 is populated on this
branch (`valid_imu_mask = 0b0001`); pressure channels are zero with
`valid_pressure_mask = 0` because the sensors are not wired yet.

`status_flags` bits (see `legday_core::fused::status`):

| Bits | Meaning |
| --- | --- |
| 0-1 | Orientation source: 0 none, 1 DMP, 2 software (Madgwick). |
| 2 | FIFO overflow or resync happened just before this frame. |
| 3 | Timestamp reconstructed from FIFO drain time and nominal period. |
| 4-5 / 6-7 | DMP accel / gyro calibration accuracy (0-3). |
| 15 | Synthetic data (unchanged from `sd-card-test`). |

Files are `S0000000.Q6B` etc. inside the same category directories the SD test
uses; `.BIN` keeps meaning 72-byte frames.

## Why the FIFO instead of polling every 5 ms

`EX_MicroSD_Data/S0000000.BIN`, recorded by the earlier synthetic test, has a
mean rate of 182 Hz instead of 200 Hz with sample gaps up to 118 ms: SD-card
writes stall the sampling loop. With the ICM-20948's 4 kB hardware FIFO the IMU
keeps sampling during a stall and the loop drains the backlog afterwards. In
raw mode a sample is 12 bytes (about 1.5 s of buffer); in DMP mode a packet is
about 40 bytes (about 0.45 s), which still covers normal SD latency.

Timestamps for drained samples are reconstructed as
`drain_time - (n - 1 - i) * period`, hence status bit 3. The absolute jitter
is bounded by one loop pass; the relative spacing is the IMU's own clock.

## Sample rate: 225 Hz, not 200 Hz

With the DLPF enabled the ICM-20948 samples at 1125 / (1 + divider) Hz, so the
choices near 200 Hz are 225 Hz (divider 4) and 187.5 Hz (divider 5). This
branch uses 225 Hz, which is also the fastest rate SparkFun documents for the
DMP. The window is therefore 225 frames (27,000 bytes) for one second.
`legday-core` still says 200 Hz / 200 samples; reconcile that in the data
contract before training data is collected.

## DMP notes for the team

- The firmware image is 14,301 bytes (`crates/legday-icm20948/src/dmp3a.bin`)
  and is written to the IMU over I2C at boot in 16-byte bursts; with read-back
  verification that takes on the order of a second. It is held in the IMU's RAM,
  so it is reloaded after every IMU power cycle.
- DMP raw accel/gyro are in hardware units at 4 g / 2000 dps, matching the raw
  register path, so the 72-byte classifier input is the same either way.
- Quat6 (game rotation vector) has arbitrary yaw and only ever reports the
  hemisphere with `w >= 0`. A single quaternion flips sign when the rotation
  passes 180 degrees from the reference, which is the "q and -q" ambiguity. For
  the classifier, prefer the *relative* thigh-to-shank quaternion
  (`QuaternionQ30::relative`) and hemisphere-align consecutive samples
  (`QuaternionQ30::aligned_to`), or feed a derived knee-flexion angle instead
  of the raw quaternion.
- Bus budget: one DMP packet costs about 1 ms of 400 kHz I2C. Four IMUs at
  225 Hz would need about 90 percent of one bus. Either split IMUs across the
  S3's two I2C controllers, or use SPI (the breakouts support it, the ICM-20948
  allows 7 MHz), or run the DMP on fewer IMUs. Raw-only FIFO mode is far
  cheaper (about 0.35 ms per sample).
- Licence: the DMP image is InvenSense firmware redistributed under SparkFun's
  MIT-licensed library; see `crates/legday-icm20948/README.md`.

## Measured on hardware (October 19, 2026, Tim's Nano ESP32 + Adafruit ICM-20948)

| Measurement | Result |
| --- | --- |
| IMU detection | `WHO_AM_I = 0xEA` at 0x69 |
| DMP firmware load + verify | 990 ms at boot (14,301 bytes over 400 kHz I2C) |
| DMP output rate | 228-229 frames/s sustained (225 Hz nominal; the IMU clock runs ~1.5 % fast) |
| Raw FIFO output rate | 228-229 frames/s sustained |
| DMP quaternion vs Madgwick | converge to the same value within ~0.003 per component at rest; both predicted gravity vectors match the normalised accelerometer |
| DMP accel/gyro accuracy flags | gyro reaches 3 after ~5 s at rest; accel stays 0 until the sensor is moved through several orientations |
| FIFO depth, DMP mode | plateaus at **1,024 bytes** (about 25 packets, ~115 ms at 225 Hz); overflow flagged and recovered |
| FIFO depth, raw mode | plateaus at **4,096 bytes** (341 samples, ~1.5 s); overflow flagged and recovered |
| Per-pass I2C cost | about 3.5 ms to drain 4 DMP packets, about 1.1 ms for 3 raw samples |
| Recording, DMP mode | 2,053 frames / 8.98 s / 228 Hz effective, 0 overflows, read back OK |
| Recording, raw mode | 1,824 frames / 7.97 s / 228 Hz effective, 0 overflows, read back OK |
| SD cost per recorder pass | 11-27 ms for 16 frames (1,920 bytes) at 4 MHz SPI |
| Firmware size | ~179 kB application image |

Consequences for the real firmware:

- In DMP mode the IMU can only buffer ~115 ms, so anything that blocks the
  sampling loop longer than that (an SD flush, a cluster allocation, a Wi-Fi
  burst) drops samples. The test firmware bounds SD work per loop pass to
  keep under that; the production design should move SD and radio work to
  the S3's second core with a RAM queue, as `docs/sensor-pipeline.md` already
  suggests.
- Raw-only FIFO mode has 4x the buffer and 3x lower bus cost, so if the
  classifier does not need on-chip quaternions, raw mode plus Madgwick on the
  S3 (a few microseconds per update) is the more robust configuration.
- The quaternions from both sources agree, so recorded data is interchangeable
  between the two modes except for `status_flags` bits 0-1.
