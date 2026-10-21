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
| `Scripts/sensor_frames.py` | Now decodes 72- and 120-byte files and has `summary` (rate, gaps, overflow flags, channel ranges) and `plot` (PNG of accel, gyro, quaternion and a gravity cross-check) commands. |
| `crates/legday-core/src/clock.rs` | `SampleClock`: reconstructs smooth per-sample timestamps for samples drained from a FIFO in bursts, measuring the IMU's real rate along the way. |
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
| DIAG button to GND (optional) | D9 | 18 |

The Adafruit ICM-20948 breakout answers at 0x69 by default; the firmware probes
0x69 then 0x68 so a SparkFun board with the ADR jumper closed also works.

## Pressure sensors

The MF01A is a force-sensing resistor: its resistance falls from megaohms
(no load) to a few kiloohms (full load). The ADC needs a voltage, so wire each
sensor as the top half of a divider:

```text
3V3 ── MF01A ──┬── A0 (GPIO1)      A0..A3 = GPIO1..4 = ADC1 channels 0..3
               │
              10 kΩ
               │
              GND
```

Unloaded, A0 sits near 0 V; loaded, it rises toward 3.3 V. 10 kΩ is a
starting point; a larger resistor gives more sensitivity at light loads, a
smaller one delays saturation at heavy loads, so choose it on the bench
with a foot on the sensor. Use ADC1 pins only (A0-A3): ADC2 is shared with
Wi-Fi on the ESP32-S3 and cannot be read while the radio is on.

The firmware reads all four channels once per loop pass with 11 dB
attenuation (12-bit counts, ~4095 at 3.3 V, non-linear above ~2.5 V) and
stores them in every frame's `pressure_adc`. `CONNECTED_PRESSURE_MASK` in
`imu-sensor-test.rs` sets `valid_pressure_mask`; it is 0 until a divider is
wired, so set bit 0 after wiring A0. Unconnected pins read floating noise.

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

The `plot` command writes a PNG with four panels: accel in g, gyro in
deg/s, the quaternion components, and the quaternion's predicted gravity
direction overlaid on the normalised accelerometer. When the sensor is not
accelerating, the solid and dashed lines of the last panel coincide; that is
the quickest visual check that the DMP output is believable.

```sh
python3 Scripts/sensor_frames.py plot /Volumes/SDCARD/WALK/S0000000.Q6B walk0.png
```

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

Timestamps for drained samples are reconstructed by `SampleClock`
(`legday-core`), hence status bit 3. A drain tells the MCU only that its
samples were taken *before* now, so the clock keeps regular spacing at the
measured IMU rate and only pulls an estimate earlier when a drain proves it
must be. On the bench this reduced timestamp jitter from about 4 ms (naive
back-calculation) to under 1 ms, and it reports the true rate (228.5 Hz:
the IMU crystal runs 1.5 % fast).

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

## Checking a recording without a reference instrument

`EX_MicroSD_Data/Q6B/s0000001.Q6B` (18 s: still, shaken, then rotated in
free space) was checked three ways, none of which need external equipment:

| Check | Result |
| --- | --- |
| Rotation rate implied by consecutive quaternions vs. gyro magnitude, over 3,711 samples above 1 rad/s | median ratio 1.000 |
| Quaternion-predicted gravity vs. accelerometer direction while still | about 2 degrees apart |
| Quaternion vector length | never exceeds 1 (max 0.9999998) |

The accelerometer clipped at ±4 g during shaking and the gyro at ±2000 dps
during the fastest flicks. The DMP firmware constants assume those ranges;
running at the shank can exceed 4 g, so an 8 g or 16 g DMP configuration
(different `ACC_SCALE` constants) is worth evaluating before data collection.

## Four IMUs: bus topology and timing

The ICM-20948 has two I2C addresses (AD0 low = 0x68, high = 0x69), so two
IMUs share a bus with no multiplexer; that is the "daisy chain" option, one
bus per leg. A TCA9548A multiplexer is only needed for more than two on one
bus, and it adds a channel-select write before every access.

Bus time is the constraint, not CPU time. At 400 kHz one byte costs about
22 µs, and reading the FIFO count plus one DMP packet is about 0.9 ms:

| Topology | Bus busy per 4.44 ms sample period | Verdict |
| --- | --- | --- |
| 4 IMUs, DMP, one bus (mux or chain) | ~3.6 ms (81 %) | too tight; any stall drops samples |
| 4 IMUs, DMP, two buses (two per leg) | ~1.8 ms each (40 %) | workable if the two buses run concurrently |
| 4 IMUs, raw FIFO, one bus | ~1.3 ms (30 %) | fine |
| 4 IMUs, DMP, SPI at 7 MHz | ~0.05 ms each | trivially fine |

"Concurrently" matters: the S3 has two I2C controllers, but the blocking
driver used here waits for each transfer. The production firmware should use
the async driver (or DMA) so both buses transfer at once, or put the IMUs on
SPI, which the breakouts support. Either way, one combined frame of all four
IMUs plus the four ADC channels every 5 ms is achievable; the DMP path just
needs the bus split.

Each IMU runs on its own crystal (1-2 % apart), so the four FIFOs drift
relative to each other. Pairing samples into one frame needs either a shared
FSYNC pulse (one GPIO trace to every breakout; worth adding to the PCB) or
matching by the reconstructed timestamps, which `SampleClock` provides.

## Diagnostics: measured bus, latency and wake costs

`crates/main_app/bin/imu_diag.rs` measures what the four-IMU decision depends
on. Trigger it with a button from **D9 / GPIO18 to GND** or the `x` key; it
runs about 45 s (keep the IMU still) and then normal streaming resumes. Save
the report with `python3 Scripts/serial_capture.py <port> 60 diag.log --send x@3`.

Results from Tim's bench, October 21, 2026 (two runs, agreeing within 1 %):

| Measurement | DMP | Raw |
| --- | --- | --- |
| Bytes per sample | 34 | 12 |
| Bus time per sample, drained every period | 1,041 µs | 511 µs |
| Same, with an emulated second IMU on the bus | 2,069 µs | 1,022 µs |
| Longest stall survived without losing samples | 125 ms | 1,200 ms |
| Sleep 2 s, wake to first sample | 38 ms, firmware retained | 37 ms |

I2C at 400 kHz costs 89 µs per transaction plus 23.7 µs per byte, and every
drain is two transactions (FIFO count, then data). The fixed part is most of
the raw-mode cost, which is why batching helps raw more than DMP.

Projected to four IMUs, draining every 4.44 ms period:

| Layout | Path | Bus busy per period | Worst sample age | Verdict |
| --- | --- | --- | --- | --- |
| 1 I2C bus (mux or 2 addresses) | DMP | 4.1 ms (93 %) | 8.6 ms | fails |
| | raw | 2.0 ms (46 %) | 6.5 ms | ok |
| 2 I2C buses x 2, concurrent | DMP | 2.1 ms (47 %) | 6.5 ms | ok |
| | raw | 1.0 ms (23 %) | 5.5 ms | ok |
| 2 I2C buses x 2, blocking driver | DMP | 4.1 ms | 8.6 ms | fails |
| SPI 7 MHz x 4 (calculated) | DMP | 0.21 ms (5 %) | 4.7 ms | ok |
| | raw | 0.11 ms (3 %) | 4.6 ms | ok |

"Worst sample age" is the drain interval plus the bus time, the oldest a
sample can be when it reaches RAM. At 225 Hz the interval alone is 4.44 ms, so
staying under 5 ms needs less than 0.56 ms of bus time per period; only SPI
gets there. The SPI rows assume 5 µs of driver overhead per transaction;
even at 20 µs they stay under 0.4 ms.

Other measurements: Madgwick fusion 3.4 µs per update (0.3 % of one core for
four IMUs), DMP firmware load 486 ms per IMU over I2C (984 ms with
verification), DMP gyro calibration 2-10 s at rest, four pressure ADC reads
32 µs, and the prototype classifier 2.2 ms for features plus 0.3 ms to predict.
SD writes while recording took up to 50 ms, already 40 % of the DMP's stall
budget.

## Yaw drift and the magnetometer

Quat6 integrates the gyro for heading, so yaw drifts. Two things limit it:
the DMP estimates and removes gyro bias on the fly (its gyro accuracy flag
reaches 3 after a few seconds at rest), and the project never needs absolute
heading. Knee flexion is the *relative* rotation between thigh and shank, so
a common drift cancels and only the difference between two IMUs' drift
remains. Options, cheapest first:

1. **Re-zero at a known posture.** When the classifier (or the pressure
   sensors) say the person is standing still, redefine the relative
   orientation as "straight". That removes accumulated drift without any
   extra sensor.
2. **Use the joint as a constraint.** A knee is close to a hinge, and the
   hinge axis can be found from the two gyros alone (Seel et al., 2014,
   "IMU-based joint angle measurement for gait analysis"). Projecting both
   orientations onto that axis gives a drift-free flexion angle.
3. **Magnetometer, occasionally.** The DMP can emit the raw compass (header
   bit 0x2000) or Quat9 (bit 0x0400) in the *same* packets as Quat6; no
   firmware swap is needed, only extra bits in `DATA_OUT_CTL1`, and the
   I2C-master setup for the AK09916 is already done in `dmp_initialize`.
   Enable it at a low rate, accept a reading only when the field magnitude
   is near the local value (about 50 µT) and steady, and use it to correct
   heading every few minutes. Indoors near steel or motors it will be
   rejected most of the time, which is fine for an occasional correction.
   This path is wired up but was not tested on this branch.

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
