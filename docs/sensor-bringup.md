# Sensor acquisition: software-first checkpoint

Work in this Rust repository. The sibling `sensor-network` Arduino prototype is
an earlier standalone reference, not the team's firmware implementation.

## Available now without assembled hardware

The first reader is `src/bin/sensor-test.rs`, a separate bench program for the
**Adafruit ESP32-C6 Feather** in the hardware photos. It checks the ICM-20948 chip
identity, resets/wakes it, configures and verifies accel/gyro ranges, checks data
ready and emits six-axis measurements with timestamps. No dependencies were added.
The original application and S3 LED test retain their existing behavior.

The repository defaults to the **Arduino Nano ESP32 / ESP32-S3**. These are different
boards. The new sensor binary is gated to C6 and uses Feather GPIO19 SDA, GPIO18
SCL and GPIO20 STEMMA power. Do not flash it to the Nano.

On a machine with the repository's Rust/Espressif environment:

```sh
cargo build-c6 --locked --release --bin sensor-test
# Only after hardware assembly, flash and monitor:
cargo run-c6 --locked --release --bin sensor-test
```

The new C6 firmware has NOT been compiled or run on hardware here. This Windows
machine has neither Cargo/Rust nor an installed WSL distribution. Follow the
repository README's toolchain setup before attempting those commands; ensure
the C6 RISC-V target and toolchain are available as well.

## Exercise the data path now

From the repository root, with Python available:

```sh
python tools/sensors.py demo --out recordings/synthetic_demo.csv --seconds 10
python tools/sensors.py validate recordings/synthetic_demo.csv
python -m unittest discover -s tools -p "test_*.py" -v
```

Demo data is explicitly **synthetic**, both in metadata and activity label. It is
only for software integration; it does not validate sensors or train a gait model.
These offline commands need only Python's standard library.

Once hardware is ready:

```sh
python -m pip install -r tools/requirements.txt
python tools/sensors.py ports
python tools/sensors.py capture --port COM5 --out recordings/bench_001.csv --seconds 30 --label stationary --placement bench
python tools/sensors.py validate recordings/bench_001.csv
```

Replace COM5 with the enumerated port (or a Linux/macOS device path). Close
espflash/other serial monitors before capture. Capture may reset the board; its
duration includes startup. CSV and companion metadata refuse overwrite. Metadata
tracks malformed rows, sequence gaps, resets and firmware diagnostics. A reset
makes a session fail validation; split/re-record it before using it for ML.

## Initial ML data contract: legday_imu6_v1

One CSV row is one sensor read:

| Fields | Meaning |
| --- | --- |
| seq | Poll attempt counter; gaps include polls without fresh data and read failures |
| t_us | MCU microseconds since boot at start of the acquisition attempt |
| read_us | Time spent checking ready and fetching accel/gyro |
| sensor_id | imu0; placement and mounting orientation belong in session metadata |
| ax_m_s2, ay_m_s2, az_m_s2 | Sensor-frame acceleration in m/s^2, including gravity |
| gx_rad_s, gy_rad_s, gz_rad_s | Sensor-frame angular velocity in rad/s |

Initial ranges are +/-8 g and +/-1000 degrees/s; firmware scales to the units
above. Host polling targets 50 Hz, with a hardware divider of 19 and enabled low
pass filters. Actual output intervals must be measured. The timestamp is host MCU
acquisition time, not the exact physical measurement instant. No FIFO is used;
intermediate sensor samples can be skipped, so this is not lossless acquisition.

This stage intentionally does not output magnetometer, temperature, force, joint
angles or fused orientation. A missing field is not fabricated as a zero.
Magnetometer support and the TCA9548A multi-IMU reader are subsequent checkpoints.

Label only single-activity sessions with the capture label; gait transitions need
separate time-interval annotations. Before training, agree placement/axis conventions,
calibration, final rate, windowing and missing-sample handling with the ML teammate.
Keep the withheld lower-quad reference recorded; exclude it from input features
explicitly at dataset construction time to avoid reference leakage.

## Hardware handoff

Ask the assembly team to connect one IMU directly by STEMMA QT, or soldered headers:
Feather 3.3V -> IMU VIN, GND -> GND, SDA -> SDA, SCL -> SCL. Keep wiring unpowered
while assembling. This pin map is for the Adafruit breakout, not a bare ICM chip.

First observed checks: no identity/configuration errors, changing values when
tilted/rotated, acceleration magnitude near 9.81 m/s^2 at rest and low angular
velocity at rest. Record 30 seconds and inspect timing/error counts. A data-ready
timeout or read error is a failed checkpoint, not a zero-valued measurement.

Then add one IMU per mux channel, followed by FSR dividers/raw ADC, and validate
the full network's actual rate. Hardware assembly, calibration and network
redundancy remain unverified.

## Primary references

- [esp-hal I2C API](https://docs.rs/esp-hal/latest/esp_hal/i2c/master/struct.I2c.html)
- [TDK ICM-20948 datasheet, rev 1.5](https://product.tdk.com/system/files/dam/doc/product/sensor/mortion-inertial/imu/data_sheet/ds-000189-icm-20948-v1.5.pdf), sections 8 and 10 and sensor sensitivities
- [Adafruit Feather pinout](https://learn.adafruit.com/adafruit-esp32-c6-feather/pinouts)
