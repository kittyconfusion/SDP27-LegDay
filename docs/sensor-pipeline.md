# Sensor and classifier pipeline

## Initial data contract

The first acquisition milestone uses four ICM-20948 IMUs and four force-sensitive
resistors. Each synchronized frame contains:

| Input | Count | Stored type | Feature order |
| --- | ---: | --- | --- |
| Accelerometer | 4 x 3 | signed 16-bit raw count | IMU 0-3, X/Y/Z |
| Gyroscope | 4 x 3 | signed 16-bit raw count | IMU 0-3, X/Y/Z |
| Foot pressure | 4 | unsigned ADC count | left heel, left ball, right heel, right ball |

This gives 28 model inputs per time step. Magnetometer, quaternion, temperature,
and derived features can be added later under a new schema version. The physical
location and axis orientation assigned to IMU indices 0-3 must be frozen before
collecting training data.

`legday-core::SensorFrame` also carries a monotonic microsecond timestamp, sequence
number, validity masks, and status flags. Its 72-byte little-endian encoding is
stable and contains no Rust struct padding.

At 200 Hz, raw recording traffic is 14.4 kB/s. A 200-frame (one-second) rolling
window occupies 14.4 kB as encoded frames and represents 5,600 scalar input
values. Inference can run every five samples (every 25 ms) while still using the
previous second of motion. Observation-window length and decision-update latency
are separate design parameters.

## Runtime ownership

```text
IMU/FIFO + ADC acquisition (200 Hz)
              |
              v
      synchronized SensorFrame
              |
       +------+------+
       |             |
       v             v
rolling inference  recording queue
window in RAM      (double buffered)
       |             |
       v             v
 classifier       SD-card writer
       |
       v
 motion class + confidence + timestamp
```

The acquisition task owns sensor timing and is the only producer of frames. The
classifier and recorder consume completed frames. SD-card writes must never block
the sampling deadline; a bounded queue or pair of block-sized buffers separates
sampling from variable SD latency. A recording button should start and stop a
session. For pre-trigger capture, retain a short rolling history and write it when
recording starts instead of trying to hold an entire session in internal SRAM.

## ICM-20948 starting point

Begin with raw accelerometer and gyroscope burst reads. The device has a 4 kB FIFO
and data-ready/FIFO interrupts, so the eventual driver can batch reads instead of
performing 24 unrelated register transactions per time step. Four identical I2C
addresses require selecting one TCA9548A channel before communicating with each
IMU. Start at 400 kHz I2C and measure completed-frame jitter and dropped frames.

The ICM-20948 DMP can generate six- and nine-axis quaternions, but it is not a
free hardware output. SparkFun's implementation uploads roughly 14.3 kB of opaque
firmware into each IMU's processor memory, performs a long configuration sequence,
and receives DMP packets through the FIFO. That payload is unrelated to the S3's
ULP memory. The current `icm20948-rs` crate claims blocking and async I2C/SPI,
magnetometer, FIFO, interrupt, wake-on-motion, and raw-data support, but describes
its DMP API as unstable. Evaluate that crate with one IMU before writing a driver.
DMP integration should remain a later, separate milestone. Raw accel/gyro data is
sufficient to establish the collection, labeling, training, and inference pipeline.

## Pressure starting point

Each MF01A sensor is a variable resistor and needs a fixed resistor to form a
3.3 V voltage divider feeding an ADC1-capable pin. Record raw calibrated ADC counts
first. Do not assume counts are Newtons: the sensor has part-to-part variation,
hysteresis, drift, and a nominal range ending near 98 N. A person's foot load can
exceed that range, so the mechanical stack must spread force and avoid saturation.
For classification, these channels are initially contact/load proxies rather than
laboratory force measurements.

## Recording and training format

Record binary `SensorFrame` values on the MCU and convert them to CSV or a columnar
format on the Mac. Binary recording avoids float formatting and reduces SD-write
bandwidth. Every session also needs metadata outside the frame stream:

- schema and firmware version;
- nominal sample rate and actual dropped-frame count;
- IMU location, orientation, full-scale ranges, and calibration identity;
- ADC configuration and pressure-sensor identity;
- subject/session identifier and label timeline;
- model preprocessing version.

Labels should be time intervals rather than a label manually attached to every
sample. Preserve transition periods such as stand-to-walk instead of silently
forcing them into steady-state classes.

## Classifier boundary

Training can use floating-point Python, but firmware inference should consume a
versioned, deterministic preprocessing pipeline. A practical deployed artifact is
an `include_bytes!` model blob containing quantized weights, biases, input scales,
and class ordering. Rust's closest equivalent to a C header of constants is a
module with `const`/`static` values or an embedded byte array.

Keep the classifier in a hardware-independent `no_std` crate. Its public API
should accept a chronological sensor window and return class scores plus timing;
it should know nothing about I2C, GPIO, SD cards, Wi-Fi, or Bluetooth.

## Work that can proceed without the harness

1. Freeze sensor index, axis, units, validity, timestamp, and recording schemas.
2. Replay synthetic and recorded frames through `WindowBuffer` on a computer.
3. Write the binary-to-training-data converter and interval-labeling tool.
4. Define a baseline classifier and measure its accuracy before a 1-D CNN.
5. Measure firmware inference time and RAM with synthetic windows on the S3.
6. Add one IMU, then four multiplexed IMUs, then ADC pressure channels, then SD.
7. Add networking only after acquisition timing and the file format are stable.

Run the hardware-independent tests and demonstration with:

```sh
bash Scripts/core-demo.sh
```

Exercise the future recording format without hardware:

```sh
python3 Scripts/sensor_frames.py generate /tmp/legday-synthetic.bin --samples 400
python3 Scripts/sensor_frames.py decode /tmp/legday-synthetic.bin /tmp/legday-synthetic.csv
```

## References

- [TDK ICM-20948 product and datasheet page](https://www.invensense.tdk.com/en-us/products/9-axis/icm-20948)
- [SparkFun ICM-20948 DMP notes](https://github.com/sparkfun/SparkFun_ICM-20948_ArduinoLibrary/blob/main/DMP.md)
- [`icm20948-rs` driver documentation](https://docs.rs/crate/icm20948-rs/latest/source/README.md)
- [Adafruit FSR measurement guide](https://learn.adafruit.com/force-sensitive-resistor-fsr?view=all)
- [MF01A-N-221-A05 datasheet](https://cdn-shop.adafruit.com/product-files/5475/MF01A-A05.pdf)
- [`esp-hal` I2C documentation](https://docs.rs/esp-hal/latest/esp_hal/i2c/master/index.html)
- [`esp-hal` ADC documentation](https://docs.espressif.com/projects/rust/esp-hal/1.2.0-rc.0/esp32s3/esp_hal/analog/adc/struct.Adc.html)
- [`esp-hal` SDMMC/SDIO documentation](https://docs.espressif.com/projects/rust/esp-hal/1.2.0-rc.0/esp32s3/esp_hal/sdmmc/index.html)
