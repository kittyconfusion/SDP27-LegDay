# Sensor integration

## Changes

- `record.rs`: initializes once, continuously reads sensors, exposes `latest_sensor_frame()`, and buffers frames to SD when recording is on.
- `bin.rs` / `sensors/mod.rs`: connect the reader and frame encoder to the recorder.
- `sensors/imu.rs`: reads both I2C buses concurrently; devices on each bus remain sequential. Adds the already-locked Embassy join dependency.
- `sensors/frame.rs`: encodes Tim's 120-byte little-endian format with timestamps, sequence, raw readings, and validity masks. Quaternion fields are unavailable (source=0), pending Tim's work.
- SD blocks carry their session number to prevent mixing recordings. Padding is cleared; a full queue drops the new block with a warning.
- `tests/sensor-data.rs`: format/mapping tests. `examples/sensor-check.rs`: separate USB hardware test.

Training uses four IMUs; Demo disables only slot 1 (non-prosthetic lower). All four pressure channels and fixed frame slots remain. Check validity bits before using readings.

## Agree with Nathan and Tim

- **Rate:** `config.rs` now sets both the recording loop and nominal IMU output to 225 Hz (divider 4). Recorder and USB test use 400 kHz I2C for transfer headroom; the test still prints slowly. Verify fresh counts, synchronization, and wiring/display at 400 kHz. Ranges/filters remain unchanged: +/-8g, +/-1000 degrees/s. The unused legacy `bin/imu_controller.rs` is not initialized by this recorder.
- **Quaternions/timing:** Tim supplies orientation and flags. Current timestamp is MCU read start; sensors are not synchronized.
- **File readers:** each 8192-byte SD block starts with a two-byte **big-endian inclusive used length**, followed by 120-byte **little-endian** frames and padding. Decode only `block[2..used]`. Old 40-byte recordings are incompatible; agree on versioning/decoder updates.
- **Live output:** latest-frame access is ready; PC transport still needs connecting. Slow readers skip frames.
- **Reliability:** agree on overflow, reconnect, and pressure calibration rules. Failed startup IMUs currently require restart; ADC readings cannot detect unplugged pressure sensors. Nathan's SD controller still hides some write failures.
- **Mode:** agree how the application selects Training/Demo; currently a constant in `record.rs`.
