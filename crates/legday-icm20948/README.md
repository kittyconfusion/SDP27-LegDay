# legday-icm20948

A small, hardware-independent ICM-20948 driver written against the
`embedded-hal` 1.0 `I2c` and `DelayNs` traits, so it compiles for the ESP32-S3,
the ESP32-C6, or a host test with a mock bus.

It was written for the IMU bring-up branch and covers exactly what the LegDay
pipeline needs:

- WHO_AM_I probe, reset, power management, bank switching.
- Accelerometer / gyroscope full-scale, DLPF, and sample-rate configuration.
- Direct 12-byte accel + gyro register reads.
- Hardware FIFO streaming of accel + gyro samples (12 bytes/sample), so SD-card
  write stalls do not drop samples.
- (`dmp` feature) Loading the InvenSense DMP firmware image into the IMU and
  streaming DMP FIFO packets containing raw accel, raw gyro, and the 6-axis
  "game rotation vector" quaternion (Quat6) at up to 225 Hz.

The DMP sequence is a port of the SparkFun `ICM_20948` Arduino library's
`initializeDMP`, `enableDMPSensor`, `setDMPODRrate`, and `readDMPdataFromFIFO`
routines, which are themselves derived from InvenSense's eMD SmartMotion
examples. The 225 Hz tuning constants come from SparkFun's `DMP.md` notes.

## Firmware image licence

`src/dmp3a.bin` (14,301 bytes) is the InvenSense DMP3 firmware image
`eMD-SmartMotion-ICM20948-1.1.0-MP`, redistributed by SparkFun in
`src/util/icm20948_img.dmp3a.h` of the MIT-licensed
[SparkFun_ICM-20948_ArduinoLibrary](https://github.com/sparkfun/SparkFun_ICM-20948_ArduinoLibrary).
InvenSense's software licence permits using the image with InvenSense parts.
It is not team-authored code; disable the `dmp` feature if the image must not be
bundled into a build.
