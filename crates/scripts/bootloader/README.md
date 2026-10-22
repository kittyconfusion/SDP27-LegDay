# ESP32-S3 second-stage bootloader (QIO, 80 MHz)

`esp32s3-qio-80m.bin` is the ESP-IDF second-stage bootloader that Espressif
builds for Arduino, converted from ELF to a flashable image. It is used
instead of the DIO bootloader espflash bundles, because it switches the flash
controller to **QIO** (4 data lines) before starting our firmware.

Why it matters: every instruction- or data-cache miss reads the external
flash. Measured on the Nano ESP32 (October 22, 2026), the cost of a miss per
float read through the classifier's dot product:

| Flash setting | Cycles per float on a cache miss |
| --- | --- |
| DIO 40 MHz (espflash default) | 118.6 |
| DIO 80 MHz | 62.1 |
| QIO 80 MHz (this bootloader) | 37.0 |

How it works: the image header stays **DIO**, so the chip's boot ROM loads
the bootloader in the conservative mode, and this bootloader then enables QIO
(`bootloader_enable_qio_mode`). This is exactly what Arduino does for this
board (`nano_nora.build.flash_mode=dio`, `nano_nora.build.boot=qio`).
Writing QIO into the header instead makes the boot ROM itself try QIO, and
the board boot-loops; reflash in DIO to recover.

The flash chip is a GigaDevice GD25Q128 (ID `c8 4018`); its quad-enable bit
(status register 2, bit 1) was already set from the factory, so nothing on
the chip had to change. The `k` key in `imu-sensor-test` prints the flash
controller's live mode (`QIO` / `DIO`) from its registers.

A 120 MHz variant exists (`bootloader_qio_120m`) but measured the same as
80 MHz with our firmware: ESP-IDF applications finish raising the clock at
start-up (with timing calibration), which a non-IDF Rust application does not.

## Provenance

- Source: <https://github.com/espressif/esp32-arduino-libs>, branch
  `idf-release/v5.1`, file `esp32s3/bin/bootloader_qio_80m.elf`
  (ESP-IDF, Apache License 2.0).
- ELF SHA-256: `0f6b63e58cd9717f294de12b8850f45fbdaad0159a0decf56d44a629e512281b`
- Converted with esptool 4.12:
  `esptool.py --chip esp32s3 elf2image --flash_mode dio --flash_freq 80m --flash_size 16MB`
- Image SHA-256: `0654a691d790651a2d4c1a63ee55d13555e987d5cfe9e7a8bb19f593c40faa74`
