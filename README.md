# Legasaurous Rex

Senior design firmware for the Arduino Nano ESP32 (ESP32-S3). Use this repository
as the shared firmware project. It uses Rust `no_std`, `esp-hal`, and Embassy /
`esp-rtos`. The earlier local `SDP_2027` folder is an ESP-IDF reference snapshot;
its APIs and build target differ from this repository.

The primary target is the ESP32-S3. The main application also has an ESP32-C6
feature configuration, available through `cargo build-c6` and `cargo run-c6`.
Keep reusable code GPIO-agnostic where practical so it can compile for either
chip. Drivers that implement the standard `embedded-hal` traits can be shared
across supported hardware, and Rust can call C/C++ libraries through FFI when a
needed component has no suitable Rust implementation.

Basic familiarity with [the Rust Book](https://doc.rust-lang.org/book/) and the
[esp-hal examples](https://github.com/esp-rs/esp-hal/tree/main/examples) will make
the firmware easier to follow.

On Windows, use WSL and attach the board's USB device to WSL with
[usbipd](https://learn.microsoft.com/en-us/windows/wsl/connect-usb). Install the
Rust and Espressif tools inside that WSL environment.

## First hardware test

From this repository on macOS or Linux:

```sh
# Compile both programs without touching the board.
bash scripts/firmware.sh build

# Compile, flash, and monitor the basic LED test.
bash scripts/firmware.sh smoke /dev/cu.usbmodem101
```

The port above was verified on Tim's Mac on October 12, 2026; it may change after
reconnecting the board. Set `ESPFLASH_PORT`, pass the current port as the second
argument, or omit the port to let espflash select it. Linux ports commonly look
like `/dev/ttyACM0`. Use a USB data cable and close other serial monitors first.

The smoke test prints `LegDay Rust smoke-test v1` and an increasing counter every
250 ms while toggling the D13 LED on **ESP32 GPIO48**. It initializes neither the
radio nor PSRAM. The LED should change state four times per second. Press `Ctrl+C` to
close the monitor; the firmware continues running on the board.

Other commands:

```sh
bash scripts/firmware.sh smoke-build             # build only the LED test
bash scripts/firmware.sh monitor /dev/cu.usbmodem101  # monitor without flashing
bash scripts/firmware.sh app /dev/cu.usbmodem101  # flash the original radio scaffold
```

`app` initializes Wi-Fi and BLE controllers and prints `Hello world!`, but does
not yet connect to Wi-Fi, advertise BLE, acquire sensors, or send classifications.
Only the smoke test has been run on hardware in the verification below.

## Matching host tools

The tested installation uses:

| Component | Verified version |
| --- | --- |
| Espressif Rust toolchain release | `1.97.0.0` (compiler `1.97.0-nightly`, commit `8ea53bcd7`, July 8, 2026) |
| espup | `0.18.0` |
| Xtensa GCC | `esp-14.2.0_20251107` / GCC `14.2.0` |
| espflash | `4.6.0` |
| esp-hal | `1.2.2` |
| esp-rtos | `0.4.0` |
| esp-radio | `1.0.0-beta.1` |
| esp-println | `0.18.0` |

The `esp` channel in `rust-toolchain.toml` is the **local toolchain name**, not a
version pin. Teammates should install the same release and keep `Cargo.lock`
committed. Use `--locked` builds; review dependency updates together, particularly
the radio beta and unstable HAL APIs.

For a new development machine with rustup / Cargo installed:

```sh
cargo install espup --version 0.18.0 --locked
espup install --targets esp32s3 --toolchain-version 1.97.0.0 \
    --crosstool-toolchain-version 14.2.0_20251107
cargo install espflash --version 4.6.0 --locked
```

These are setup instructions, not installations performed during the hardware
test. **Do not pass `--std` to espup for this project:** that skips GCC, which
this `no_std` target needs for linking. The build wrapper sources
`~/export-esp.sh` and finds GCC in the usual espup or ESP-IDF installation paths.
For another installation location, set `ESP_XTENSA_GCC_BIN` to its `bin` directory.
An existing linker on `PATH` takes precedence; use the table to check its version.

For manual Cargo commands, source the export file and ensure
`xtensa-esp32s3-elf-gcc` is on `PATH`. Then:

```sh
cargo build --locked --release --bins
cargo run --locked --release --bin smoke-test
```

`cargo run` uses the runner in `.cargo/config.toml`. The wrapper also sets the
board's 16 MB flash size explicitly when flashing. `ldproxy` and an ESP-IDF source
checkout are not required for this project's application build.

## Board memory

The [Nano ESP32 datasheet](https://docs.arduino.cc/resources/datasheets/ABX00083-datasheet.pdf)
specifies 16 MB flash, 8 MB octal PSRAM, and 512 KB internal SRAM, plus 16 KB RTC
SRAM. The CPU is a dual-core Xtensa LX7 with a maximum clock of 240 MHz.

Flash stores firmware and persistent data. SRAM and PSRAM hold working data;
PSRAM is not extra flash and must be configured before use. Neither current
program enables it. The app scaffold reserves two internal heap regions totaling
139,280 bytes; that is its chosen heap allocation, not total physical RAM.
The smoke test uses no dynamic heap. Hardware interrogation confirmed 16 MB flash;
PSRAM capacity has not yet been probed or tested on this particular board.

## Working together

Use branches of this repository for both MCU and classifier work. Keep board
drivers, timing, and transport in the firmware, and develop the classifier as a
separate pure Rust module or crate with explicit inputs and outputs. That will
allow recorded samples to be replayed on a computer and the same inference logic
to run on the MCU. `no_std` code can use `core` and, when an allocator is installed,
`alloc`; it cannot assume desktop `std`, files, or OS threads are available.

Before implementing inference, agree on timestamps, sample rate, sensor ordering,
units, calibration, window length, missing-sample handling, and classification
output fields. Train on a computer if needed, then measure inference latency and
memory on the MCU. The next hardware milestone is one IMU streaming timestamped
samples over USB; radio delivery can follow once the sample format is stable.

## Verification and references

See [the bring-up record](docs/bringup-2026-10-12.md) for the comparison of both
folders and the observed hardware output.

- [Rust on ESP Book](https://docs.espressif.com/projects/rust/book/)
- [Espressif Rust documentation](https://docs.espressif.com/projects/rust/)
- [ESP32-S3 datasheet](https://documentation.espressif.com/esp32-s3_datasheet_en.pdf)
- [Arduino Nano ESP32 datasheet](https://docs.arduino.cc/resources/datasheets/ABX00083-datasheet.pdf)
- [Rust Book](https://doc.rust-lang.org/book/)
- [Board pinout](https://docs.arduino.cc/resources/pinouts/ABX00083-full-pinout.pdf)
- [IMUs](https://www.adafruit.com/product/4554)
- [I2C multiplexer](https://www.adafruit.com/product/2717)
- [SD breakout](https://www.adafruit.com/product/4682)
- [Force sensor](https://www.adafruit.com/product/5475)
