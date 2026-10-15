### Legasaurous Rex

The code is written for an ESP32-S3 Arduino Nano board using embedded rust and no_std. The codebase targets *esp-hal [v1.2](https://docs.espressif.com/projects/rust/)* and *esp-radio v1.0.0-beta.2*. *esp-hal* acts as a lower-level and completely rewritten alternative to Espressif's official ESP-IDF framework rather than as a binding.

The code should run on both the ESP32 S3 and C6. However, differences do exist and most code outside of main should be gpio pin agnostic. Alternative commands have been made for the C6: *cargo build-c6* and *cargo run-c6*.

It is suggested (by yours truly) that [basic familiarity with Rust](https://doc.rust-lang.org/book/) and the various crates and [examples](https://github.com/esp-rs/esp-hal/tree/main/examples) provided by *esp-hal* be glanced over (at the very least) and understood (preferably). Rust, while opinionated, ensures that most errors happen at compile-time.

External crates implementing standarized *embedded-hal* traits can be used on both the main Xtensa cores as well as the ULP Core when using *[esp-lp-hal](https://github.com/esp-rs/esp-hal/tree/main/esp-lp-hal)*. Additionally, it is possible to use and call C/C++ libraries if need by from within Rust.

**On Windows** Please configure within WSL because the linker hates my (and probably your) rust linker on Windows. You will need to install all of the dependancies within your instance and install [usbpid](https://learn.microsoft.com/en-us/windows/wsl/connect-usb) to attach the ESP usb serial connection to the WSL VM.



[Getting Started](https://docs.espressif.com/projects/rust/book/getting-started/toolchain.html)

[Board Specs](https://docs.arduino.cc/hardware/nano-esp32/#tech-specs)

[Board Pinout](https://docs.arduino.cc/resources/pinouts/ABX00083-full-pinout.pdf)

[esp32s3 Datasheet](https://documentation.espressif.com/esp32-s3_datasheet_en.pdf)

[Board Datasheet](https://docs.arduino.cc/resources/datasheets/ABX00083-datasheet.pdf)

[IMUs](https://www.adafruit.com/product/4554)

[I2C Multiplexer](https://www.adafruit.com/product/2717)

[SD Breakout](https://www.adafruit.com/product/4682)

[Force Sensor](https://www.adafruit.com/product/5475)
