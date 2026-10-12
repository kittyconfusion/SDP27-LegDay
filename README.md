### Legasaurous Rex

The code is written for an ESP32-S3 Arduino Nano board using [embedded rust](https://github.com/esp-rs). The codebase targets *esp-hal [v1.2](https://docs.espressif.com/projects/rust/)* and *esp-radio v1.0.0-beta.2*. *esp-hal* acts as a lower-level and completely rewritten alternative to Espressif's official ESP-IDF framework rather than being a [Rust binding](https://github.com/esp-rs/esp-idf) around it. In addition, extenal crates implementing standarized *embedded-hal* traits can be used on both the main Xtensa cores as well as the ULP Core when using *[esp-lp-hal](https://github.com/esp-rs/esp-hal/tree/main/esp-lp-hal)* (while still limited to 8 KB of shared [RTC Slow Memory](https://documentation.espressif.com/esp32-s3_datasheet_en.pdf)). 

It is suggested (by yours truly) that [basic familiarity with Rust](https://doc.rust-lang.org/book/) and the various 

[Getting Started Guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html)s

[Board Specs](https://docs.arduino.cc/hardware/nano-esp32/#tech-specs)

[Board Pinout](https://docs.arduino.cc/resources/pinouts/ABX00083-full-pinout.pdf)

[Board Datasheet](https://docs.arduino.cc/resources/datasheets/ABX00083-datasheet.pdf)

[IMUs](https://www.adafruit.com/product/4554)

[I2C Multiplexer](https://www.adafruit.com/product/2717)

[SD Breakout](https://www.adafruit.com/product/4682)

[Force Sensor](https://www.adafruit.com/product/5475)
