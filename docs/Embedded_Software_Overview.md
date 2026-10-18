The code has been written for _ESP32-S3_ and _ESP32-C6_ boards using a variant of rust for embedded systems. We have decided to use _[esp-hal](https://github.com/esp-rs/esp-hal/tree/main)_, a low-level hardware abstraction layer, for development. _esp-hal_ is _no_std_, meaning that it does not implement the Rust standard library and basic functionality is provided through _esp-hal_ and optional add-ons. The low power coprocessor can be programmed using _[esp-lp-hal](https://github.com/esp-rs/esp-hal/tree/main/esp-lp-hal)_.

This choice was made to simplify development and overhead compared to the primary alternative, _esp-idf_.

The folder `crates/legday-core` contains custom library functions for chip-agnostic interfacing with sensors and files.

**We are currently targeting:**

_[esp-hal v1.2](https://docs.espressif.com/projects/rust/)_ and associated dependancies for base system functions.

_[esp-radio v1.0.0-beta.2](https://docs.espressif.com/projects/rust/esp-radio/1.0.0-beta.2/index.html)_ for WiFi and Bluetooth functionality.

**Examples** for _esp-hal_ and _esp-radio_ can be found [here](https://github.com/esp-rs/esp-hal/tree/main/examples).

Basic familiarity with Rust would be helpful before heading deep into the code. Specifically, [Chapters 1-6](https://doc.rust-lang.org/book/) of the official Rust language guidebook. Later chapters are not always applicable because of _no_std_.

External crates (libraries) implementing a standarized set of _embedded-hal_ traits can be used on both the main Xtensa cores as well as the ULP Core. Additionally, it is possible to use and call C/C++ libraries if need by from within Rust.
