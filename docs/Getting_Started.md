## Embedded Programming

The physical portion of the project uses both _Arduino Nano ESP32-S3_ and _Adafruit Feather ESP32-C6_ development boards programmed with embedded rust. These are both used to run sensor and network code, but only the _S3_ is expected to run machine learning workloads and implement power-saving measures. These two setups will require differing pin layouts.

**On Windows** Please install the toolchain in the next paragraph from within [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to avoid persistant linker issues. It will also be necessary to route the ESP32's serial interface to the WSL virtual machine to build and run. This can be done using [usbpid](https://learn.microsoft.com/en-us/windows/wsl/connect-usb).

Please follow the directions from [Sections 3.1 and 3.2](https://docs.espressif.com/projects/rust/book/getting-started/index.html) of the official _Rust on ESP Book_. Both support for Xtensa _(ESP32-S3)_ and RISC-V chips _(ESP32-C6)_ is recommended.

Once the toolchain is functional, the code can be compiled and run, respectively, with the following terminal commands.
_Arduino Nano ESP32-S3_ : `cargo build` and `cargo run`
_Adafruit Feather ESP32-C6_ : `cargo build-c6` and `cargo run-c6`

## Network Script

## Blender
