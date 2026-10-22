#!/usr/bin/env bash
# Cargo runner for the ESP32-S3 (`cargo run`): flash and open the monitor.
#
# Cargo calls this with the built ELF as the last argument. It exists so the
# QIO bootloader can be passed by absolute path: Cargo resolves the runner's
# own path relative to the project, but not paths inside its arguments.
#
# Flash settings: DIO in the image header (the boot ROM loads in DIO), 80 MHz,
# and the bundled QIO bootloader switches the flash to 4 lines. Measured and
# explained in crates/scripts/bootloader/README.md.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec espflash flash --baud=921600 --monitor --chip esp32s3 --list-all-ports \
    --flash-size 16mb --flash-mode dio --flash-freq 80mhz \
    --bootloader "$script_dir/bootloader/esp32s3-qio-80m.bin" \
    "$@"
