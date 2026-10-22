#!/usr/bin/env bash
# Run from any directory. Keep machine-specific paths out of Cargo config.
set -euo pipefail

firmware_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(cd -- "$firmware_dir/.." && pwd)"
cd "$firmware_dir"

action="${1:-help}"
case "$action" in
    build|c6-build|main-build|sd-build|imu-build|main|sd|imu|app|monitor) ;;
    *)
        echo "Usage: $0 {build|c6-build|main-build|sd-build|imu-build|main|sd|imu|monitor} [serial-port]"
        echo "main/sd/imu build, flash, and monitor; Ctrl+C exits the monitor."
        exit 0
        ;;
esac

if [[ -f "$HOME/export-esp.sh" ]]; then
    source "$HOME/export-esp.sh"
fi

# The previous std-only espup setup skipped installing GCC into the esp toolchain.
# Reuse the ESP-IDF-managed installation when available, or espup's installation.
if [[ "$action" != monitor && "$action" != c6-build ]] && ! command -v xtensa-esp32s3-elf-gcc >/dev/null 2>&1; then
    if [[ -n "${ESP_XTENSA_GCC_BIN:-}" ]]; then
        export PATH="$ESP_XTENSA_GCC_BIN:$PATH"
    else
        shopt -s nullglob
        gcc_candidates=(
            "$HOME"/.espressif/tools/xtensa-esp-elf/*/xtensa-esp-elf/bin
            "$HOME"/.rustup/toolchains/esp/xtensa-esp*-elf/*/xtensa-esp*-elf/bin
        )
        gcc_bin=""
        for candidate in "${gcc_candidates[@]-}"; do
            if [[ -x "$candidate/xtensa-esp32s3-elf-gcc" ]]; then
                gcc_bin="$candidate"
            fi
        done
        if [[ -n "$gcc_bin" ]]; then
            export PATH="$gcc_bin:$PATH"
        fi
    fi
    if ! command -v xtensa-esp32s3-elf-gcc >/dev/null 2>&1; then
        echo "Xtensa GCC is missing. Install espup for no_std (without --std)." >&2
        echo "See README.md for the tested toolchain versions." >&2
        exit 1
    fi
fi

serial_port="${2:-${ESPFLASH_PORT:-}}"
set --
if [[ -n "$serial_port" ]]; then
    set -- --port "$serial_port"
fi

case "$action" in
    build) cargo build --locked --release --bins ;;
    c6-build)
        cargo build --locked --release --no-default-features --features esp32c6 \
            --target riscv32imac-unknown-none-elf
        ;;
    main-build) cargo build --locked --release --bin sdp27-leg-day ;;
    sd-build) cargo build --locked --release --bin sd-card-test ;;
    imu-build) cargo build --locked --release --bin imu-sensor-test ;;
    main|app|sd|imu)
        binary=sdp27-leg-day
        if [[ "$action" == sd ]]; then binary=sd-card-test; fi
        if [[ "$action" == imu ]]; then binary=imu-sensor-test; fi
        cargo build --locked --release --bin "$binary"
        # DIO at 80 MHz doubles flash bandwidth over espflash's default (40 MHz):
        # every instruction- or data-cache miss is served twice as fast, which
        # took the classifier from 31 ms to 22 ms with weights in flash. QIO is
        # faster still but needs the flash's quad-enable bit, which espflash's
        # bundled bootloader does not set (the board boot-loops; reflash in DIO
        # to recover). Measured October 22, 2026; see docs/classifier.md.
        exec espflash flash --chip esp32s3 --flash-size 16mb \
            --flash-mode dio --flash-freq 80mhz --monitor \
            "$@" "$workspace_dir/target/xtensa-esp32s3-none-elf/release/$binary"
        ;;
    monitor)
        exec espflash monitor --chip esp32s3 "$@"
        ;;
esac
