#!/usr/bin/env bash
# Run from any directory. Keep machine-specific paths out of Cargo config.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

action="${1:-help}"
case "$action" in
    build|c6-build|main-build|smoke-build|main|smoke|app|monitor) ;;
    *)
        echo "Usage: $0 {build|c6-build|main-build|smoke-build|main|smoke|monitor} [serial-port]"
        echo "main/smoke build, flash, and monitor; Ctrl+C exits the monitor."
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
    smoke-build) cargo build --locked --release --bin smoke-test ;;
    main|app|smoke)
        binary=sdp27-leg-day
        if [[ "$action" == smoke ]]; then binary=smoke-test; fi
        cargo build --locked --release --bin "$binary"
        exec espflash flash --chip esp32s3 --flash-size 16mb --monitor \
            "$@" "target/xtensa-esp32s3-none-elf/release/$binary"
        ;;
    monitor)
        exec espflash monitor --chip esp32s3 "$@"
        ;;
esac
