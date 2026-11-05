#!/usr/bin/env bash
# Independent experimental image; does not build or alter the sensor executable.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../.."
action="${1:-help}"
transport="${2:-offline}"
case "$action" in
    build|flash) ;;
    *) echo "Usage: bash crates/scripts/networking.sh {build|flash} {offline|wifi|ble} [serial-port]"; exit 0 ;;
esac
case "$transport" in offline|wifi|ble) ;; *) echo 'Choose offline, wifi, or ble' >&2; exit 1 ;; esac
if [[ -f "$HOME/export-esp.sh" ]]; then source "$HOME/export-esp.sh"; fi
if ! command -v xtensa-esp32s3-elf-gcc >/dev/null 2>&1; then
    shopt -s nullglob
    for candidate in "$HOME"/.espressif/tools/xtensa-esp-elf/*/xtensa-esp-elf/bin \
        "$HOME"/.rustup/toolchains/esp/xtensa-esp*-elf/*/xtensa-esp*-elf/bin; do
        if [[ -x "$candidate/xtensa-esp32s3-elf-gcc" ]]; then export PATH="$candidate:$PATH"; fi
    done
fi
if [[ -f .networking.local.env ]]; then source .networking.local.env; fi
if [[ "$transport" == wifi && "$action" == flash ]]; then
    : "${LEGDAY_WIFI_SSID:?Set Wi-Fi SSID in .networking.local.env}"
    : "${LEGDAY_WIFI_PASSWORD:?Set Wi-Fi password in .networking.local.env}"
    : "${LEGDAY_PC_IP:?Set laptop IPv4 address in .networking.local.env}"
    if [[ "$LEGDAY_WIFI_SSID" == 'your-2.4GHz-SSID' || "$LEGDAY_WIFI_PASSWORD" == 'your-WPA2-password' ]]; then
        echo 'Replace the placeholders in .networking.local.env (in the repository root) before flashing Wi-Fi.' >&2
        exit 1
    fi
fi
cargo build --locked --release -p networking-playground --no-default-features --features "$transport"
# Save each variant so switching modes does not erase the comparison ELF.
elf="target/xtensa-esp32s3-none-elf/release/networking-playground"
cp "$elf" "$elf-$transport"
if [[ "$action" == flash ]]; then
    port="${3:-${ESPFLASH_PORT:-}}"
    args=()
    if [[ -n "$port" ]]; then args+=(--port "$port"); fi
    exec espflash flash --chip esp32s3 --flash-size 16mb --flash-mode dio --flash-freq 80mhz \
        --bootloader "$script_dir/bootloader/esp32s3-qio-80m.bin" --monitor "${args[@]}" "$elf-$transport"
fi
