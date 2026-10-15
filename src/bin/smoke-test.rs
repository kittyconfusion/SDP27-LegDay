#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock,
    delay::Delay,
    gpio::{Level, Output, OutputConfig},
};

// Both Rust frameworks use the ESP-IDF image format for the bootloader.
esp_bootloader_esp_idf::esp_app_desc!();

#[esp_hal::main]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    // Nano ESP32 D13 / LED_BUILTIN is GPIO48, not GPIO13.
    let mut led = Output::new(peripherals.GPIO48, Level::Low, OutputConfig::default());
    let delay = Delay::new();

    esp_println::println!(
        "LegDay Rust smoke-test v1 | package {} | ESP32-S3 | 240 MHz",
        env!("CARGO_PKG_VERSION")
    );
    esp_println::println!("GPIO48 toggles every 250 ms; radio and PSRAM are not initialized");

    let mut toggles: u64 = 0;
    loop {
        led.toggle();
        toggles = toggles.wrapping_add(1);
        esp_println::println!("LegDay Rust smoke-test v1 | LED toggle {toggles}");
        delay.delay_millis(250);
    }
}
