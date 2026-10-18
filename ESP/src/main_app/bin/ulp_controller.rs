#![cfg(feature = "esp32s3")]

use esp_hal::gpio;
use esp_hal::load_lp_code;
use esp_hal::lp_core;
use esp_hal::lp_core::{UlpCore, UlpCoreWakeupSource};
use esp_hal::peripherals::Peripherals;
use esp_println::println;
const ADDR: usize = 0x5000_0400;

pub fn init(peripherals: &mut Peripherals, ulp: &mut UlpCore) {
    ulp.stop();
    //println!("ulp core stopped");
    //let mut lp_core_code = load_lp_code!("target/riscv32imac-unknown-none-elf/release/hamstrings");

    // lp_core_code.run(&mut lp_core, UlpCoreWakeupSource::HpCpu, lp_pin);

    //println!("lp core run");
}
