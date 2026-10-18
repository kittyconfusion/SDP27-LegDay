#![no_std]
#![no_main]

//use esp_lp_hal::{self, delay::Delay, prelude::entry};
use panic_halt as _; // Don't include esp_backtrace for space reasons

const ADDRESS: u32 = 0x400;

//#[entry]
fn main() -> ! {
    let ptr = ADDRESS as *mut u32;
    let mut c: u32 = 0;
    loop {
        c = c.wrapping_add(1);
        unsafe {
            ptr.write_volatile(c);
            //Delay.delay_millis(1000);
        }
    }
}
