#![no_std]
#![no_main]
#![deny(clippy::mem_forget)] // Should not be used so deny
#![deny(clippy::large_stack_frames)] // General compiler error if allocate > 512kb at a time (main function given exemption)

mod bin;

#[cfg(feature = "esp32s3")]
use bin::ulp_controller;

use embassy_executor::Spawner;
use embassy_time::{Duration, Timer};
use esp_hal::clock::CpuClock;

#[cfg(feature = "esp32s3")]
use esp_hal::lp_core::UlpCore;
#[cfg(feature = "esp32s3")]
use esp_hal::system::Stack;
#[cfg(feature = "esp32s3")]
use static_cell::ConstStaticCell;

use esp_hal::timer::timg::TimerGroup;
use esp_println::println;
use log;

// Alternative to providing our own panic_handler
// should print backtrace to serial upon fatal error
use esp_backtrace as _;

extern crate alloc;

// Creates a default app-descriptor required by the esp-idf bootloader. aka makes things work right.
esp_bootloader_esp_idf::esp_app_desc!();

// Only need the Core 1 stack if the core actually exists
#[cfg(feature = "esp32s3")]
static CORE1_STACK: ConstStaticCell<Stack<8192>> = ConstStaticCell::new(Stack::new());

// suppress compiler complaining about memory allocation
#[allow(clippy::large_stack_frames)]
// alternative to the synchronous version, #[esp_hal::entry], and sets up a lot of stuff
#[esp_rtos::main]
async fn main(_spawner: Spawner) -> ! {
    esp_println::logger::init_logger_from_env(); // Makes log:: work

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    // This is ram used by the bootloader and will be left otherwise unused
    // able to be used for dynamic memory allocation
    #[cfg(feature = "esp32s3")]
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    #[cfg(feature = "esp32c6")]
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 65536);

    // COEX (the coexistance engine for wifi and bluetooth at the same time) needs more RAM
    // esp_alloc::heap_allocator!(size: 64 * 1024);

    let timg0 = TimerGroup::new(peripherals.TIMG0);
    esp_rtos::start(timg0.timer0, peripherals.FROM_CPU_INTR0);

    log::info!("Embassy initialized!");

    #[cfg(feature = "esp32s3")]
    {
        let ulp_core = UlpCore::new(peripherals.ULP_RISCV_CORE);

        log::info!("S3 initialized");
        esp_rtos::start_second_core(
            peripherals.CPU_CTRL,
            peripherals.FROM_CPU_INTR1,
            CORE1_STACK.take(),
            || {
                // This is the second core (core 1)'s main loop!
                log::info!("Core 1 running!");
                // and immediately doing nothing (except for background async tasks)
            },
        );
        loop {
            log::info!("");
            Timer::after(Duration::from_secs(5)).await;
        }
    }
    #[cfg(feature = "esp32c6")]
    {
        log::info!("C6 initialized");
        loop {
            log::info!("Hello world!");
            Timer::after(Duration::from_secs(5)).await;
        }
    }
}

// create chip specific code with
// if cfg!(feature = "esp32s3")  // chooses whether to run at run-time
// #[cfg(feature = "esp32s3")]   // chooses whether to include (at all) at compile-time
// for inspiration have a look at the examples at https://github.com/esp-rs/esp-hal/tree/main/examples
// initial generator parameters: -o esp32s3 -o esp32s3-wroom-1-octal-psram -o unstable-hal -o alloc -o embassy -o wifi -o ble-trouble -o log -o vscode -o esp
