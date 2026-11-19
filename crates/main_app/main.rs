#![no_std]
#![no_main]
#![deny(clippy::mem_forget)] // Should not be used so deny
#![deny(clippy::large_stack_frames)] // General compiler error if allocate > 512kb at a time (main function given exemption)

#[cfg(feature = "esp32c6")]
compile_error!("main-app will only compile on an esp32s3");

mod bin;

use bin::gpio_controller::{Status, gpio_loop};
use bin::sensors::{
    ImuReader, PressureReader, SensorReader,
    config::{AcquisitionMode, I2C_BUS_KHZ, SAMPLE_HZ},
    frame::{self, SensorFrame},
};
use bin::ulp_controller;

// One shared rate sets both polling and the hardware divider in config.rs.
// TODO(Nathan, Tim): verify sustained fresh counts and synchronization on hardware;
// a nominal 225-Hz setting does not guarantee every frame has four fresh IMUs.
const RECORD_HZ: u64 = SAMPLE_HZ;
// Tim's 120-byte little-endian payload replaces the old 40-byte placeholder.
const BYTES_PER_RECORD: usize = frame::FRAME_BYTES;
// Change to Demo to omit only the removable non-prosthetic lower reference.
// TODO(Nathan, Tim): agree how the final application selects training/demo.
const ACQUISITION_MODE: AcquisitionMode = AcquisitionMode::Training;
// Define the circular ring buffer used for sharing data between sensor and sd card threads
// Define the circular ring buffer used for sharing data between sensor and sd card threads
const FRAME_BUFFER: usize = 10;

// THREADING
use embassy_executor::Spawner;
use embassy_time::{Duration, Instant, Ticker, Timer};
use esp_rtos::embassy::Executor;

// HARDWARE INTERFACES
use esp_hal::{
    Async,
    clock::CpuClock,
    delay::Delay,
    gpio::{Input, InputConfig, Level, Output, OutputConfig, Pull},
    i2c,
    i2c::master::I2c,
    lp_core::UlpCore,
    spi,
    spi::master::Spi,
    system::Stack,
    time::Rate,
    timer::timg::TimerGroup,
};

// SHARED DATA
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::mutex::Mutex;

// RESERVE LARGE MEMORY BLOCKS
use static_cell::StaticCell;

// SHARED BUFFER
use ringbuffer::{ConstGenericRingBuffer, RingBuffer};

use esp_backtrace as _;
use log::{self, debug, error, info, warn};

extern crate alloc;

// FANCY STRING FORMATTING
use alloc::format;

use crate::bin::tasks::shared_status::{
    GlobalConfig, preset_classifying_no_network, preset_demo_day, preset_sd_readout, preset_sensors,
};

//  === THREAD-SAFE VARIABLES ===
static RECORD_STATUS: Mutex<CriticalSectionRawMutex, Status> = Mutex::new(Status::Setup);

// With what functionality do we want to run with
static GLOBAL_CONFIG: GlobalConfig = preset_sensors();

// Acquisition continues while SD recording is off. A future simulation transport
// can copy this snapshot without owning/reinitializing the hardware reader.
// TODO(Nathan, Tim): this is latest-only, not a lossless stream. Transport and
// quaternion generation remain their owners' work; use sequence to spot gaps.
static LATEST_SENSOR_FRAME: Mutex<CriticalSectionRawMutex, Option<SensorFrame>> = Mutex::new(None);

pub async fn latest_sensor_frame() -> Option<SensorFrame> {
    *LATEST_SENSOR_FRAME.lock().await
}

// Main thread-shared data buffer. RingBuffer loops around instead of overflowing and handles a whole lot for us
static RING: Mutex<CriticalSectionRawMutex, ConstGenericRingBuffer<SensorFrame, FRAME_BUFFER>> =
    Mutex::new(ConstGenericRingBuffer::<SensorFrame, FRAME_BUFFER>::new());

// NOTE: Be careful about passing this BETWEEN cores because can end up in spinloop, just take care
// For other things may want to use critical_section mutex that is caveated
// CriticalSectionRawMutex may be needed for other circumstances
// Note that SPI0 and SPI1 are reserved for Flash and PSRAM so use SPI2 and SPI3
static I2C0_BUS: StaticCell<Mutex<CriticalSectionRawMutex, I2c<'static, Async>>> =
    StaticCell::new();
static I2C1_BUS: StaticCell<Mutex<CriticalSectionRawMutex, I2c<'static, Async>>> =
    StaticCell::new();
// static SPI2_BUS: StaticCell<Mutex<CriticalSectionRawMutex, Spi<'static, Async>>> =
//     StaticCell::new();

// CORE 1 HAS 32 K OF STACK SPACE
// ESP-RADIO recommends 16K + 16K for classifier
static CORE1_STACK: StaticCell<Stack<{ 1024 * 16 * 2 }>> = StaticCell::new();

// Creates a default app-descriptor required by the esp-idf bootloader. aka makes things work right.
esp_bootloader_esp_idf::esp_app_desc!();

// suppress compiler complaining about memory allocation
#[allow(clippy::large_stack_frames)]
#[esp_rtos::main]
async fn main(spawner: Spawner) -> ! {
    esp_println::logger::init_logger_from_env(); // Makes error!, info!, etc. work
    info!("Start'er up");

    // ======= BOILERPLACE INITIALIZATION ========

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let delay = Delay::new();

    // Reclaim RAM used for startup
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    // Start RTOS
    let timg0 = TimerGroup::new(peripherals.TIMG0);
    esp_rtos::start(timg0.timer0, peripherals.FROM_CPU_INTR0);

    // Enable the use of extenal 8 mb psram
    esp_alloc::psram_allocator!(peripherals.PSRAM, esp_hal::psram);

    // Currently a stub. Low power coprocessor
    let mut ulp_core = UlpCore::new(peripherals.ULP_RISCV_CORE);
    ulp_controller::init(&mut ulp_core);

    // ======= GPIO PINS =======
    // SPI 2
    //let sclk = peripherals.GPIO47;
    //let miso = peripherals.GPIO38;
    //let mosi = peripherals.GPIO21;
    //let cs = peripherals.GPIO18;

    // I2C 0
    let sda0 = peripherals.GPIO11;
    let scl0 = peripherals.GPIO12;

    // I2C 1
    let sda1 = peripherals.GPIO13;
    let scl1 = peripherals.GPIO14;

    // GPIO
    let power_light = peripherals.GPIO18;
    let network_connect_light = peripherals.GPIO17;
    let classifier_light = peripherals.GPIO6;
    let on_switch = peripherals.GPIO7;

    // ======= THREAD SETUP =======
    // Core 0 needs to setup both I2C busses
    // THREAD: Sensors and Data -> Classifier Buffer
    // THREAD: GPIO             -> Global enum and status display
    //
    // Core 1
    // THREAD: Network
    // THREAD: Classifier       -> Pass data to network

    // Start Core 1, create SPI bus, and start SD thread
    esp_rtos::start_second_core(
        peripherals.CPU_CTRL,
        peripherals.FROM_CPU_INTR1,
        CORE1_STACK.init(Stack::new()),
        move || {
            // Core 1 needs its own executor so that Core 0 can still spawn tasks
            // For the future: Executors can have special interrupt-driven priorities to ensure they run on time
            static EXECUTOR: StaticCell<Executor> = StaticCell::new();
            let executor = EXECUTOR.init(Executor::new());
            executor.run(|spawner| {
                if GLOBAL_CONFIG == preset_classifying_no_network() {
                    // Create the classifying task thread
                }
                if GLOBAL_CONFIG == preset_demo_day() {
                    // Create the network task thread
                }
                info!("Core 1 fully initialized");
            });
        },
    );

    // ======= CORE 0 THREADING =======
    // Setup async I2C busses and add them to their mutex to allow multiple-thread use
    let i2c0 = I2c::new(
        peripherals.I2C0,
        i2c::master::Config::default().with_frequency(Rate::from_khz(I2C_BUS_KHZ)),
    )
    .unwrap()
    .with_sda(sda0)
    .with_scl(scl0)
    .into_async();

    let i2c1 = I2c::new(
        peripherals.I2C1,
        i2c::master::Config::default().with_frequency(Rate::from_khz(I2C_BUS_KHZ)),
    )
    .unwrap()
    .with_sda(sda1)
    .with_scl(scl1)
    .into_async();

    // These are currently only used by 1 thread but it should be trivial to share usage
    let shared_i2c0 = I2C0_BUS.init(Mutex::new(i2c0));
    let shared_i2c1: &mut Mutex<CriticalSectionRawMutex, I2c<'static, Async>> =
        I2C1_BUS.init(Mutex::new(i2c1));

    let pressure = PressureReader::new(
        peripherals.ADC1,
        peripherals.GPIO1,
        peripherals.GPIO2,
        peripherals.GPIO3,
        peripherals.GPIO4,
    );

    //let rec_switch = Input::new(recording_switch, InputConfig::default().with_pull(Pull::Up));
    //let rec_light = Output::new(recording_light, Level::Low, OutputConfig::default());

    if GLOBAL_CONFIG != preset_sd_readout() {
        // Start up sensor thread task

        // Start up gpio thread task
    }

    info!("Core 0 fully initialized");

    // Core 0 cannot return from main. This makes sure that idle time is spent in
    // yielding sleep as opposed to a spinlock
    // In the future, a sleep mode communicating with the ulpcore will likely have logic here
    loop {
        Timer::after_secs(1).await;
    }
}
