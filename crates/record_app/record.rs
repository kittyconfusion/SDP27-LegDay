#![no_std]
#![no_main]
#![deny(clippy::mem_forget)] // Should not be used so deny
#![deny(clippy::large_stack_frames)] // General compiler error if allocate > 512kb at a time (main function given exemption)

// Run record-app with "cargo run --package quadriceps"

mod bin;

use bin::gpio_controller::{Status, gpio_loop};
use bin::sd_controller::SdController;

const RECORD_HZ: u64 = 200; // Frequency of sensor data collection
// 4 IMU * (6 BYTES) + 4 PRESSURE * (2 BYTES) + 1 TIMESTAMP_MS * (8 BYTES) = 40
const BYTES_PER_RECORD: usize = 24 + 8 + 8;
// Define the circular ring buffer used for sharing data between sensor and sd card threads
const BUFFER_SIZE: usize = 8192;
const NUM_BUFFERS: usize = 10;

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

//  === THREAD-SAFE VARIABLES ===
static RECORD_STATUS: Mutex<CriticalSectionRawMutex, Status> = Mutex::new(Status::Setup);

// Main thread-shared data buffer. RingBuffer loops around instead of overflowing and handles a whole lot for us
static RING: Mutex<
    CriticalSectionRawMutex,
    ConstGenericRingBuffer<[u8; BUFFER_SIZE], NUM_BUFFERS>,
> = Mutex::new(ConstGenericRingBuffer::<[u8; BUFFER_SIZE], NUM_BUFFERS>::new());

// NOTE: Be careful about passing this BETWEEN cores because can end up in spinloop, just take care
// For other things may want to use critical_section mutex that is caveated
// CriticalSectionRawMutex may be needed for other circumstances
static I2C0_BUS: StaticCell<Mutex<CriticalSectionRawMutex, I2c<'static, Async>>> =
    StaticCell::new();
static I2C1_BUS: StaticCell<Mutex<CriticalSectionRawMutex, I2c<'static, Async>>> =
    StaticCell::new();

// CORE 1 HAS 32 K OF STACK SPACE
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

    #[cfg(feature = "esp32s3")]
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    #[cfg(feature = "esp32c6")]
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 65536);

    let timg0 = TimerGroup::new(peripherals.TIMG0);
    esp_rtos::start(timg0.timer0, peripherals.FROM_CPU_INTR0);

    // ======= GPIO PINS =======
    // SPI
    let sclk = peripherals.GPIO47;
    let miso = peripherals.GPIO38;
    let mosi = peripherals.GPIO21;
    let cs = peripherals.GPIO18;
    // I2C 0
    let sda0 = peripherals.GPIO11;
    let scl0 = peripherals.GPIO12;
    // I2C 1
    let sda1 = peripherals.GPIO13;
    let scl1 = peripherals.GPIO14;

    let recording_light = peripherals.GPIO6;
    let recording_switch = peripherals.GPIO7;

    // ======= THREAD SETUP =======
    // Core 0 needs to setup both I2C busses
    // THREAD: Sensors and Data -> Buffer
    // THREAD: GPIO
    // Core 1 needs to setup SPI bus
    // THREAD: SD Writer

    // Start Core 1, create SPI bus, and start SD thread
    esp_rtos::start_second_core(
        peripherals.CPU_CTRL,
        peripherals.FROM_CPU_INTR1,
        CORE1_STACK.init(Stack::new()),
        move || {
            // Setup as async even though the SD library is entirely blocking
            // This is a regretable oversight (will be changed in the future but it works as is)
            let spi = Spi::new(
                peripherals.SPI2, // SPI 0 and 1 are for Flash and PSRAM
                spi::master::Config::default()
                    .with_frequency(Rate::from_khz(400))
                    .with_mode(spi::Mode::_0),
            )
            .unwrap()
            .with_sck(sclk)
            .with_miso(miso)
            .with_mosi(mosi)
            .into_async();

            // Core 1 needs its own executor so that Core 0 can still spawn tasks
            // For the future: Executors can have special interrupt-driven priorities to ensure they run on time
            static EXECUTOR: StaticCell<Executor> = StaticCell::new();
            let executor = EXECUTOR.init(Executor::new());
            executor.run(|spawner| {
                spawner.spawn(
                    // Start the SD thread
                    sensor_writer(
                        spi,
                        Output::new(cs, Level::High, OutputConfig::default()),
                        delay,
                    )
                    .unwrap(),
                );
                info!("Core 1 fully initialized");
            });
        },
    );

    // ======= CORE 0 THREADING =======
    // Setup async I2C busses and add them to their mutex to allow multiple-thread use
    let i2c0 = I2c::new(
        peripherals.I2C0,
        i2c::master::Config::default().with_frequency(Rate::from_khz(100)),
    )
    .unwrap()
    .with_sda(sda0)
    .with_scl(scl0)
    .into_async();

    let i2c1 = I2c::new(
        peripherals.I2C1,
        i2c::master::Config::default().with_frequency(Rate::from_khz(100)),
    )
    .unwrap()
    .with_sda(sda1)
    .with_scl(scl1)
    .into_async();

    // These are currently only used by 1 thread but it should be trivial to share usage
    let shared_i2c0 = I2C0_BUS.init(Mutex::new(i2c0));
    let shared_i2c1 = I2C1_BUS.init(Mutex::new(i2c1));

    spawner.spawn(sensor_recorder(shared_i2c0, shared_i2c1).unwrap());

    // Wait for RECORD_STATUS to be initialized by the SD card thread
    // Starts in Record::Setup and eventually moved to OFF/ON with next file number
    loop {
        let current_status = { *RECORD_STATUS.lock().await };
        if current_status != Status::Setup {
            break;
        }
        Timer::after_millis(10).await;
    }

    let rec_switch = Input::new(recording_switch, InputConfig::default().with_pull(Pull::Up));
    let rec_light = Output::new(recording_light, Level::Low, OutputConfig::default());

    // Know that file naming has been initialized. Start the gpio task now that display can be initialized.
    spawner.spawn(gpio_loop(&RECORD_STATUS, rec_switch, rec_light, shared_i2c0).unwrap());

    info!("Core 0 fully initialized");

    // Core 0 cannot return from main. This makes sure that idle time is spent in
    // yielding sleep as opposed to a spinlock
    loop {
        Timer::after_secs(1).await;
    }
}

// SD Card Thread Main Loop
#[embassy_executor::task]
async fn sensor_writer(spi: Spi<'static, Async>, cs: Output<'static>, delay: Delay) {
    let mut ticker = Ticker::every(Duration::from_hz(1)); // Run every second
    let mut sd = SdController::new(spi, cs, delay);

    // We do not want to overwrite an existing file. Therefore keep trying to figure out
    // what the next available file number is
    let mut current_file_num = loop {
        let file_num = sd.find_last_numerical_file();
        if file_num != u16::MAX {
            break file_num; // breaks the loop and returns the value of file_num
        }
        error!("Unable to open SD card. Trying again.");
        ticker.next().await; // could not open sd card so wait another second and try again
    };

    // Get out of global recording setup mode and "upload" current_file_num
    {
        // Do not lock for longer than the minimum time
        *RECORD_STATUS.lock().await = Status::RecordingOff(current_file_num);
    }

    // While there is data to write
    //   Figure out what file number we should use
    //   Write it
    // Yield until the next second increment
    loop {
        while let Some(read_data) = { RING.lock().await.dequeue() } {
            {
                let current_record_status = RECORD_STATUS.lock().await;
                current_file_num = match *current_record_status {
                    Status::RecordingOff(num) => num,
                    Status::RecordingOn(num) => num,
                    Status::Setup => current_file_num,
                }
            }
            sd.write(&read_data, &format!("{current_file_num}.leg"))
                .expect("File is do broke :-(");
        }

        ticker.next().await;
    }
}

// Sensor Data Collection Main Loop
// Also "uploads" data to buffer
// and Starts / Stops data recording
#[embassy_executor::task]
async fn sensor_recorder(
    i2c0: &'static Mutex<CriticalSectionRawMutex, I2c<'static, Async>>,
    i2c1: &'static Mutex<CriticalSectionRawMutex, I2c<'static, Async>>,
) {
    let mut recorder = Recording::new(); // Buffer Controller
    // Remember the last state so we know if recording should be started / stopped
    let mut last_status = *{ RECORD_STATUS.lock().await };
    let mut ticker = Ticker::every(Duration::from_hz(RECORD_HZ));

    // ===================================
    // TODO: Setup the IMUs / IMU Controller for each I2C bus
    // ===================================

    loop {
        // Determine if GPIO has requested a start of stop to data collection
        // either stop_recording or start_recording will be called if need be
        let current_status = *{ RECORD_STATUS.lock().await };

        if last_status != current_status {
            match current_status {
                Status::RecordingOff(_) => {
                    info!("Changing recording status to OFF");
                    recorder.stop_recording().await;
                }
                Status::RecordingOn(_) => {
                    info!("Changing recording status to ON");
                    recorder.start_recording();
                }
                Status::Setup => {}
            }
        }
        last_status = current_status;

        // create a new sensor data block, collect data, and push to the current sensor data buffer
        if recorder.currently_recording {
            // expects sensor_record to be in big endian order. This can be easily switched if that doesn't make sense
            let mut sensor_record: [u8; BYTES_PER_RECORD] = [0; BYTES_PER_RECORD];

            // insert timestamp at the end of a record
            sensor_record[BYTES_PER_RECORD - 8..]
                .copy_from_slice(&Instant::now().as_micros().to_be_bytes());

            recorder.write_to_buffer(&sensor_record).await;
        }
        ticker.next().await;
    }
}

pub struct Recording {
    // STRUCTURE
    // 2  bytes Block Size in bytes, inclusive
    // 24 bytes Raw IMU Data x4
    // 8  bytes Raw Pressure Data x4
    // 24 bytes Raw IMU Data x4
    // 8  bytes Raw Pressure Data x4
    // .............................
    // end padded with 0 until BUFFER_SIZE
    write_buffer: [u8; BUFFER_SIZE],
    // Current place in the write_buffer
    // don't need > 64k block so even though usize is u32, will cast to u16 when transcribing
    cursor: usize,
    currently_recording: bool,
}

impl Recording {
    pub const fn new() -> Self {
        Self {
            write_buffer: [0; BUFFER_SIZE],
            cursor: 2,
            currently_recording: false,
        }
    }

    pub async fn stop_recording(&mut self) {
        self.currently_recording = false;

        // always upload remaining data to buffer
        if self.cursor > 2 {
            self.upload_buffer().await;
        }
    }

    pub fn start_recording(&mut self) {
        self.cursor = 2;
        self.currently_recording = true;
    }

    // Transcribe the final write_buffer used length and upload its data to the thread-shared ring buffer
    pub async fn upload_buffer(&mut self) {
        self.write_buffer[0..2].copy_from_slice(&(self.cursor as u16).to_be_bytes());

        {
            RING.lock().await.enqueue(self.write_buffer);
        }

        self.cursor = 2; // Reserved for final write_buffer length
    }

    // Writes to the write_buffer an input sensor record and initiate an "upload" if almost full
    pub async fn write_to_buffer(&mut self, data: &[u8]) {
        let end = self.cursor + data.len();

        if end < self.write_buffer.len() {
            self.write_buffer[self.cursor..end].copy_from_slice(data);
            self.cursor = end;
        } else {
            error!("Write to buffer is attempting to overflow!");
        }

        // let's initiate a write due to a full (or about to be full) buffer!
        if end + BYTES_PER_RECORD >= self.write_buffer.len() {
            self.upload_buffer().await;
        }
    }
}
