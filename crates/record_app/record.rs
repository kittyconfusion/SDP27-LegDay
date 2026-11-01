#![no_std]
#![no_main]
#![deny(clippy::mem_forget)] // Should not be used so deny
#![deny(clippy::large_stack_frames)] // General compiler error if allocate > 512kb at a time (main function given exemption)

mod bin;

use bin::sd_controller::SdController;

const RECORD_HZ: u64 = 200;
// 4 IMU * (6 BYTES) + 4 PRESSURE * (2 BYTES) + 1 TIMESTAMP_MS * (8 BYTES) = 40
const BYTES_PER_RECORD: usize = 24 + 8 + 8;
const BUFFER_SIZE: usize = 8192;
const NUM_BUFFERS: usize = 10;

// BYES_PER_SECOND * RECORDS_PER_SAVE should be below ~200 kbit

use core::cell::RefCell;
use embassy_executor::Spawner;
use embassy_time::{Duration, Instant, Ticker, Timer};
use esp_hal::{
    Async,
    clock::CpuClock,
    delay::Delay,
    gpio::{Input, Level, Output, OutputConfig},
    i2c,
    i2c::master::I2c,
    spi,
    spi::master::Spi,
    system::Stack,
    time::Rate,
};
use esp_rtos::embassy::{Executor, InterruptExecutor};

use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::mutex::Mutex;

use static_cell::{ConstStaticCell, StaticCell};

use esp_hal::timer::timg::{self, TimerGroup};
use log::{self, error};
use ringbuffer::{ConstGenericRingBuffer, RingBuffer};

// Alternative to providing our own panic_handler
// should print backtrace to serial upon fatal error
use esp_backtrace as _;

extern crate alloc;
use alloc::format;

use crate::bin::imu_controller::I2cController;

static RING: Mutex<
    CriticalSectionRawMutex,
    ConstGenericRingBuffer<[u8; BUFFER_SIZE], NUM_BUFFERS>,
> = Mutex::new(ConstGenericRingBuffer::<[u8; BUFFER_SIZE], NUM_BUFFERS>::new());

// NOTE: Be careful about passing this BETWEEN cores because can end up in spinloop, just take care
// For other things may want to use critical_section mutex but that is caveated
static I2C0_BUS: StaticCell<Mutex<CriticalSectionRawMutex, I2c<'static, Async>>> =
    StaticCell::new();
static I2C1_BUS: StaticCell<Mutex<CriticalSectionRawMutex, I2c<'static, Async>>> =
    StaticCell::new();

static CORE1_STACK: StaticCell<Stack<8192>> = StaticCell::new();

// Creates a default app-descriptor required by the esp-idf bootloader. aka makes things work right.
esp_bootloader_esp_idf::esp_app_desc!();

// suppress compiler complaining about memory allocation
#[allow(clippy::large_stack_frames)]
#[esp_rtos::main]
async fn main(spawner: Spawner) -> ! {
    esp_println::logger::init_logger_from_env(); // Makes error!, info!, etc. work

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let delay = Delay::new();

    #[cfg(feature = "esp32s3")]
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    #[cfg(feature = "esp32c6")]
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 65536);

    let timg0 = TimerGroup::new(peripherals.TIMG0);
    esp_rtos::start(timg0.timer0, peripherals.FROM_CPU_INTR0);

    let sclk = peripherals.GPIO0;
    let miso = peripherals.GPIO2;
    let mosi = peripherals.GPIO11;
    let cs = peripherals.GPIO12;

    let sda0 = peripherals.GPIO4;
    let scl0 = peripherals.GPIO5;
    let sda1 = peripherals.GPIO6;
    let scl1 = peripherals.GPIO7;

    // let's spawn some tasks!
    // problematic blocking code for sd writing runs on core0
    // gpio control and the key sensor read task run on core1

    esp_rtos::start_second_core(
        peripherals.CPU_CTRL,
        peripherals.FROM_CPU_INTR1,
        CORE1_STACK.init(Stack::new()),
        move || {
            let spi = Spi::new(
                peripherals.SPI2,
                spi::master::Config::default()
                    .with_frequency(Rate::from_khz(400))
                    .with_mode(spi::Mode::_0),
            )
            .unwrap()
            .with_sck(sclk)
            .with_miso(miso)
            .with_mosi(mosi)
            .into_async();

            static EXECUTOR: StaticCell<Executor> = StaticCell::new();
            let executor = EXECUTOR.init(Executor::new());
            executor.run(|spawner| {
                spawner.spawn(
                    sensor_writer(
                        spi,
                        Output::new(cs, Level::High, OutputConfig::default()),
                        delay,
                    )
                    .unwrap(),
                );
            });
        },
    );

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

    let shared_i2c0 = I2C0_BUS.init(Mutex::new(i2c0));
    let shared_i2c1 = I2C1_BUS.init(Mutex::new(i2c1));

    spawner.spawn(sensor_recorder(shared_i2c0, shared_i2c1).unwrap());
    loop {
        Timer::after_secs(1).await;
    }
}

#[embassy_executor::task]
async fn sensor_writer(spi: Spi<'static, Async>, cs: Output<'static>, delay: Delay) {
    let mut ticker = Ticker::every(Duration::from_hz(1));
    let mut sd = SdController::new(spi, cs, delay);

    let current_file_num = 1 + loop {
        let file_num = sd.find_last_numerical_file();
        if file_num != u16::MAX {
            break file_num; // breaks the loop and returns the value of fileNum
        }
        ticker.next().await; // could not open sd card
    };

    loop {
        let mut buffer_length = { RING.lock().await.len() }; // Automatically release the lock when scope is dropped

        while buffer_length > 0 {
            let read_buffer: [u8; BUFFER_SIZE];
            {
                let mut buffer = RING.lock().await;

                read_buffer = buffer
                    .dequeue()
                    .expect("Read buffer unable to dequeue. This should be protected against.");
                buffer_length = buffer.len();
            } // release the lock before the long synchronous sd card write

            sd.write(&read_buffer, &format!("{current_file_num}.leg"))
                .unwrap();
        }
        ticker.next().await;
    }
}

#[embassy_executor::task]
async fn gpio() {}

#[embassy_executor::task]
async fn sensor_recorder(
    i2c0: &'static Mutex<CriticalSectionRawMutex, I2c<'static, Async>>,
    i2c1: &'static Mutex<CriticalSectionRawMutex, I2c<'static, Async>>,
) {
    let mut recorder = Recording::new();
    let mut i2c_control_0 = I2cController::new(i2c0, true).await;
    let mut i2c_control_1 = I2cController::new(i2c1, false).await;
    let mut ticker = Ticker::every(Duration::from_hz(RECORD_HZ));

    loop {
        if recorder.currently_recording {
            // expects sensor_record to be in big endian order. This can be easily switched if that doesn't make sense
            let mut sensor_record: [u8; BYTES_PER_RECORD] = [1; BYTES_PER_RECORD];

            // timestamp
            sensor_record[BYTES_PER_RECORD - 8..]
                .copy_from_slice(&Instant::now().as_micros().to_be_bytes());

            recorder.write_to_buffer(&sensor_record);
        }
        ticker.next().await;
    }
}

pub struct Recording {
    // Structure: 2b block size (inclusize of themselves), 6b IMU x4, 2b Press x4, 8b time, 6b IMU x4, 2b Press x4, 8b time, ...
    write_buffer: [u8; BUFFER_SIZE],
    // don't need > 64k block so even though usize = u32, will cast to u16 when transcribing
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

    pub async fn write_empty_record(&mut self) {
        // expects sensor_record to be in big endian order. This can be easily switched if that doesn't make sense
        let mut sensor_record: [u8; BYTES_PER_RECORD] = [0; BYTES_PER_RECORD];
        // timestamp
        sensor_record[BYTES_PER_RECORD - 8..]
            .copy_from_slice(&Instant::now().as_micros().to_be_bytes());

        self.write_to_buffer(&sensor_record);
    }

    pub async fn stop_recording(&mut self) {
        self.currently_recording = false;

        // upload remaining data to buffer
        if self.cursor > 2 {
            self.upload_buffer();
        }
    }

    pub async fn start_recording(&mut self) {
        self.cursor = 2;
        self.currently_recording = true;
    }

    pub async fn upload_buffer(&mut self) {
        self.write_buffer[0..2].copy_from_slice(&(self.cursor as u16).to_be_bytes());

        {
            RING.lock().await.enqueue(self.write_buffer);
        }

        self.cursor = 2;
    }
    pub fn write_to_buffer(&mut self, data: &[u8]) {
        let end = self.cursor + data.len();

        if end < self.write_buffer.len() {
            self.write_buffer[self.cursor..end].copy_from_slice(data);
            self.cursor = end;
        } else {
            error!("Write to buffer is attempting to overflow!");
        }

        // let's initiate a write due to a full (or about to be full) buffer!
        if end + BYTES_PER_RECORD >= self.write_buffer.len() {
            self.upload_buffer();
        }
    }
}
