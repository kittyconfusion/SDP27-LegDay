#![no_std]
#![no_main]
#![deny(clippy::mem_forget)] // Should not be used so deny
#![deny(clippy::large_stack_frames)] // General compiler error if allocate > 512kb at a time (main function given exemption)

mod bin;

const RECORD_HZ: u64 = 200;
// 4 IMU * (6 BYTES) + 4 PRESSURE * (2 BYTES) + 1 TIMESTAMP_MS * (8 BYTES) = 40
const BYTES_PER_RECORD: usize = 24 + 8 + 8;
const BUFFER_SIZE: usize = 8192;
const NUM_BUFFERS: usize = 10;

// BYES_PER_SECOND * RECORDS_PER_SAVE should be below ~200 kbit

use core::{cell::RefCell, mem};
use embassy_executor::Spawner;
use embassy_time::{Duration, Instant, Ticker, Timer};
use esp_hal::{clock::CpuClock, interrupt::Priority};
use esp_rtos::embassy::InterruptExecutor;

use static_cell::StaticCell;

use esp_rtos::TimerSource;

use critical_section::{Mutex, with};
use esp_hal::timer::timg::{self, TimerGroup};
use esp_println::println;
use log::{self, Record, error};
use ringbuffer::{ConstGenericRingBuffer, RingBuffer};

// Alternative to providing our own panic_handler
// should print backtrace to serial upon fatal error
use esp_backtrace as _;

extern crate alloc;

static RING: Mutex<RefCell<ConstGenericRingBuffer<[u8; BUFFER_SIZE], NUM_BUFFERS>>> =
    Mutex::new(RefCell::new(ConstGenericRingBuffer::<
        [u8; BUFFER_SIZE],
        NUM_BUFFERS,
    >::new()));

// use esp_alloc::Internal/External Memory to specify sram vs psram

// Creates a default app-descriptor required by the esp-idf bootloader. aka makes things work right.
esp_bootloader_esp_idf::esp_app_desc!();
//let timestamp_ms_u16 = Instant::now().duration_since_epoch().as_millis() as u16;
// suppress compiler complaining about memory allocation
#[allow(clippy::large_stack_frames)]
#[esp_rtos::main]
async fn main(spawner: Spawner) -> ! {
    esp_println::logger::init_logger_from_env(); // Makes log:: work

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    #[cfg(feature = "esp32s3")]
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    #[cfg(feature = "esp32c6")]
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 65536);

    let timg0 = TimerGroup::new(peripherals.TIMG0);
    esp_rtos::start(timg0.timer0, peripherals.FROM_CPU_INTR0);

    // set up pin layout
    #[cfg(feature = "esp32s3")]
    {}
    #[cfg(feature = "esp32c6")]
    {}

    static EXECUTOR: StaticCell<InterruptExecutor<2>> = StaticCell::new();
    let executor = InterruptExecutor::new(peripherals.FROM_CPU_INTR2);
    let executor = EXECUTOR.init(executor);

    let high_spawner = executor.start(Priority::Priority3);
    high_spawner.spawn(sensor_recorder().unwrap());

    spawner.spawn(sensor_writer().unwrap());

    loop {}
}

#[embassy_executor::task]
async fn sensor_writer() {
    let mut ticker = Ticker::every(Duration::from_hz(1));

    loop {
        let mut buffer_length = 0;
        let mut read_buffer: [u8; BUFFER_SIZE] = [1; BUFFER_SIZE];

        critical_section::with(|cs| {
            buffer_length = RING.borrow(cs).borrow_mut().len();
        });

        while buffer_length > 0 {
            critical_section::with(|cs| {
                let mut buffer = RING.borrow(cs).borrow_mut();

                read_buffer = buffer
                    .dequeue()
                    .expect("Read buffer unable to dequeue. This should be protected against.");
                buffer_length = buffer.len();
            });
        }
        ticker.next().await;
    }
}

#[embassy_executor::task]
async fn sensor_recorder() {
    let mut recorder = Recording::new();
    let mut ticker = Ticker::every(Duration::from_hz(RECORD_HZ));

    loop {
        // expects sensor_record to be in big endian order. This can be easily switched if that doesn't make sense
        let mut sensor_record: [u8; BYTES_PER_RECORD] = [1; BYTES_PER_RECORD];
        // timestamp
        sensor_record[BYTES_PER_RECORD - 8..]
            .copy_from_slice(&Instant::now().as_micros().to_be_bytes());

        recorder.write_to_buffer(&sensor_record);

        ticker.next().await;
    }
}

// Double buffer approach to avoid copying over data itself and instead swap over indexes
// core::mem::swap may be useful in the future but reference lifetimes become a big deal and we only need 2 identical buffers anyways
pub struct Recording {
    // write to 0, read to 1
    // double buffer
    write_buffer: [u8; BUFFER_SIZE],
    cursor: usize,
    currently_recording: bool,
}

impl Recording {
    pub const fn new() -> Self {
        Self {
            write_buffer: [0; BUFFER_SIZE],
            cursor: 0,
            currently_recording: false,
        }
    }

    pub async fn create_record(&mut self, data: &[u8]) {
        // expects sensor_record to be in big endian order. This can be easily switched if that doesn't make sense
        let sensor_record: [u8; BYTES_PER_RECORD - 8] = [1; BYTES_PER_RECORD - 8];
        let timestamp: [u8; 8] = Instant::now().as_micros().to_be_bytes(); // only want/need the lower 4 bytes

        self.write_to_buffer(&sensor_record);
        self.write_to_buffer(&timestamp);
    }

    pub fn write_to_buffer(&mut self, data: &[u8]) {
        let end = self.cursor + data.len();

        if end <= self.write_buffer.len() {
            self.write_buffer[self.cursor..end].copy_from_slice(data);
            self.cursor = end;
        } else {
            error!("Write to buffer is attempting to overflow!");
        }

        // let's initiate a write due to a full (or about to be full) buffer!
        if end + BYTES_PER_RECORD >= self.write_buffer.len() {
            critical_section::with(|cs| {
                let mut buffer = RING.borrow(cs).borrow_mut();

                buffer.enqueue(self.write_buffer);
            });

            self.cursor = 0;
        }
    }
}

// create chip specific code with
// if cfg!(feature = "esp32s3")  // chooses whether to run at run-time
// #[cfg(feature = "esp32s3")]   // chooses whether to include (at all) at compile-time
