#![no_std]
#![no_main]
#![deny(clippy::mem_forget)] // Should not be used so deny
#![deny(clippy::large_stack_frames)] // General compiler error if allocate > 512kb at a time (main function given exemption)

// Run record-app with "cargo run --package quadriceps"

mod bin;

use bin::gpio_controller::{Status, gpio_loop};
use bin::sd_controller::SdController;
use bin::sensors::{
    ImuReader, PressureReader, SensorReader,
    config::{AcquisitionMode, I2C_BUS_KHZ, SAMPLE_HZ},
    frame::{self, SensorFrame},
};
use legday_core::fused::{FusedFrame, FusedWindowBuffer};

// One shared rate sets both polling and the hardware divider in config.rs.
// TODO(Nathan, Tim): verify sustained fresh counts and synchronization on hardware;
// a nominal 225-Hz setting does not guarantee every frame has four fresh IMUs
const RECORD_HZ: u64 = SAMPLE_HZ;

// Last second at the nominal recording cadence; actual time coverage follows timestamps
const HISTORY_FRAMES: usize = RECORD_HZ as usize;

// Tim's 120-byte little-endian payload replaces the old 40-byte placeholder
const BYTES_PER_RECORD: usize = frame::FRAME_BYTES;

// Change to Demo to omit only the removable non-prosthetic lower reference
// TODO(Nathan, Tim): agree how the final application selects training/demo
const ACQUISITION_MODE: AcquisitionMode = AcquisitionMode::Training;

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

//=========================== THREAD-SAFE VARIABLES ===================================
static RECORD_STATUS: Mutex<CriticalSectionRawMutex, Status> = Mutex::new(Status::Setup);

// Acquisition continues while SD recording is off. A future simulation transport
// can copy this snapshot without owning/reinitializing the hardware reader
// This is latest-only; use the separate history below for a consistent window.
static LATEST_SENSOR_FRAME: Mutex<CriticalSectionRawMutex, Option<SensorFrame>> = Mutex::new(None);

pub async fn latest_sensor_frame() -> Option<SensorFrame> {
    *LATEST_SENSOR_FRAME.lock().await
}

// Separate from the SD queue: retain the latest 225 frames even when not recording.
// Static storage avoids putting the ~27 kB window on a task's stack or heap.
static SENSOR_HISTORY: Mutex<CriticalSectionRawMutex, FusedWindowBuffer<HISTORY_FRAMES>> =
    Mutex::new(FusedWindowBuffer::new());

// Copy the newest frames, oldest first, into caller-owned storage. A full-window
// destination should be static/heap-backed. Run inference after this returns,
// with the lock released, so classification does not hold up acquisition
pub async fn copy_sensor_history(output: &mut [SensorFrame]) -> usize {
    let history = SENSOR_HISTORY.lock().await;
    let count = output.len().min(history.len());
    let first = history.len() - count;
    for (index, destination) in output[..count].iter_mut().enumerate() {
        *destination = history.get(first + index).unwrap().encode_le();
    }
    count
}

// Main thread-shared data buffer. RingBuffer loops around instead of overflowing and handles a whole lot for us
static RING: Mutex<
    CriticalSectionRawMutex,
    ConstGenericRingBuffer<(u16, [u8; BUFFER_SIZE]), NUM_BUFFERS>,
> = Mutex::new(ConstGenericRingBuffer::<
    (u16, [u8; BUFFER_SIZE]),
    NUM_BUFFERS,
>::new());

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

    // ======================= GPIO PINS =======================================
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

    // ======================= THREAD SETUP ====================================
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

    // ============================ CORE 0 THREADING =====================================
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
    let shared_i2c1 = I2C1_BUS.init(Mutex::new(i2c1));

    let pressure = PressureReader::new(
        peripherals.ADC1,
        peripherals.GPIO1,
        peripherals.GPIO2,
        peripherals.GPIO3,
        peripherals.GPIO4,
    );
    spawner.spawn(sensor_recorder(shared_i2c0, shared_i2c1, pressure).unwrap());

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

    // Know that file naming has been initialized. Start the gpio task now that display can be initialized
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
    let current_file_num = loop {
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
        while let Some((file_num, read_data)) = { RING.lock().await.dequeue() } {
            // Bind queued data to its original session, even if the switch has
            // already started another recording before the SD task drains it.
            // TODO(Nathan): SdController::write currently hides some write errors;
            // decide retry/recovery there before claiming lossless SD recording.
            sd.write(&read_data, &format!("{file_num}.leg"))
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
    pressure: PressureReader<'static>,
) {
    let mut recorder = Recording::new(); // Buffer Controller
    // Remember the last state so we know if recording should be started / stopped
    let mut last_status = Status::Setup;

    let mut sensors = SensorReader::new(ImuReader::new(i2c0, i2c1, ACQUISITION_MODE), pressure);

    for (slot, result) in sensors.initialize().await.iter().enumerate() {
        info!("IMU slot {slot} initialization: {result:?}");
    }

    // Missing sensors do not block the others. No automatic hot-plug/reset:
    // TODO(Nathan, Tim): agree on reconnect policy; for now reboot after attaching.
    let mut ticker = Ticker::every(Duration::from_hz(RECORD_HZ));
    let mut sequence = 0u32;
    #[cfg(feature = "sensor-fusion")]
    let mut fusion = legday_fusion::frame::FrameFusion::new(0.1);
    info!(
        "Sensor pipeline: software fusion={}, history={} frames, frame={} bytes",
        cfg!(feature = "sensor-fusion"),
        HISTORY_FRAMES,
        BYTES_PER_RECORD
    );

    let mut report_at = Instant::now().as_micros();
    let mut polls = 0u32;
    let mut fresh_counts = [0u32; 4];
    let mut overruns = 0u32;

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
                Status::RecordingOn(file_num) => {
                    info!("Changing recording status to ON");
                    // Flush the old session if rapid switch changes skipped OFF.
                    recorder.stop_recording().await;
                    recorder.start_recording(file_num);
                }
                Status::Setup => {}
            }
        }
        last_status = current_status;

        // Always acquire for live consumers; the switch only gates SD buffering.
        // The reader overlaps the two bus transfers, while the existing GPIO
        // task/display and core-1 SD task continue independently.

        let sample = sensors.read().await;
        let sensor_record = frame::encode(&sample, sequence);
        #[cfg(feature = "sensor-fusion")]
        let sensor_record = {
            let mut decoded = legday_core::fused::FusedFrame::decode_le(&sensor_record);
            fusion.update(&mut decoded);
            decoded.encode_le()
        };
        sequence = sequence.wrapping_add(1);
        // Keep the final (optionally fused) frame in RAM before publishing or recording it.
        SENSOR_HISTORY
            .lock()
            .await
            .push(FusedFrame::decode_le(&sensor_record));
        *LATEST_SENSOR_FRAME.lock().await = Some(sensor_record);

        if recorder.currently_recording {
            recorder.write_to_buffer(&sensor_record).await;
        }

        polls += 1;

        for (slot, count) in fresh_counts.iter_mut().enumerate() {
            if sample.fresh_imu_mask() & (1 << slot) != 0 {
                *count += 1;
            }
        }

        // Do not spin trying to catch up missed deadlines. Log measured rates
        // so the nominal polling cadence cannot be confused with fresh IMU output.
        if Instant::now().as_micros() - sample.read_started_us >= 1_000_000 / RECORD_HZ {
            overruns += 1;
            ticker.reset();
        }

        let now = Instant::now().as_micros();

        if now - report_at >= 1_000_000 {
            info!(
                "Sensor interval {} us: polls={}, fresh per slot={:?}, overruns={}",
                now - report_at,
                polls,
                fresh_counts,
                overruns
            );

            let (history_len, history_span_us, oldest_checksum) = {
                let history = SENSOR_HISTORY.lock().await;
                let span = match (history.get(0), history.latest()) {
                    (Some(first), Some(last)) => {
                        last.raw.timestamp_us.saturating_sub(first.raw.timestamp_us)
                    }
                    _ => 0,
                };

                // Inspect one retained payload, not just the latest-frame snapshot.
                let checksum = history.get(0).map_or(0u32, |frame| {
                    frame.encode_le().iter().map(|byte| u32::from(*byte)).sum()
                });

                (history.len(), span, checksum)
            };
            info!(
                "Sensor history: {}/{} frames, first-to-last={} us, oldest checksum={}",
                history_len, HISTORY_FRAMES, history_span_us, oldest_checksum
            );

            report_at = now;
            polls = 0;
            fresh_counts = [0; 4];
            overruns = 0;
        }
        ticker.next().await;
    }
}

pub struct Recording {
    // STRUCTURE
    // 2  bytes Block Size in bytes, inclusive
    // Followed by whole 120-byte little-endian frames, then zero padding.
    // TODO(Nathan, Tim): agree on file version/decoder. The outer inclusive
    // length stays BIG endian for Nathan's existing SD block convention.
    // A .leg file is not a flat stream of 120-byte frames. Old 40-byte files
    // are incompatible; decoders must explicitly select this new format.
    // .............................
    // end padded with 0 until BUFFER_SIZE
    write_buffer: [u8; BUFFER_SIZE],
    // Current place in the write_buffer
    // don't need > 64k block so even though usize is u32, will cast to u16 when transcribing
    cursor: usize,
    currently_recording: bool,
    file_num: u16,
}

impl Recording {
    pub const fn new() -> Self {
        Self {
            write_buffer: [0; BUFFER_SIZE],
            cursor: 2,
            currently_recording: false,
            file_num: 0,
        }
    }

    pub async fn stop_recording(&mut self) {
        self.currently_recording = false;

        // always upload remaining data to buffer
        if self.cursor > 2 {
            self.upload_buffer().await;
        }
    }

    pub fn start_recording(&mut self, file_num: u16) {
        self.write_buffer.fill(0);
        self.file_num = file_num;
        self.cursor = 2;
        self.currently_recording = true;
    }

    // Transcribe the final write_buffer used length and upload its data to the thread-shared ring buffer
    pub async fn upload_buffer(&mut self) {
        self.write_buffer[0..2].copy_from_slice(&(self.cursor as u16).to_be_bytes());

        {
            let mut ring = RING.lock().await;
            // Never silently overwrite an older block in a full ring.
            // TODO(Nathan, Tim): agree overflow policy/backpressure. Currently
            // drop the new block, log it, and leave frame sequence gaps visible.
            if ring.is_full() {
                warn!(
                    "Sensor SD queue full: dropping block for session {}",
                    self.file_num
                );
            } else {
                ring.enqueue((self.file_num, self.write_buffer));
            }
        }

        self.write_buffer.fill(0); // Partial final blocks must not retain old samples.
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
