#![no_std]
#![no_main]
// Inline assembly on Xtensa is a nightly-only feature (the esp toolchain is
// nightly). Used by the cycle probes in imu_cycles.rs.
#![feature(asm_experimental_arch)]

//! # IMU bring-up test firmware
//!
//! Proves the sensor-to-RAM-to-SD path with raw IMU data and orientation
//! computed on the S3. Everything the real firmware will need is here in
//! miniature:
//!
//! ```text
//!   IMU FIFOs ──I2C──▶ acquire ──▶ pressure ADC ──▶ assemble ──▶ 1 s RAM window ──▶ classifier (later)
//!   (up to 4)          (all IMUs)                   (bias + quaternion          │
//!                                                     for all IMUs, then         └──RECORD──▶ SD card
//!                                                     one 120-byte frame)
//! ```
//!
//! * Each IMU samples at 225 Hz on its own clock and queues raw accel + gyro
//!   in its 4 kB hardware FIFO (about 1.5 s of buffering).
//! * Every pass of the main loop drains all IMUs, reads the four pressure
//!   channels, then builds one frame per sample instant: gyro bias removed,
//!   every IMU's quaternion updated, everything packed into one 120-byte
//!   [`FusedFrame`] (the 72-byte raw `SensorFrame` plus one quaternion per
//!   IMU). See `imu_pipeline.rs` for the stages and how IMUs are kept in step.
//! * The frames go into a static one-second ring buffer that the classifier
//!   will read. RECORD (button or `r`) streams that buffer and all later
//!   frames to a new file on the SD card.
//! * The DMP is no longer used here. The diagnostics (`x`) still exercise it
//!   so its cost can be compared with this raw path.
//!
//! ## Wiring (Arduino Nano ESP32 labels)
//!
//! | Signal                     | Pin             | GPIO        |
//! |----------------------------|-----------------|-------------|
//! | IMU SDA (Qwiic blue)       | A4              | 11          |
//! | IMU SCL (Qwiic yellow)     | A5              | 12          |
//! | Pressure dividers 0..3     | A0-A3           | 1-4         |
//! | SD CLK / MISO / MOSI / CS  | D13/D12/D11/D10 | 48/47/38/21 |
//! | RECORD button to GND       | D7              | 10          |
//! | CATEGORY button to GND     | D8              | 17          |
//! | DIAG button to GND         | D6              | 9           |
//! | SD card detect (DET)       | D9              | 18 (unused) |
//!
//! A second IMU on the same Qwiic chain needs its ADR jumper bridged (address
//! 0x68); it is detected at boot and fills frame slot 1.
//!
//! ## Serial commands (type into the monitor)
//!
//! `r` record start/stop · `c` next category · `d` hex-dump the newest frame ·
//! `p` pause draining 2 s (measures FIFO depth) · `x` run the bus/latency/wake
//! diagnostics (see `imu_diag.rs`) · `b` time CNN models of several sizes ·
//! `h` help.
//!
//! ## How to read this file
//!
//! Top to bottom: tunable constants, static buffers, small helper types,
//! the recorder, then `main`, which is "set everything up, then loop".

use core::{cell::RefCell, fmt::Write as _};

use embedded_hal_bus::{i2c::RefCellDevice, spi::ExclusiveDevice};

use embedded_sdmmc::{
    Directory, File, Mode, SdCard, TimeSource, Timestamp, VolumeIdx, VolumeManager,
};

use esp_hal::{
    Blocking,
    analog::adc::{Adc, AdcConfig, AdcPin, Attenuation},
    clock::CpuClock,
    delay::Delay,
    gpio::{Input, InputConfig, Level, Output, OutputConfig, Pull},
    i2c::master::{Config as I2cConfig, I2c},
    peripherals::{ADC1, GPIO1, GPIO2, GPIO3, GPIO4},
    spi::{
        Mode as SpiMode,
        master::{Config as SpiConfig, Spi},
    },
    time::{Duration, Instant, Rate},
    usb::usb_serial_jtag::UsbSerialJtag,
};

use esp_println::println;
use heapless::String;

use legday_classifier::fusion::{down_in_body, q30_to_f32, reconstruct_w, sqrt};

use legday_core::{
    FRAME_WIRE_BYTES,
    fused::{FUSED_FRAME_WIRE_BYTES, FusedFrame, FusedFrameBatch, FusedWindowBuffer},
    recording::MotionCategory,
};

use legday_icm20948::{
    AccelRange, Config as ImuConfig, GyroRange, I2C_ADDR_AD0_HIGH, I2C_ADDR_AD0_LOW, Icm20948,
};

use static_cell::{ConstStaticCell, StaticCell};

use esp_backtrace as _;

/// Bus, latency and wake-cost measurements (DIAG button or `x`).
mod imu_diag;

/// Acquire -> fuse -> frame stages for up to four IMUs.
mod imu_pipeline;

/// Runs the CNN on the live window and reports its answer and timing.
mod imu_classifier;

/// Cycle-level probes of the chip's arithmetic (the `k` key).
mod imu_cycles;

use imu_pipeline::Pipeline;

esp_bootloader_esp_idf::esp_app_desc!();

// --------------------------------------------------------------------------------
// Tunable parameters
// --------------------------------------------------------------------------------

/// IMU output rate = 1125 / (1 + div) Hz. 4 gives 225 Hz; 1 would give
/// 562.5 Hz (raw mode is not limited to 225 Hz the way the DMP is).
const SAMPLE_RATE_DIV: u8 = 4;

/// Accelerometer range. ±16 g so heel strikes and jumps do not clip; the
/// DMP forced ±4 g, which clipped in the first shaking test.
const ACCEL_RANGE: AccelRange = AccelRange::G16;

/// Gyroscope range: ±2000 deg/s, the widest available.
const GYRO_RANGE: GyroRange = GyroRange::Dps2000;

/// Which I2C address fills which frame slot. Slots 0 and 1 share the Qwiic
/// chain on I2C0 (one leg, the "daisy chain" layout); slots 2 and 3 are
/// reserved for a second bus. Freeze which body location each slot means
/// before collecting training data.
const IMU_LAYOUT: [(usize, u8); 2] = [(0, I2C_ADDR_AD0_HIGH), (1, I2C_ADDR_AD0_LOW)];

/// One second of history at 225 Hz; 225 x 120 bytes = 27,000 bytes of SRAM.
const WINDOW_FRAMES: usize = 225;

/// Bits of `valid_pressure_mask` to set: one bit per connected sensor on
/// A0..A3. All four channels are always sampled and stored; this only says which
/// to trust. Set bit 0 once a voltage divider is wired to A0 for pressure sensors
const CONNECTED_PRESSURE_MASK: u8 = 0b0000;

/// Frames handed to the SD card per loop pass while recording: 16 x 120 =
/// 1,920 bytes (about 4 x 512-byte sectors), written in 10-50 ms. Small enough
/// that the IMU FIFO (4 kB, ~1.5 s in raw mode) never overflows during a write.
const SD_FRAMES_PER_PASS: usize = 16;

/// How often the directory entry is flushed while recording. A flush is a read-modify-
/// -write of the File allocation table (a.k.a, FAT) that can stall for tens of milliseconds.
const SD_FLUSH_PERIOD: Duration = Duration::from_secs(5);

/// FAT 8.3 extension for 120-byte fused recordings (`.BIN` stays 72-byte).
const FILE_EXTENSION: &str = "Q6B";

/// Madgwick is a sensor-fusion algorithm that estimates orientation from accel & gyro
/// and outputs quaternions.
/// Madgwick gain: higher trusts the accelerometer more and settles faster.
const MADGWICK_BETA: f32 = 0.1;

const STATUS_PERIOD: Duration = Duration::from_secs(1);
const BUTTON_PERIOD: Duration = Duration::from_millis(5);
const DEBOUNCE_SAMPLES: u8 = 6;
const PAUSE_MS: u32 = 2_000;

// ------------------------------------------------------------------------------------------
// Static buffers (allocated at compile time, never on the stack)
// ------------------------------------------------------------------------------------------

/// The one-second rolling history every consumer reads from.
static WINDOW: ConstStaticCell<FusedWindowBuffer<WINDOW_FRAMES>> =
    ConstStaticCell::new(FusedWindowBuffer::new());

/// Staging area for frames encoded for the SD card.
static BATCH: ConstStaticCell<FusedFrameBatch> = ConstStaticCell::new(FusedFrameBatch::new());

/// The I2C bus, wrapped so several IMU drivers can share it (two IMUs on
/// one Qwiic chain, at different addresses).
static I2C_BUS: StaticCell<RefCell<I2c<'static, Blocking>>> = StaticCell::new();

// ---------------------------------------------------------------------------------------------
// Bus and SD-card plumbing
// ---------------------------------------------------------------------------------------------

/// One IMU driver on the shared I2C bus. Also used by the diagnostics module.
type Imu = Icm20948<RefCellDevice<'static, I2c<'static, Blocking>>>;

/// The SD card as seen by `embedded-sdmmc`: an SPI bus with a chip-select pin.
type SdBlockDevice = SdCard<ExclusiveDevice<Spi<'static, Blocking>, Output<'static>, Delay>, Delay>;
/// A directory handle (`4, 4, 1` = max open dirs, files, volumes).
type SdDirectory<'a> = Directory<'a, SdBlockDevice, FixedTime, 4, 4, 1>;
/// An open file handle.
type SdFile<'a> = File<'a, SdBlockDevice, FixedTime, 4, 4, 1>;

/// The FAT library wants a wall clock for file timestamps; we have none, so
/// every file is dated 1980-01-01.
struct FixedTime;

impl TimeSource for FixedTime {
    fn get_timestamp(&self) -> Timestamp {
        Timestamp::from_calendar(1980, 1, 1, 0, 0, 0).unwrap()
    }
}

// ----------------------------------------------------------------------------------------------
// Small helper types
// ----------------------------------------------------------------------------------------------

/// Software debounce: a level must be stable for `DEBOUNCE_SAMPLES` polls
/// (30 ms at the 5 ms poll rate) before it counts.
struct DebouncedButton {
    candidate_low: bool,
    stable_low: bool,
    count: u8,
}

impl DebouncedButton {
    const fn new() -> Self {
        Self {
            candidate_low: false,
            stable_low: false,
            count: 0,
        }
    }

    /// Feed one poll; returns `true` exactly once per confirmed press.
    fn update(&mut self, raw_low: bool) -> bool {
        if raw_low != self.candidate_low {
            // Level changed: start counting again.
            self.candidate_low = raw_low;
            self.count = 1;
            return false;
        }

        self.count = self.count.saturating_add(1);

        if self.count >= DEBOUNCE_SAMPLES && self.stable_low != self.candidate_low {
            self.stable_low = self.candidate_low;
            return self.stable_low; // a press is a confirmed transition to low
        }

        false
    }
}

/// Something the user asked for, via a button or a serial key.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    ToggleRecord,
    NextCategory,
    DumpFrame,
    Pause,
    Diagnostics,
    ModelBench,
    CycleProbes,
    Help,
}

/// Counters printed on the once-per-second status line.
#[derive(Default)]
struct Stats {
    frames_this_period: u32,
    total_frames: u32,
    sd_write_max_us: u64,
    record_dropped: u32,
    adc_us_max: u64,
}

/// The four force-sensor voltage dividers on A0..A3.
struct PressureInputs {
    adc: Adc<'static, ADC1<'static>, Blocking>,
    a0: AdcPin<GPIO1<'static>, ADC1<'static>>,
    a1: AdcPin<GPIO2<'static>, ADC1<'static>>,
    a2: AdcPin<GPIO3<'static>, ADC1<'static>>,
    a3: AdcPin<GPIO4<'static>, ADC1<'static>>,
}

impl PressureInputs {
    /// Reads all four channels (12-bit counts, 0 = 0 V, ~4095 = 3.3 V).
    /// Unconnected pins float and read noise.
    fn read(&mut self) -> [u16; 4] {
        [
            self.adc.read_blocking(&mut self.a0),
            self.adc.read_blocking(&mut self.a1),
            self.adc.read_blocking(&mut self.a2),
            self.adc.read_blocking(&mut self.a3),
        ]
    }
}

/// Advances a periodic deadline by one period. If the loop was blocked for
/// longer than a period (diagnostics, a mode switch), the deadline jumps to
/// one period from now instead of firing once per missed period to catch up.
fn next_deadline(deadline: Instant, period: Duration) -> Instant {
    let next = deadline + period;
    let now = Instant::now();

    if next < now { now + period } else { next }
}

/// Microseconds since boot.
fn now_us() -> u64 {
    Instant::now().duration_since_epoch().as_micros()
}

fn print_help() {
    println!(
        "Serial commands: r=record start/stop, c=next category, d=dump newest frame, p=pause 2 s (FIFO depth), x=diagnostics, b=model benchmark, h=help"
    );
}

/// Prints one frame as hex, split at the 72-byte raw/quaternion boundary.
fn dump_frame(frame: &FusedFrame) {
    let encoded = frame.encode_le();

    println!(
        "FRAME seq {} ts {} us flags 0x{:04X} ({} bytes: raw 0..{} | quat {}..{})",
        frame.raw.sequence,
        frame.raw.timestamp_us,
        frame.raw.status_flags,
        FUSED_FRAME_WIRE_BYTES,
        FRAME_WIRE_BYTES,
        FRAME_WIRE_BYTES,
        FUSED_FRAME_WIRE_BYTES
    );

    for (row, chunk) in encoded.chunks(24).enumerate() {
        let offset = row * 24;
        let label = if offset < FRAME_WIRE_BYTES {
            "raw "
        } else {
            "quat"
        };

        let mut line = String::<96>::new();

        for byte in chunk {
            write!(&mut line, "{:02X} ", byte).unwrap();
        }

        println!(
            "  {} [{:>3}..{:>3}] {}",
            label,
            offset,
            offset + chunk.len(),
            line.as_str()
        );
    }

    let q = frame.quat6[0];
    let (x, y, z) = (q30_to_f32(q.x), q30_to_f32(q.y), q30_to_f32(q.z));

    println!(
        "  imu0 accel {:?} gyro {:?} pressure {:?} quat6 w={:+.4} x={:+.4} y={:+.4} z={:+.4}",
        frame.raw.imus[0].accel,
        frame.raw.imus[0].gyro,
        frame.raw.pressure_adc,
        reconstruct_w(x, y, z),
        x,
        y,
        z
    );
}

// -------------------------------------------------------------------------------------
// Recorder: a cursor that walks the RAM window out to the SD card
// -------------------------------------------------------------------------------------

/// While a file is open, `next_seq` is the sequence number of the next frame to write.
/// Each loop pass writes a bounded number of frames from there, so the one-second
/// pre-roll streams out at ~3x real time and a pass is never blocked longer than
/// the IMU FIFO can cover.
struct Recorder<'r, 'v> {
    /// Root directory, or `None` when no card mounted (recording is refused).
    root: Option<&'r SdDirectory<'v>>,
    file: Option<SdFile<'v>>,
    name: String<12>,
    category: MotionCategory,
    next_seq: u32,
    last_flush: Instant,
}

impl<'r, 'v> Recorder<'r, 'v> {
    fn new(root: Option<&'r SdDirectory<'v>>) -> Self {
        Self {
            root,
            file: None,
            name: String::new(),
            category: MotionCategory::Walk,
            next_seq: 0,
            last_flush: Instant::now(),
        }
    }

    fn is_active(&self) -> bool {
        self.file.is_some()
    }

    /// Opens the next free `Sxxxxxxx.Q6B` in the category directory and
    /// points the cursor at the oldest frame in the window.
    fn start(&mut self, category: MotionCategory, window: &FusedWindowBuffer<WINDOW_FRAMES>) {
        let Some(root) = self.root else {
            println!("RECORD ignored: no SD card mounted");
            return;
        };

        let directory = root.open_dir(category.directory()).unwrap();

        // Find the first unused session number.
        let mut name = String::<12>::new();

        for session in 0..=9_999_999_u32 {
            name.clear();
            write!(&mut name, "S{:07}.{}", session, FILE_EXTENSION).unwrap();
            if !directory.directory_entry_exists(name.as_str()) {
                break;
            }
        }

        let file = directory
            .open_file_in_dir(name.as_str(), Mode::ReadWriteCreate)
            .unwrap();
        directory.close().unwrap();

        self.next_seq = window.get(0).map(|frame| frame.raw.sequence).unwrap_or(0);
        self.last_flush = Instant::now();
        self.name = name;
        self.category = category;
        self.file = Some(file);

        println!(
            "RECORDING STARTED: /{}/{} with {} pre-roll frames queued",
            category.directory(),
            self.name.as_str(),
            window.len()
        );
    }

    /// Writes up to `limit` frames from the cursor; returns how many.
    /// Frames older than the window has kept are reported as lost.
    fn write_from_cursor(
        &mut self,
        window: &FusedWindowBuffer<WINDOW_FRAMES>,
        batch: &mut FusedFrameBatch,
        limit: usize,
        stats: &mut Stats,
    ) -> usize {
        let (Some(file), Some(oldest), Some(newest)) =
            (self.file.as_ref(), window.get(0), window.latest())
        else {
            return 0;
        };

        let oldest_seq = oldest.raw.sequence;

        // Wrapping arithmetic so a sequence rollover does not confuse "behind".
        let behind = oldest_seq.wrapping_sub(self.next_seq);

        if behind != 0 && behind < 0x8000_0000 {
            println!("RECORDER fell behind the window: {} frames lost", behind);
            stats.record_dropped += behind;
            self.next_seq = oldest_seq;
        }

        let mut written = 0;

        while newest.raw.sequence.wrapping_sub(self.next_seq) < 0x8000_0000 && written < limit {
            let index = self.next_seq.wrapping_sub(oldest_seq) as usize;

            if batch.push(window.get(index).unwrap()) {
                file.write(batch.encoded()).unwrap();
                batch.clear();
            }

            self.next_seq = self.next_seq.wrapping_add(1);
            written += 1;
        }
        written
    }

    /// One loop pass worth of recording work.
    fn step(
        &mut self,
        window: &FusedWindowBuffer<WINDOW_FRAMES>,
        batch: &mut FusedFrameBatch,
        stats: &mut Stats,
    ) {
        if !self.is_active() {
            return;
        }

        let started = now_us();
        let written = self.write_from_cursor(window, batch, SD_FRAMES_PER_PASS, stats);

        // Push a partially filled batch to the card once it is worth a write.
        let file = self.file.as_ref().unwrap();

        if batch.frames() >= SD_FRAMES_PER_PASS {
            file.write(batch.encoded()).unwrap();
            batch.clear();
        }

        if self.last_flush.elapsed() >= SD_FLUSH_PERIOD {
            file.flush().unwrap();
            self.last_flush = Instant::now();
        }

        if written > 0 {
            stats.sd_write_max_us = stats.sd_write_max_us.max(now_us() - started);
        }
    }

    /// Writes everything up to the newest frame, closes the file, then reads
    /// its first and last frame back as a proof the recording is intact.
    fn stop(
        &mut self,
        window: &FusedWindowBuffer<WINDOW_FRAMES>,
        batch: &mut FusedFrameBatch,
        stats: &mut Stats,
    ) {
        self.write_from_cursor(window, batch, usize::MAX, stats);
        let Some(file) = self.file.take() else { return };

        if !batch.is_empty() {
            file.write(batch.encoded()).unwrap();
            batch.clear();
        }

        file.flush().unwrap();

        let bytes = file.length();

        file.close().unwrap();

        println!(
            "RECORDING STOPPED: /{}/{} = {} bytes = {} fused frames safely closed",
            self.category.directory(),
            self.name.as_str(),
            bytes,
            bytes as usize / FUSED_FRAME_WIRE_BYTES
        );

        self.verify(bytes);
    }

    /// Re-opens the finished file and decodes its first and last frames.
    fn verify(&self, bytes: u32) {
        let Some(root) = self.root else { return };
        let directory = root.open_dir(self.category.directory()).unwrap();
        let file = directory
            .open_file_in_dir(self.name.as_str(), Mode::ReadOnly)
            .unwrap();

        let mut first = [0_u8; FUSED_FRAME_WIRE_BYTES];
        let mut last = [0_u8; FUSED_FRAME_WIRE_BYTES];
        let first_len = file.read(&mut first).unwrap();

        file.seek_from_end(FUSED_FRAME_WIRE_BYTES as u32).unwrap();

        let last_len = file.read(&mut last).unwrap();

        file.close().unwrap();
        directory.close().unwrap();

        let whole_frames = bytes as usize % FUSED_FRAME_WIRE_BYTES == 0;

        if first_len != FUSED_FRAME_WIRE_BYTES
            || last_len != FUSED_FRAME_WIRE_BYTES
            || !whole_frames
        {
            println!(
                "READBACK FAILED: read {}/{} bytes, length {}",
                first_len, last_len, bytes
            );
            return;
        }

        let first = FusedFrame::decode_le(&first);
        let last = FusedFrame::decode_le(&last);
        let frames = u64::from(last.raw.sequence.wrapping_sub(first.raw.sequence)) + 1;
        let span_us = last.raw.timestamp_us.saturating_sub(first.raw.timestamp_us);

        println!(
            "READBACK OK: seq {}..{} ({} frames, {} in file), {} ms span, {} Hz effective, flags 0x{:04X}",
            first.raw.sequence,
            last.raw.sequence,
            frames,
            bytes as usize / FUSED_FRAME_WIRE_BYTES,
            span_us / 1_000,
            if span_us > 0 {
                (frames - 1) * 1_000_000 / span_us
            } else {
                0
            },
            last.raw.status_flags
        );
    }
}

// ---------------------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------------------

#[esp_hal::main]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));

    // The board's 8 MB octal PSRAM, mapped into the address space through the
    // data cache. Only the cycle probes (`k`) and the model benchmark (`b`)
    // use it so far, to measure what PSRAM-resident weights would cost.
    let psram = esp_hal::psram::Psram::new(
        peripherals.PSRAM,
        esp_hal::psram::PsramConfig {
            mode: esp_hal::psram::PsramMode::OctalSpi,
            ram_frequency: esp_hal::psram::SpiRamFreq::Freq80m,
            ..Default::default()
        },
    );
    let (psram_base, psram_bytes) = psram.raw_parts();
    imu_cycles::PSRAM_BASE.store(psram_base as usize, core::sync::atomic::Ordering::Release);
    imu_cycles::PSRAM_BYTES.store(psram_bytes, core::sync::atomic::Ordering::Release);
    let mut delay = Delay::new();

    // Receive side of the USB serial port, for the single-key commands.
    let mut usb_serial = UsbSerialJtag::new(peripherals.USB_DEVICE);

    println!();
    println!("LegDay IMU bring-up test (branch tim/testing-memory-and-sensors)");
    println!(
        "PSRAM: {} kB at 0x{:08X}",
        psram_bytes / 1_024,
        psram_base as usize
    );
    println!("I2C: A4/GPIO11 SDA, A5/GPIO12 SCL @ 400 kHz | ADC: A0-A3 pressure");
    println!("SPI: D13/GPIO48 CLK, D12/GPIO47 SO, D11/GPIO38 SI, D10/GPIO21 CS");
    println!("Buttons: D7/GPIO10 RECORD, D8/GPIO17 CATEGORY, D6/GPIO9 DIAG (all to GND)");
    println!(
        "Frame: {} bytes fused ({} raw + {} quat), window {} frames = {} bytes",
        FUSED_FRAME_WIRE_BYTES,
        FRAME_WIRE_BYTES,
        FUSED_FRAME_WIRE_BYTES - FRAME_WIRE_BYTES,
        WINDOW_FRAMES,
        WINDOW_FRAMES * core::mem::size_of::<FusedFrame>()
    );

    // ------------------- IMUs: find every address in IMU_LAYOUT that answers ---------------------
    let i2c = I2c::new(
        peripherals.I2C0,
        I2cConfig::default().with_frequency(Rate::from_khz(400)),
    )
    .unwrap()
    .with_sda(peripherals.GPIO11)
    .with_scl(peripherals.GPIO12);

    let bus: &'static RefCell<I2c<'static, Blocking>> = I2C_BUS.init(RefCell::new(i2c));

    let imu_config = ImuConfig {
        accel_range: ACCEL_RANGE,
        gyro_range: GYRO_RANGE,
        sample_rate_div: SAMPLE_RATE_DIV,
        ..ImuConfig::default()
    };

    let period_us = u64::from(imu_config.period_us());
    let mut pipeline = Pipeline::new(imu_config, MADGWICK_BETA);

    while pipeline.connected_mask() == 0 {
        for (slot, address) in IMU_LAYOUT {
            let mut imu: Imu = Icm20948::new(RefCellDevice::new(bus), address);

            match imu.who_am_i() {
                Ok(legday_icm20948::WHO_AM_I_VALUE) => {
                    println!(
                        "ICM-20948 found at 0x{:02X} -> frame slot {}",
                        address, slot
                    );
                    pipeline.add_imu(slot, imu);
                }
                Ok(other) => println!(
                    "0x{:02X}: WHO_AM_I = 0x{:02X}, not an ICM-20948",
                    address, other
                ),
                Err(_) => {} // nothing at this address; normal with one IMU
            }
        }

        if pipeline.connected_mask() == 0 {
            println!("No IMU answered. Check Qwiic blue->A4, yellow->A5, 3V3, GND. Retrying...");
            delay.delay_millis(1_000);
        }
    }

    println!(
        "IMU ODR {}.{:03} Hz (nominal period {} us), accel ±{} g, gyro ±2000 dps, IMU mask 0b{:04b}",
        imu_config.odr_millihz() / 1_000,
        imu_config.odr_millihz() % 1_000,
        period_us,
        2_u32 << (ACCEL_RANGE as u32),
        pipeline.connected_mask()
    );

    pipeline.start(&mut delay);
    println!(
        "Raw FIFO streaming; orientation computed on the S3 (Madgwick beta {})",
        MADGWICK_BETA
    );

    // ---------------------------- Pressure sensors on ADC1 -----------------------------------------
    let mut adc_config = AdcConfig::new();
    let mut pressure = PressureInputs {
        a0: adc_config.enable_pin(peripherals.GPIO1, Attenuation::_11dB),
        a1: adc_config.enable_pin(peripherals.GPIO2, Attenuation::_11dB),
        a2: adc_config.enable_pin(peripherals.GPIO3, Attenuation::_11dB),
        a3: adc_config.enable_pin(peripherals.GPIO4, Attenuation::_11dB),
        adc: Adc::new(peripherals.ADC1, adc_config),
    };

    // ------------------------- SD card: init at 400 kHz, then run the bus at 4 MHz -----------------
    let cs = Output::new(peripherals.GPIO21, Level::High, OutputConfig::default());
    let mut spi_bus = Spi::new(
        peripherals.SPI2,
        SpiConfig::default()
            .with_frequency(Rate::from_khz(400))
            .with_mode(SpiMode::_0),
    )
    .unwrap()
    .with_sck(peripherals.GPIO48)
    .with_miso(peripherals.GPIO47)
    .with_mosi(peripherals.GPIO38);
    spi_bus.write(&[0xff; 10]).unwrap(); // cards need >= 74 idle clocks before the first command

    let sdcard = SdCard::new(
        ExclusiveDevice::new(spi_bus, cs, Delay::new()).unwrap(),
        Delay::new(),
    );

    let sd_present = match sdcard.num_bytes() {
        Ok(bytes) => {
            println!("SD initialized: {} bytes", bytes);
            sdcard.spi(|device| {
                device
                    .bus_mut()
                    .apply_config(
                        &SpiConfig::default()
                            .with_frequency(Rate::from_mhz(4))
                            .with_mode(SpiMode::_0),
                    )
                    .unwrap();
            });
            true
        }

        Err(error) => {
            println!("SD INIT FAILED: {:?} (continuing without recording)", error);
            false
        }
    };

    let volume_manager = VolumeManager::new(sdcard, FixedTime);
    let volume = sd_present
        .then(|| volume_manager.open_volume(VolumeIdx(0)).ok())
        .flatten();

    let root = volume
        .as_ref()
        .map(|volume| volume.open_root_dir().unwrap());

    if let Some(root) = root.as_ref() {
        for category in MotionCategory::ALL {
            if !root.directory_entry_exists(category.directory()) {
                root.make_dir_in_dir(category.directory()).unwrap();
                println!("Created /{}", category.directory());
            }
        }
    }

    let mut recorder = Recorder::new(root.as_ref());

    // ------------------------------- Buttons ------------------------------------------------
    let record_pin = Input::new(
        peripherals.GPIO10,
        InputConfig::default().with_pull(Pull::Up),
    );

    let category_pin = Input::new(
        peripherals.GPIO17,
        InputConfig::default().with_pull(Pull::Up),
    );

    // DIAG is on D6: D9 carries the SD breakout's card-detect (DET) line,
    // which is grounded whenever no card is inserted and would read as a press.
    let diag_pin = Input::new(
        peripherals.GPIO9,
        InputConfig::default().with_pull(Pull::Up),
    );

    let mut record_button = DebouncedButton::new();
    let mut category_button = DebouncedButton::new();
    let mut diag_button = DebouncedButton::new();
    let mut category = MotionCategory::Walk;

    // -------------------------------- Loop state --------------------------------------------
    let window = WINDOW.take();
    let batch = BATCH.take();
    let mut diag_buffers = imu_diag::Buffers::take(); // taken once so diagnostics can rerun

    let mut classifier = match imu_classifier::LiveClassifier::new() {
        Ok(classifier) => Some(classifier),
        Err(error) => {
            println!("Classifier disabled: demo model rejected: {:?}", error);
            None
        }
    };
    let mut stats = Stats::default();
    let sample_period = Duration::from_micros(period_us);
    let mut next_acquire = Instant::now();
    let mut next_button_poll = Instant::now();
    let mut next_status = Instant::now() + STATUS_PERIOD;
    let started_at = Instant::now();
    let mut actions: heapless::Vec<Action, 8> = heapless::Vec::new();

    println!("IMU TEST READY: streaming into the RAM window");
    println!(
        "Selected /{}; CATEGORY changes it while idle, RECORD starts/stops a .{} file",
        category.directory(),
        FILE_EXTENSION
    );

    print_help();

    loop {
        // ============ 1-3. Acquire all IMUs, read pressure, assemble frames ==============
        // (the stages are explained in imu_pipeline.rs). This runs once per
        // sample period: polling more often only finds empty FIFOs and wastes
        // a ~200 µs I2C transaction each time.

        if Instant::now() >= next_acquire {
            next_acquire = next_deadline(next_acquire, sample_period);

            pipeline.acquire();

            let adc_started = now_us();
            let pressure_counts = pressure.read();
            stats.adc_us_max = stats.adc_us_max.max(now_us() - adc_started);

            let new_frames = pipeline.assemble(pressure_counts, CONNECTED_PRESSURE_MASK, window);

            stats.frames_this_period += new_frames as u32;
            stats.total_frames += new_frames as u32;

            // Classify the newest second every few frames.
            if let Some(classifier) = classifier.as_mut() {
                classifier.on_new_frames(new_frames, window);
            }
        }

        // ===================== 4. Collect input: buttons every 5 ms, serial keys always ===================

        if Instant::now() >= next_button_poll {
            next_button_poll = next_deadline(next_button_poll, BUTTON_PERIOD);

            if category_button.update(category_pin.is_low()) {
                let _ = actions.push(Action::NextCategory);
            }

            if diag_button.update(diag_pin.is_low()) {
                let _ = actions.push(Action::Diagnostics);
            }

            if record_button.update(record_pin.is_low()) {
                let _ = actions.push(Action::ToggleRecord);
            }
        }

        while let Ok(byte) = usb_serial.read_byte() {
            let action = match byte.to_ascii_lowercase() {
                b'r' => Some(Action::ToggleRecord),
                b'c' => Some(Action::NextCategory),
                b'd' => Some(Action::DumpFrame),
                b'p' => Some(Action::Pause),
                b'x' => Some(Action::Diagnostics),
                b'b' => Some(Action::ModelBench),
                b'k' => Some(Action::CycleProbes),
                b'h' | b'?' => Some(Action::Help),
                _ => None,
            };

            if let Some(action) = action {
                let _ = actions.push(action);
            }
        }

        // ====================== 5. Perform the requested actions ==============================

        for action in actions.iter().copied() {
            match action {
                Action::Help => print_help(),

                Action::DumpFrame => match window.latest() {
                    Some(frame) => dump_frame(frame),
                    None => println!("window is empty"),
                },

                Action::Pause => {
                    println!(
                        "PAUSE: not draining for {} ms; watch 'fifo max' and ovf next",
                        PAUSE_MS
                    );

                    delay.delay_millis(PAUSE_MS);
                }

                Action::NextCategory if recorder.is_active() => {
                    println!("CATEGORY ignored while recording")
                }

                Action::NextCategory => {
                    category = category.next();
                    println!("Selected /{}", category.directory());
                }

                Action::CycleProbes => {
                    // The probes borrow the classifier's scratch for their operands.
                    let mut none = [0.0_f32; 0];
                    let scratch = match classifier.as_mut() {
                        Some(classifier) => classifier.buffers().1,
                        None => &mut none[..],
                    };
                    imu_cycles::run(scratch);
                }

                Action::ModelBench if recorder.is_active() => {
                    println!("MODEL BENCH ignored while recording")
                }

                Action::ModelBench => {
                    // Blocks the loop for a few seconds; the IMU FIFO covers
                    // about 1.2 s, so frames from that time are dropped.
                    if let Some(classifier) = classifier.as_mut() {
                        imu_diag::run_model_bench(classifier.buffers());
                    }

                    pipeline.start(&mut delay);
                    window.clear();
                }

                Action::Diagnostics if recorder.is_active() => {
                    println!("DIAGNOSTICS ignored while recording")
                }

                Action::Diagnostics => {
                    // The diagnostics take over the first IMU, then streaming
                    // restarts. History from before the run is discarded.
                    pipeline.stop();

                    if let Some(primary) = pipeline.primary() {
                        // The model benchmark borrows the classifier's buffers.
                        let mut spare_input = [0.0_f32; 0];
                        let mut spare_scratch = [0.0_f32; 0];
                        let model_buffers = match classifier.as_mut() {
                            Some(classifier) => classifier.buffers(),
                            None => (&mut spare_input[..], &mut spare_scratch[..]),
                        };

                        imu_diag::run(
                            &mut primary.imu,
                            &mut delay,
                            &mut diag_buffers,
                            &mut pressure,
                            model_buffers,
                            period_us,
                        );
                    }

                    pipeline.start(&mut delay);
                    window.clear();
                }

                Action::ToggleRecord if recorder.is_active() => {
                    recorder.stop(window, batch, &mut stats)
                }

                Action::ToggleRecord => recorder.start(category, window),
            }
        }

        actions.clear();

        // ======================= 6. Recorder: bounded SD work for this pass ============================
        // Runs after the actions so a fresh recording writes its first frames
        // before the next drain can overwrite the oldest ones.
        recorder.step(window, batch, &mut stats);

        // ======================= 7. Status lines once per second ========================================
        if Instant::now() >= next_status {
            next_status = next_deadline(next_status, STATUS_PERIOD);
            print_status(&stats, &mut pipeline, window, &recorder, started_at);

            if let Some(classifier) = classifier.as_mut() {
                print_classifier(classifier);
                classifier.timing = Default::default();
            }

            stats.frames_this_period = 0;
            stats.sd_write_max_us = 0;
            stats.adc_us_max = 0;
            pipeline.timing = Default::default();
        }

        // ====================== 8. Idle until the next sample is due =============================

        let wait_us = next_acquire
            .duration_since_epoch()
            .as_micros()
            .saturating_sub(now_us());

        if wait_us > 0 {
            delay.delay_micros(wait_us as u32);
        }
    }
}

/// Prints the classifier's latest answer and the cost of producing it.
fn print_classifier(classifier: &imu_classifier::LiveClassifier) {
    let timing = classifier.timing;
    let runs = timing.runs.max(1) as f32;

    let answer = match classifier.last {
        Some(prediction) => {
            let mut text = String::<40>::new();
            let _ = write!(
                &mut text,
                "{} {:.0}%",
                classifier.model.class_name(prediction.class),
                prediction.probability * 100.0
            );
            text
        }
        None => String::try_from("(window filling)").unwrap_or_default(),
    };

    println!(
        "         classifier: {} [demo model, synthetic training data] | {} runs/s | input {:.0} us | inference {:.1} ms (max {:.1} ms)",
        answer.as_str(),
        timing.runs,
        timing.input_us_total as f32 / runs,
        timing.infer_us_total as f32 / runs / 1_000.0,
        timing.infer_us_max as f32 / 1_000.0
    );
}

/// Prints the once-per-second status block: rates and error counters, the
/// measured cost of each pipeline stage, then one line per connected IMU.
fn print_status(
    stats: &Stats,
    pipeline: &mut Pipeline,
    window: &FusedWindowBuffer<WINDOW_FRAMES>,
    recorder: &Recorder,
    started_at: Instant,
) {
    // Error counters summed over all IMUs.
    let (fifo_max, overflows, bus_errors, realigns) = pipeline.slots.iter().flatten().fold(
        (0_u16, 0_u32, 0_u32, 0_u32),
        |(fifo, ovf, err, realign), slot| {
            (
                fifo.max(slot.stats.fifo_max_bytes),
                ovf + slot.stats.overflows,
                err + slot.stats.bus_errors,
                realign + slot.stats.realigns,
            )
        },
    );

    println!(
        "[{:>5}s] {} fps | IMUs 0b{:04b} | total {} | fifo max {} B | ovf {} i2c-err {} realign {} | sd max {} us lost {} | win {}/{}{}",
        started_at.elapsed().as_secs(),
        stats.frames_this_period,
        pipeline.connected_mask(),
        stats.total_frames,
        fifo_max,
        overflows,
        bus_errors,
        realigns,
        stats.sd_write_max_us,
        stats.record_dropped,
        window.len(),
        window.capacity(),
        if recorder.is_active() { " REC" } else { "" }
    );

    // Measured stage costs. "per pass" covers all IMUs at once; "per frame"
    // is one instant of all IMUs.
    let timing = pipeline.timing;
    let per_pass = |total: u64| total as f32 / timing.passes.max(1) as f32;
    let per_frame = |total: u64| total as f32 / timing.frames.max(1) as f32;

    // Bus utilization: share of the last status period the I2C bus was busy.
    let bus_busy_percent = timing.bus_us_total as f32 / STATUS_PERIOD.as_micros() as f32 * 100.0;

    println!(
        "         pipeline: bus busy {:.1}% | bus {:.0} us/frame ({:.0} us/pass, max {} us) | fusion {:.1} us/frame | frame build {:.1} us/frame | ADC max {} us",
        bus_busy_percent,
        per_frame(timing.bus_us_total),
        per_pass(timing.bus_us_total),
        timing.bus_us_max,
        per_frame(timing.fusion_us_total),
        per_frame(timing.frame_us_total),
        stats.adc_us_max
    );

    // One line per IMU: latest raw reading, gyro calibration, orientation.
    let latest = window.latest().copied().unwrap_or(FusedFrame::ZERO);
    let scale = *pipeline.scale();

    for (index, slot) in pipeline.slots.iter().enumerate() {
        let Some(slot) = slot else { continue };

        let accel = latest.raw.imus[index].accel;
        let gyro = latest.raw.imus[index].gyro;
        let accel_g = scale.accel_g(accel);
        let accel_norm =
            sqrt(accel_g[0] * accel_g[0] + accel_g[1] * accel_g[1] + accel_g[2] * accel_g[2])
                .max(1e-3);
        let bias = slot.bias().bias_counts();
        let quat = slot.quaternion();
        let down = down_in_body(quat);

        println!(
            "         imu{} accel [{:>6} {:>6} {:>6}] = {:.2} g  gyro [{:>6} {:>6} {:>6}]  bias [{:+.1} {:+.1} {:+.1}] level {}  period {} us",
            index,
            accel[0],
            accel[1],
            accel[2],
            accel_norm,
            gyro[0],
            gyro[1],
            gyro[2],
            bias[0],
            bias[1],
            bias[2],
            slot.bias().level(),
            slot.period_us()
        );

        println!(
            "              q=[{:+.4} {:+.4} {:+.4} {:+.4}] down=[{:+.2} {:+.2} {:+.2}]  accel/|a|=[{:+.2} {:+.2} {:+.2}]  pressure {:?}",
            quat[0],
            quat[1],
            quat[2],
            quat[3],
            down[0],
            down[1],
            down[2],
            accel_g[0] / accel_norm,
            accel_g[1] / accel_norm,
            accel_g[2] / accel_norm,
            latest.raw.pressure_adc
        );
    }
}
