#![no_std]
#![no_main]

//! # IMU bring-up test firmware
//!
//! Proves the sensor-to-RAM-to-SD path with one ICM-20948 and (optionally)
//! the pressure sensors. Everything the real firmware will need is here in
//! miniature:
//!
//! ```text
//!   ICM-20948 ──FIFO──▶ drain (I2C) ──▶ FusedFrame ──▶ 1 s RAM window ──▶ classifier (later)
//!   pressure ADC ───────────────────────────┘                 │
//!                                                  RECORD ───▶ SD card (.Q6B files)
//! ```
//!
//! * The IMU samples at 225 Hz on its own clock and queues results in its
//!   hardware FIFO. Every pass of the main loop drains whatever has queued
//!   up (usually 1-5 samples), so the loop never has to be exactly on time.
//! * In **DMP mode** each FIFO packet already carries raw accel, raw gyro
//!   and the 6-axis quaternion computed on the IMU. In **raw mode** the FIFO
//!   carries accel + gyro only and a Madgwick filter on the S3 computes the
//!   quaternion. DMP is tried first; raw mode is the fallback.
//! * Every sample becomes one 120-byte [`FusedFrame`] (the 72-byte raw
//!   `SensorFrame` followed by one quaternion per IMU) in a static ring
//!   buffer holding the last second. The classifier will read that buffer.
//! * The RECORD button (or `r` on the serial monitor) streams the window and
//!   all later frames to a new file on the SD card; pressing it again closes
//!   the file and reads it back to prove it is intact.
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
//!
//! ## Serial commands (type into the monitor)
//!
//! `r` record start/stop · `c` next category · `m` switch DMP/software ·
//! `d` hex-dump the newest frame · `p` pause draining 2 s (measures FIFO
//! depth) · `h` help.
//!
//! ## How to read this file
//!
//! Top to bottom: tunable constants, static buffers, small helper types,
//! the recorder, then `main`, which is "set everything up, then loop".

use core::fmt::Write as _;

use embedded_hal_bus::spi::ExclusiveDevice;
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
use legday_classifier::fusion::{
    ImuScale, Madgwick, down_in_body, q30_to_f32, reconstruct_w, sqrt,
};
use legday_core::{
    FRAME_WIRE_BYTES,
    clock::SampleClock,
    fused::{
        FUSED_FRAME_WIRE_BYTES, FusedFrame, FusedFrameBatch, FusedWindowBuffer, Quat6Raw, status,
    },
    recording::MotionCategory,
};
use legday_icm20948::{
    Config as ImuConfig, Error as ImuError, Icm20948, RawSample,
    dmp::{DmpOutputs, DmpPacket, PACKETS_PER_CHUNK},
};
use static_cell::ConstStaticCell;

use esp_backtrace as _;

esp_bootloader_esp_idf::esp_app_desc!();

// --------------------------------------------------------------------------------
// Tunable parameters
// --------------------------------------------------------------------------------

/// Try the on-IMU DMP first; fall back to software fusion if it fails.
const PREFER_DMP: bool = true;

/// Read the 14 kB DMP firmware back after loading (adds ~0.5 s at boot).
const VERIFY_DMP_FIRMWARE: bool = true;

/// IMU output rate = 1125 / (1 + div) Hz. 4 gives 225 Hz, the fastest rate
/// SparkFun documents for the DMP. Exact '200 Hz' is not available.
const SAMPLE_RATE_DIV: u8 = 4;

/// One second of history at 225 Hz; 225 x 120 bytes = 27,000 bytes of SRAM.
const WINDOW_FRAMES: usize = 225;

/// Bits of `valid_pressure_mask` to set: one bit per connected sensor on
/// A0..A3. All four channels are always sampled and stored; this only says which
/// to trust. Set bit 0 once a voltage divider is wired to A0 for pressure sensors
const CONNECTED_PRESSURE_MASK: u8 = 0b0000;

/// Frames handed to the SD card per loop pass while recording: 16 x 120 =
/// 1,920 bytes (about 4x 512KB sectors) written in 10-25 ms. Small enough that the
/// IMU FIFO (1 kB in DMP mode, ~115 ms) never overflows during a write.
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

// ---------------------------------------------------------------------------------------------
// SD-card plumbing
// ---------------------------------------------------------------------------------------------

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

/// Where the quaternion comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AcquisitionMode {
    /// Accel + gyro from the raw hardware FIFO; quaternion from Madgwick on the S3.
    RawFifo,
    /// Accel + gyro + Quat6 from DMP packets in the hardware FIFO.
    Dmp,
}

impl AcquisitionMode {
    const fn name(self) -> &'static str {
        match self {
            Self::RawFifo => "RAW+SW-FUSION",
            Self::Dmp => "DMP",
        }
    }
}

/// Something the user asked for, via a button or a serial key.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    ToggleRecord,
    NextCategory,
    ToggleMode,
    DumpFrame,
    Pause,
    Help,
}

/// Counters printed on the once-per-second status line.
#[derive(Default)]
struct Stats {
    frames_this_period: u32,
    total_frames: u32,
    fifo_max_bytes: u16,
    overflows: u32,
    resyncs: u32,
    bus_errors: u32,
    max_burst: usize,
    drain_time_max_us: u64,
    sd_write_max_us: u64,
    record_dropped: u32,
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

/// Microseconds since boot.
fn now_us() -> u64 {
    Instant::now().duration_since_epoch().as_micros()
}

fn print_help() {
    println!(
        "Serial commands: r=record start/stop, c=next category, m=switch DMP/software, d=dump newest frame, p=pause 2 s (FIFO depth), h=help"
    );
}

// -----------------------------------------------------------------------------------
// IMU start-up and frame building
// -----------------------------------------------------------------------------------

/// Resets the IMU and starts streaming: DMP if requested and it loads,
/// otherwise the raw FIFO. Returns the mode that is actually running.
fn start_acquisition<I2C: embedded_hal::i2c::I2c>(
    imu: &mut Icm20948<I2C>,
    delay: &mut Delay,
    config: &ImuConfig,
    prefer_dmp: bool,
) -> AcquisitionMode
where
    I2C::Error: core::fmt::Debug,
{
    imu.reset(delay).unwrap();

    if prefer_dmp {
        let started = now_us();

        // Load firmware, ask for all three outputs in every packet, go.
        let result: Result<(), ImuError<I2C::Error>> = imu
            .dmp_initialize(delay, config.sample_rate_div, VERIFY_DMP_FIRMWARE)
            .and_then(|_| imu.dmp_enable_outputs(DmpOutputs::ALL))
            .and_then(|_| imu.dmp_set_all_odrs(0))
            .and_then(|_| imu.dmp_start());

        match result {
            Ok(()) => {
                let state = imu.dmp_state();

                println!(
                    "DMP ready in {} ms: firmware {} bytes verified={} pll=0x{:02X} gyro_sf={}",
                    (now_us() - started) / 1_000,
                    legday_icm20948::dmp::FIRMWARE.len(),
                    VERIFY_DMP_FIRMWARE,
                    state.pll,
                    state.gyro_sf
                );

                return AcquisitionMode::Dmp;
            }
            Err(error) => {
                println!(
                    "DMP init FAILED: {:?}; falling back to raw FIFO + Madgwick",
                    error
                );

                imu.reset(delay).unwrap();
            }
        }
    }

    imu.configure(config).unwrap();
    imu.start_raw_fifo().unwrap();

    println!(
        "Raw FIFO streaming (12 bytes/sample), Madgwick beta {}",
        MADGWICK_BETA
    );

    AcquisitionMode::RawFifo
}

/// Builds the parts of a frame that do not depend on the IMU data.
fn base_frame(timestamp_us: u64, sequence: u32, pressure: [u16; 4]) -> FusedFrame {
    let mut frame = FusedFrame::ZERO;
    frame.raw.timestamp_us = timestamp_us;
    frame.raw.sequence = sequence;
    frame.raw.status_flags = status::TIMESTAMP_RECONSTRUCTED;
    frame.raw.pressure_adc = pressure;
    frame.raw.valid_pressure_mask = CONNECTED_PRESSURE_MASK;
    frame
}

/// Fills IMU 0 from a DMP packet. `accuracy` remembers the last reported
/// calibration accuracies, because the DMP only sends them when they change.
fn fill_from_dmp(frame: &mut FusedFrame, packet: &DmpPacket, accuracy: &mut (u16, u16)) {
    if let Some(accel) = packet.accel {
        frame.raw.imus[0].accel = accel;
        frame.raw.valid_imu_mask = 0b0001;
    }

    if let Some(gyro) = packet.gyro {
        frame.raw.imus[0].gyro = gyro;
    }

    if let Some(value) = packet.accel_accuracy {
        accuracy.0 = value & status::ACCURACY_MASK;
    }

    if let Some(value) = packet.gyro_accuracy {
        accuracy.1 = value & status::ACCURACY_MASK;
    }

    frame.raw.status_flags |=
        (accuracy.0 << status::ACCEL_ACCURACY_SHIFT) | (accuracy.1 << status::GYRO_ACCURACY_SHIFT);

    if let Some(quat) = packet.quat6 {
        frame.quat6[0] = Quat6Raw {
            x: quat[0],
            y: quat[1],
            z: quat[2],
        };
        frame.raw.status_flags |= status::ORIENTATION_DMP;
    }
}

/// Fills IMU 0 from a raw sample, running the software filter for the quaternion.
fn fill_from_raw(
    frame: &mut FusedFrame,
    sample: &RawSample,
    madgwick: &mut Madgwick,
    scale: &ImuScale,
    dt: f32,
) {
    madgwick.update(
        scale.gyro_rad_s(sample.gyro),
        scale.accel_g(sample.accel),
        dt,
    );

    let quat = madgwick.quat6_q30();

    frame.raw.imus[0].accel = sample.accel;
    frame.raw.imus[0].gyro = sample.gyro;
    frame.raw.valid_imu_mask = 0b0001;
    frame.quat6[0] = Quat6Raw {
        x: quat[0],
        y: quat[1],
        z: quat[2],
    };

    frame.raw.status_flags |= status::ORIENTATION_SOFTWARE;
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
    fn start(
        &mut self,
        category: MotionCategory,
        window: &FusedWindowBuffer<WINDOW_FRAMES>,
        mode: AcquisitionMode,
    ) {
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
            "RECORDING STARTED: /{}/{} with {} pre-roll frames queued ({} mode)",
            category.directory(),
            self.name.as_str(),
            window.len(),
            mode.name()
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
    let mut delay = Delay::new();

    // Receive side of the USB serial port, for the single-key commands.
    let mut usb_serial = UsbSerialJtag::new(peripherals.USB_DEVICE);

    println!();
    println!("LegDay IMU bring-up test (branch tim/testing-memory-and-sensors)");
    println!("I2C: A4/GPIO11 SDA, A5/GPIO12 SCL @ 400 kHz | ADC: A0-A3 pressure");
    println!("SPI: D13/GPIO48 CLK, D12/GPIO47 SO, D11/GPIO38 SI, D10/GPIO21 CS");
    println!("Buttons: D7/GPIO10 RECORD to GND, D8/GPIO17 CATEGORY to GND");
    println!(
        "Frame: {} bytes fused ({} raw + {} quat), window {} frames = {} bytes",
        FUSED_FRAME_WIRE_BYTES,
        FRAME_WIRE_BYTES,
        FUSED_FRAME_WIRE_BYTES - FRAME_WIRE_BYTES,
        WINDOW_FRAMES,
        WINDOW_FRAMES * core::mem::size_of::<FusedFrame>()
    );

    // ------------------- IMU: probe both possible addresses until one answers ---------------------
    let i2c = I2c::new(
        peripherals.I2C0,
        I2cConfig::default().with_frequency(Rate::from_khz(400)),
    )
    .unwrap()
    .with_sda(peripherals.GPIO11)
    .with_scl(peripherals.GPIO12);

    let mut imu = Icm20948::new(i2c, legday_icm20948::I2C_ADDR_AD0_HIGH);

    'probe: loop {
        for address in [
            legday_icm20948::I2C_ADDR_AD0_HIGH,
            legday_icm20948::I2C_ADDR_AD0_LOW,
        ] {
            imu.set_address(address);

            match imu.who_am_i() {
                Ok(legday_icm20948::WHO_AM_I_VALUE) => {
                    println!("ICM-20948 found at 0x{:02X} (WHO_AM_I = 0xEA)", address);
                    break 'probe;
                }
                Ok(other) => println!(
                    "0x{:02X}: WHO_AM_I = 0x{:02X}, not an ICM-20948",
                    address, other
                ),
                Err(error) => println!("0x{:02X}: {:?}", address, error),
            }
        }

        println!("No IMU answered. Check Qwiic blue->A4, yellow->A5, 3V3, GND. Retrying...");
        delay.delay_millis(1_000);
    }

    let imu_config = ImuConfig {
        sample_rate_div: SAMPLE_RATE_DIV,
        ..ImuConfig::default()
    };

    let period_us = u64::from(imu_config.period_us());

    println!(
        "IMU ODR {}.{:03} Hz, nominal period {} us",
        imu_config.odr_millihz() / 1_000,
        imu_config.odr_millihz() % 1_000,
        period_us
    );

    let mut mode = start_acquisition(&mut imu, &mut delay, &imu_config, PREFER_DMP);

    println!("Acquisition mode: {}", mode.name());

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

    let mut record_button = DebouncedButton::new();
    let mut category_button = DebouncedButton::new();
    let mut category = MotionCategory::Walk;

    // -------------------------------- Loop state --------------------------------------------
    let window = WINDOW.take();
    let batch = BATCH.take();
    let scale = ImuScale::ICM20948_4G_2000DPS;
    let mut madgwick = Madgwick::new(MADGWICK_BETA);
    let mut madgwick_seeded = false; // first sample initialises the filter from gravity
    let mut clock = SampleClock::new(period_us);
    let mut packets = [DmpPacket::default(); PACKETS_PER_CHUNK];
    let mut raw_samples = [RawSample::default(); 32];
    let mut sequence = 0_u32;
    let mut stats = Stats::default();
    let mut overflow_pending = false; // mark the next frame after a FIFO overflow
    let mut dmp_accuracy = (0_u16, 0_u16);
    let mut last_dmp_quat = [1.0_f32, 0.0, 0.0, 0.0];
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
        // ===================== 1. Drain the IMU FIFO into the RAM window =============================
        let drain_started = now_us();
        let pressure_counts = pressure.read();
        let mut new_frames = 0_usize;

        match mode {
            AcquisitionMode::Dmp => {
                let read = imu.dmp_read_packets(&mut packets).unwrap_or_else(|error| {
                    stats.bus_errors += 1;
                    if stats.bus_errors <= 5 {
                        println!("I2C error while draining DMP FIFO: {:?}", error);
                    }
                    Default::default()
                });

                stats.fifo_max_bytes = stats.fifo_max_bytes.max(read.fifo_bytes);
                stats.resyncs += u32::from(read.resynced);
                stats.overflows += u32::from(read.overflowed);

                if read.resynced || read.overflowed {
                    overflow_pending = true;
                    clock.resync(); // samples were lost; re-measure the rate
                }

                // One frame per packet, timestamps from the sample clock.
                let mut index = 0;

                clock.burst(drain_started, read.packets, |timestamp_us| {
                    let packet = &packets[index];

                    index += 1;

                    let mut frame = base_frame(timestamp_us, sequence, pressure_counts);

                    fill_from_dmp(&mut frame, packet, &mut dmp_accuracy);

                    if let Some(q) = packet.quat6 {
                        let (x, y, z) = (q30_to_f32(q[0]), q30_to_f32(q[1]), q30_to_f32(q[2]));
                        last_dmp_quat = [reconstruct_w(x, y, z), x, y, z];
                    }

                    // The software filter runs too, purely to cross-check the DMP.
                    if let (Some(accel), Some(gyro)) = (packet.accel, packet.gyro) {
                        let accel_g = scale.accel_g(accel);

                        if !madgwick_seeded {
                            madgwick.reset_to_gravity(accel_g);
                            madgwick_seeded = true;
                        }

                        madgwick.update(scale.gyro_rad_s(gyro), accel_g, period_us as f32 * 1e-6);
                    }

                    if core::mem::take(&mut overflow_pending) {
                        frame.raw.status_flags |= status::FIFO_OVERFLOW;
                    }

                    sequence = sequence.wrapping_add(1);
                    window.push(frame);
                    new_frames += 1;
                });
            }
            AcquisitionMode::RawFifo => {
                let read = imu
                    .read_raw_fifo_samples(&mut raw_samples)
                    .unwrap_or_else(|error| {
                        stats.bus_errors += 1;
                        if stats.bus_errors <= 5 {
                            println!("I2C error while draining raw FIFO: {:?}", error);
                        }

                        Default::default()
                    });

                stats.fifo_max_bytes = stats.fifo_max_bytes.max(read.fifo_bytes);

                if read.overflowed {
                    stats.overflows += 1;
                    overflow_pending = true;
                    clock.resync(); // samples were lost; re-measure the rate
                }

                let mut index = 0;

                clock.burst(drain_started, read.samples, |timestamp_us| {
                    let sample = &raw_samples[index];

                    index += 1;

                    if !madgwick_seeded {
                        madgwick.reset_to_gravity(scale.accel_g(sample.accel));
                        madgwick_seeded = true;
                    }

                    let mut frame = base_frame(timestamp_us, sequence, pressure_counts);

                    fill_from_raw(
                        &mut frame,
                        sample,
                        &mut madgwick,
                        &scale,
                        period_us as f32 * 1e-6,
                    );

                    if core::mem::take(&mut overflow_pending) {
                        frame.raw.status_flags |= status::FIFO_OVERFLOW;
                    }

                    sequence = sequence.wrapping_add(1);
                    window.push(frame);
                    new_frames += 1;
                });
            }
        }

        stats.frames_this_period += new_frames as u32;
        stats.total_frames += new_frames as u32;
        stats.max_burst = stats.max_burst.max(new_frames);
        stats.drain_time_max_us = stats.drain_time_max_us.max(now_us() - drain_started);

        // ===================== 2. Collect input: buttons every 5 ms, serial keys always ===================

        if Instant::now() >= next_button_poll {
            next_button_poll += BUTTON_PERIOD;

            if category_button.update(category_pin.is_low()) {
                let _ = actions.push(Action::NextCategory);
            }

            if record_button.update(record_pin.is_low()) {
                let _ = actions.push(Action::ToggleRecord);
            }
        }

        while let Ok(byte) = usb_serial.read_byte() {
            let action = match byte.to_ascii_lowercase() {
                b'r' => Some(Action::ToggleRecord),
                b'c' => Some(Action::NextCategory),
                b'm' => Some(Action::ToggleMode),
                b'd' => Some(Action::DumpFrame),
                b'p' => Some(Action::Pause),
                b'h' | b'?' => Some(Action::Help),
                _ => None,
            };

            if let Some(action) = action {
                let _ = actions.push(action);
            }
        }

        // ====================== 3. Perform the requested actions ==============================

        for action in actions.iter().copied() {
            match action {
                Action::Help => print_help(),
                Action::DumpFrame => match window.latest() {
                    Some(frame) => dump_frame(frame),
                    None => println!("window is empty"),
                },
                Action::Pause => {
                    println!(
                        "PAUSE: not draining for {} ms; watch 'fifo max' and ovf/resync next",
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

                Action::ToggleMode if recorder.is_active() => {
                    println!("MODE change ignored while recording")
                }

                Action::ToggleMode => {
                    // Stop the current stream, restart in the other mode, and
                    // forget history that was produced under the old mode.

                    println!("Switching acquisition mode...");

                    let _ = match mode {
                        AcquisitionMode::Dmp => imu.dmp_stop(),
                        AcquisitionMode::RawFifo => imu.stop_raw_fifo(),
                    };

                    mode = start_acquisition(
                        &mut imu,
                        &mut delay,
                        &imu_config,
                        mode == AcquisitionMode::RawFifo,
                    );

                    window.clear();
                    clock.reset();
                    madgwick.reset();
                    madgwick_seeded = false;
                    overflow_pending = false;
                    stats = Stats::default();

                    println!("Acquisition mode: {}", mode.name());
                }

                Action::ToggleRecord if recorder.is_active() => {
                    recorder.stop(window, batch, &mut stats)
                }

                Action::ToggleRecord => recorder.start(category, window, mode),
            }
        }

        actions.clear();

        // ======================= 4. Recorder: bounded SD work for this pass ============================
        // Runs after the actions so a fresh recording writes its first frames
        // before the next drain can overwrite the oldest ones.
        recorder.step(window, batch, &mut stats);

        // ======================= 5. Status line once per second ========================================
        if Instant::now() >= next_status {
            next_status += STATUS_PERIOD;

            let latest = window.latest().copied().unwrap_or(FusedFrame::ZERO);
            let accel = latest.raw.imus[0].accel;
            let gyro = latest.raw.imus[0].gyro;
            let accel_g = scale.accel_g(accel);
            let accel_norm =
                sqrt(accel_g[0] * accel_g[0] + accel_g[1] * accel_g[1] + accel_g[2] * accel_g[2])
                    .max(1e-3);
            let sw_quat = madgwick.quaternion_positive_w();
            let sw_down = down_in_body(sw_quat);

            println!(
                "[{:>5}s] {} {} fps ({} us period) | total {} | fifo max {} B | burst max {} | drain max {} us | sd max {} us | ovf {} resync {} i2c-err {} lost {} | win {}/{}{}",
                started_at.elapsed().as_secs(),
                mode.name(),
                stats.frames_this_period,
                clock.period_us(now_us()),
                stats.total_frames,
                stats.fifo_max_bytes,
                stats.max_burst,
                stats.drain_time_max_us,
                stats.sd_write_max_us,
                stats.overflows,
                stats.resyncs,
                stats.bus_errors,
                stats.record_dropped,
                window.len(),
                window.capacity(),
                if recorder.is_active() { " REC" } else { "" }
            );

            println!(
                "         accel [{:>6} {:>6} {:>6}] = {:.2} g  gyro [{:>6} {:>6} {:>6}]  pressure {:?}  seq {}",
                accel[0],
                accel[1],
                accel[2],
                accel_norm,
                gyro[0],
                gyro[1],
                gyro[2],
                latest.raw.pressure_adc,
                latest.raw.sequence
            );

            if mode == AcquisitionMode::Dmp {
                let dmp_down = down_in_body(last_dmp_quat);

                println!(
                    "         DMP  q=[{:+.4} {:+.4} {:+.4} {:+.4}] down=[{:+.2} {:+.2} {:+.2}] acc/gyro accuracy {}/{}",
                    last_dmp_quat[0],
                    last_dmp_quat[1],
                    last_dmp_quat[2],
                    last_dmp_quat[3],
                    dmp_down[0],
                    dmp_down[1],
                    dmp_down[2],
                    dmp_accuracy.0,
                    dmp_accuracy.1
                );
            }
            println!(
                "         SW   q=[{:+.4} {:+.4} {:+.4} {:+.4}] down=[{:+.2} {:+.2} {:+.2}]  accel/|a|=[{:+.2} {:+.2} {:+.2}]",
                sw_quat[0],
                sw_quat[1],
                sw_quat[2],
                sw_quat[3],
                sw_down[0],
                sw_down[1],
                sw_down[2],
                accel_g[0] / accel_norm,
                accel_g[1] / accel_norm,
                accel_g[2] / accel_norm
            );

            stats.frames_this_period = 0;
            stats.max_burst = 0;
            stats.drain_time_max_us = 0;
            stats.sd_write_max_us = 0;
        }

        // ====================== 6. Idle: nothing queued, so do not hammer the I2C bus =============================

        if new_frames == 0 {
            delay.delay_micros(500);
        }
    }
}
