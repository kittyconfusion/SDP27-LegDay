//! # Bus, latency and wake-cost diagnostics
//!
//! Triggered by the DIAG button (D9 / GPIO18 to GND) or the `x` serial key.
//! It takes about a minute, prints a report, and then normal streaming
//! resumes. **Leave the IMU still on the bench while it runs**: one step times
//! the DMP's gyro calibration, which only converges at rest.
//!
//! Everything is measured on the real IMU and I2C bus, except the SPI rows
//! of the final table, which are calculated from the measured byte counts and
//! are labelled `calc`.
//!
//! The report has six sections:
//!
//! 1. **I2C primitives**: the fixed cost of one transaction and the cost of
//!    each extra byte, which everything else is built from.
//! 2. **Streaming cost**: bus time per sample in DMP and raw mode when the
//!    FIFO is drained every sample period, with and without an emulated
//!    second IMU on the same bus (the "two IMUs per leg" layout).
//! 3. **Stall tolerance**: the longest pause (a Wi-Fi burst, an SD flush)
//!    each mode's FIFO survives without losing samples.
//! 4. **Wake costs**: DMP firmware load time, sleep-to-first-sample time,
//!    whether the DMP firmware survives IMU sleep, and gyro calibration time.
//! 5. **CPU costs** on the S3: software fusion, packet parsing, frame
//!    encoding, the pressure ADC, and the prototype classifier.
//! 6. **Summary**: the measurements projected onto 4-IMU layouts.

use core::hint::black_box;

use esp_hal::delay::Delay;
use esp_println::println;

use legday_classifier::{
    fusion::Madgwick,
    prototype::{PrototypeClassifier, WINDOW_FEATURES, WindowFeatures},
};

use legday_core::{SensorFrame, WindowBuffer, fused::FusedFrame};

use legday_icm20948::{
    Config as ImuConfig, RAW_FIFO_SAMPLE_BYTES, RawSample,
    dmp::{self, DmpOutputs, DmpPacket, PACKETS_PER_CHUNK, header},
    regs::bank0,
};

use static_cell::ConstStaticCell;

use super::{Imu, PressureInputs, SAMPLE_RATE_DIV, WINDOW_FRAMES, now_us};

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// How long each streaming measurement runs.
const STREAM_MS: u64 = 3_000;

/// Pauses tried by the stall test, shortest first. The sweep stops at the
/// first pause that loses samples.
const STALLS_MS: [u32; 12] = [25, 50, 75, 100, 125, 150, 200, 300, 500, 800, 1_200, 1_600];

/// How long the IMU is put to sleep in the wake tests.
const SLEEP_MS: u32 = 2_000;

/// Give up on the gyro calibration test after this long.
const GYRO_CAL_TIMEOUT_US: u64 = 20_000_000;

/// SPI clock for the calculated rows (the ICM-20948's datasheet maximum).
const SPI_CLOCK_MHZ: f32 = 7.0;

/// Assumed fixed cost per SPI transaction (chip select plus driver setup).
/// This is an estimate, not a measurement; it is printed with the results.
const SPI_TRANSACTION_OVERHEAD_US: f32 = 5.0;

/// Bus utilization thresholds for the verdict column.
const UTILIZATION_OK: f32 = 50.0;
const UTILIZATION_MARGINAL: f32 = 80.0;

/// The data-age target from the project brief.
const TARGET_AGE_MS: f32 = 5.0;

// ---------------------------------------------------------------------------
// Static buffers for the classifier timing (kept off the stack)
// ---------------------------------------------------------------------------

/// One second of raw frames, the classifier's input shape.
static CLASSIFIER_WINDOW: ConstStaticCell<WindowBuffer<WINDOW_FRAMES>> =
    ConstStaticCell::new(WindowBuffer::new());

/// A six-class prototype model to time `predict` against.
static CLASSIFIER_MODEL: ConstStaticCell<PrototypeClassifier<6, WINDOW_FEATURES>> =
    ConstStaticCell::new(PrototypeClassifier::new());

/// Memory the diagnostics need, taken once at boot so the test can run any
/// number of times (a `ConstStaticCell` can only be taken once).
pub struct Buffers {
    window: &'static mut WindowBuffer<WINDOW_FRAMES>,
    model: &'static mut PrototypeClassifier<6, WINDOW_FEATURES>,
    packets: [DmpPacket; PACKETS_PER_CHUNK],
    raw: [RawSample; 32],
}

impl Buffers {
    pub fn take() -> Self {
        Self {
            window: CLASSIFIER_WINDOW.take(),
            model: CLASSIFIER_MODEL.take(),
            packets: [DmpPacket::default(); PACKETS_PER_CHUNK],
            raw: [RawSample::default(); 32],
        }
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Which acquisition path a test uses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    Dmp,
    Raw,
}

impl Path {
    const fn name(self) -> &'static str {
        match self {
            Self::Dmp => "DMP",
            Self::Raw => "raw",
        }
    }
}

/// What one drain of the IMU FIFO returned.
#[derive(Clone, Copy, Default)]
struct Drain {
    samples: usize,

    /// Bytes of sample data read (excludes the 2-byte FIFO count read).
    payload_bytes: usize,

    /// The FIFO overflowed or had to be resynchronised: samples were lost.
    overflowed: bool,

    /// Latest DMP gyro calibration accuracy (0..3), if a packet reported it.
    gyro_accuracy: Option<u16>,

    /// Latest DMP quaternion vector part, if a packet carried one.
    quat6: Option<[i32; 3]>,
}

impl Drain {
    /// Adds a later drain's results to this one.
    fn absorb(&mut self, other: Drain) {
        self.samples += other.samples;
        self.payload_bytes += other.payload_bytes;
        self.overflowed |= other.overflowed;
        self.gyro_accuracy = other.gyro_accuracy.or(self.gyro_accuracy);
        self.quat6 = other.quat6.or(self.quat6);
    }
}

/// Runs `op` `repeats` times and returns the average duration in µs.
/// Averaging gets below the systimer's 1 µs resolution.
fn average_us(imu: &mut Imu, repeats: u32, mut op: impl FnMut(&mut Imu)) -> f32 {
    let started = now_us();

    for _ in 0..repeats {
        op(imu);
    }

    (now_us() - started) as f32 / repeats as f32
}

/// Least-squares straight line through `(x, y)` points: returns
/// `(intercept, slope)`.
fn fit_line(points: &[(f32, f32)]) -> (f32, f32) {
    let n = points.len() as f32;
    let (sum_x, sum_y) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + y));
    let (sum_xx, sum_xy) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), (x, y)| (a + x * x, b + x * y));

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    (intercept, slope)
}

/// Size in bytes of a decoded DMP packet, rebuilt from the fields it carried.
fn packet_bytes(packet: &DmpPacket) -> usize {
    let mut bytes = 2 + 2; // header + footer

    if packet.header & header::HEADER2 != 0 {
        bytes += 2;
    }

    if packet.accel.is_some() {
        bytes += 6;
    }

    if packet.gyro.is_some() {
        bytes += 12; // gyro + gyro bias
    }

    if packet.quat6.is_some() {
        bytes += 12;
    }

    bytes += 2 * usize::from(packet.accel_accuracy.is_some());
    bytes += 2 * usize::from(packet.gyro_accuracy.is_some());

    bytes
}

// ---------------------------------------------------------------------------
// Starting and draining the IMU
// ---------------------------------------------------------------------------

/// Resets the IMU and starts DMP streaming. Returns the time spent loading
/// and configuring the DMP firmware, in µs.
fn start_dmp(imu: &mut Imu, delay: &mut Delay, verify: bool) -> u64 {
    imu.reset(delay).unwrap();

    let started = now_us();
    imu.dmp_initialize(delay, SAMPLE_RATE_DIV, verify).unwrap();
    let load_us = now_us() - started;

    imu.dmp_enable_outputs(DmpOutputs::ALL).unwrap();
    imu.dmp_set_all_odrs(0).unwrap();
    imu.dmp_start().unwrap();

    load_us
}

/// Resets the IMU and starts raw accel + gyro FIFO streaming.
fn start_raw(imu: &mut Imu, delay: &mut Delay) {
    imu.reset(delay).unwrap();

    let config = ImuConfig {
        sample_rate_div: SAMPLE_RATE_DIV,
        ..ImuConfig::default()
    };
    imu.configure(&config).unwrap();
    imu.start_raw_fifo().unwrap();
}

/// Reads whatever the FIFO holds right now (at most one 240-byte chunk in
/// DMP mode, at most 32 samples in raw mode).
fn drain_once(imu: &mut Imu, path: Path, buffers: &mut Buffers) -> Drain {
    match path {
        Path::Dmp => {
            let read = imu.dmp_read_packets(&mut buffers.packets).unwrap();

            let mut drain = Drain {
                samples: read.packets,
                overflowed: read.overflowed || read.resynced,
                ..Drain::default()
            };

            for packet in &buffers.packets[..read.packets] {
                drain.payload_bytes += packet_bytes(packet);
                drain.gyro_accuracy = packet.gyro_accuracy.or(drain.gyro_accuracy);
                drain.quat6 = packet.quat6.or(drain.quat6);
            }

            drain
        }

        Path::Raw => {
            let read = imu.read_raw_fifo_samples(&mut buffers.raw).unwrap();

            Drain {
                samples: read.samples,
                payload_bytes: read.samples * RAW_FIFO_SAMPLE_BYTES,
                overflowed: read.overflowed,
                ..Drain::default()
            }
        }
    }
}

/// Drains repeatedly until the FIFO is empty or an overflow is reported.
fn drain_all(imu: &mut Imu, path: Path, buffers: &mut Buffers) -> Drain {
    let mut total = Drain::default();

    // The cap is only a safety net; a full 4 kB FIFO empties in ~20 calls.
    for _ in 0..200 {
        let drain = drain_once(imu, path, buffers);
        total.absorb(drain);

        if drain.samples == 0 || drain.overflowed {
            break;
        }
    }

    total
}

/// Polls until at least one sample arrives. Returns how long that took and
/// what was read, or `None` on timeout.
fn wait_for_sample(
    imu: &mut Imu,
    path: Path,
    buffers: &mut Buffers,
    timeout_us: u64,
) -> Option<(u64, Drain)> {
    let started = now_us();

    while now_us() - started < timeout_us {
        let drain = drain_once(imu, path, buffers);

        if drain.samples > 0 {
            return Some((now_us() - started, drain));
        }
    }

    None
}

/// Reads the same number of bytes a second IMU would cost, from registers
/// that have no read side effects (0x3B..0x6A), in pieces of at most 48
/// bytes. When draining every sample period the payload is one DMP packet
/// (~40 B) or one raw sample (12 B), so this matches a real second IMU's
/// transactions exactly.
fn emulate_second_imu(imu: &mut Imu, payload_bytes: usize) {
    let mut scratch = [0_u8; 48];

    // The FIFO count read every drain starts with.
    imu.read_regs(0, bank0::FIFO_COUNT_H, &mut scratch[..2])
        .unwrap();

    let mut left = payload_bytes;
    while left > 0 {
        let piece = left.min(scratch.len());
        imu.read_regs(0, bank0::EXT_SLV_SENS_DATA_00, &mut scratch[..piece])
            .unwrap();
        left -= piece;
    }
}

// ---------------------------------------------------------------------------
// Section 1: I2C primitives
// ---------------------------------------------------------------------------

/// Measured I2C costs at the configured bus speed.
struct BusCosts {
    /// Fixed cost of one transaction (start, address, register, stop).
    overhead_us: f32,

    /// Cost of each additional byte read.
    per_byte_us: f32,
}

fn section_bus_primitives(imu: &mut Imu, delay: &mut Delay) -> BusCosts {
    println!();
    println!("[1/6] I2C primitives at 400 kHz");

    // After a reset the FIFO is off, so reading FIFO_R_W has no side effects.
    imu.reset(delay).unwrap();

    let write_us = average_us(imu, 200, |imu| {
        imu.write_reg(0, bank0::FIFO_EN_1, 0).unwrap()
    });
    let count_us = average_us(imu, 200, |imu| {
        imu.fifo_count().unwrap();
    });

    println!("  1-byte register write:      {:>7.1} us", write_us);
    println!("  FIFO count read (2 B):      {:>7.1} us", count_us);

    // Burst reads of increasing length. A straight-line fit through them
    // separates the fixed cost per transaction from the cost per byte.
    let lengths = [1_usize, 12, 40, 120, 240];
    let mut points = [(0.0_f32, 0.0_f32); 5];
    let mut buffer = [0_u8; 240];

    for (point, &length) in points.iter_mut().zip(&lengths) {
        let us = average_us(imu, 50, |imu| imu.read_fifo(&mut buffer[..length]).unwrap());
        *point = (length as f32, us);

        println!("  burst read {:>3} B:          {:>7.1} us", length, us);
    }

    let (overhead_us, per_byte_us) = fit_line(&points);

    println!(
        "  => {:.1} us per transaction + {:.2} us per byte (ideal at 400 kHz: 22.50 us/byte)",
        overhead_us, per_byte_us
    );

    BusCosts {
        overhead_us,
        per_byte_us,
    }
}

// ---------------------------------------------------------------------------
// Section 2: streaming cost per sample
// ---------------------------------------------------------------------------

/// Totals from one streaming run.
#[derive(Default)]
struct Stream {
    samples: u32,
    drains: u32,

    /// Time spent on the bus (real drain plus any emulated second IMU).
    bus_us: u64,

    payload_bytes: u64,

    /// Longest single drain, used for the worst-case data age.
    max_drain_us: u64,

    /// Most samples returned by one drain.
    max_burst: usize,

    overflows: u32,
}

impl Stream {
    /// Bus time per sample period, for everything on this bus.
    fn us_per_sample(&self) -> f32 {
        self.bus_us as f32 / self.samples.max(1) as f32
    }

    /// Average payload per sample per IMU.
    fn bytes_per_sample(&self) -> f32 {
        self.payload_bytes as f32 / self.samples.max(1) as f32
    }
}

/// Drains on a fixed schedule for `STREAM_MS`, timing only the bus work.
fn run_stream(
    imu: &mut Imu,
    path: Path,
    buffers: &mut Buffers,
    interval_us: u64,
    second_imu: bool,
) -> Stream {
    let mut stream = Stream::default();

    // Start from an empty FIFO so the first drain is not a backlog.
    drain_all(imu, path, buffers);

    let end_us = now_us() + STREAM_MS * 1_000;
    let mut next_us = now_us();

    while now_us() < end_us {
        // Busy-wait for the next drain slot: this stands in for the rest of
        // the control loop, and keeps the timing below free of other work.
        while now_us() < next_us {}
        next_us += interval_us;

        let started = now_us();

        let drain = drain_once(imu, path, buffers);
        if second_imu {
            emulate_second_imu(imu, drain.payload_bytes);
        }

        let busy_us = now_us() - started;

        stream.samples += drain.samples as u32;
        stream.drains += 1;
        stream.bus_us += busy_us;
        stream.payload_bytes += drain.payload_bytes as u64;
        stream.max_drain_us = stream.max_drain_us.max(busy_us);
        stream.max_burst = stream.max_burst.max(drain.samples);
        stream.overflows += u32::from(drain.overflowed);
    }

    stream
}

fn print_stream(label: &str, stream: &Stream, interval_us: u64, period_us: u64) {
    println!(
        "  {:<26} {:>6.0} us/sample  {:>5.1} B/sample  max drain {:>5} us  max burst {:>2}  worst age {:>5.2} ms  {:>4.1}% of period  ovf {}",
        label,
        stream.us_per_sample(),
        stream.bytes_per_sample(),
        stream.max_drain_us,
        stream.max_burst,
        (interval_us + stream.max_drain_us) as f32 / 1_000.0,
        stream.us_per_sample() / period_us as f32 * 100.0,
        stream.overflows
    );
}

/// Per-sample bus costs measured in section 2, all with per-period draining.
struct StreamCosts {
    dmp_us: f32,
    dmp_bytes: f32,
    dmp_two_imus_us: f32,
    raw_us: f32,
    raw_bytes: f32,
    raw_two_imus_us: f32,
}

fn section_streaming(
    imu: &mut Imu,
    delay: &mut Delay,
    buffers: &mut Buffers,
    period_us: u64,
) -> StreamCosts {
    println!();
    println!(
        "[2/6] Streaming cost, one real IMU (period {} us)",
        period_us
    );
    println!("  'per period' drains every sample period (what a 5 ms control loop needs);");
    println!(
        "  'batched' drains every 4 periods; '+ 2nd IMU' adds an emulated second IMU on the bus."
    );
    println!(
        "  'worst age' = drain interval + longest drain: the oldest a sample can be when it reaches RAM."
    );

    let batched_us = 4 * period_us;

    start_dmp(imu, delay, false);
    delay.delay_millis(500); // let the DMP settle before timing

    let dmp = run_stream(imu, Path::Dmp, buffers, period_us, false);
    print_stream("DMP, per period", &dmp, period_us, period_us);

    let dmp_batched = run_stream(imu, Path::Dmp, buffers, batched_us, false);
    print_stream("DMP, batched", &dmp_batched, batched_us, period_us);

    let dmp_two = run_stream(imu, Path::Dmp, buffers, period_us, true);
    print_stream("DMP + 2nd IMU, per period", &dmp_two, period_us, period_us);

    start_raw(imu, delay);
    delay.delay_millis(300);

    let raw = run_stream(imu, Path::Raw, buffers, period_us, false);
    print_stream("raw, per period", &raw, period_us, period_us);

    let raw_batched = run_stream(imu, Path::Raw, buffers, batched_us, false);
    print_stream("raw, batched", &raw_batched, batched_us, period_us);

    let raw_two = run_stream(imu, Path::Raw, buffers, period_us, true);
    print_stream("raw + 2nd IMU, per period", &raw_two, period_us, period_us);

    StreamCosts {
        dmp_us: dmp.us_per_sample(),
        dmp_bytes: dmp.bytes_per_sample(),
        dmp_two_imus_us: dmp_two.us_per_sample(),
        raw_us: raw.us_per_sample(),
        raw_bytes: raw.bytes_per_sample(),
        raw_two_imus_us: raw_two.us_per_sample(),
    }
}

// ---------------------------------------------------------------------------
// Section 3: stall tolerance
// ---------------------------------------------------------------------------

/// Pauses for increasing lengths and reports the longest pause the FIFO
/// survived. Returns that pause in ms.
fn stall_sweep(imu: &mut Imu, delay: &mut Delay, path: Path, buffers: &mut Buffers) -> u32 {
    let mut survived_ms = 0;

    for &stall_ms in &STALLS_MS {
        // Empty the FIFO and clear any stale overflow flag first.
        drain_all(imu, path, buffers);
        let _ = imu.take_fifo_overflow();

        delay.delay_millis(stall_ms);

        let queued_bytes = imu.fifo_count().unwrap();
        let drained = drain_all(imu, path, buffers);

        println!(
            "  {} stall {:>4} ms: FIFO held {:>4} B ({:>3} samples) -> {}",
            path.name(),
            stall_ms,
            queued_bytes,
            drained.samples,
            if drained.overflowed {
                "OVERFLOW, samples lost"
            } else {
                "ok"
            }
        );

        if drained.overflowed {
            return survived_ms;
        }

        survived_ms = stall_ms;
    }

    survived_ms
}

/// Longest survivable stall for each path, in ms.
struct StallLimits {
    dmp_ms: u32,
    raw_ms: u32,
}

fn section_stalls(imu: &mut Imu, delay: &mut Delay, buffers: &mut Buffers) -> StallLimits {
    println!();
    println!("[3/6] Stall tolerance: the longest pause with no lost samples");

    start_dmp(imu, delay, false);
    delay.delay_millis(300);
    let dmp_ms = stall_sweep(imu, delay, Path::Dmp, buffers);

    start_raw(imu, delay);
    delay.delay_millis(300);
    let raw_ms = stall_sweep(imu, delay, Path::Raw, buffers);

    println!(
        "  => DMP survives {} ms, raw survives {} ms",
        dmp_ms, raw_ms
    );

    StallLimits { dmp_ms, raw_ms }
}

// ---------------------------------------------------------------------------
// Section 4: wake costs
// ---------------------------------------------------------------------------

/// Compares the DMP program code in IMU memory against the firmware image.
/// Only the code region (from the program start address on) is compared:
/// the data region below it holds settings and state the DMP changes while
/// it runs, so it legitimately differs from the image.
fn firmware_code_mismatches(imu: &mut Imu) -> usize {
    let code_offset = usize::from(dmp::PROGRAM_START_ADDRESS - dmp::FIRMWARE_LOAD_ADDRESS);
    let code = &dmp::FIRMWARE[code_offset..];
    let mut chunk = [0_u8; 16];
    let mut mismatches = 0;

    for (index, expected) in code.chunks(16).enumerate() {
        let address = dmp::PROGRAM_START_ADDRESS + (index * 16) as u16;
        imu.dmp_read_mems(address, &mut chunk[..expected.len()])
            .unwrap();

        mismatches += chunk
            .iter()
            .zip(expected)
            .filter(|(read, want)| read != want)
            .count();
    }

    mismatches
}

fn section_wake(imu: &mut Imu, delay: &mut Delay, buffers: &mut Buffers) {
    println!();
    println!("[4/6] Wake costs (keep the IMU still)");

    // (a) Firmware load time, with and without reading it back.
    let verified_us = start_dmp(imu, delay, true);
    let fast_us = start_dmp(imu, delay, false);
    println!(
        "  DMP load + configure: {} ms with read-back verify, {} ms without",
        verified_us / 1_000,
        fast_us / 1_000
    );

    // (b) DMP: sleep the IMU, wake it, and see whether packets resume
    // without reloading the firmware.
    delay.delay_millis(500);
    let before = drain_all(imu, Path::Dmp, buffers);

    imu.set_sleep(true).unwrap();
    delay.delay_millis(SLEEP_MS);
    imu.set_sleep(false).unwrap();

    match wait_for_sample(imu, Path::Dmp, buffers, 1_000_000) {
        Some((us, after)) => {
            println!(
                "  DMP sleep {} s -> wake: first packet after {:.1} ms, no reload needed",
                SLEEP_MS / 1_000,
                us as f32 / 1_000.0
            );

            // A still IMU should report the same orientation after waking.
            if let (Some(q0), Some(q1)) = (before.quat6, after.quat6) {
                println!(
                    "    quaternion before sleep {:?}, after wake {:?} (Q30)",
                    q0, q1
                );
            }
        }

        None => {
            // Try restarting the DMP processor without reloading its code.
            imu.dmp_start().unwrap();

            match wait_for_sample(imu, Path::Dmp, buffers, 1_000_000) {
                Some((us, _)) => println!(
                    "  DMP sleep -> wake: no packets until DMP restart; then first packet after {:.1} ms, no reload",
                    us as f32 / 1_000.0
                ),
                None => println!(
                    "  DMP sleep -> wake: no packets even after DMP restart; firmware reload needed"
                ),
            }
        }
    }

    // (c) Is the program code still intact in IMU memory after the sleep?
    let mismatches = firmware_code_mismatches(imu);
    println!(
        "  DMP program code after sleep: {} of {} bytes differ from the image ({})",
        mismatches,
        dmp::FIRMWARE.len() - usize::from(dmp::PROGRAM_START_ADDRESS - dmp::FIRMWARE_LOAD_ADDRESS),
        if mismatches == 0 {
            "retained"
        } else {
            "NOT retained"
        }
    );

    // (d) Raw mode: sleep -> wake -> first sample.
    start_raw(imu, delay);
    delay.delay_millis(300);
    drain_all(imu, Path::Raw, buffers);

    imu.set_sleep(true).unwrap();
    delay.delay_millis(SLEEP_MS);

    let woke_us = now_us();
    imu.set_sleep(false).unwrap();
    imu.reset_fifo().unwrap(); // discard anything queued before sleeping

    match wait_for_sample(imu, Path::Raw, buffers, 1_000_000) {
        Some(_) => println!(
            "  raw sleep {} s -> wake: first sample after {:.1} ms",
            SLEEP_MS / 1_000,
            (now_us() - woke_us) as f32 / 1_000.0
        ),
        None => println!("  raw sleep -> wake: no sample within 1 s"),
    }

    // (e) Software reset -> first raw sample (the closest we can get to a
    // power cycle without a power switch on the IMU).
    let reset_us = now_us();
    start_raw(imu, delay);

    match wait_for_sample(imu, Path::Raw, buffers, 1_000_000) {
        Some(_) => println!(
            "  software reset -> first raw sample: {:.1} ms (includes 60 ms of fixed delays in reset())",
            (now_us() - reset_us) as f32 / 1_000.0
        ),
        None => println!("  software reset -> no raw sample within 1 s"),
    }

    // (f) Gyro calibration: time from a fresh DMP start until the DMP
    // reports full gyro accuracy (3). Only converges if the IMU is still.
    start_dmp(imu, delay, false);
    let calibration_start_us = now_us();
    let mut accuracy = 0;

    while accuracy < 3 && now_us() - calibration_start_us < GYRO_CAL_TIMEOUT_US {
        delay.delay_millis(5);

        if let Some(reported) = drain_all(imu, Path::Dmp, buffers).gyro_accuracy {
            accuracy = reported;
        }
    }

    if accuracy >= 3 {
        println!(
            "  DMP gyro calibration reached accuracy 3 after {:.1} s",
            (now_us() - calibration_start_us) as f32 / 1e6
        );
    } else {
        println!(
            "  DMP gyro calibration only reached accuracy {} in {} s (was the IMU moving?)",
            accuracy,
            GYRO_CAL_TIMEOUT_US / 1_000_000
        );
    }
}

// ---------------------------------------------------------------------------
// Section 5: CPU costs
// ---------------------------------------------------------------------------

/// Times `op` over `repeats` runs and returns µs per run.
fn average_cpu_us(repeats: u32, mut op: impl FnMut(u32)) -> f32 {
    let started = now_us();

    for index in 0..repeats {
        op(index);
    }

    (now_us() - started) as f32 / repeats as f32
}

/// A representative 40-byte DMP packet (accel, gyro, Quat6, both accuracies).
fn sample_packet() -> [u8; 40] {
    let mut bytes = [0_u8; 40];

    bytes[0..2].copy_from_slice(&0xC848_u16.to_be_bytes()); // header
    bytes[2..4].copy_from_slice(&0x6000_u16.to_be_bytes()); // header2: both accuracies
    bytes[4..6].copy_from_slice(&8_192_i16.to_be_bytes()); // accel x = 1 g
    bytes[10..12].copy_from_slice(&100_i16.to_be_bytes()); // gyro x
    bytes[22..26].copy_from_slice(&0x1000_0000_i32.to_be_bytes()); // quat x
    bytes[34..36].copy_from_slice(&3_u16.to_be_bytes()); // accel accuracy
    bytes[36..38].copy_from_slice(&3_u16.to_be_bytes()); // gyro accuracy

    bytes
}

/// Returns the Madgwick update time in µs (used by the summary).
fn section_cpu(buffers: &mut Buffers, pressure: &mut PressureInputs, period_us: u64) -> f32 {
    println!();
    println!("[5/6] CPU costs on the S3 (240 MHz, one core)");

    // Software fusion: one 6-axis Madgwick update.
    let mut filter = Madgwick::new(0.1);
    let madgwick_us = average_cpu_us(20_000, |index| {
        let wobble = (index % 64) as f32 * 0.001;
        filter.update(
            black_box([0.01 + wobble, -0.02, 0.03]),
            black_box([0.1, wobble, 0.99]),
            black_box(1.0 / 225.0),
        );
    });
    black_box(filter.quaternion());

    // DMP packet parsing.
    let packet = sample_packet();
    let allowed =
        header::ACCEL | header::GYRO | header::GYRO_CALIBR | header::QUAT6 | header::HEADER2 | 0x7;
    let parse_us = average_cpu_us(20_000, |_| {
        black_box(dmp::parse_packet(black_box(&packet), allowed).ok());
    });

    // Encoding one 120-byte frame for the SD card.
    let frame = FusedFrame::ZERO;
    let encode_us = average_cpu_us(20_000, |_| {
        black_box(black_box(&frame).encode_le());
    });

    // Reading the four pressure channels.
    let pressure_us = average_cpu_us(200, |_| {
        black_box(pressure.read());
    });

    // Prototype classifier: fill one second of synthetic frames, then time
    // feature extraction over the window and a nearest-centroid prediction.
    for index in 0..WINDOW_FRAMES {
        let mut raw = SensorFrame::ZERO;
        let value = (index % 50) as i16;
        raw.imus[0].accel = [value, -value, 8_192];
        raw.imus[0].gyro = [value * 2, 0, -value];
        raw.pressure_adc = [index as u16; 4];
        buffers.window.push(raw);
    }

    let features_us = average_cpu_us(20, |_| {
        black_box(WindowFeatures::from_window(black_box(&*buffers.window)));
    });

    let features = WindowFeatures::from_window(buffers.window).unwrap();
    for class in 0..6 {
        let mut shifted = features.values;
        shifted[0] += class as i32 * 100;
        buffers.model.observe(class, &shifted);
    }

    let predict_us = average_cpu_us(200, |_| {
        black_box(buffers.model.predict(black_box(&features.values)));
    });

    // What these cost per second at 4 IMUs and 225 Hz.
    let samples_per_s = 1_000_000.0 / period_us as f32;
    let fusion_load = madgwick_us * 4.0 * samples_per_s / 1e6 * 100.0;

    println!(
        "  Madgwick update:              {:>8.2} us  (4 IMUs at 225 Hz = {:.2}% of one core)",
        madgwick_us, fusion_load
    );
    println!("  DMP packet parse:             {:>8.2} us", parse_us);
    println!("  120-byte frame encode:        {:>8.2} us", encode_us);
    println!("  4 pressure ADC reads:         {:>8.2} us", pressure_us);
    println!(
        "  classifier features (1 s):    {:>8.1} us  ({} frames x 28 channels)",
        features_us, WINDOW_FRAMES
    );
    println!("  classifier predict (6 class): {:>8.1} us", predict_us);

    madgwick_us
}

// ---------------------------------------------------------------------------
// Section 6: summary
// ---------------------------------------------------------------------------

/// Calculated SPI cost per sample for one IMU: a FIFO count read and a data
/// read, each with one command byte, at the datasheet clock.
fn spi_us_per_sample(payload_bytes: f32) -> f32 {
    let bytes = payload_bytes + 2.0 + 2.0; // + count (2 B) + two command bytes
    bytes * 8.0 / SPI_CLOCK_MHZ + 2.0 * SPI_TRANSACTION_OVERHEAD_US
}

/// Prints one row of the summary table.
fn summary_row(layout: &str, path: &str, busy_us: f32, period_us: u64, cpu_us: f32) {
    let utilization = busy_us / period_us as f32 * 100.0;
    let worst_age_ms = (period_us as f32 + busy_us) / 1_000.0;

    let verdict = if utilization > 100.0 {
        "FAILS (cannot keep up)"
    } else if utilization > UTILIZATION_MARGINAL {
        "FAILS (no headroom)"
    } else if utilization > UTILIZATION_OK {
        "marginal"
    } else {
        "ok"
    };

    println!(
        "  {:<34} {:<4} {:>6.0} us {:>5.1}%  {:>6.0} us  {:>5.2} ms {:<3}  {}",
        layout,
        path,
        busy_us,
        utilization,
        cpu_us,
        worst_age_ms,
        if worst_age_ms <= TARGET_AGE_MS {
            "yes"
        } else {
            "no"
        },
        verdict
    );
}

fn section_summary(
    bus: &BusCosts,
    costs: &StreamCosts,
    stalls: &StallLimits,
    madgwick_us: f32,
    period_us: u64,
) {
    println!();
    println!(
        "[6/6] Summary: 4 IMUs, every sample drained every period ({} us)",
        period_us
    );
    println!(
        "  busy = bus time per period on the busiest bus; CPU = time the CPU waits on buses per"
    );
    println!("  period with today's blocking driver; age = worst age of a sample on reaching RAM.");
    println!(
        "  2-IMU figures use the measured emulated-2nd-IMU runs; SPI rows are calculated ({} MHz, {} us/transaction assumed).",
        SPI_CLOCK_MHZ, SPI_TRANSACTION_OVERHEAD_US
    );
    println!();
    println!(
        "  {:<34} {:<4} {:>9} {:>6}  {:>9}  {:>8} {:<3}  {}",
        "layout", "path", "busy", "util", "CPU", "age", "<5", "verdict"
    );

    // Measured: one IMU, and two IMUs on one bus.
    let dmp_one = costs.dmp_us;
    let raw_one = costs.raw_us;
    let dmp_two = costs.dmp_two_imus_us;
    let raw_two = costs.raw_two_imus_us;

    // Four on one bus = two "two IMU" loads back to back.
    summary_row(
        "4 IMUs, 1 I2C bus (mux or 2 addr)",
        "DMP",
        2.0 * dmp_two,
        period_us,
        2.0 * dmp_two,
    );
    summary_row("", "raw", 2.0 * raw_two, period_us, 2.0 * raw_two);

    // Two buses, two IMUs each. Concurrent needs the async driver or a
    // second core; blocking waits on each bus in turn.
    summary_row(
        "2 I2C buses x 2, concurrent",
        "DMP",
        dmp_two,
        period_us,
        dmp_two,
    );
    summary_row("", "raw", raw_two, period_us, raw_two);
    summary_row(
        "2 I2C buses x 2, blocking (today)",
        "DMP",
        2.0 * dmp_two,
        period_us,
        2.0 * dmp_two,
    );
    summary_row("", "raw", 2.0 * raw_two, period_us, 2.0 * raw_two);

    // SPI, calculated: four chip selects on one bus.
    let spi_dmp = 4.0 * spi_us_per_sample(costs.dmp_bytes);
    let spi_raw = 4.0 * spi_us_per_sample(costs.raw_bytes);
    summary_row(
        "SPI 7 MHz, 1 bus x 4 (calc)",
        "DMP",
        spi_dmp,
        period_us,
        spi_dmp,
    );
    summary_row("", "raw", spi_raw, period_us, spi_raw);

    println!();
    println!(
        "  one IMU measured: DMP {:.0} us/sample ({:.1} B), raw {:.0} us/sample ({:.1} B); bus {:.1} us/txn + {:.2} us/B",
        dmp_one, costs.dmp_bytes, raw_one, costs.raw_bytes, bus.overhead_us, bus.per_byte_us
    );
    println!(
        "  stall tolerance (per IMU FIFO, independent of IMU count): DMP {} ms, raw {} ms",
        stalls.dmp_ms, stalls.raw_ms
    );
    println!(
        "  software fusion for 4 IMUs: {:.0} us per period ({:.2}% of one core)",
        4.0 * madgwick_us,
        4.0 * madgwick_us / period_us as f32 * 100.0
    );
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Runs every section and prints the report. Leaves the IMU in an
/// unspecified state; the caller restarts normal acquisition afterwards.
pub fn run(
    imu: &mut Imu,
    delay: &mut Delay,
    buffers: &mut Buffers,
    pressure: &mut PressureInputs,
    period_us: u64,
) {
    let started_us = now_us();

    println!();
    println!(
        "==================== DIAGNOSTICS (about 1 minute; keep the IMU still) ===================="
    );

    let bus = section_bus_primitives(imu, delay);
    let costs = section_streaming(imu, delay, buffers, period_us);
    let stalls = section_stalls(imu, delay, buffers);
    section_wake(imu, delay, buffers);
    let madgwick_us = section_cpu(buffers, pressure, period_us);
    section_summary(&bus, &costs, &stalls, madgwick_us, period_us);

    println!();
    println!(
        "==================== DIAGNOSTICS DONE in {} s; resuming normal streaming ====================",
        (now_us() - started_us) / 1_000_000
    );
}
