#![no_std]
#![no_main]

//! IMU bring-up test for the `tim/testing-memory-and-sensors` branch.
//!
//! One ICM-20948 on the Nano ESP32's default I2C pins streams accel + gyro
//! samples at 225 Hz. If the DMP firmware loads, the breakout board also
//! produces a 6-axis quaternion (Quat6) per sample; otherwise a Madgwick
//! filter on the S3 produces the quaternion instead. Every sample becomes a
//! 120-byte `FusedFrame` (72-byte raw `SensorFrame` + 4 x 12-byte Quat6) in a
//! one-second static RAM window, and the RECORD / CATEGORY buttons write the
//! window plus live frames to the SD card exactly like `sd-card-test`.
//!
//! The same actions are available as single-key commands typed into the
//! serial monitor: `r` record, `c` category, `m` switch DMP <-> software
//! fusion, `d` dump the newest frame as hex, `p` pause draining for 2 s to
//! measure FIFO depth, `h` help.
//!
//! Wiring (Arduino Nano ESP32 labels):
//!   IMU  Qwiic blue  -> A4 / GPIO11 (SDA), Qwiic yellow -> A5 / GPIO12 (SCL)
//!   SD   D13/GPIO48 CLK, D12/GPIO47 MISO, D11/GPIO38 MOSI, D10/GPIO21 CS
//!   RECORD button D7/GPIO10 to GND, CATEGORY button D8/GPIO17 to GND

use core::fmt::Write as _;

use embedded_hal_bus::spi::ExclusiveDevice;
use embedded_sdmmc::{Mode, SdCard, TimeSource, Timestamp, VolumeIdx, VolumeManager};
use esp_hal::{
    clock::CpuClock,
    delay::Delay,
    gpio::{Input, InputConfig, Level, Output, OutputConfig, Pull},
    i2c::master::{Config as I2cConfig, I2c},
    spi::{
        Mode as SpiMode,
        master::{Config as SpiConfig, Spi},
    },
    time::{Duration, Instant, Rate},
    usb::usb_serial_jtag::UsbSerialJtag,
};
use esp_println::println;
use heapless::String;
use legday_classifier::fusion::{ImuScale, Madgwick, down_in_body, q30_to_f32, reconstruct_w};
use legday_core::{
    FRAME_WIRE_BYTES,
    fused::{
        FUSED_FRAME_WIRE_BYTES, FusedFrame, FusedFrameBatch, FusedWindowBuffer, Quat6Raw, status,
    },
    recording::MotionCategory,
};
use legday_icm20948::{
    Config as ImuConfig, Error as ImuError, FifoRead, Icm20948, RawSample,
    dmp::{DmpOutputs, DmpPacket, DmpReadStatus, mem},
};
use static_cell::ConstStaticCell;

use esp_backtrace as _;

esp_bootloader_esp_idf::esp_app_desc!();

/// Try the on-IMU DMP first; fall back to software fusion if it fails.
const PREFER_DMP: bool = true;
/// Read the 14 kB firmware back after loading (adds ~0.5 s at boot).
const VERIFY_DMP_FIRMWARE: bool = true;
/// IMU ODR = 1125 / (1 + div) Hz. 4 -> 225 Hz, the fastest documented DMP rate.
const SAMPLE_RATE_DIV: u8 = 4;
/// One second of history at 225 Hz: 225 x 120 bytes = 27,000 bytes of SRAM.
const WINDOW_FRAMES: usize = 225;
const STATUS_PERIOD: Duration = Duration::from_millis(1_000);
const BUTTON_PERIOD: Duration = Duration::from_millis(5);
const DEBOUNCE_SAMPLES: u8 = 6;
/// FAT 8.3 extension for 120-byte fused recordings (`.BIN` stays 72-byte).
const FILE_EXTENSION: &str = "Q6B";
const MADGWICK_BETA: f32 = 0.1;
/// Frames handed to the SD card per loop pass while recording. Keeps every
/// pass short so the IMU FIFO (about 1 kB in DMP mode) never overflows;
/// 16 x 120 bytes = 1920 bytes, roughly 4 sectors, written in a few ms.
const SD_FRAMES_PER_PASS: usize = 16;
/// Directory-entry/FAT flush interval while recording. Each flush is a
/// read-modify-write that can stall for tens of milliseconds.
const SD_FLUSH_PERIOD: Duration = Duration::from_secs(5);
/// How long the `p` command stops draining the IMU, to measure FIFO depth.
const PAUSE_PERIOD: Duration = Duration::from_millis(2_000);

static WINDOW: ConstStaticCell<FusedWindowBuffer<WINDOW_FRAMES>> =
    ConstStaticCell::new(FusedWindowBuffer::new());
static BATCH: ConstStaticCell<FusedFrameBatch> = ConstStaticCell::new(FusedFrameBatch::new());

struct FixedTime;

impl TimeSource for FixedTime {
    fn get_timestamp(&self) -> Timestamp {
        Timestamp::from_calendar(1980, 1, 1, 0, 0, 0).unwrap()
    }
}

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

    /// Returns `true` once per debounced press.
    fn update(&mut self, raw_low: bool) -> bool {
        if raw_low != self.candidate_low {
            self.candidate_low = raw_low;
            self.count = 1;
            return false;
        }
        self.count = self.count.saturating_add(1);
        if self.count >= DEBOUNCE_SAMPLES && self.stable_low != self.candidate_low {
            self.stable_low = self.candidate_low;
            return self.stable_low;
        }
        false
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AcquisitionMode {
    /// Raw accel + gyro from the hardware FIFO, quaternion from Madgwick on the S3.
    RawFifo,
    /// Raw accel + gyro + Quat6 from DMP packets in the hardware FIFO.
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    ToggleRecord,
    NextCategory,
    ToggleMode,
    DumpFrame,
    Pause,
    Help,
}

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

fn now_us() -> u64 {
    Instant::now().duration_since_epoch().as_micros()
}

/// Resets the IMU and starts either DMP or raw-FIFO acquisition.
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
        let result: Result<(), ImuError<I2C::Error>> = imu
            .dmp_initialize(delay, config.sample_rate_div, VERIFY_DMP_FIRMWARE)
            .and_then(|_| imu.dmp_enable_outputs(DmpOutputs::ALL))
            .and_then(|_| imu.dmp_set_odr(mem::ODR_QUAT6, mem::ODR_CNTR_QUAT6, 0))
            .and_then(|_| imu.dmp_set_odr(mem::ODR_ACCEL, mem::ODR_CNTR_ACCEL, 0))
            .and_then(|_| imu.dmp_set_odr(mem::ODR_GYRO, mem::ODR_CNTR_GYRO, 0))
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
                println!("DMP init FAILED: {:?}", error);
                println!("Falling back to raw FIFO + Madgwick on the S3");
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

fn print_help() {
    println!(
        "Serial commands: r=record start/stop, c=next category, m=switch DMP/software, d=dump newest frame, p=pause 2 s (FIFO depth), h=help"
    );
}

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
    let quat = frame.quat6[0];
    let (x, y, z) = (q30_to_f32(quat.x), q30_to_f32(quat.y), q30_to_f32(quat.z));
    println!(
        "  imu0 accel {:?} gyro {:?} quat6 w={:+.4} x={:+.4} y={:+.4} z={:+.4}",
        frame.raw.imus[0].accel,
        frame.raw.imus[0].gyro,
        reconstruct_w(x, y, z),
        x,
        y,
        z
    );
}

#[esp_hal::main]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    let mut delay = Delay::new();
    let mut usb_serial = UsbSerialJtag::new(peripherals.USB_DEVICE);

    println!();
    println!("LegDay IMU bring-up test (branch tim/testing-memory-and-sensors)");
    println!("I2C: A4/GPIO11 SDA, A5/GPIO12 SCL @ 400 kHz");
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

    // ---------------------------------------------------------------- IMU --
    let i2c = I2c::new(
        peripherals.I2C0,
        I2cConfig::default().with_frequency(Rate::from_khz(400)),
    )
    .unwrap()
    .with_sda(peripherals.GPIO11)
    .with_scl(peripherals.GPIO12);
    let mut imu = Icm20948::new(i2c, legday_icm20948::I2C_ADDR_AD0_HIGH);

    loop {
        let mut found = None;
        for address in [
            legday_icm20948::I2C_ADDR_AD0_HIGH,
            legday_icm20948::I2C_ADDR_AD0_LOW,
        ] {
            imu.set_address(address);
            match imu.who_am_i() {
                Ok(legday_icm20948::WHO_AM_I_VALUE) => {
                    println!("ICM-20948 found at 0x{:02X} (WHO_AM_I = 0xEA)", address);
                    found = Some(address);
                    break;
                }
                Ok(other) => println!(
                    "0x{:02X}: replied WHO_AM_I = 0x{:02X}, not an ICM-20948",
                    address, other
                ),
                Err(error) => println!("0x{:02X}: {:?}", address, error),
            }
        }
        if found.is_some() {
            break;
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

    // ------------------------------------------------------------ SD card --
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
    spi_bus.write(&[0xff; 10]).unwrap();
    let spi_device = ExclusiveDevice::new(spi_bus, cs, Delay::new()).unwrap();
    let sdcard = SdCard::new(spi_device, Delay::new());

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
    let volume = if sd_present {
        match volume_manager.open_volume(VolumeIdx(0)) {
            Ok(volume) => Some(volume),
            Err(error) => {
                println!(
                    "FAT VOLUME FAILED: {:?} (continuing without recording)",
                    error
                );
                None
            }
        }
    } else {
        None
    };
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

    // ------------------------------------------------------------ buttons --
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
    let mut active_file: Option<embedded_sdmmc::File<'_, _, _, 4, 4, 1>> = None;
    let mut active_name = String::<12>::new();
    let mut active_category = category;
    let mut next_record_seq = 0_u32;
    let mut last_flush = Instant::now();

    // -------------------------------------------------------------- state --
    let window = WINDOW.take();
    let batch = BATCH.take();
    let mut madgwick = Madgwick::new(MADGWICK_BETA);
    let mut madgwick_seeded = false;
    let scale = ImuScale::ICM20948_4G_2000DPS;
    let dt = period_us as f32 * 1e-6;
    let mut packets = [DmpPacket::default(); legday_icm20948::dmp::PACKETS_PER_CHUNK];
    let mut raw_samples = [RawSample::default(); 32];
    let mut sequence = 0_u32;
    let mut stats = Stats::default();
    let mut overflow_pending = false;
    let mut last_dmp_quat = [1.0_f32, 0.0, 0.0, 0.0];
    let mut last_accuracy = (0_u16, 0_u16);
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
        // ------------------------------------------- drain the IMU FIFO --
        let drain_started = now_us();
        let mut new_frames = 0_usize;

        match mode {
            AcquisitionMode::Dmp => {
                let read = match imu.dmp_read_packets(&mut packets) {
                    Ok(read) => read,
                    Err(error) => {
                        stats.bus_errors += 1;
                        if stats.bus_errors <= 5 {
                            println!("I2C error while draining DMP FIFO: {:?}", error);
                        }
                        DmpReadStatus::default()
                    }
                };
                stats.fifo_max_bytes = stats.fifo_max_bytes.max(read.fifo_bytes);
                if read.resynced {
                    stats.resyncs += 1;
                    overflow_pending = true;
                }
                if read.overflowed {
                    stats.overflows += 1;
                    overflow_pending = true;
                }
                let count = read.packets;
                for (index, packet) in packets[..count].iter().enumerate() {
                    let timestamp_us = drain_started - (count - 1 - index) as u64 * period_us;
                    let mut frame = FusedFrame::ZERO;
                    frame.raw.timestamp_us = timestamp_us;
                    frame.raw.sequence = sequence;
                    frame.raw.status_flags = status::TIMESTAMP_RECONSTRUCTED;
                    if let Some(accel) = packet.accel {
                        frame.raw.imus[0].accel = accel;
                        frame.raw.valid_imu_mask = 0b0001;
                    }
                    if let Some(gyro) = packet.gyro {
                        frame.raw.imus[0].gyro = gyro;
                    }
                    if let Some(accuracy) = packet.accel_accuracy {
                        last_accuracy.0 = accuracy & status::ACCURACY_MASK;
                    }
                    if let Some(accuracy) = packet.gyro_accuracy {
                        last_accuracy.1 = accuracy & status::ACCURACY_MASK;
                    }
                    frame.raw.status_flags |= (last_accuracy.0 << status::ACCEL_ACCURACY_SHIFT)
                        | (last_accuracy.1 << status::GYRO_ACCURACY_SHIFT);
                    if let Some(quat) = packet.quat6 {
                        frame.quat6[0] = Quat6Raw {
                            x: quat[0],
                            y: quat[1],
                            z: quat[2],
                        };
                        frame.raw.status_flags |= status::ORIENTATION_DMP;
                        let x = q30_to_f32(quat[0]);
                        let y = q30_to_f32(quat[1]);
                        let z = q30_to_f32(quat[2]);
                        last_dmp_quat = [reconstruct_w(x, y, z), x, y, z];
                    }
                    if let (Some(accel), Some(gyro)) = (packet.accel, packet.gyro) {
                        // Software fusion runs alongside the DMP for comparison only.
                        let accel_g = scale.accel_g(accel);
                        if !madgwick_seeded {
                            madgwick.reset_to_gravity(accel_g);
                            madgwick_seeded = true;
                        }
                        madgwick.update(scale.gyro_rad_s(gyro), accel_g, dt);
                    }
                    if overflow_pending {
                        frame.raw.status_flags |= status::FIFO_OVERFLOW;
                        overflow_pending = false;
                    }
                    sequence = sequence.wrapping_add(1);
                    window.push(frame);
                    new_frames += 1;
                }
            }
            AcquisitionMode::RawFifo => {
                let read = match imu.read_raw_fifo_samples(&mut raw_samples) {
                    Ok(read) => read,
                    Err(error) => {
                        stats.bus_errors += 1;
                        if stats.bus_errors <= 5 {
                            println!("I2C error while draining raw FIFO: {:?}", error);
                        }
                        FifoRead::default()
                    }
                };
                stats.fifo_max_bytes = stats.fifo_max_bytes.max(read.fifo_bytes);
                if read.overflowed {
                    stats.overflows += 1;
                    overflow_pending = true;
                }
                let count = read.samples;
                for (index, sample) in raw_samples[..count].iter().enumerate() {
                    let timestamp_us = drain_started - (count - 1 - index) as u64 * period_us;
                    let accel_g = scale.accel_g(sample.accel);
                    if !madgwick_seeded {
                        madgwick.reset_to_gravity(accel_g);
                        madgwick_seeded = true;
                    }
                    madgwick.update(scale.gyro_rad_s(sample.gyro), accel_g, dt);
                    let quat = madgwick.quat6_q30();
                    let mut frame = FusedFrame::ZERO;
                    frame.raw.timestamp_us = timestamp_us;
                    frame.raw.sequence = sequence;
                    frame.raw.valid_imu_mask = 0b0001;
                    frame.raw.status_flags =
                        status::TIMESTAMP_RECONSTRUCTED | status::ORIENTATION_SOFTWARE;
                    frame.raw.imus[0].accel = sample.accel;
                    frame.raw.imus[0].gyro = sample.gyro;
                    frame.quat6[0] = Quat6Raw {
                        x: quat[0],
                        y: quat[1],
                        z: quat[2],
                    };
                    if overflow_pending {
                        frame.raw.status_flags |= status::FIFO_OVERFLOW;
                        overflow_pending = false;
                    }
                    sequence = sequence.wrapping_add(1);
                    window.push(frame);
                    new_frames += 1;
                }
            }
        }

        stats.frames_this_period += new_frames as u32;
        stats.total_frames += new_frames as u32;
        stats.max_burst = stats.max_burst.max(new_frames);
        stats.drain_time_max_us = stats.drain_time_max_us.max(now_us() - drain_started);

        // ------------------------------------------- collect user input --
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
            let action = match byte {
                b'r' | b'R' => Some(Action::ToggleRecord),
                b'c' | b'C' => Some(Action::NextCategory),
                b'm' | b'M' => Some(Action::ToggleMode),
                b'd' | b'D' => Some(Action::DumpFrame),
                b'p' | b'P' => Some(Action::Pause),
                b'h' | b'H' | b'?' => Some(Action::Help),
                _ => None,
            };
            if let Some(action) = action {
                let _ = actions.push(action);
            }
        }

        // ---------------------------------------------- perform actions --
        for action in actions.iter().copied() {
            match action {
                Action::Help => print_help(),
                Action::Pause => {
                    println!(
                        "PAUSE: not draining for {} ms; watch 'fifo max' and ovf/resync on the next status lines",
                        PAUSE_PERIOD.as_millis()
                    );
                    delay.delay_millis(PAUSE_PERIOD.as_millis() as u32);
                }
                Action::DumpFrame => {
                    if let Some(frame) = window.latest() {
                        dump_frame(frame);
                    } else {
                        println!("window is empty");
                    }
                }
                Action::NextCategory => {
                    if active_file.is_none() {
                        category = category.next();
                        println!("Selected /{}", category.directory());
                    } else {
                        println!("CATEGORY ignored while recording");
                    }
                }
                Action::ToggleMode => {
                    if active_file.is_some() {
                        println!("MODE change ignored while recording");
                    } else {
                        let prefer_dmp = mode == AcquisitionMode::RawFifo;
                        println!("Switching acquisition mode...");
                        if mode == AcquisitionMode::Dmp {
                            let _ = imu.dmp_stop();
                        } else {
                            let _ = imu.stop_raw_fifo();
                        }
                        mode = start_acquisition(&mut imu, &mut delay, &imu_config, prefer_dmp);
                        window.clear();
                        madgwick_seeded = false;
                        madgwick.reset();
                        overflow_pending = false;
                        stats = Stats::default();
                        println!("Acquisition mode: {}", mode.name());
                    }
                }
                Action::ToggleRecord => {
                    if let Some(file) = active_file.take() {
                        // Drain the cursor to the newest frame, then close.
                        if let (Some(oldest), Some(newest)) = (window.get(0), window.latest()) {
                            let oldest_seq = oldest.raw.sequence;
                            while newest.raw.sequence.wrapping_sub(next_record_seq) < 0x8000_0000 {
                                let index = next_record_seq.wrapping_sub(oldest_seq) as usize;
                                if let Some(frame) = window.get(index) {
                                    if batch.push(frame) {
                                        file.write(batch.encoded()).unwrap();
                                        batch.clear();
                                    }
                                }
                                next_record_seq = next_record_seq.wrapping_add(1);
                            }
                        }
                        if !batch.is_empty() {
                            file.write(batch.encoded()).unwrap();
                            batch.clear();
                        }
                        file.flush().unwrap();
                        let bytes = file.length();
                        file.close().unwrap();
                        println!(
                            "RECORDING STOPPED: /{}/{} = {} bytes = {} fused frames safely closed",
                            active_category.directory(),
                            active_name.as_str(),
                            bytes,
                            bytes as usize / FUSED_FRAME_WIRE_BYTES
                        );

                        // Read-back check: first and last frame of the file we just wrote.
                        if let Some(root) = root.as_ref() {
                            let dir = root.open_dir(active_category.directory()).unwrap();
                            let file = dir
                                .open_file_in_dir(active_name.as_str(), Mode::ReadOnly)
                                .unwrap();
                            let mut first = [0_u8; FUSED_FRAME_WIRE_BYTES];
                            let mut last = [0_u8; FUSED_FRAME_WIRE_BYTES];
                            let first_len = file.read(&mut first).unwrap();
                            file.seek_from_end(FUSED_FRAME_WIRE_BYTES as u32).unwrap();
                            let last_len = file.read(&mut last).unwrap();
                            file.close().unwrap();
                            dir.close().unwrap();
                            if first_len == FUSED_FRAME_WIRE_BYTES
                                && last_len == FUSED_FRAME_WIRE_BYTES
                                && bytes as usize % FUSED_FRAME_WIRE_BYTES == 0
                            {
                                let first = FusedFrame::decode_le(&first);
                                let last = FusedFrame::decode_le(&last);
                                let span_frames =
                                    last.raw.sequence.wrapping_sub(first.raw.sequence) as u64 + 1;
                                let span_us =
                                    last.raw.timestamp_us.saturating_sub(first.raw.timestamp_us);
                                println!(
                                    "READBACK OK: seq {}..{} ({} frames, {} in file), {} ms span, {} Hz effective, flags 0x{:04X}",
                                    first.raw.sequence,
                                    last.raw.sequence,
                                    span_frames,
                                    bytes as usize / FUSED_FRAME_WIRE_BYTES,
                                    span_us / 1_000,
                                    if span_us > 0 {
                                        (span_frames - 1) * 1_000_000 / span_us
                                    } else {
                                        0
                                    },
                                    last.raw.status_flags
                                );
                            } else {
                                println!(
                                    "READBACK FAILED: read {}/{} bytes, length {} not a multiple of {}",
                                    first_len, last_len, bytes, FUSED_FRAME_WIRE_BYTES
                                );
                            }
                        }
                    } else if let Some(root) = root.as_ref() {
                        let category_dir = root.open_dir(category.directory()).unwrap();
                        let mut filename: Option<String<12>> = None;
                        for session in 0..=9_999_999_u32 {
                            let mut candidate = String::<12>::new();
                            write!(&mut candidate, "S{:07}.{}", session, FILE_EXTENSION).unwrap();
                            if !category_dir.directory_entry_exists(candidate.as_str()) {
                                filename = Some(candidate);
                                break;
                            }
                        }
                        let filename = filename.expect("all session filenames are occupied");
                        let file = category_dir
                            .open_file_in_dir(filename.as_str(), Mode::ReadWriteCreate)
                            .unwrap();
                        category_dir.close().unwrap();

                        // Pre-roll: start the cursor at the oldest frame in the
                        // one-second window; the recorder step streams it out.
                        batch.clear();
                        next_record_seq = window
                            .get(0)
                            .map(|frame| frame.raw.sequence)
                            .unwrap_or(sequence);
                        last_flush = Instant::now();
                        println!(
                            "RECORDING STARTED: /{}/{} with {} pre-roll frames queued ({} mode)",
                            category.directory(),
                            filename.as_str(),
                            window.len(),
                            mode.name()
                        );
                        active_name = filename;
                        active_category = category;
                        active_file = Some(file);
                    } else {
                        println!("RECORD ignored: no SD card mounted");
                    }
                }
            }
        }
        actions.clear();

        // ------------------------------------------------- recorder step --
        // The recorder is a cursor into the RAM window: it writes up to
        // SD_FRAMES_PER_PASS frames per pass, so the one-second pre-roll and
        // live frames drain to the card at ~3x real time without ever
        // blocking a pass for longer than the IMU FIFO can buffer. It runs
        // after the actions so a fresh recording writes its first frames
        // before the next drain can overwrite the oldest ones.
        if let Some(file) = active_file.as_ref() {
            if let (Some(oldest), Some(newest)) = (window.get(0), window.latest()) {
                let oldest_seq = oldest.raw.sequence;
                let behind = oldest_seq.wrapping_sub(next_record_seq);
                if behind != 0 && behind < 0x8000_0000 {
                    println!("RECORDER fell behind the window: {} frames lost", behind);
                    stats.record_dropped += behind;
                    next_record_seq = oldest_seq;
                }
                let write_started = now_us();
                let mut written = 0;
                while newest.raw.sequence.wrapping_sub(next_record_seq) < 0x8000_0000
                    && written < SD_FRAMES_PER_PASS
                {
                    let index = next_record_seq.wrapping_sub(oldest_seq) as usize;
                    let frame = window.get(index).unwrap();
                    if batch.push(frame) {
                        file.write(batch.encoded()).unwrap();
                        batch.clear();
                    }
                    next_record_seq = next_record_seq.wrapping_add(1);
                    written += 1;
                }
                if batch.frames() >= SD_FRAMES_PER_PASS {
                    file.write(batch.encoded()).unwrap();
                    batch.clear();
                }
                if last_flush.elapsed() >= SD_FLUSH_PERIOD {
                    file.flush().unwrap();
                    last_flush = Instant::now();
                }
                if written > 0 {
                    stats.sd_write_max_us = stats.sd_write_max_us.max(now_us() - write_started);
                }
            }
        }

        // ------------------------------------------------------- status --
        if Instant::now() >= next_status {
            next_status += STATUS_PERIOD;
            let latest = window.latest().copied().unwrap_or(FusedFrame::ZERO);
            let accel = latest.raw.imus[0].accel;
            let gyro = latest.raw.imus[0].gyro;
            let accel_g = scale.accel_g(accel);
            let accel_norm = legday_classifier::fusion::sqrt(
                accel_g[0] * accel_g[0] + accel_g[1] * accel_g[1] + accel_g[2] * accel_g[2],
            );
            let software_quat = madgwick.quaternion_positive_w();
            let software_down = down_in_body(software_quat);

            println!(
                "[{:>5}s] {} {} fps | total {} | fifo max {} B | burst max {} | drain max {} us | sd max {} us | ovf {} resync {} i2c-err {} lost {} | win {}/{}{}",
                started_at.elapsed().as_secs(),
                mode.name(),
                stats.frames_this_period,
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
                if active_file.is_some() { " REC" } else { "" }
            );
            println!(
                "         accel [{:>6} {:>6} {:>6}] = {:.2} g  gyro [{:>6} {:>6} {:>6}] counts  seq {}",
                accel[0],
                accel[1],
                accel[2],
                accel_norm,
                gyro[0],
                gyro[1],
                gyro[2],
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
                    last_accuracy.0,
                    last_accuracy.1
                );
            }
            println!(
                "         SW   q=[{:+.4} {:+.4} {:+.4} {:+.4}] down=[{:+.2} {:+.2} {:+.2}]  accel/|a|=[{:+.2} {:+.2} {:+.2}]",
                software_quat[0],
                software_quat[1],
                software_quat[2],
                software_quat[3],
                software_down[0],
                software_down[1],
                software_down[2],
                accel_g[0] / accel_norm.max(1e-3),
                accel_g[1] / accel_norm.max(1e-3),
                accel_g[2] / accel_norm.max(1e-3)
            );
            stats.frames_this_period = 0;
            stats.max_burst = 0;
            stats.drain_time_max_us = 0;
            stats.sd_write_max_us = 0;
        }

        if new_frames == 0 {
            // Nothing queued yet; a short nap keeps I2C polling well below the ODR.
            delay.delay_micros(500);
        }
    }
}
