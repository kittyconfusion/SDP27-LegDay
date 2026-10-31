#![no_std]
#![no_main]

use core::fmt::Write as _;

use embedded_hal_bus::spi::ExclusiveDevice;
use embedded_sdmmc::{Mode, SdCard, TimeSource, Timestamp, VolumeIdx, VolumeManager};
use esp_hal::{
    delay::Delay,
    gpio::{Input, InputConfig, Level, Output, OutputConfig, Pull},
    spi::{
        Mode as SpiMode,
        master::{Config as SpiConfig, Spi},
    },
    time::{Instant, Rate},
};
use esp_println::println;
use heapless::String;
use legday_core::lib_recording::{
    Imu6Raw, SAMPLE_RATE_HZ, SensorFrame, WINDOW_SAMPLES, WindowBuffer,
};
use legday_core::recording::{FrameBatch, MotionCategory};
use static_cell::ConstStaticCell;

use esp_backtrace as _;

esp_bootloader_esp_idf::esp_app_desc!();

const SAMPLE_PERIOD_MS: u32 = 1_000 / SAMPLE_RATE_HZ as u32;
const DEBOUNCE_SAMPLES: u8 = 6;
const SYNTHETIC_STATUS_FLAG: u16 = 0x8000;

static WINDOW: ConstStaticCell<WindowBuffer<WINDOW_SAMPLES>> =
    ConstStaticCell::new(WindowBuffer::new());
static BATCH: ConstStaticCell<FrameBatch> = ConstStaticCell::new(FrameBatch::new());

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

#[esp_hal::main]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default());
    let delay = Delay::new();

    println!("LegDay SD-card pipeline test");
    println!("SPI: D13/GPIO48 CLK, D12/GPIO47 SO, D11/GPIO38 SI, D10/GPIO21 CS");
    println!("Buttons: D7/GPIO10 RECORD to GND, D8/GPIO17 CATEGORY to GND");

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

    // SD cards require at least 74 clocks at <=400 kHz with CS deasserted.
    spi_bus.write(&[0xff; 10]).unwrap();
    let spi_device = ExclusiveDevice::new(spi_bus, cs, Delay::new()).unwrap();
    let sdcard = SdCard::new(spi_device, Delay::new());

    let card_bytes = match sdcard.num_bytes() {
        Ok(bytes) => bytes,
        Err(error) => {
            println!("SD INIT FAILED: {:?}", error);
            println!("Check FAT32 format, 3.3 V power, common ground, and SPI wiring.");
            loop {
                delay.delay_millis(1_000);
            }
        }
    };
    println!("SD initialized: {} bytes", card_bytes);

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
    println!("SPI data clock raised to 4 MHz");

    let volume_manager = VolumeManager::new(sdcard, FixedTime);
    let volume = match volume_manager.open_volume(VolumeIdx(0)) {
        Ok(volume) => volume,
        Err(error) => {
            println!("FAT VOLUME FAILED: {:?}", error);
            loop {
                delay.delay_millis(1_000);
            }
        }
    };
    let root = volume.open_root_dir().unwrap();

    for category in MotionCategory::ALL {
        if !root.directory_entry_exists(category.directory()) {
            root.make_dir_in_dir(category.directory()).unwrap();
            println!("Created /{}", category.directory());
        }
    }

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
    let mut sequence = 0_u32;
    let mut batches_since_flush = 0_u8;
    let window = WINDOW.take();
    let batch = BATCH.take();

    println!("SD CARD TEST READY");
    println!(
        "Selected /{}; press CATEGORY while idle to change",
        category.directory()
    );
    println!("Press RECORD to start; press it again to flush and close");

    loop {
        delay.delay_millis(SAMPLE_PERIOD_MS);

        if category_button.update(category_pin.is_low()) {
            if active_file.is_none() {
                category = category.next();
                println!("Selected /{}", category.directory());
            } else {
                println!("CATEGORY ignored while recording");
            }
        }

        if record_button.update(record_pin.is_low()) {
            if let Some(file) = active_file.take() {
                if !batch.is_empty() {
                    file.write(batch.encoded()).unwrap();
                    batch.clear();
                }
                file.flush().unwrap();
                let bytes = file.length();
                file.close().unwrap();
                println!("RECORDING STOPPED: {} bytes safely closed", bytes);
                batches_since_flush = 0;
            } else {
                let category_dir = root.open_dir(category.directory()).unwrap();
                let mut filename: Option<String<12>> = None;
                for session in 0..=9_999_999_u32 {
                    let mut candidate = String::<12>::new();
                    write!(&mut candidate, "S{:07}.BIN", session).unwrap();
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

                batch.clear();
                for index in 0..window.len() {
                    let frame = window.get(index).unwrap();
                    if batch.push(frame) {
                        file.write(batch.encoded()).unwrap();
                        batch.clear();
                    }
                }
                if !batch.is_empty() {
                    file.write(batch.encoded()).unwrap();
                    batch.clear();
                }
                file.flush().unwrap();
                println!(
                    "RECORDING STARTED: /{}/{} with {} pre-roll frames",
                    category.directory(),
                    filename.as_str(),
                    window.len()
                );
                active_file = Some(file);
            }
        }

        let frame = synthetic_frame(sequence, category);
        sequence = sequence.wrapping_add(1);
        window.push(frame);

        if let Some(file) = active_file.as_ref() {
            if batch.push(&frame) {
                file.write(batch.encoded()).unwrap();
                batch.clear();
                batches_since_flush += 1;
                if batches_since_flush >= 4 {
                    file.flush().unwrap();
                    batches_since_flush = 0;
                    println!("recording: {} frames captured", sequence);
                }
            }
        }
    }
}

fn synthetic_frame(sequence: u32, category: MotionCategory) -> SensorFrame {
    let wave = (sequence % 400) as i16 - 200;
    let pressure_wave = ((sequence % 200) * 20) as u16;
    let mut imus = [Imu6Raw::default(); 4];
    for (index, imu) in imus.iter_mut().enumerate() {
        let offset = index as i16 * 100;
        imu.accel = [wave * 8 + offset, -wave * 4, 16_384 + offset];
        imu.gyro = [wave * 3, -wave * 2 + offset, wave];
    }

    let category_id = MotionCategory::ALL
        .iter()
        .position(|candidate| *candidate == category)
        .unwrap() as u16;

    SensorFrame {
        timestamp_us: Instant::now().duration_since_epoch().as_micros(),
        sequence,
        valid_imu_mask: 0b1111,
        valid_pressure_mask: 0b1111,
        status_flags: SYNTHETIC_STATUS_FLAG | category_id,
        imus,
        pressure_adc: [
            pressure_wave,
            4_000 - pressure_wave,
            pressure_wave / 2,
            4_000 - pressure_wave / 2,
        ],
    }
}
