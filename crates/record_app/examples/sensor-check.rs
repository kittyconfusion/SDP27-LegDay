//! Standalone Nano ESP32-S3 sensor check. No SD, recording, display, LEDs,
//! classifier, or quaternions. Nathan's recorder remains a separate program.
//!
//! Compile only (does not flash):
//! cargo check --locked --release -p quadriceps --example sensor-check
//!
//! The slow diagnostic polling below is for readable USB output, not a
//! recording schedule. Raw register polling may skip intermediate samples.

#![no_std]
#![no_main]

#[path = "../bin/sensors/mod.rs"]
mod sensors;

use embassy_executor::Spawner;
use embassy_sync::mutex::Mutex;
use embassy_time::Timer;
use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock,
    i2c::master::{Config as I2cConfig, I2c},
    time::Rate,
    timer::timg::TimerGroup,
};
use esp_println::println;
use sensors::{
    ImuReader, PressureReader, SensorReader,
    config::{AcquisitionMode, I2C_BUS_KHZ, IMU_LAYOUT, SAMPLE_HZ},
    imu::SharedI2c,
    sample::ImuReading,
};
use static_cell::StaticCell;

esp_bootloader_esp_idf::esp_app_desc!();

// To check the removable reference later, select Demo here. Slot 1 then stays
// Disabled, while the remaining three IMUs and all pressure inputs are read.
const MODE: AcquisitionMode = AcquisitionMode::Training;
static I2C0_BUS: StaticCell<SharedI2c> = StaticCell::new();
static I2C1_BUS: StaticCell<SharedI2c> = StaticCell::new();

#[esp_rtos::main]
async fn main(_spawner: Spawner) -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    // Match Nathan's scheduler/heap setup; no radio is initialized.
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);
    let timg0 = TimerGroup::new(peripherals.TIMG0);
    esp_rtos::start(timg0.timer0, peripherals.FROM_CPU_INTR0);

    println!("Sensor check: {:?}; raw counts only", MODE);
    println!("I2C0 A4/A5 (GPIO11/12); I2C1 A6/A7 (GPIO13/14)");
    println!("Pressure A0..A3 (GPIO1..4); diagnostic polling every 100 ms");
    println!(
        "IMU settings: +/-8 g, +/-1000 deg/s, nominal {} Hz",
        SAMPLE_HZ
    );

    // Same pins and configured bus speed as record.rs. This standalone
    // program owns its peripherals; it does not run alongside his recorder.
    let i2c0 = I2c::new(
        peripherals.I2C0,
        I2cConfig::default().with_frequency(Rate::from_khz(I2C_BUS_KHZ)),
    )
    .expect("I2C0 configuration")
    .with_sda(peripherals.GPIO11)
    .with_scl(peripherals.GPIO12)
    .into_async();
    let i2c1 = I2c::new(
        peripherals.I2C1,
        I2cConfig::default().with_frequency(Rate::from_khz(I2C_BUS_KHZ)),
    )
    .expect("I2C1 configuration")
    .with_sda(peripherals.GPIO13)
    .with_scl(peripherals.GPIO14)
    .into_async();

    let shared_i2c0 = I2C0_BUS.init(Mutex::new(i2c0));
    let shared_i2c1 = I2C1_BUS.init(Mutex::new(i2c1));
    let imus = ImuReader::new(shared_i2c0, shared_i2c1, MODE);
    let pressure = PressureReader::new(
        peripherals.ADC1,
        peripherals.GPIO1,
        peripherals.GPIO2,
        peripherals.GPIO3,
        peripherals.GPIO4,
    );
    let mut reader = SensorReader::new(imus, pressure);
    let report = reader.initialize().await;
    for (assignment, result) in IMU_LAYOUT.iter().zip(&report) {
        println!(
            "Init {:?} {:?} address 0x{:02x}: {:?}",
            assignment.slot, assignment.bus, assignment.address, result,
        );
    }
    println!(
        "All required IMUs initialized: {}",
        reader.all_required_imus_initialized(),
    );

    loop {
        // Wait before the first read too: initialization discards startup data.
        Timer::after_millis(100).await;
        let sample = reader.read().await;
        println!(
            "Read {}..{} us; fresh IMU mask 0b{:04b}",
            sample.read_started_us,
            sample.read_completed_us,
            sample.fresh_imu_mask(),
        );
        for (assignment, reading) in IMU_LAYOUT.iter().zip(&sample.imus) {
            match reading {
                ImuReading::Fresh(reading) => println!(
                    "  {:?}: accel {:?}; gyro {:?}; read {}..{} us",
                    assignment.slot,
                    reading.value.accel,
                    reading.value.gyro,
                    reading.read_started_us,
                    reading.read_completed_us,
                ),
                other => println!("  {:?}: {:?}", assignment.slot, other),
            }
        }
        println!(
            "  Pressure A0/A1 non-prosthetic, A2/A3 prosthetic: {:?}",
            sample.pressure.map(|reading| reading.value),
        );
    }
}
