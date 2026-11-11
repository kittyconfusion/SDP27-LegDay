//! Sensor acquisition for Nathan's data-collection-commented branch.
//! record.rs initializes this reader once and publishes each acquired frame.
//!
//! record.rs passes the existing shared I2C0/I2C1 mutexes to ImuReader::new,
//! constructs PressureReader from ADC1 and GPIO1..GPIO4, combines the readers,
//! awaits initialize(), reports each result, and calls read() on its schedule.
//! Nothing in this module spawns a
//! task or selects a recording rate. Do not initialize the old IMU controller
//! and this reader against the same devices simultaneously.
//!
//! Training reads all four IMUs. Demo disables only non-prosthetic lower
//! (I2C0, 0x68), preserving slot order and reading all four pressure channels.
//! Raw counts, per-read timing, and explicit errors are returned to the caller.
//! The two buses are polled concurrently; devices on each bus are sequential.
//! These are not synchronized physical sample times.
//!
//! Hardware reference: TDK ICM-20948 DS-000189 rev 1.5, sections 8 and 10.
//! https://product.tdk.com/system/files/dam/doc/product/sensor/mortion-inertial/imu/data_sheet/ds-000189-icm-20948-v1.5.pdf
//! HAL reference: esp-hal 1.2 API, consistent with the calls already in this repo.
//! https://docs.espressif.com/projects/rust/esp-hal/1.2.0/esp32s3/esp_hal/index.html

#[cfg(not(feature = "esp32s3"))]
compile_error!("This sensor pinout is for the Nano ESP32-S3, not the Feather ESP32-C6.");

pub mod config;
pub mod frame;
pub mod imu;
pub mod pressure;
pub mod sample;

use embassy_time::Instant;
use esp_hal::i2c::master::Error;

pub use imu::ImuReader;
pub use pressure::PressureReader;
pub use sample::{ImuInitialization, SensorSample};

pub struct SensorReader<'d> {
    imus: ImuReader,
    pressure: PressureReader<'d>,
}

impl<'d> SensorReader<'d> {
    pub fn new(imus: ImuReader, pressure: PressureReader<'d>) -> Self {
        Self { imus, pressure }
    }

    pub async fn initialize(&mut self) -> [ImuInitialization<Error>; config::IMU_COUNT] {
        self.imus.initialize().await
    }

    pub fn all_required_imus_initialized(&self) -> bool {
        self.imus.all_required_initialized()
    }

    pub async fn read(&mut self) -> SensorSample<Error> {
        let read_started_us = Instant::now().as_micros();
        let imus = self.imus.read().await;
        let pressure = self.pressure.read();
        SensorSample {
            read_started_us,
            read_completed_us: Instant::now().as_micros(),
            imus,
            pressure,
        }
    }
}
