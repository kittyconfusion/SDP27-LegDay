//! Direct-connected ICM-20948 bench reader for the Adafruit ESP32-C6 Feather.
//! Six-axis bring-up only: no magnetometer, force sensors, mux or radio yet.

use embassy_sync::{
    blocking_mutex::raw::CriticalSectionRawMutex,
    mutex::{Mutex, MutexGuard},
};
use embassy_time::{Duration, Instant, Ticker, Timer, block_for};
use esp_hal::{
    Async,
    delay::Delay,
    gpio::{Level, Output, OutputConfig},
    i2c::master::{Config, Error, I2c},
};
use esp_println::println;
use log::error;

const PERIOD_US: u64 = 20_000;
const HEADER: &str =
    "seq,t_us,read_us,sensor_id,ax_m_s2,ay_m_s2,az_m_s2,gx_rad_s,gy_rad_s,gz_rad_s";

pub struct IMU_Controller {
    i2c_bus: &'static Mutex<CriticalSectionRawMutex, I2c<'static, Async>>,
}
impl IMU_Controller {
    pub async fn new(
        i2c_bus: &'static Mutex<CriticalSectionRawMutex, I2c<'static, Async>>,
    ) -> Self {
        Self { i2c_bus: i2c_bus }
    }

    async fn register(
        &self,
        i2c: &mut MutexGuard<'_, CriticalSectionRawMutex, I2c<'static, Async>>,
        address: u8,
        reg: u8,
    ) -> Result<u8, Error> {
        let mut byte = [0];
        i2c.write_read_async(address, &[reg], &mut byte).await?;
        Ok(byte[0])
    }

    pub async fn check_sensor_existance(&self) {
        {
            let mut i2c = self.i2c_bus.lock().await;
            let mut detected = None;
            for address in [0x69_u8, 0x68_u8] {
                if i2c.write(address, &[0x7f, 0]).is_ok()
                    && matches!(self.register(&mut i2c, address, 0x00).await, Ok(0xea))
                {
                    detected = Some(address);
                }

                let Some(_) = detected else {
                    loop {
                        error!("# ERROR: ICM-20948 not detected at {address}; check wiring/power");
                        Timer::after_millis(1000).await;
                    }
                };
            }
        }
    }

    // these functions all run synchronously (figure timing is important-ish)
    // just lock down the i2c bus for the entire time
    pub async fn initialize(&self, address: u8) -> Result<(), Error> {
        let mut i2c = self.i2c_bus.lock().await;
        // Register banks and values: TDK DS-000189 rev 1.5, sections 8 and 10.
        i2c.write(address, &[0x7f, 0x00])?; // bank 0
        i2c.write(address, &[0x06, 0x80])?; // device reset
        block_for(Duration::from_millis(100));
        i2c.write(address, &[0x7f, 0x00])?;
        i2c.write(address, &[0x06, 0x01])?; // wake, automatic clock selection
        i2c.write(address, &[0x07, 0x00])?; // enable accel and gyro axes
        block_for(Duration::from_millis(100));
        i2c.write(address, &[0x7f, 0x20])?; // bank 2
        i2c.write(address, &[0x00, 19])?; // gyro divider
        i2c.write(address, &[0x01, 0x25])?; // +/-1000 dps, DLPF config 4, enabled
        i2c.write(address, &[0x10, 0x00, 19])?; // accel 12-bit divider
        i2c.write(address, &[0x14, 0x25])?; // +/-8 g, DLPF config 4, enabled

        // Read back the settings that determine scaling; reject mismatches.
        for (reg, value) in [
            (0x00, 19),
            (0x01, 0x25),
            (0x10, 0),
            (0x11, 19),
            (0x14, 0x25),
        ] {
            if self.register(&mut i2c, address, reg).await? != value {
                panic!("IMU configuration readback mismatch at register {reg:#x}");
            }
        }
        i2c.write(address, &[0x7f, 0x00])?;
        i2c.write(address, &[0x11, 0x01])?; // enable raw-data-ready status

        block_for(Duration::from_millis(100));
        Ok(())
    }

    pub async fn collect_data(&self, address: u8) {
        let result = async {
            // putting this in the inner parenthesis causes problems for some reason
            // but want to make sure the lock goes out of scope as fast as possible
            let mut i2c = self.i2c_bus.lock().await;

            if self.register(&mut i2c, address, 0x1a).await? & 1 == 0 {
                return Ok(None);
            }
            let mut bytes = [0_u8; 12];
            i2c.write_read_async(address, &[0x2d], &mut bytes).await?;
            Ok::<_, Error>(Some(bytes))
        }
        .await;

        match result {
            Ok(Some(bytes)) => {
                //let read_us = micros() - start;
                let mut values = [0_f32; 6];
                for (axis, pair) in bytes.chunks_exact(2).enumerate() {
                    let raw = i16::from_be_bytes([pair[0], pair[1]]) as f32;
                    values[axis] = if axis < 3 {
                        raw * (9.80665 / 4096.0)
                    } else {
                        raw * (core::f32::consts::PI / (180.0 * 32.8))
                    };
                }
                Ok(Some(values))
                // println!(
                //    "{seq},{start},{read_us},imu0,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                //    values[0], values[1], values[2], values[3], values[4], values[5]
                //);
                //last_success = micros();
            }
            Ok(None) => Ok(None), // No new sample: do not present stale values as new data.
            Err(error) => {
                error!("# ERROR: I2C read failed: {error:?}");
                Ok(None)
            }
        }
    }
}

fn micros() -> u64 {
    Instant::now().as_micros()
}

/*
async fn main() -> ! {
    println!("# legday_imu6_v1; board=adafruit_feather_esp32c6; polling_target_hz=50");

    initialize(&mut i2c, address, &delay).expect("IMU initialization failed");
    println!("# imu0 address={address:#x}; accel_range_g=8; gyro_range_dps=1000; data=measured");
    println!("{HEADER}");
    let mut next_read = micros();
    let mut last_success = next_read;
    let mut sequence = 0_u64;
    loop {
        let now = micros();
        if now < next_read {
            delay.delay_millis(1);
            continue;
        }
        next_read = now + PERIOD_US;
        let seq = sequence;
        sequence += 1;
        let start = micros();

        if micros() - last_success > 1_000_000 {
            println!(
                "# ERROR: no successful fresh sample for over one second; reset/check hardware"
            );
            last_success = micros();
        }
    }
}
*/
