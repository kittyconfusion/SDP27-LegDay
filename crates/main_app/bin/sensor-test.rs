//! Direct-connected ICM-20948 bench reader for the Adafruit ESP32-C6 Feather.
//! Six-axis bring-up only: no magnetometer, force sensors, mux or radio yet.
#![no_std]
#![no_main]

#[cfg(feature = "esp32s3")]
compile_error!("sensor-test requires --no-default-features --features esp32c6");

use esp_backtrace as _;
use esp_hal::{
    Blocking,
    delay::Delay,
    gpio::{Level, Output, OutputConfig},
    i2c::master::{Config, Error, I2c},
    time::{Instant, Rate},
};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

const PERIOD_US: u64 = 20_000;
const HEADER: &str = "seq,t_us,read_us,sensor_id,ax_m_s2,ay_m_s2,az_m_s2,gx_rad_s,gy_rad_s,gz_rad_s";

fn micros() -> u64 {
    Instant::now().duration_since_epoch().as_micros()
}

fn register(i2c: &mut I2c<'_, Blocking>, address: u8, reg: u8) -> Result<u8, Error> {
    let mut byte = [0];
    i2c.write_read(address, &[reg], &mut byte)?;
    Ok(byte[0])
}

fn initialize(i2c: &mut I2c<'_, Blocking>, address: u8, delay: &Delay) -> Result<(), Error> {
    // Register banks and values: TDK DS-000189 rev 1.5, sections 8 and 10.
    i2c.write(address, &[0x7f, 0x00])?; // bank 0
    i2c.write(address, &[0x06, 0x80])?; // device reset
    delay.delay_millis(100);
    i2c.write(address, &[0x7f, 0x00])?;
    i2c.write(address, &[0x06, 0x01])?; // wake, automatic clock selection
    i2c.write(address, &[0x07, 0x00])?; // enable accel and gyro axes
    delay.delay_millis(100);
    i2c.write(address, &[0x7f, 0x20])?; // bank 2
    i2c.write(address, &[0x00, 19])?; // gyro divider
    i2c.write(address, &[0x01, 0x25])?; // +/-1000 dps, DLPF config 4, enabled
    i2c.write(address, &[0x10, 0x00, 19])?; // accel 12-bit divider
    i2c.write(address, &[0x14, 0x25])?; // +/-8 g, DLPF config 4, enabled
    // Read back the settings that determine scaling; reject mismatches.
    for (reg, value) in [(0x00, 19), (0x01, 0x25), (0x10, 0), (0x11, 19), (0x14, 0x25)] {
        if register(i2c, address, reg)? != value {
            panic!("IMU configuration readback mismatch at register {reg:#x}");
        }
    }
    i2c.write(address, &[0x7f, 0x00])?;
    i2c.write(address, &[0x11, 0x01])?; // enable raw-data-ready status
    delay.delay_millis(100);
    Ok(())
}

#[esp_hal::main]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default());
    let delay = Delay::new();
    // Keep this output alive: it supplies the Feather's STEMMA QT power.
    let _sensor_power = Output::new(peripherals.GPIO20, Level::High, OutputConfig::default());
    delay.delay_millis(1000);
    let mut i2c = I2c::new(
        peripherals.I2C0,
        Config::default().with_frequency(Rate::from_khz(100)),
    )
    .expect("I2C configuration")
    .with_sda(peripherals.GPIO19)
    .with_scl(peripherals.GPIO18);

    println!("# legday_imu6_v1; board=adafruit_feather_esp32c6; polling_target_hz=50");
    let mut detected = None;
    for address in [0x69_u8, 0x68_u8] {
        if i2c.write(address, &[0x7f, 0]).is_ok()
            && matches!(register(&mut i2c, address, 0x00), Ok(0xea))
        {
            detected = Some(address);
            break;
        }
    }
    let Some(address) = detected else {
        loop {
            println!("# ERROR: ICM-20948 not detected at 0x69 or 0x68; check wiring/power");
            delay.delay_millis(1000);
        }
    };
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
        let result = (|| {
            if register(&mut i2c, address, 0x1a)? & 1 == 0 {
                return Ok(None);
            }
            let mut bytes = [0_u8; 12];
            i2c.write_read(address, &[0x2d], &mut bytes)?;
            Ok::<_, Error>(Some(bytes))
        })();
        match result {
            Ok(Some(bytes)) => {
                let read_us = micros() - start;
                let mut values = [0_f32; 6];
                for (axis, pair) in bytes.chunks_exact(2).enumerate() {
                    let raw = i16::from_be_bytes([pair[0], pair[1]]) as f32;
                    values[axis] = if axis < 3 {
                        raw * (9.80665 / 4096.0)
                    } else {
                        raw * (core::f32::consts::PI / (180.0 * 32.8))
                    };
                }
                println!("{seq},{start},{read_us},imu0,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                    values[0], values[1], values[2], values[3], values[4], values[5]);
                last_success = micros();
            }
            Ok(None) => {} // No new sample: do not present stale values as new data.
            Err(error) => println!("# ERROR: seq={seq} I2C read failed: {error:?}"),
        }
        if micros() - last_success > 1_000_000 {
            println!("# ERROR: no successful fresh sample for over one second; reset/check hardware");
            last_success = micros();
        }
    }
}
