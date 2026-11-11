//! In-memory readings only: no serialization or recording format is defined.
//! Times describe MCU read operations, not synchronized physical sample times.

use super::config::{IMU_COUNT, PRESSURE_COUNT};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timed<T> {
    pub read_started_us: u64,
    pub read_completed_us: u64,
    pub value: T,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawImu {
    pub accel: [i16; 3],
    pub gyro: [i16; 3],
}

impl RawImu {
    /// ICM-20948 burst order: AX, AY, AZ, GX, GY, GZ, each signed big-endian.
    pub fn from_register_bytes(bytes: [u8; 12]) -> Self {
        let word = |at: usize| i16::from_be_bytes([bytes[at], bytes[at + 1]]);
        Self {
            accel: [word(0), word(2), word(4)],
            gyro: [word(6), word(8), word(10)],
        }
    }
}

#[derive(Debug)]
pub enum ImuError<E> {
    Bus(E),
    UnexpectedIdentity {
        actual: u8,
    },
    ConfigurationMismatch {
        bank: u8,
        register: u8,
        expected: u8,
        actual: u8,
    },
}

#[derive(Debug)]
pub enum ImuReading<E> {
    Disabled,
    NotInitialized,
    /// No new data-ready indication; no old measurement is returned.
    NotReady,
    Failed(ImuError<E>),
    Fresh(Timed<RawImu>),
}

#[derive(Debug)]
pub enum ImuInitialization<E> {
    Disabled,
    Ready,
    Failed(ImuError<E>),
}

#[derive(Debug)]
pub struct SensorSample<E> {
    pub read_started_us: u64,
    pub read_completed_us: u64,
    /// Non-prosthetic upper/lower, then prosthetic upper/lower.
    pub imus: [ImuReading<E>; IMU_COUNT],
    /// Raw ADC counts, A0..A3. A completed conversion does not establish that
    /// a pressure sensor is connected, calibrated, or within its force range.
    pub pressure: [Timed<u16>; PRESSURE_COUNT],
}

impl<E> SensorSample<E> {
    pub fn fresh_imu_mask(&self) -> u8 {
        self.imus.iter().enumerate().fold(0, |mask, (index, reading)| {
            if matches!(reading, ImuReading::Fresh(_)) {
                mask | (1 << index)
            } else {
                mask
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_signed_axes_in_sensor_order() {
        let sample = RawImu::from_register_bytes([
            0x80, 0x00, 0xff, 0xff, 0x00, 0x01,
            0x7f, 0xff, 0x00, 0x00, 0xfe, 0xdc,
        ]);
        assert_eq!(sample.accel, [-32768, -1, 1]);
        assert_eq!(sample.gyro, [32767, 0, -292]);
    }

    #[test]
    fn invalid_readings_do_not_become_zero_measurements() {
        let sample: SensorSample<()> = SensorSample {
            read_started_us: 10,
            read_completed_us: 20,
            imus: [
                ImuReading::Fresh(Timed {
                    read_started_us: 11,
                    read_completed_us: 12,
                    value: RawImu { accel: [0; 3], gyro: [0; 3] },
                }),
                ImuReading::Disabled,
                ImuReading::NotReady,
                ImuReading::Failed(ImuError::Bus(())),
            ],
            pressure: [Timed {
                read_started_us: 19,
                read_completed_us: 20,
                value: 0,
            }; PRESSURE_COUNT],
        };
        assert_eq!(sample.fresh_imu_mask(), 0b0001);
    }
}
