//! Tim's 120-byte payload, encoded explicitly so Rust struct padding cannot leak.
use super::{
    config::{IMU_ACCEL_CONFIG, IMU_GYRO_CONFIG},
    sample::{ImuReading, SensorSample},
};

pub const FRAME_BYTES: usize = 120;
pub type SensorFrame = [u8; FRAME_BYTES];

/// Range codes use the sensor's FS_SEL encoding (bits 2:1). Mark them valid
/// instead of claiming Tim's example +/-16g, +/-2000dps for Nathan's settings.
pub const RANGE_FLAGS: u16 = 1 << 12
    | (((IMU_ACCEL_CONFIG >> 1) & 3) as u16) << 8
    | (((IMU_GYRO_CONFIG >> 1) & 3) as u16) << 10;

pub fn encode<E>(sample: &SensorSample<E>, sequence: u32) -> SensorFrame {
    let mut bytes = [0; FRAME_BYTES];
    // TODO(Nathan, Tim): this is MCU acquisition start, not a reconstructed
    // common sensor timestamp. Leave SampleClock/realignment flags clear.
    bytes[0..8].copy_from_slice(&sample.read_started_us.to_le_bytes());
    bytes[8..12].copy_from_slice(&sequence.to_le_bytes());
    bytes[12] = sample.fresh_imu_mask();
    // All four ADC channels are configured/read. ADC cannot detect unplugged
    // pressure sensors. TODO(Nathan, Tim): agree on disconnect/calibration rules;
    // 4095 means ADC saturation, not a calibrated force measurement.
    bytes[13] = 0b1111;
    bytes[14..16].copy_from_slice(&RANGE_FLAGS.to_le_bytes());
    for (slot, reading) in sample.imus.iter().enumerate() {
        if let ImuReading::Fresh(reading) = reading {
            for (axis, value) in reading
                .value
                .accel
                .iter()
                .chain(reading.value.gyro.iter())
                .enumerate()
            {
                let offset = 16 + slot * 12 + axis * 2;
                bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    for (slot, reading) in sample.pressure.iter().enumerate() {
        let offset = 64 + slot * 2;
        bytes[offset..offset + 2].copy_from_slice(&reading.value.to_le_bytes());
    }
    // TODO(Tim, Nathan): Tim supplies quaternion generation and the orientation
    // source bits later. Bytes 72..120 remain zero with source=0 (UNAVAILABLE).
    // Consumers must not reconstruct these placeholders as identity rotations.
    bytes
}

#[cfg(test)]
mod tests {
    use super::super::sample::{RawImu, Timed};
    use super::*;
    fn sample() -> SensorSample<()> {
        SensorSample {
            read_started_us: 0x0807060504030201,
            read_completed_us: 0x0807060504030300,
            imus: core::array::from_fn(|slot| {
                ImuReading::Fresh(Timed {
                    read_started_us: 0,
                    read_completed_us: 1,
                    value: RawImu {
                        accel: [slot as i16, -2, 0x1234],
                        gyro: [i16::MIN, i16::MAX, -1],
                    },
                })
            }),
            pressure: core::array::from_fn(|slot| Timed {
                read_started_us: 0,
                read_completed_us: 1,
                value: [0, 1, 0x1234, 4095][slot],
            }),
        }
    }
    #[test]
    fn exact_offsets_and_little_endian() {
        let bytes = encode(&sample(), 0x44332211);
        assert_eq!(bytes.len(), 120);
        assert_eq!(
            &bytes[..12],
            &[1, 2, 3, 4, 5, 6, 7, 8, 0x11, 0x22, 0x33, 0x44]
        );
        assert_eq!(bytes[12], 15);
        assert_eq!(bytes[13], 15);
        assert_eq!(&bytes[14..16], &[0, 0x1a]);
        for slot in 0..4 {
            assert_eq!(
                &bytes[16 + slot * 12..28 + slot * 12],
                &[
                    slot as u8, 0, 0xfe, 0xff, 0x34, 0x12, 0, 0x80, 0xff, 0x7f, 0xff, 0xff
                ]
            );
        }
        assert_eq!(&bytes[64..72], &[0, 0, 1, 0, 0x34, 0x12, 0xff, 0x0f]);
        assert_eq!(&bytes[72..], &[0; 48]);
        assert_eq!(u16::from_le_bytes([bytes[14], bytes[15]]) & 0x800f, 0);
    }
    #[test]
    fn demo_keeps_missing_slot_and_frame_size() {
        let mut sample = sample();
        sample.imus[1] = ImuReading::Disabled;
        let bytes = encode(&sample, 0);
        assert_eq!(bytes[12], 0b1101);
        assert_eq!(&bytes[28..40], &[0; 12]);
        assert_eq!(&bytes[40..42], &[2, 0]);
    }
    #[test]
    fn stale_or_failed_data_is_never_marked_fresh() {
        let mut sample = sample();
        sample.imus[0] = ImuReading::NotReady;
        sample.imus[1] = ImuReading::NotInitialized;
        sample.imus[2] = ImuReading::Failed(super::super::sample::ImuError::Bus(()));
        let bytes = encode(&sample, u32::MAX);
        assert_eq!(bytes[12], 8);
        assert_eq!(&bytes[16..52], &[0; 36]);
        assert_eq!(&bytes[8..12], &[255; 4]);
    }
}
