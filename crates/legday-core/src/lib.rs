#![no_std]

//! Hardware-independent sensor data types used by acquisition, recording, and
//! classification code. This crate deliberately has no ESP or operating-system
//! dependencies, so the same representation can be tested on a computer and
//! compiled into ESP32-S3 or ESP32-C6 firmware.

pub mod clock;
pub mod fused;
pub mod recording;

pub const IMU_COUNT: usize = 4;
pub const IMU_CHANNELS: usize = 6;
pub const PRESSURE_COUNT: usize = 4;
pub const FEATURE_COUNT: usize = IMU_COUNT * IMU_CHANNELS + PRESSURE_COUNT;

/// Proposed initial sampling configuration.
///
/// A one-second history does not imply one-second inference latency. A model can
/// consume this rolling history every five samples, producing a new result every
/// 25 ms while retaining the context of a full gait cycle.
pub const SAMPLE_RATE_HZ: usize = 200;
pub const WINDOW_SAMPLES: usize = SAMPLE_RATE_HZ;
pub const INFERENCE_HOP_SAMPLES: usize = 5;

/// Stable encoded size of one frame written to a recording.
pub const FRAME_WIRE_BYTES: usize = 72;

/// One accelerometer and gyroscope sample from an IMU.
///
/// Values remain as signed sensor counts here. Calibration, unit conversion,
/// normalization, and quantization belong in a versioned preprocessing stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Imu6Raw {
    pub accel: [i16; 3],
    pub gyro: [i16; 3],
}

/// A synchronized snapshot of all classifier inputs.
///
/// Feature order is four IMUs in physical-layout order, each containing
/// `ax, ay, az, gx, gy, gz`, followed by four pressure ADC counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SensorFrame {
    pub timestamp_us: u64,
    pub sequence: u32,
    pub valid_imu_mask: u8,
    pub valid_pressure_mask: u8,
    pub status_flags: u16,
    pub imus: [Imu6Raw; IMU_COUNT],
    pub pressure_adc: [u16; PRESSURE_COUNT],
}

impl SensorFrame {
    pub const ZERO: Self = Self {
        timestamp_us: 0,
        sequence: 0,
        valid_imu_mask: 0,
        valid_pressure_mask: 0,
        status_flags: 0,
        imus: [Imu6Raw {
            accel: [0; 3],
            gyro: [0; 3],
        }; IMU_COUNT],
        pressure_adc: [0; PRESSURE_COUNT],
    };

    /// Copies the frame into the integer feature order consumed by preprocessing.
    pub fn write_features_i32(&self, output: &mut [i32; FEATURE_COUNT]) {
        let mut output_index = 0;
        for imu in &self.imus {
            for value in imu.accel.iter().chain(imu.gyro.iter()) {
                output[output_index] = i32::from(*value);
                output_index += 1;
            }
        }
        for value in self.pressure_adc {
            output[output_index] = i32::from(value);
            output_index += 1;
        }
    }

    /// Encodes one frame without struct padding or host-endianness assumptions.
    pub fn encode_le(&self) -> [u8; FRAME_WIRE_BYTES] {
        let mut bytes = [0_u8; FRAME_WIRE_BYTES];
        bytes[0..8].copy_from_slice(&self.timestamp_us.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.sequence.to_le_bytes());
        bytes[12] = self.valid_imu_mask;
        bytes[13] = self.valid_pressure_mask;
        bytes[14..16].copy_from_slice(&self.status_flags.to_le_bytes());

        let mut offset = 16;
        for imu in &self.imus {
            for value in imu.accel.iter().chain(imu.gyro.iter()) {
                bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
                offset += 2;
            }
        }
        for value in self.pressure_adc {
            bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            offset += 2;
        }
        bytes
    }

    /// Decodes one frame written by [`SensorFrame::encode_le`].
    pub fn decode_le(bytes: &[u8; FRAME_WIRE_BYTES]) -> Self {
        let mut frame = Self::ZERO;
        frame.timestamp_us = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        frame.sequence = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        frame.valid_imu_mask = bytes[12];
        frame.valid_pressure_mask = bytes[13];
        frame.status_flags = u16::from_le_bytes(bytes[14..16].try_into().unwrap());

        let mut offset = 16;
        for imu in &mut frame.imus {
            for value in imu.accel.iter_mut().chain(imu.gyro.iter_mut()) {
                *value = i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                offset += 2;
            }
        }
        for value in &mut frame.pressure_adc {
            *value = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            offset += 2;
        }
        frame
    }
}

/// A heap-free rolling window retaining the most recent `N` frames.
pub struct WindowBuffer<const N: usize> {
    frames: [SensorFrame; N],
    next: usize,
    len: usize,
}

impl<const N: usize> WindowBuffer<N> {
    pub const fn new() -> Self {
        assert!(N > 0, "window capacity must be nonzero");
        Self {
            frames: [SensorFrame::ZERO; N],
            next: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, frame: SensorFrame) {
        self.frames[self.next] = frame;
        self.next = (self.next + 1) % N;
        self.len = core::cmp::min(self.len + 1, N);
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn capacity(&self) -> usize {
        N
    }

    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// Returns a frame in chronological order, with index zero being oldest.
    pub fn get(&self, chronological_index: usize) -> Option<&SensorFrame> {
        if chronological_index >= self.len {
            return None;
        }
        let oldest = if self.len == N { self.next } else { 0 };
        Some(&self.frames[(oldest + chronological_index) % N])
    }

    pub fn latest(&self) -> Option<&SensorFrame> {
        self.len.checked_sub(1).and_then(|index| self.get(index))
    }
}

impl<const N: usize> Default for WindowBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, not(target_os = "none")))]
mod tests {
    use super::*;

    fn frame(sequence: u32) -> SensorFrame {
        SensorFrame {
            timestamp_us: u64::from(sequence) * 5_000,
            sequence,
            valid_imu_mask: 0b1111,
            valid_pressure_mask: 0b1111,
            status_flags: 0x1234,
            imus: [Imu6Raw {
                accel: [sequence as i16, -2, 3],
                gyro: [4, -5, 6],
            }; IMU_COUNT],
            pressure_adc: [10, 20, 30, 40],
        }
    }

    #[test]
    fn wire_encoding_round_trips() {
        let original = frame(42);
        let encoded = original.encode_le();
        assert_eq!(encoded.len(), FRAME_WIRE_BYTES);
        assert_eq!(SensorFrame::decode_le(&encoded), original);
    }

    #[test]
    fn feature_order_has_24_imu_then_4_pressure_values() {
        let original = frame(7);
        let mut features = [0_i32; FEATURE_COUNT];
        original.write_features_i32(&mut features);
        assert_eq!(&features[0..6], &[7, -2, 3, 4, -5, 6]);
        assert_eq!(&features[24..28], &[10, 20, 30, 40]);
    }

    #[test]
    fn full_window_retains_latest_frames_in_order() {
        let mut window = WindowBuffer::<3>::new();
        for sequence in 0..5 {
            window.push(frame(sequence));
        }
        assert!(window.is_full());
        assert_eq!(window.get(0).unwrap().sequence, 2);
        assert_eq!(window.get(1).unwrap().sequence, 3);
        assert_eq!(window.latest().unwrap().sequence, 4);
    }
}
