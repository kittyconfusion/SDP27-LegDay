//! Extended frame used by the IMU bring-up firmware: the stable 72-byte raw
//! [`SensorFrame`] followed by one DMP-style Quat6 orientation per IMU.
//!
//! The 120-byte encoding is deliberately a superset of the raw encoding so a
//! classifier that only wants raw data can slice bytes `0..72` out of every
//! 120-byte chunk without any conversion. See [`RAW_FRAME_RANGE`] and
//! [`QUAT6_RANGE`].

use crate::{FRAME_WIRE_BYTES, IMU_COUNT, SensorFrame, recording::FRAMES_PER_SD_BATCH};

/// Three signed Q30 vector components per IMU; `w` is reconstructed.
pub const QUAT6_BYTES_PER_IMU: usize = 12;
/// Stable encoded size of one fused frame: 72 raw bytes + 4 x 12 quaternion bytes.
pub const FUSED_FRAME_WIRE_BYTES: usize = FRAME_WIRE_BYTES + IMU_COUNT * QUAT6_BYTES_PER_IMU;
/// Byte range of the raw [`SensorFrame`] inside an encoded fused frame.
pub const RAW_FRAME_RANGE: core::ops::Range<usize> = 0..FRAME_WIRE_BYTES;
/// Byte range of the four Quat6 vectors inside an encoded fused frame.
pub const QUAT6_RANGE: core::ops::Range<usize> = FRAME_WIRE_BYTES..FUSED_FRAME_WIRE_BYTES;
/// Encoded size of one SD-card batch of fused frames.
pub const FUSED_SD_BATCH_BYTES: usize = FRAMES_PER_SD_BATCH * FUSED_FRAME_WIRE_BYTES;

/// Meaning of `SensorFrame::status_flags` bits as written by the bring-up
/// firmware. Bit 15 remains the synthetic-data marker used by the SD test.
pub mod status {
    pub const ORIENTATION_SOURCE_MASK: u16 = 0x0003;
    pub const ORIENTATION_NONE: u16 = 0x0000;
    /// Quaternion came from the ICM-20948 DMP (on the breakout board).
    pub const ORIENTATION_DMP: u16 = 0x0001;
    /// Quaternion came from the software filter running on the ESP32-S3.
    pub const ORIENTATION_SOFTWARE: u16 = 0x0002;
    /// The IMU FIFO overflowed or was resynchronised shortly before this frame.
    pub const FIFO_OVERFLOW: u16 = 0x0004;
    /// The timestamp was reconstructed from the FIFO drain time and the ODR.
    pub const TIMESTAMP_RECONSTRUCTED: u16 = 0x0008;
    /// DMP accelerometer calibration accuracy (0-3) occupies bits 4-5.
    pub const ACCEL_ACCURACY_SHIFT: u16 = 4;
    /// DMP gyroscope calibration accuracy (0-3) occupies bits 6-7.
    pub const GYRO_ACCURACY_SHIFT: u16 = 6;
    pub const ACCURACY_MASK: u16 = 0x0003;
    pub const SYNTHETIC: u16 = 0x8000;
}

/// DMP Quat6 ("game rotation vector") sample: `x`, `y`, `z` in signed Q30.
///
/// The scalar part is implied non-negative and can be rebuilt with
/// `legday_classifier::quaternion::QuaternionQ30::from_dmp_quat6`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Quat6Raw {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Quat6Raw {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };
}

/// One raw frame plus one orientation estimate per IMU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FusedFrame {
    pub raw: SensorFrame,
    pub quat6: [Quat6Raw; IMU_COUNT],
}

impl FusedFrame {
    pub const ZERO: Self = Self {
        raw: SensorFrame::ZERO,
        quat6: [Quat6Raw::ZERO; IMU_COUNT],
    };

    /// Encodes the raw frame in bytes `0..72` followed by the quaternions.
    pub fn encode_le(&self) -> [u8; FUSED_FRAME_WIRE_BYTES] {
        let mut bytes = [0_u8; FUSED_FRAME_WIRE_BYTES];
        bytes[RAW_FRAME_RANGE].copy_from_slice(&self.raw.encode_le());
        let mut offset = FRAME_WIRE_BYTES;
        for quat in &self.quat6 {
            for value in [quat.x, quat.y, quat.z] {
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                offset += 4;
            }
        }
        bytes
    }

    pub fn decode_le(bytes: &[u8; FUSED_FRAME_WIRE_BYTES]) -> Self {
        let raw_bytes: &[u8; FRAME_WIRE_BYTES] = bytes[RAW_FRAME_RANGE].try_into().unwrap();
        let mut frame = Self {
            raw: SensorFrame::decode_le(raw_bytes),
            quat6: [Quat6Raw::ZERO; IMU_COUNT],
        };
        let mut offset = FRAME_WIRE_BYTES;
        for quat in &mut frame.quat6 {
            let mut values = [0_i32; 3];
            for value in &mut values {
                *value = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
            *quat = Quat6Raw {
                x: values[0],
                y: values[1],
                z: values[2],
            };
        }
        frame
    }

    /// Borrows the raw 72-byte encoding inside an encoded fused frame.
    pub fn raw_bytes(encoded: &[u8; FUSED_FRAME_WIRE_BYTES]) -> &[u8; FRAME_WIRE_BYTES] {
        encoded[RAW_FRAME_RANGE].try_into().unwrap()
    }
}

/// A heap-free rolling window of fused frames, oldest first.
pub struct FusedWindowBuffer<const N: usize> {
    frames: [FusedFrame; N],
    next: usize,
    len: usize,
}

impl<const N: usize> FusedWindowBuffer<N> {
    pub const fn new() -> Self {
        assert!(N > 0, "window capacity must be nonzero");
        Self {
            frames: [FusedFrame::ZERO; N],
            next: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, frame: FusedFrame) {
        self.frames[self.next] = frame;
        self.next = (self.next + 1) % N;
        self.len = core::cmp::min(self.len + 1, N);
    }

    /// Forgets all history (for example after the acquisition mode changes).
    pub fn clear(&mut self) {
        self.next = 0;
        self.len = 0;
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn capacity(&self) -> usize {
        N
    }

    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// Returns a frame in chronological order, with index zero being oldest.
    pub fn get(&self, chronological_index: usize) -> Option<&FusedFrame> {
        if chronological_index >= self.len {
            return None;
        }
        let oldest = if self.len == N { self.next } else { 0 };
        Some(&self.frames[(oldest + chronological_index) % N])
    }

    pub fn latest(&self) -> Option<&FusedFrame> {
        self.len.checked_sub(1).and_then(|index| self.get(index))
    }

    /// Chronological iterator over the raw 72-byte frames only, which is what
    /// the raw-data classifier consumes.
    pub fn raw_frames(&self) -> impl Iterator<Item = &SensorFrame> + '_ {
        (0..self.len).filter_map(move |index| self.get(index).map(|frame| &frame.raw))
    }

    /// Copies the raw halves into a plain [`crate::WindowBuffer`] of the same
    /// capacity, for code that was written against the 72-byte contract.
    pub fn copy_raw_into(&self, window: &mut crate::WindowBuffer<N>) {
        for frame in self.raw_frames() {
            window.push(*frame);
        }
    }
}

impl<const N: usize> Default for FusedWindowBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-size staging buffer of encoded fused frames for SD-card writes.
pub struct FusedFrameBatch {
    bytes: [u8; FUSED_SD_BATCH_BYTES],
    frames: usize,
}

impl FusedFrameBatch {
    pub const fn new() -> Self {
        Self {
            bytes: [0; FUSED_SD_BATCH_BYTES],
            frames: 0,
        }
    }

    /// Appends one frame; returns `true` once the batch is full.
    pub fn push(&mut self, frame: &FusedFrame) -> bool {
        assert!(!self.is_full(), "clear a full batch before pushing");
        let start = self.frames * FUSED_FRAME_WIRE_BYTES;
        self.bytes[start..start + FUSED_FRAME_WIRE_BYTES].copy_from_slice(&frame.encode_le());
        self.frames += 1;
        self.is_full()
    }

    pub const fn frames(&self) -> usize {
        self.frames
    }

    pub const fn is_empty(&self) -> bool {
        self.frames == 0
    }

    pub const fn is_full(&self) -> bool {
        self.frames == FRAMES_PER_SD_BATCH
    }

    pub fn encoded(&self) -> &[u8] {
        &self.bytes[..self.frames * FUSED_FRAME_WIRE_BYTES]
    }

    pub fn clear(&mut self) {
        self.frames = 0;
    }
}

impl Default for FusedFrameBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, not(target_os = "none")))]
mod tests {
    use super::*;
    use crate::Imu6Raw;
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    fn frame(sequence: u32) -> FusedFrame {
        let mut frame = FusedFrame::ZERO;
        frame.raw.sequence = sequence;
        frame.raw.timestamp_us = u64::from(sequence) * 4_444;
        frame.raw.valid_imu_mask = 0b0001;
        frame.raw.status_flags = status::ORIENTATION_DMP | status::TIMESTAMP_RECONSTRUCTED;
        frame.raw.imus[0] = Imu6Raw {
            accel: [1, -2, 8_192],
            gyro: [-3, 4, 5],
        };
        frame.quat6[0] = Quat6Raw {
            x: 0x1000_0000,
            y: -0x0800_0000,
            z: 7,
        };
        frame
    }

    #[test]
    fn fused_encoding_is_120_bytes_and_round_trips() {
        let original = frame(9);
        let encoded = original.encode_le();
        assert_eq!(encoded.len(), 120);
        assert_eq!(FusedFrame::decode_le(&encoded), original);
    }

    #[test]
    fn raw_prefix_matches_plain_sensor_frame_encoding() {
        let original = frame(3);
        let encoded = original.encode_le();
        assert_eq!(FusedFrame::raw_bytes(&encoded), &original.raw.encode_le());
        assert_eq!(
            SensorFrame::decode_le(FusedFrame::raw_bytes(&encoded)),
            original.raw
        );
        assert_eq!(&encoded[QUAT6_RANGE][..4], &0x1000_0000_i32.to_le_bytes());
    }

    #[test]
    fn window_exposes_raw_frames_in_order() {
        let mut window = FusedWindowBuffer::<3>::new();
        for sequence in 0..5 {
            window.push(frame(sequence));
        }
        let sequences: Vec<u32> = window.raw_frames().map(|raw| raw.sequence).collect();
        assert_eq!(sequences, vec![2, 3, 4]);

        let mut plain = crate::WindowBuffer::<3>::new();
        window.copy_raw_into(&mut plain);
        assert_eq!(plain.latest().unwrap().sequence, 4);
    }

    #[test]
    fn batch_concatenates_fused_frames() {
        let mut batch = FusedFrameBatch::new();
        assert!(!batch.push(&frame(1)));
        assert!(!batch.push(&frame(2)));
        assert_eq!(batch.encoded().len(), 2 * FUSED_FRAME_WIRE_BYTES);
        assert_eq!(
            &batch.encoded()[FUSED_FRAME_WIRE_BYTES..],
            &frame(2).encode_le()
        );
    }
}
