//! Adapter from the shared raw frame to software Quat6 fields.
//!
//! Updates only slots marked fresh. The current polling recorder supplies MCU
//! read-start times, so elapsed time here is an acquisition-time estimate, not
//! a hardware sample timestamp. This does not synchronize sensors or fill gaps.
//!
//! Where it sits in the recorder (only when `record_app` is built with its
//! `sensor-fusion` feature):
//!
//! ```text
//!   sensors -> 120-byte frame, quaternion bytes all zero
//!           -> FrameFusion::update   (fills the quaternions, sets the flags)
//!           -> publish / SD card
//! ```
//!
//! Vocabulary: a *slot* is one of the four fixed IMU positions in a frame.
//! A slot is *fresh* when its bit in `valid_imu_mask` is set, meaning the
//! frame carries a new reading from that IMU.

use legday_core::{
    IMU_COUNT,
    fused::{FusedFrame, Quat6Raw, status},
};

use crate::{GyroBias, ImuScale, Madgwick, accel_trust};

/// Accelerometer counts per g for each range code in the frame's status
/// flags: code 0, 1, 2, 3 = ±2, ±4, ±8, ±16 g. (The same table is in
/// `legday-classifier`'s `preprocess.rs`.)

const ACCEL_SCALES: [f32; 4] = [16_384.0, 8_192.0, 4_096.0, 2_048.0];

/// Gyroscope counts per deg/s for each range code: code 0, 1, 2, 3 =
/// ±250, ±500, ±1000, ±2000 deg/s.

const GYRO_SCALES: [f32; 4] = [131.0, 65.5, 32.8, 16.4];

/// Longest time between two fresh samples of one IMU (250 ms) that is still
/// integrated. After a longer gap the filter starts over from gravity.

const MAX_GAP_US: u64 = 250_000;

/// One filter and gyro-bias estimator per fixed sensor slot; no heap or I/O.
/// Bias-learning thresholds are inherited from the 225-Hz prototype and count
/// fresh samples. Verify actual fresh-sample rates before assuming those timings.
pub struct FrameFusion {
    /// Orientation filter for each slot.
    filters: [Madgwick; IMU_COUNT],

    /// Gyroscope zero-offset estimator for each slot.
    biases: [GyroBias; IMU_COUNT],

    /// Timestamp of each slot's last integrated sample. `None` means the
    /// next fresh sample starts over from gravity.
    last_us: [Option<u64>; IMU_COUNT],

    /// The range bits (8-11 of the status flags) of the previous frame, kept
    /// to notice when the sensor ranges change.
    range_flags: Option<u16>,
}

impl FrameFusion {
    /// `beta` is the Madgwick filter gain (see [`Madgwick`]); the recorder
    /// passes 0.1

    pub const fn new(beta: f32) -> Self {
        Self {
            filters: [Madgwick::new(beta); IMU_COUNT],

            // Placeholder scale (±2000 deg/s). It is replaced on the first
            // frame, once the real range has been read from the flags
            biases: [GyroBias::new(16.4); IMU_COUNT],
            last_us: [None; IMU_COUNT],
            range_flags: None,
        }
    }

    /// Preserves raw readings and timestamps, and writes orientations only for
    /// fresh IMUs. Requires recorded range settings instead of guessing units.
    /// First samples and samples after long gaps seed tilt from gravity; yaw
    /// starts at zero. Unavailable slots keep zero quaternion bytes

    pub fn update(&mut self, frame: &mut FusedFrame) {
        // Start from "no orientation": zero quaternions, source bits cleared

        frame.quat6 = [Quat6Raw::ZERO; IMU_COUNT];
        frame.raw.status_flags &= !status::ORIENTATION_SOURCE_MASK;
        let flags = frame.raw.status_flags;

        // Without recorded ranges, counts cannot be converted to units
        if flags & status::RANGES_RECORDED == 0 {
            return;
        }

        // Two-bit range codes -> scale factors
        let accel_code = usize::from((flags >> status::ACCEL_RANGE_SHIFT) & status::RANGE_MASK);
        let gyro_code = usize::from((flags >> status::GYRO_RANGE_SHIFT) & status::RANGE_MASK);
        let scale = ImuScale {
            accel_lsb_per_g: ACCEL_SCALES[accel_code],
            gyro_lsb_per_dps: GYRO_SCALES[gyro_code],
        };

        // A change of range (including the very first frame) invalidates the
        // timing history and the learned bias, which is held in counts, so
        // both start over
        let ranges = flags & 0x0f00;

        if self.range_flags != Some(ranges) {
            self.range_flags = Some(ranges);
            self.last_us = [None; IMU_COUNT];
            self.biases = [GyroBias::new(scale.gyro_lsb_per_dps); IMU_COUNT];
        }

        // `fresh`: one bit per slot holding a new reading
        // `oriented`: one bit per slot given a quaternion below
        // `level`: lowest gyro calibration level (0-3) among those slots
        let fresh = frame.raw.valid_imu_mask & 0x0f;
        let mut oriented = 0u8;
        let mut level = 3u16;

        for slot in 0..IMU_COUNT {
            if fresh & (1 << slot) == 0 {
                continue;
            }
            let sample = frame.raw.imus[slot];
            let accel = scale.accel_g(sample.accel);
            let now = frame.raw.timestamp_us;

            // Time since this slot's last integrated sample. `None` on its
            // first sample, or if the timestamp went backwards

            let elapsed = self.last_us[slot].and_then(|previous| now.checked_sub(previous));

            if elapsed == Some(0) {
                continue; // Never integrate the same timestamp twice
            }

            if let Some(dt_us) = elapsed.filter(|dt| *dt <= MAX_GAP_US) {
                // Normal case: remove the gyro bias, convert counts to rad/s
                // and advance the filter by the elapsed time (in seconds)

                let counts = self.biases[slot].correct(sample.gyro, accel);
                let k = core::f32::consts::PI / (180.0 * scale.gyro_lsb_per_dps);
                let gyro = counts.map(|value| value * k);

                self.filters[slot].update_with_gain(
                    gyro,
                    accel,
                    dt_us as f32 * 1e-6,
                    accel_trust(accel),
                );
            } else {
                // First sample, or too long a gap: start over from the tilt
                // that gravity shows in this accelerometer reading

                if sample.accel == [0; 3] {
                    self.last_us[slot] = None;
                    continue; // No gravity direction from which to seed tilt
                }

                self.filters[slot].reset_to_gravity(accel);
            }

            // Store the result as Q30 `x, y, z` and count the slot as oriented

            self.last_us[slot] = Some(now);
            let [x, y, z] = self.filters[slot].quat6_q30();

            frame.quat6[slot] = Quat6Raw { x, y, z };
            oriented |= 1 << slot;
            level = level.min(self.biases[slot].level());
        }

        // The existing format has one orientation-source flag for the frame.
        // Advertise software orientation only if every fresh IMU has one.

        if fresh != 0 && oriented == fresh {
            // Overwrite the gyro accuracy bits with the lowest bias level.
            frame.raw.status_flags &= !(status::ACCURACY_MASK << status::GYRO_ACCURACY_SHIFT);
            frame.raw.status_flags |=
                status::ORIENTATION_SOFTWARE | (level << status::GYRO_ACCURACY_SHIFT);
        }
    }
}

// Unit tests, built only by `cargo test`
#[cfg(test)]

mod tests {
    use super::*;

    /// A frame with only IMU 0 fresh, ranges recorded as ±8 g and
    /// ±1000 deg/s (code 2 for both), and 1 g (4096 counts) along +Y
    fn raw() -> FusedFrame {
        let mut frame = FusedFrame::ZERO;
        frame.raw.timestamp_us = 1_000;
        frame.raw.valid_imu_mask = 1;
        frame.raw.status_flags = status::RANGES_RECORDED | (2 << 8) | (2 << 10);
        frame.raw.imus[0].accel = [0, 4096, 0];
        frame
    }

    #[test]
    fn fills_orientation_without_changing_measurements_or_missing_slots() {
        let mut frame = raw();
        let before = frame.raw;

        FrameFusion::new(0.1).update(&mut frame);

        assert_eq!(frame.raw.imus, before.imus);
        assert_eq!(frame.raw.pressure_adc, before.pressure_adc);
        assert_eq!(frame.raw.timestamp_us, before.timestamp_us);
        assert_eq!(frame.raw.valid_imu_mask, before.valid_imu_mask);
        assert_eq!(
            frame.raw.status_flags & status::ORIENTATION_SOURCE_MASK,
            status::ORIENTATION_SOFTWARE
        );

        assert_ne!(frame.quat6[0], Quat6Raw::ZERO);
        assert_eq!(&frame.quat6[1..], &[Quat6Raw::ZERO; 3]);
    }

    #[test]
    fn unavailable_or_duplicate_measurements_do_not_advance_filter() {
        let mut fusion = FrameFusion::new(0.1);
        let mut frame = raw();

        fusion.update(&mut frame);

        let q = fusion.filters[0].quaternion();

        // A later frame with no fresh IMU leaves the filter untouched.
        frame.raw.valid_imu_mask = 0;
        frame.raw.timestamp_us += 5_000;
        fusion.update(&mut frame);

        assert_eq!(fusion.filters[0].quaternion(), q);
        assert_eq!(frame.raw.status_flags & status::ORIENTATION_SOURCE_MASK, 0);

        // So does a fresh frame that repeats the first frame's timestamp.
        frame = raw();
        fusion.update(&mut frame);

        assert_eq!(fusion.filters[0].quaternion(), q);
        assert_eq!(frame.raw.status_flags & status::ORIENTATION_SOURCE_MASK, 0);
    }

    #[test]
    fn does_not_guess_missing_range_metadata() {
        let mut frame = raw();

        frame.raw.status_flags = 0;

        FrameFusion::new(0.1).update(&mut frame);

        assert_eq!(frame.raw.status_flags & status::ORIENTATION_SOURCE_MASK, 0);
        assert_eq!(frame.quat6, [Quat6Raw::ZERO; IMU_COUNT]);
    }

    #[test]
    fn integrates_elapsed_time_between_fresh_samples_not_poll_intervals() {
        let mut fusion = FrameFusion::new(0.1);
        let mut frame = raw();

        frame.raw.imus[0].accel = [0, 0, 4096];
        fusion.update(&mut frame);

        // A poll 5 ms later with nothing fresh, then a fresh sample 15 ms
        // after that: the filter must integrate the full 20 ms.

        frame.raw.timestamp_us += 5_000;
        frame.raw.valid_imu_mask = 0;
        fusion.update(&mut frame);
        frame.raw.timestamp_us += 15_000;
        frame.raw.valid_imu_mask = 1;
        frame.raw.imus[0].gyro = [3280, 0, 0]; // 100 deg/s at +/-1000 deg/s.
        fusion.update(&mut frame);

        let q = fusion.filters[0].quaternion();

        // 20 ms at 100 deg/s = two degrees; quaternion x = sin(one degree).
        assert!((q[1] - 0.0174524).abs() < 1e-5);
    }
}
