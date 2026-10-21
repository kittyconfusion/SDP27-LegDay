//! Software orientation fallback: a Madgwick IMU filter (accelerometer +
//! gyroscope, no magnetometer) producing the same kind of "game rotation
//! vector" the ICM-20948 DMP calls Quat6.
//!
//! This exists so the pipeline can run even when the DMP is unavailable, and
//! so DMP output can be cross-checked on the ESP32-S3. It uses only `f32`
//! multiply/add plus a software inverse square root, so it needs no `libm`.
//! At 225 Hz one update costs a few microseconds on a 240 MHz LX7 core.

use core::f32::consts::PI;

/// `1 / sqrt(x)` without `std` or `libm` (bit-trick seed + Newton polish).
/// Relative error is below 1e-6 for normal, positive inputs.
pub fn inv_sqrt(x: f32) -> f32 {
    if x <= 0.0 || !x.is_finite() {
        return 0.0;
    }

    let half = 0.5 * x;
    let mut y = f32::from_bits(0x5f37_59df - (x.to_bits() >> 1));

    for _ in 0..3 {
        y *= 1.5 - half * y * y;
    }

    y
}

/// `sqrt(x)` for `x >= 0`, built on [`inv_sqrt`].
pub fn sqrt(x: f32) -> f32 {
    if x <= 0.0 { 0.0 } else { x * inv_sqrt(x) }
}

/// Q30 fixed-point helpers matching the DMP Quat6 representation.
pub const Q30_ONE: f32 = 1_073_741_824.0;

pub fn q30_to_f32(value: i32) -> f32 {
    value as f32 / Q30_ONE
}

pub fn f32_to_q30(value: f32) -> i32 {
    let scaled = value * Q30_ONE;

    if scaled >= i32::MAX as f32 {
        i32::MAX
    } else if scaled <= i32::MIN as f32 {
        i32::MIN
    } else {
        scaled as i32
    }
}

/// Rebuilds the non-negative scalar part of a unit quaternion from its vector.
pub fn reconstruct_w(x: f32, y: f32, z: f32) -> f32 {
    sqrt(1.0 - (x * x + y * y + z * z))
}

/// Gravity ("down") direction expressed in the sensor body frame for a unit
/// quaternion `[w, x, y, z]` that rotates body vectors into the world frame.
/// At rest this should match the normalised accelerometer vector; if it
/// matches the negated vector instead, the quaternion uses the opposite
/// (world-to-body) convention.
pub fn down_in_body(q: [f32; 4]) -> [f32; 3] {
    let [w, x, y, z] = q;
    [
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

/// Scale factors for converting raw IMU counts into physical units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuScale {
    pub accel_lsb_per_g: f32,
    pub gyro_lsb_per_dps: f32,
}

impl ImuScale {
    /// ICM-20948 at +/-4 g and +/-2000 dps (the ranges the DMP firmware uses).
    pub const ICM20948_4G_2000DPS: Self = Self {
        accel_lsb_per_g: 8_192.0,
        gyro_lsb_per_dps: 16.4,
    };

    /// ICM-20948 at +/-16 g and +/-2000 dps (the raw-mode default: no
    /// clipping when a leg strikes the ground).
    pub const ICM20948_16G_2000DPS: Self = Self {
        accel_lsb_per_g: 2_048.0,
        gyro_lsb_per_dps: 16.4,
    };

    pub fn accel_g(&self, raw: [i16; 3]) -> [f32; 3] {
        [
            raw[0] as f32 / self.accel_lsb_per_g,
            raw[1] as f32 / self.accel_lsb_per_g,
            raw[2] as f32 / self.accel_lsb_per_g,
        ]
    }

    pub fn gyro_rad_s(&self, raw: [i16; 3]) -> [f32; 3] {
        let k = PI / 180.0 / self.gyro_lsb_per_dps;

        [raw[0] as f32 * k, raw[1] as f32 * k, raw[2] as f32 * k]
    }
}

/// Madgwick's gradient-descent orientation filter, IMU (6-axis) variant.
///
/// `beta` trades gyroscope trust against accelerometer correction; 0.1 is a
/// reasonable default for a limb-mounted sensor, larger values converge faster
/// but let linear acceleration disturb the estimate more.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Madgwick {
    q: [f32; 4],
    beta: f32,
}

impl Madgwick {
    pub const fn new(beta: f32) -> Self {
        Self {
            q: [1.0, 0.0, 0.0, 0.0],
            beta,
        }
    }

    pub fn reset(&mut self) {
        self.q = [1.0, 0.0, 0.0, 0.0];
    }

    /// Re-initialises the estimate from a (roughly stationary) accelerometer
    /// sample so the filter starts at the correct tilt instead of spending
    /// seconds converging from identity. Yaw is left at zero.
    pub fn reset_to_gravity(&mut self, accel: [f32; 3]) {
        let norm_sq = accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2];

        if norm_sq <= 0.0 {
            self.reset();
            return;
        }

        let recip = inv_sqrt(norm_sq);
        let [ax, ay, az] = [accel[0] * recip, accel[1] * recip, accel[2] * recip];

        // Quaternion rotating the measured down vector onto +Z:
        // (1 + a.z, a x z) with z = (0, 0, 1), then normalised.
        let mut q = [1.0 + az, ay, -ax, 0.0];
        let q_norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];

        if q_norm_sq < 1e-8 {
            // Exactly upside down: any 180-degree turn about a horizontal axis.
            q = [0.0, 1.0, 0.0, 0.0];
        } else {
            let recip = inv_sqrt(q_norm_sq);

            for value in &mut q {
                *value *= recip;
            }
        }

        self.q = q;
    }

    /// Unit quaternion `[w, x, y, z]`.
    pub const fn quaternion(&self) -> [f32; 4] {
        self.q
    }

    /// Quaternion with `w >= 0`, as the DMP reports it (`q` and `-q` are the
    /// same rotation, so this picks one hemisphere for storage).
    pub fn quaternion_positive_w(&self) -> [f32; 4] {
        if self.q[0] < 0.0 {
            [-self.q[0], -self.q[1], -self.q[2], -self.q[3]]
        } else {
            self.q
        }
    }

    /// DMP-style Q30 vector part with implied non-negative `w`.
    pub fn quat6_q30(&self) -> [i32; 3] {
        let q = self.quaternion_positive_w();

        [f32_to_q30(q[1]), f32_to_q30(q[2]), f32_to_q30(q[3])]
    }

    /// One filter step. `gyro` in rad/s, `accel` in any consistent unit
    /// (only its direction is used), `dt` in seconds.
    pub fn update(&mut self, gyro: [f32; 3], accel: [f32; 3], dt: f32) {
        self.update_with_gain(gyro, accel, dt, 1.0);
    }

    /// One filter step with the accelerometer correction scaled by
    /// `accel_gain` (0 = gyro only, 1 = the configured `beta`). Pass
    /// [`accel_trust`] here so hard accelerations, such as a heel strike, do
    /// not drag the orientation estimate toward a false "down".
    pub fn update_with_gain(&mut self, gyro: [f32; 3], accel: [f32; 3], dt: f32, accel_gain: f32) {
        let beta = self.beta * accel_gain;
        let [q0, q1, q2, q3] = self.q;
        let [gx, gy, gz] = gyro;
        let [mut ax, mut ay, mut az] = accel;

        let mut q_dot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
        let mut q_dot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
        let mut q_dot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
        let mut q_dot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

        let accel_norm_sq = ax * ax + ay * ay + az * az;

        if accel_norm_sq > 0.0 {
            let recip = inv_sqrt(accel_norm_sq);
            ax *= recip;
            ay *= recip;
            az *= recip;

            let two_q0 = 2.0 * q0;
            let two_q1 = 2.0 * q1;
            let two_q2 = 2.0 * q2;
            let two_q3 = 2.0 * q3;
            let four_q0 = 4.0 * q0;
            let four_q1 = 4.0 * q1;
            let four_q2 = 4.0 * q2;
            let eight_q1 = 8.0 * q1;
            let eight_q2 = 8.0 * q2;
            let q0q0 = q0 * q0;
            let q1q1 = q1 * q1;
            let q2q2 = q2 * q2;
            let q3q3 = q3 * q3;

            let mut s0 = four_q0 * q2q2 + two_q2 * ax + four_q0 * q1q1 - two_q1 * ay;
            let mut s1 = four_q1 * q3q3 - two_q3 * ax + 4.0 * q0q0 * q1 - two_q0 * ay - four_q1
                + eight_q1 * q1q1
                + eight_q1 * q2q2
                + four_q1 * az;

            let mut s2 = 4.0 * q0q0 * q2 + two_q0 * ax + four_q2 * q3q3 - two_q3 * ay - four_q2
                + eight_q2 * q1q1
                + eight_q2 * q2q2
                + four_q2 * az;

            let mut s3 = 4.0 * q1q1 * q3 - two_q1 * ax + 4.0 * q2q2 * q3 - two_q2 * ay;

            let step_norm_sq = s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;

            if step_norm_sq > 0.0 && beta > 0.0 {
                let recip = inv_sqrt(step_norm_sq);
                s0 *= recip;
                s1 *= recip;
                s2 *= recip;
                s3 *= recip;
                q_dot0 -= beta * s0;
                q_dot1 -= beta * s1;
                q_dot2 -= beta * s2;
                q_dot3 -= beta * s3;
            }
        }

        let mut q = [
            q0 + q_dot0 * dt,
            q1 + q_dot1 * dt,
            q2 + q_dot2 * dt,
            q3 + q_dot3 * dt,
        ];

        let recip = inv_sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

        for value in &mut q {
            *value *= recip;
        }

        self.q = q;
    }
}

impl Default for Madgwick {
    fn default() -> Self {
        Self::new(0.1)
    }
}

// ---------------------------------------------------------------------------
// Replacements for the DMP's on-chip helpers
// ---------------------------------------------------------------------------

/// How far the accelerometer magnitude may stray from 1 g before it stops
/// being trusted as a gravity reference at all.
const ACCEL_TRUST_BAND_G: f32 = 0.25;

/// How much to trust the accelerometer as a measure of "down", from 1.0
/// (magnitude exactly 1 g, so it is mostly gravity) falling linearly to 0.0
/// at `ACCEL_TRUST_BAND_G` away (a heel strike or a jump, so it is mostly
/// motion). Feed the result to [`Madgwick::update_with_gain`].
pub fn accel_trust(accel_g: [f32; 3]) -> f32 {
    let magnitude =
        sqrt(accel_g[0] * accel_g[0] + accel_g[1] * accel_g[1] + accel_g[2] * accel_g[2]);

    let deviation = (magnitude - 1.0).abs();

    (1.0 - deviation / ACCEL_TRUST_BAND_G).clamp(0.0, 1.0)
}

/// Stillness test: the accelerometer reads 1 g within this tolerance ...
const STILL_ACCEL_TOLERANCE_G: f32 = 0.08;

/// ... and the gyro, after removing the current bias estimate, reads less
/// than this. Must exceed the sensor's worst-case raw offset (about 5 deg/s
/// for the ICM-20948) or an uncalibrated sensor would never count as still.
const STILL_GYRO_LIMIT_DPS: f32 = 6.0;

/// Consecutive still samples required before learning starts (~0.2 s at
/// 225 Hz), so brief pauses mid-motion are not mistaken for rest.
const STILL_SAMPLES_TO_START: u32 = 45;

/// Learning starts as a plain average, then settles to this slow rate
/// (time constant ~2 s at 225 Hz) to track temperature drift.
const SLOW_LEARNING_RATE: f32 = 0.002;

/// Still samples learned before each calibration level is reported:
/// level 1 at 0.2 s, level 2 at 1 s, level 3 at 2 s (at 225 Hz).
const LEVEL_SAMPLES: [u32; 3] = [45, 225, 450];

/// Learns the gyroscope's zero-rate offset whenever the sensor is at rest.
///
/// This replaces the DMP's background gyro calibration. A gyro at rest
/// should read zero but reads a small offset (bias) that changes with
/// temperature. Uncorrected, the orientation filter integrates the bias as
/// a slow, steady rotation: drift. Whenever the sensor has been still for a
/// moment, the average gyro reading *is* the bias, so it is learned then and
/// subtracted from every reading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GyroBias {
    /// Current estimate, in raw gyro counts.
    bias: [f32; 3],

    /// Counts per deg/s, to express the stillness limit in counts.
    lsb_per_dps: f32,

    /// Consecutive still samples seen so far.
    still_run: u32,

    /// Total still samples absorbed into the estimate.
    learned: u32,
}

impl GyroBias {
    pub const fn new(gyro_lsb_per_dps: f32) -> Self {
        Self {
            bias: [0.0; 3],
            lsb_per_dps: gyro_lsb_per_dps,
            still_run: 0,
            learned: 0,
        }
    }

    /// Current bias estimate in raw counts. Save it before power-down and
    /// restore it with [`GyroBias::restore`] to skip relearning after wake.
    pub const fn bias_counts(&self) -> [f32; 3] {
        self.bias
    }

    /// Starts from a previously saved estimate, reported as fully learned.
    pub fn restore(&mut self, bias_counts: [f32; 3]) {
        self.bias = bias_counts;
        self.learned = LEVEL_SAMPLES[2];
    }

    /// Calibration level 0-3, the same scale the DMP uses: 0 never still
    /// long enough, 3 after about 2 seconds of accumulated rest.
    pub fn level(&self) -> u16 {
        LEVEL_SAMPLES
            .iter()
            .filter(|&&needed| self.learned >= needed)
            .count() as u16
    }

    /// Feeds one sample. Updates the estimate if the sensor is still, and
    /// returns the bias-corrected gyro reading in counts.
    pub fn correct(&mut self, gyro_counts: [i16; 3], accel_g: [f32; 3]) -> [f32; 3] {
        let gyro = [
            f32::from(gyro_counts[0]),
            f32::from(gyro_counts[1]),
            f32::from(gyro_counts[2]),
        ];

        let residual = [
            gyro[0] - self.bias[0],
            gyro[1] - self.bias[1],
            gyro[2] - self.bias[2],
        ];

        // Stillness: gravity alone on the accelerometer, little rotation.
        let accel_magnitude =
            sqrt(accel_g[0] * accel_g[0] + accel_g[1] * accel_g[1] + accel_g[2] * accel_g[2]);

        let rotation_dps =
            sqrt(residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2])
                / self.lsb_per_dps;

        let still = (accel_magnitude - 1.0).abs() < STILL_ACCEL_TOLERANCE_G
            && rotation_dps < STILL_GYRO_LIMIT_DPS;

        self.still_run = if still {
            self.still_run.saturating_add(1)
        } else {
            0
        };

        if self.still_run >= STILL_SAMPLES_TO_START {
            // Running average at first, so the first second of rest gives a
            // good estimate quickly; then a slow rate that tracks drift.

            self.learned = self.learned.saturating_add(1);

            let rate = (1.0 / self.learned as f32).max(SLOW_LEARNING_RATE);

            for axis in 0..3 {
                self.bias[axis] += rate * (gyro[axis] - self.bias[axis]);
            }
        }

        [
            gyro[0] - self.bias[0],
            gyro[1] - self.bias[1],
            gyro[2] - self.bias[2],
        ]
    }
}

#[cfg(all(test, not(target_os = "none")))]
mod tests {
    use super::*;

    #[test]
    fn inverse_square_root_is_accurate() {
        for value in [0.01_f32, 0.5, 1.0, 2.0, 9.0, 12_345.0] {
            let expected = 1.0 / value.sqrt();
            assert!((inv_sqrt(value) - expected).abs() <= expected * 2e-6);
        }

        assert_eq!(inv_sqrt(0.0), 0.0);
    }

    #[test]
    fn q30_conversion_round_trips() {
        assert_eq!(q30_to_f32(f32_to_q30(0.25)), 0.25);
        assert_eq!(f32_to_q30(1.0), 1 << 30);
        assert_eq!(f32_to_q30(-1.0), -(1 << 30));
        assert_eq!(f32_to_q30(3.0), i32::MAX);
        assert_eq!(f32_to_q30(-3.0), i32::MIN);
    }

    #[test]
    fn filter_settles_on_gravity_when_tilted() {
        // Sensor rolled 90 degrees about X: gravity appears along +Y.
        let mut filter = Madgwick::new(0.5);

        for _ in 0..4_000 {
            filter.update([0.0; 3], [0.0, 1.0, 0.0], 1.0 / 225.0);
        }

        let down = down_in_body(filter.quaternion());

        assert!(down[1].abs() > 0.99, "down vector {down:?}");
        assert!(
            down[0].abs() < 0.05 && down[2].abs() < 0.05,
            "down vector {down:?}"
        );

        let q = filter.quaternion_positive_w();

        assert!(q[0] >= 0.0);
    }

    #[test]
    fn gravity_reset_matches_accelerometer_immediately() {
        for accel in [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.3, -0.4, 0.5],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ] {
            let mut filter = Madgwick::default();

            filter.reset_to_gravity(accel);

            let down = down_in_body(filter.quaternion());
            let norm = (accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]).sqrt();

            for axis in 0..3 {
                assert!(
                    (down[axis] - accel[axis] / norm).abs() < 1e-5,
                    "{accel:?} -> {down:?}"
                );
            }
        }
    }

    #[test]
    fn accel_trust_falls_off_away_from_one_g() {
        assert!((accel_trust([0.0, 0.0, 1.0]) - 1.0).abs() < 1e-5);
        assert!((accel_trust([0.0, 0.0, 1.125]) - 0.5).abs() < 1e-3);
        assert_eq!(accel_trust([0.0, 0.0, 3.0]), 0.0);
        assert_eq!(accel_trust([0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn zero_gain_is_pure_gyro_integration() {
        // With no accelerometer trust, a gravity vector that disagrees with
        // the orientation must not move it.

        let mut filter = Madgwick::new(0.5);

        filter.update_with_gain([0.0; 3], [1.0, 0.0, 0.0], 0.01, 0.0);

        let q = filter.quaternion();

        assert!(
            (q[0] - 1.0).abs() < 1e-5 && q[1] == 0.0 && q[2] == 0.0 && q[3] == 0.0,
            "{q:?}"
        );
    }

    #[test]
    fn gyro_bias_is_learned_at_rest_and_not_during_motion() {
        let mut bias = GyroBias::new(16.4);
        let offset = [40_i16, -25, 12]; // ~2.4, -1.5, 0.7 deg/s

        // Motion: big rotation, acceleration far from 1 g. Nothing learned.
        for _ in 0..1_000 {
            bias.correct([2_000, 0, 0], [0.0, 0.0, 2.0]);
        }

        assert_eq!(bias.level(), 0);
        assert_eq!(bias.bias_counts(), [0.0; 3]);

        // Rest: constant offset with a little noise. Learned to level 3.
        for sample in 0..600 {
            let noise = (sample % 5) as i16 - 2;

            bias.correct(
                [offset[0] + noise, offset[1] - noise, offset[2]],
                [0.0, 0.0, 1.0],
            );
        }

        assert_eq!(bias.level(), 3);

        for axis in 0..3 {
            assert!((bias.bias_counts()[axis] - f32::from(offset[axis])).abs() < 1.0);
        }

        // Once learned, a still reading comes back corrected to about zero.
        let corrected = bias.correct(offset, [0.0, 0.0, 1.0]);

        assert!(corrected.iter().all(|value| value.abs() < 1.0));
    }

    #[test]
    fn identity_orientation_sees_gravity_on_z() {
        let filter = Madgwick::default();

        assert_eq!(down_in_body(filter.quaternion()), [0.0, 0.0, 1.0]);
        assert_eq!(filter.quat6_q30(), [0, 0, 0]);
    }
}
