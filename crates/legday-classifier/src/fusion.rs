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
            if step_norm_sq > 0.0 {
                let recip = inv_sqrt(step_norm_sq);
                s0 *= recip;
                s1 *= recip;
                s2 *= recip;
                s3 *= recip;
                q_dot0 -= self.beta * s0;
                q_dot1 -= self.beta * s1;
                q_dot2 -= self.beta * s2;
                q_dot3 -= self.beta * s3;
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
    fn identity_orientation_sees_gravity_on_z() {
        let filter = Madgwick::default();
        assert_eq!(down_in_body(filter.quaternion()), [0.0, 0.0, 1.0]);
        assert_eq!(filter.quat6_q30(), [0, 0, 0]);
    }
}
