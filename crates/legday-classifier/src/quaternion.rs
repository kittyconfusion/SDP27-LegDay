/// Scale used by the ICM-20948 DMP's 32-bit quaternion output.
pub const Q30_SCALE: i32 = 1 << 30;

/// Quaternion represented as signed Q30 fixed-point values.
///
/// The ICM-20948's six-axis DMP output supplies `x`, `y`, and `z`; `w` is
/// reconstructed from the unit-quaternion constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuaternionQ30 {
    pub w: i32,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl QuaternionQ30 {
    pub const IDENTITY: Self = Self {
        w: Q30_SCALE,
        x: 0,
        y: 0,
        z: 0,
    };

    pub const fn new(w: i32, x: i32, y: i32, z: i32) -> Self {
        Self { w, x, y, z }
    }

    /// Reconstructs the non-negative scalar component from a DMP Quat6 sample.
    /// Returns `None` when the vector component is clearly outside the unit
    /// sphere. A small amount of fixed-point overshoot is clamped to zero.
    pub fn from_dmp_quat6(x: i32, y: i32, z: i32) -> Option<Self> {
        let vector_norm_sq = square(x) + square(y) + square(z);
        let unit_norm_sq = square(Q30_SCALE);
        let tolerance = unit_norm_sq / 100;
        if vector_norm_sq > unit_norm_sq + tolerance {
            return None;
        }

        let w_sq = unit_norm_sq.saturating_sub(vector_norm_sq) as u64;
        Some(Self::new(integer_sqrt(w_sq) as i32, x, y, z))
    }

    pub const fn conjugate(self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    pub const fn negated(self) -> Self {
        Self::new(-self.w, -self.x, -self.y, -self.z)
    }

    pub fn dot(self, other: Self) -> i128 {
        i128::from(self.w) * i128::from(other.w)
            + i128::from(self.x) * i128::from(other.x)
            + i128::from(self.y) * i128::from(other.y)
            + i128::from(self.z) * i128::from(other.z)
    }

    /// Makes consecutive samples use the same quaternion hemisphere. Since
    /// `q` and `-q` represent the same rotation, this prevents sign jumps from
    /// appearing as artificial motion to a classifier.
    pub fn aligned_to(self, previous: Self) -> Self {
        if self.dot(previous) < 0 {
            self.negated()
        } else {
            self
        }
    }

    /// Hamilton product in Q30 fixed point.
    pub fn multiply(self, rhs: Self) -> Self {
        let aw = i128::from(self.w);
        let ax = i128::from(self.x);
        let ay = i128::from(self.y);
        let az = i128::from(self.z);
        let bw = i128::from(rhs.w);
        let bx = i128::from(rhs.x);
        let by = i128::from(rhs.y);
        let bz = i128::from(rhs.z);

        Self::new(
            scale_product(aw * bw - ax * bx - ay * by - az * bz),
            scale_product(aw * bx + ax * bw + ay * bz - az * by),
            scale_product(aw * by - ax * bz + ay * bw + az * bx),
            scale_product(aw * bz + ax * by - ay * bx + az * bw),
        )
    }

    /// Orientation of `measured` relative to `reference`.
    ///
    /// For a leg, the reference can be the calibrated thigh orientation and
    /// the measured value the calibrated shank orientation.
    pub fn relative(reference: Self, measured: Self) -> Self {
        reference.conjugate().multiply(measured)
    }
}

fn square(value: i32) -> i128 {
    i128::from(value) * i128::from(value)
}

fn scale_product(value: i128) -> i32 {
    (value / i128::from(Q30_SCALE)) as i32
}

fn integer_sqrt(value: u64) -> u64 {
    if value < 2 {
        return value;
    }

    let mut estimate = 1_u64 << ((64 - value.leading_zeros() as u64) / 2);
    loop {
        let next = (estimate + value / estimate) / 2;
        if next >= estimate {
            return estimate;
        }
        estimate = next;
    }
}

#[cfg(all(test, not(target_os = "none")))]
mod tests {
    use super::*;

    #[test]
    fn reconstructs_identity_from_zero_vector() {
        assert_eq!(
            QuaternionQ30::from_dmp_quat6(0, 0, 0),
            Some(QuaternionQ30::IDENTITY)
        );
    }

    #[test]
    fn relative_orientation_removes_shared_orientation() {
        const SIN_COS_45_Q30: i32 = 759_250_125;
        let quarter_turn_x = QuaternionQ30::new(SIN_COS_45_Q30, SIN_COS_45_Q30, 0, 0);
        let relative = QuaternionQ30::relative(quarter_turn_x, quarter_turn_x);

        assert!((relative.w - Q30_SCALE).abs() <= 2);
        assert!(relative.x.abs() <= 2);
        assert_eq!(relative.y, 0);
        assert_eq!(relative.z, 0);
    }

    #[test]
    fn aligns_equivalent_opposite_signs() {
        let opposite_identity = QuaternionQ30::IDENTITY.negated();
        assert_eq!(
            opposite_identity.aligned_to(QuaternionQ30::IDENTITY),
            QuaternionQ30::IDENTITY
        );
    }
}
