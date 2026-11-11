//! The baseline classifier that predates the CNN: nearest centroid.
//!
//! 1. [`WindowFeatures`] summarizes a window of raw frames as 56 integers:
//!    for each of the 28 raw channels (4 IMUs x 6 axes, plus 4 pressure
//!    sensors), its mean and its mean absolute change from one frame to
//!    the next.
//! 2. [`PrototypeClassifier`] keeps, for each class, the average of every
//!    feature vector it was shown for that class: the class's "centroid"
//!    or "prototype". To classify, it picks the class whose centroid is
//!    nearest to the new feature vector.
//!
//! Training is only a running sum, so it can happen on the device, and all
//! of the arithmetic is integer. This module works on the 72-byte raw
//! frame ([`legday_core::SensorFrame`], no quaternions) in raw sensor
//! counts, unlike [`crate::preprocess`], which converts to physical units.
//!
//! The `<const CLASSES: usize, const FEATURES: usize>` parameters are
//! "const generics": sizes fixed at compile time, so the model is plain
//! arrays and needs no heap.

use legday_core::{FEATURE_COUNT, WindowBuffer};

/// Two summary values for every raw input channel: its mean and its mean
/// absolute change between consecutive samples.
pub const WINDOW_FEATURES: usize = FEATURE_COUNT * 2;

/// Compact, heap-free summary of one sensor window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowFeatures {
    /// The first half holds each channel's mean, the second half each
    /// channel's mean absolute change, both in raw sensor counts.
    pub values: [i32; WINDOW_FEATURES],
}

impl WindowFeatures {
    /// Extracts posture/load information through channel means and motion
    /// information through mean absolute sample-to-sample changes.
    /// Returns `None` for a window of fewer than two frames.
    pub fn from_window<const N: usize>(window: &WindowBuffer<N>) -> Option<Self> {
        // A change needs two frames to compare.
        if window.len() < 2 {
            return None;
        }

        // Running totals per channel, in 64 bits so they cannot overflow.
        let mut sums = [0_i64; FEATURE_COUNT];
        let mut absolute_changes = [0_u64; FEATURE_COUNT];
        let mut previous = [0_i32; FEATURE_COUNT];
        let mut current = [0_i32; FEATURE_COUNT];

        for frame_index in 0..window.len() {
            let frame = window.get(frame_index)?;

            frame.write_features_i32(&mut current);

            for feature_index in 0..FEATURE_COUNT {
                sums[feature_index] += i64::from(current[feature_index]);

                // The first frame has no previous frame to differ from.
                if frame_index > 0 {
                    let change =
                        i64::from(current[feature_index]) - i64::from(previous[feature_index]);
                    absolute_changes[feature_index] += change.unsigned_abs();
                }
            }

            previous = current;
        }

        // Totals -> averages (integer division, rounded toward zero).
        // N frames give N - 1 changes.
        let sample_count = window.len() as i64;
        let change_count = (window.len() - 1) as u64;
        let mut values = [0_i32; WINDOW_FEATURES];

        for feature_index in 0..FEATURE_COUNT {
            values[feature_index] = (sums[feature_index] / sample_count) as i32;
            values[FEATURE_COUNT + feature_index] =
                (absolute_changes[feature_index] / change_count) as i32;
        }

        Some(Self { values })
    }
}

/// A small supervised baseline which learns one average feature vector per
/// class. Training is streaming: samples do not need to remain in RAM.
///
/// Inputs should be normalized before this model is used for a real comparison;
/// otherwise high-range channels dominate Euclidean distance.
pub struct PrototypeClassifier<const CLASSES: usize, const FEATURES: usize> {
    /// For each class, the sum of every feature vector observed. Dividing
    /// by `counts` gives the centroid, so no samples have to be stored.
    sums: [[i64; FEATURES]; CLASSES],
    /// Training examples seen for each class.
    counts: [u32; CLASSES],
}

/// The result of [`PrototypeClassifier::predict`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prediction {
    /// Index of the nearest class.
    pub class_index: usize,
    /// Squared Euclidean distance to that class's centroid (smaller means
    /// closer).
    pub squared_distance: u128,
    /// Training examples that class was built from.
    pub training_samples: u32,
}

impl<const CLASSES: usize, const FEATURES: usize> PrototypeClassifier<CLASSES, FEATURES> {
    /// An untrained model. Zero classes or zero features is rejected.
    pub const fn new() -> Self {
        assert!(CLASSES > 0, "classifier must contain at least one class");
        assert!(FEATURES > 0, "classifier must contain at least one feature");

        Self {
            sums: [[0; FEATURES]; CLASSES],
            counts: [0; CLASSES],
        }
    }

    /// Adds one labeled training example. Returns `false` for an invalid class
    /// index or when that class has reached its counter limit.
    pub fn observe(&mut self, class_index: usize, features: &[i32; FEATURES]) -> bool {
        let Some(count) = self.counts.get_mut(class_index) else {
            return false;
        };

        if *count == u32::MAX {
            return false;
        }

        for (sum, feature) in self.sums[class_index].iter_mut().zip(features) {
            *sum += i64::from(*feature);
        }

        *count += 1;
        true
    }

    /// Training examples seen for a class, or `None` for an invalid index.
    pub const fn training_samples(&self, class_index: usize) -> Option<u32> {
        if class_index < CLASSES {
            Some(self.counts[class_index])
        } else {
            None
        }
    }

    /// Chooses the nearest trained class centroid. Returns `None` if no
    /// class has been trained yet.
    pub fn predict(&self, features: &[i32; FEATURES]) -> Option<Prediction> {
        let mut best: Option<Prediction> = None;

        for class_index in 0..CLASSES {
            let count = self.counts[class_index];

            // A class with no training examples cannot be chosen.
            if count == 0 {
                continue;
            }

            // Centroid = sum / count; add up the squared differences.
            let mut squared_distance = 0_u128;

            for (sum, feature) in self.sums[class_index].iter().zip(features) {
                let centroid = *sum / i64::from(count);
                let difference = i128::from(*feature) - i128::from(centroid);
                squared_distance += difference.unsigned_abs().pow(2);
            }

            // Keep this class if it is the first or the closest so far.
            if best
                .as_ref()
                .is_none_or(|prediction| squared_distance < prediction.squared_distance)
            {
                best = Some(Prediction {
                    class_index,
                    squared_distance,
                    training_samples: count,
                });
            }
        }

        best
    }
}

impl<const CLASSES: usize, const FEATURES: usize> Default
    for PrototypeClassifier<CLASSES, FEATURES>
{
    fn default() -> Self {
        Self::new()
    }
}

// Unit tests. Built only by `cargo test` on a computer: `target_os = "none"`
// is the bare-metal ESP32 target, which has no test runner.
#[cfg(all(test, not(target_os = "none")))]
mod tests {
    use super::*;
    use legday_core::{Imu6Raw, SensorFrame};

    /// A frame in which every IMU axis and pressure reading equals `value`.
    fn frame(sequence: u32, value: i16) -> SensorFrame {
        SensorFrame {
            sequence,
            imus: [Imu6Raw {
                accel: [value; 3],
                gyro: [value; 3],
            }; 4],
            pressure_adc: [value as u16; 4],
            ..SensorFrame::ZERO
        }
    }

    #[test]
    fn summarizes_mean_and_change() {
        let mut window = WindowBuffer::<3>::new();
        window.push(frame(0, 10));
        window.push(frame(1, 20));
        window.push(frame(2, 30));

        let features = WindowFeatures::from_window(&window).unwrap();

        // Readings 10, 20, 30: mean 20, and a change of 10 at every step.
        assert!(
            features.values[..FEATURE_COUNT]
                .iter()
                .all(|value| *value == 20)
        );

        assert!(
            features.values[FEATURE_COUNT..]
                .iter()
                .all(|value| *value == 10)
        );
    }

    #[test]
    fn learns_streaming_class_centroids() {
        let mut model = PrototypeClassifier::<2, 2>::new();

        assert!(model.observe(0, &[0, 2]));
        assert!(model.observe(0, &[2, 4]));
        assert!(model.observe(1, &[100, 102]));
        assert!(model.observe(1, &[102, 104]));

        assert_eq!(model.predict(&[3, 4]).unwrap().class_index, 0);
        assert_eq!(model.predict(&[99, 101]).unwrap().class_index, 1);
        assert_eq!(model.training_samples(0), Some(2));
    }
}
