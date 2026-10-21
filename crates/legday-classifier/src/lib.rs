#![no_std]

//! Hardware-independent preprocessing and inference primitives.
//!
//! Keep this crate free of ESP-specific APIs so the same algorithm can be
//! tested on recorded data on a computer and linked into embedded firmware.
//!
//! # What lives here and why
//!
//! | Module | Status | Role |
//! |---|---|---|
//! | [`prototype`] | baseline, kept | Turns a one-second window of 72-byte raw frames into a small feature vector (per-channel mean and mean absolute change) and classifies it with a nearest-centroid model that can be trained on device. It is the "does the pipeline work at all" model to beat before a 1-D CNN. |
//! | [`quaternion`] | kept | Fixed-point Q30 quaternion maths for the DMP's Quat6 output: rebuild `w`, hemisphere alignment (`q` and `-q` are the same rotation), and the relative rotation between two sensors, which is how knee flexion will be derived. Inner-loop material, not classifier input. |
//! | [`fusion`] | kept, fallback | A Madgwick 6-axis filter that computes a Quat6-equivalent on the S3 from raw accel + gyro. Used when the DMP is unavailable and as a live cross-check of the DMP. |
//!
//! The current plan feeds the classifier the raw 72-byte frames (4 IMUs x
//! accel/gyro + 4 pressure channels) and reserves quaternions for the inner
//! control loops; the modules above follow that split.

pub mod fusion;
pub mod prototype;
pub mod quaternion;

/// Shape of the current raw-data model input.
pub const INPUT_FRAMES: usize = legday_core::WINDOW_SAMPLES;
pub const RAW_FEATURES_PER_FRAME: usize = legday_core::FEATURE_COUNT;
pub const RAW_INPUT_VALUES: usize = INPUT_FRAMES * RAW_FEATURES_PER_FRAME;
