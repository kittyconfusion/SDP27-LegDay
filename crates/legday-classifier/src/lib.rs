#![no_std]

//! Hardware-independent preprocessing and inference primitives.
//!
//! Keep this crate free of ESP-specific APIs so the same algorithm can be
//! tested on recorded data on a computer and linked into embedded firmware.

pub mod prototype;
pub mod quaternion;

/// Shape of the current raw-data model input.
pub const INPUT_FRAMES: usize = legday_core::WINDOW_SAMPLES;
pub const RAW_FEATURES_PER_FRAME: usize = legday_core::FEATURE_COUNT;
pub const RAW_INPUT_VALUES: usize = INPUT_FRAMES * RAW_FEATURES_PER_FRAME;
