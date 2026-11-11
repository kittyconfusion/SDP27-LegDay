#![no_std]

#[cfg(feature = "esp32s3")]
pub mod imu_controller;
// pub mod sd_controller;

pub mod lib_recording;
pub mod recording;

// Keep the team's module path while also supporting the original classifier API.
// Both paths refer to the same types; there is only one raw-frame implementation.
pub use lib_recording::*;
pub mod clock;
pub mod fused;
