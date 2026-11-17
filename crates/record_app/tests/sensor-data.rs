// Host-only checks: rustc +stable --edition=2024 --test this-file.rs.
// Avoid building embedded HAL dependencies just to verify byte serialization.
#![allow(dead_code)]
#[path = "../bin/sensors/config.rs"]
mod config;
#[path = "../bin/sensors/frame.rs"]
mod frame;
#[path = "../bin/sensors/sample.rs"]
mod sample;
