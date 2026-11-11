// No standard library, so the crate also builds for the bare-metal ESP32-S3.
#![no_std]
// The ESP32-S3 kernels in `cnn` use inline assembly, which is still a
// nightly-only feature on Xtensa (the esp toolchain is nightly). Other
// targets, including the Mac running `cargo test`, do not need it.
#![cfg_attr(target_arch = "xtensa", feature(asm_experimental_arch))]

//! Hardware-independent preprocessing and inference for the LegDay motion
//! classifier, plus the orientation math the firmware shares with it.
//!
//! Nothing here touches ESP32 hardware, so every module runs (and is
//! tested) on a Mac exactly as it runs on the S3.
//!
//! # What lives here
//!
//! | Module | Role |
//! |---|---|
//! | [`preprocess`] | One-second window of 120-byte frames -> the model's input: 40 channels in physical units, averaged to 75 steps, standardized. Mirrors `Scripts/legday_ml.py` line for line. |
//! | [`cnn`] | Runs the 1-D CNN trained by `Scripts/train_classifier.py`: reads the exported `.ldm` file in place from flash and computes class scores. This is the real classifier. |
//! | [`prototype`] | The earlier baseline: per-channel mean and mean-absolute-change features with a nearest-centroid model that trains by averaging, so it can learn on the device. Kept as the bar the CNN must beat and as the pattern for on-device personalization. |
//! | [`fusion`] | Orientation on the S3: Madgwick filter, gyro bias learning, accelerometer trust. Used by the firmware's pipeline to fill the quaternions in every frame. |
//! | [`quaternion`] | Fixed-point Q30 quaternion math (relative rotation between two IMUs, hemisphere alignment) for the inner control loops, e.g. knee flexion. |
//!
//! # The network in one paragraph
//!
//! Four 1-D convolutions along time (kernels 7, 5, 5, 3, each followed by
//! ReLU; the second and fourth use stride 2 in place of a pooling layer),
//! an optional self-attention layer, an average over time, and one dense
//! layer that gives a score per class. The table of layer shapes is at the
//! top of [`cnn`].
//!
//! # How a classification happens on the board
//!
//! ```text
//!   1 s RAM window (225 frames)
//!     --preprocess::window_input-->  40 x 75 standardized input
//!     --cnn::Model::infer-->         one score per class
//!     --cnn::softmax-->              probabilities; the largest is the answer
//! ```
//!
//! The window is counted in frames, not in time: 225 frames is one second
//! only at the 225 Hz frame rate the model was designed around.

pub mod cnn;
// Compatibility with the original crate path, now backed by a separate package.
pub use legday_fusion as fusion;
pub mod preprocess;
pub mod prototype;
pub mod quaternion;
