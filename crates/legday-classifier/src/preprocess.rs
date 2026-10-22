//! Turning frames into model input, exactly as `Scripts/legday_ml.py` does.
//!
//! A model only works if it is fed the same numbers it was trained on, so
//! every step here mirrors the Python function named in its comment. The
//! test `cnn::tests::demo_model_matches_pytorch` checks the two agree.
//!
//! The pipeline for one classification:
//!
//! ```text
//!  225 frames (1 s)  --frame_channels-->  225 x 40 channels (physical units)
//!                    --average groups of 3-->  75 steps x 40 channels
//!                    --standardize with the model's mean / std-->  model input
//! ```
//!
//! The 40 channels per frame:
//!
//! | index | content | unit |
//! |---|---|---|
//! | 0-23  | IMU 0..3 x (ax, ay, az, gx, gy, gz) | g, deg/s |
//! | 24-27 | pressure 0..3 | fraction of ADC full scale |
//! | 28-39 | IMU 0..3 x gravity direction "down" (x, y, z) | unit vector |
//!
//! The gravity direction is derived from each IMU's quaternion rather than
//! feeding the quaternion itself, because it is free of the quaternion's two
//! ambiguities: `q` and `-q` give the same "down", and so does every heading
//! (yaw), which a 6-axis quaternion cannot measure.

use legday_core::{
    IMU_COUNT, PRESSURE_COUNT,
    fused::{FusedFrame, FusedWindowBuffer, status},
};

use crate::fusion::{down_in_body, q30_to_f32, reconstruct_w};

/// Channels per frame (24 IMU + 4 pressure + 12 gravity direction).
pub const CHANNELS: usize = IMU_COUNT * 6 + PRESSURE_COUNT + IMU_COUNT * 3;

/// Counts per g / per deg/s for each full-scale code in the status flags.
const ACCEL_LSB_PER_G: [f32; 4] = [16_384.0, 8_192.0, 4_096.0, 2_048.0];
const GYRO_LSB_PER_DPS: [f32; 4] = [131.0, 65.5, 32.8, 16.4];

/// The 12-bit ADC's maximum reading.
const ADC_FULL_SCALE: f32 = 4_095.0;

/// Why a window could not be turned into model input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputError {
    /// The model expects a different channel layout than this code produces.
    ChannelMismatch { model: usize, expected: usize },
    /// The window does not yet hold `steps x decimation` frames.
    NotEnoughFrames { needed: usize, available: usize },
    /// The output buffer is not `channels x steps` long.
    OutputLength { needed: usize, available: usize },
}

/// Accel counts per g and gyro counts per deg/s for a frame. Frames from
/// before the ranges were recorded (all DMP recordings) were ±4 g / ±2000 dps.
/// Mirrors `frame_channels` in Python (the `lsb_g` / `lsb_dps` lines).
pub fn sensor_scales(flags: u16) -> (f32, f32) {
    if flags & status::RANGES_RECORDED == 0 {
        return (ACCEL_LSB_PER_G[1], GYRO_LSB_PER_DPS[3]);
    }

    let accel_code = usize::from((flags >> status::ACCEL_RANGE_SHIFT) & status::RANGE_MASK);
    let gyro_code = usize::from((flags >> status::GYRO_RANGE_SHIFT) & status::RANGE_MASK);

    (ACCEL_LSB_PER_G[accel_code], GYRO_LSB_PER_DPS[gyro_code])
}

/// One frame -> 40 channels. Mirrors Python `frame_channels`.
pub fn frame_channels(frame: &FusedFrame, out: &mut [f32; CHANNELS]) {
    let raw = &frame.raw;
    let (lsb_per_g, lsb_per_dps) = sensor_scales(raw.status_flags);
    let has_orientation = raw.status_flags & status::ORIENTATION_SOURCE_MASK != 0;

    *out = [0.0; CHANNELS];

    for imu in 0..IMU_COUNT {
        // Missing IMUs stay zero.
        if raw.valid_imu_mask & (1 << imu) == 0 {
            continue;
        }

        let sample = &raw.imus[imu];

        for axis in 0..3 {
            out[imu * 6 + axis] = f32::from(sample.accel[axis]) / lsb_per_g;
            out[imu * 6 + 3 + axis] = f32::from(sample.gyro[axis]) / lsb_per_dps;
        }

        if has_orientation {
            let q = frame.quat6[imu];
            let (x, y, z) = (q30_to_f32(q.x), q30_to_f32(q.y), q30_to_f32(q.z));
            let down = down_in_body([reconstruct_w(x, y, z), x, y, z]);
            let base = IMU_COUNT * 6 + PRESSURE_COUNT + imu * 3;

            out[base..base + 3].copy_from_slice(&down);
        }
    }

    for sensor in 0..PRESSURE_COUNT {
        if raw.valid_pressure_mask & (1 << sensor) != 0 {
            out[IMU_COUNT * 6 + sensor] = f32::from(raw.pressure_adc[sensor]) / ADC_FULL_SCALE;
        }
    }
}

/// The newest `steps x decimation` frames of a window -> standardized model
/// input, channels first (`out[channel * steps + step]`).
///
/// Each step is the plain average of `decimation` consecutive frames, summed
/// in order (Python `decimate`), then standardized with the model's own
/// per-channel mean and standard deviation (Python `normalize`).
pub fn window_input<const N: usize>(
    window: &FusedWindowBuffer<N>,
    model_channels: usize,
    steps: usize,
    decimation: usize,
    mean: &[f32],
    std: &[f32],
    out: &mut [f32],
) -> Result<(), InputError> {
    if model_channels != CHANNELS || mean.len() != CHANNELS || std.len() != CHANNELS {
        return Err(InputError::ChannelMismatch {
            model: model_channels,
            expected: CHANNELS,
        });
    }

    let needed = steps * decimation;

    if window.len() < needed {
        return Err(InputError::NotEnoughFrames {
            needed,
            available: window.len(),
        });
    }

    if out.len() != CHANNELS * steps {
        return Err(InputError::OutputLength {
            needed: CHANNELS * steps,
            available: out.len(),
        });
    }

    let first = window.len() - needed;
    let mut channels = [0.0_f32; CHANNELS];

    for step in 0..steps {
        let mut sum = [0.0_f32; CHANNELS];

        for offset in 0..decimation {
            // `first + ...` is always < window.len(), checked above.
            let frame = window.get(first + step * decimation + offset).unwrap();
            frame_channels(frame, &mut channels);

            for (total, value) in sum.iter_mut().zip(&channels) {
                *total += *value;
            }
        }

        for channel in 0..CHANNELS {
            let average = sum[channel] / decimation as f32;
            out[channel * steps + step] = (average - mean[channel]) / std[channel];
        }
    }

    Ok(())
}

#[cfg(all(test, not(target_os = "none")))]
mod tests {
    use super::*;
    use legday_core::fused::Quat6Raw;

    #[test]
    fn channels_follow_the_documented_layout() {
        let mut frame = FusedFrame::ZERO;
        frame.raw.valid_imu_mask = 0b0010; // only IMU 1
        frame.raw.valid_pressure_mask = 0b0100; // only pressure 2
        frame.raw.status_flags = status::ORIENTATION_SOFTWARE
            | status::RANGES_RECORDED
            | (3 << status::ACCEL_RANGE_SHIFT)
            | (3 << status::GYRO_RANGE_SHIFT);
        frame.raw.imus[1].accel = [2_048, 0, -2_048]; // +1 g, 0, -1 g at ±16 g
        frame.raw.imus[1].gyro = [164, 0, 0]; // 10 deg/s at ±2000 dps
        frame.raw.pressure_adc = [4_095, 4_095, 4_095, 4_095];
        frame.quat6[1] = Quat6Raw { x: 0, y: 0, z: 0 }; // identity: down = +z

        let mut out = [0.0; CHANNELS];
        frame_channels(&frame, &mut out);

        assert_eq!(&out[6..9], &[1.0, 0.0, -1.0]);
        assert!((out[9] - 10.0).abs() < 1e-5);
        assert_eq!(out[0], 0.0); // IMU 0 not valid
        assert_eq!(&out[24..28], &[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(&out[31..34], &[0.0, 0.0, 1.0]);
        assert_eq!(&out[28..31], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn old_recordings_are_read_as_4g() {
        assert_eq!(sensor_scales(0), (8_192.0, 16.4));
    }
}
