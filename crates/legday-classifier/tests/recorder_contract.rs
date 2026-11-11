//! Exercise the team's actual encoder against the restored fusion/preprocessor.
//!
//! The `#[path = ...]` lines below compile three source files straight out
//! of `record_app` into this test, so the bytes checked here come from the
//! same encoder the firmware runs, not from a copy of it.
//!
//! Byte layout of one 120-byte frame, for the offsets used below:
//!
//! ```text
//!    0..8    timestamp (us)         14..16   status flags
//!    8..12   sequence number        16..64   4 IMUs x (accel xyz, gyro xyz), i16
//!   12       valid-IMU mask         64..72   4 pressure readings, u16
//!   13       valid-pressure mask    72..120  4 quaternions x (x, y, z), i32
//! ```
// Not every item in the borrowed files is used here.
#![allow(dead_code)]

#[path = "../../record_app/bin/sensors/config.rs"]
mod config;
#[path = "../../record_app/bin/sensors/frame.rs"]
mod frame;
#[path = "../../record_app/bin/sensors/sample.rs"]
mod sample;

use legday_classifier::preprocess::{CHANNELS, frame_channels};
use legday_core::fused::{FusedFrame, FusedWindowBuffer, Quat6Raw, status};
use legday_fusion::frame::FrameFusion;
use sample::{ImuReading, RawImu, SensorSample, Timed};

/// One reading as the recorder produces it: IMU slot 1 disabled, the other
/// three at rest with 1 g along +Y (4096 counts at ±8 g), and every
/// pressure sensor at 2048 counts.
fn measured_sample() -> SensorSample<()> {
    SensorSample {
        read_started_us: 10_000,
        read_completed_us: 12_000,
        imus: core::array::from_fn(|slot| {
            if slot == 1 {
                ImuReading::Disabled
            } else {
                ImuReading::Fresh(Timed {
                    read_started_us: 10_000,
                    read_completed_us: 11_000,
                    value: RawImu {
                        accel: [0, 4096, 0],
                        gyro: [0; 3],
                    },
                })
            }
        }),
        pressure: core::array::from_fn(|_| Timed {
            read_started_us: 11_000,
            read_completed_us: 12_000,
            value: 2048,
        }),
    }
}

#[test]
fn recorder_payload_round_trips_and_unavailable_orientation_stays_unavailable() {
    let encoded = frame::encode(&measured_sample(), 7);
    let decoded = FusedFrame::decode_le(&encoded);
    // Decoding and re-encoding gives back the same bytes.
    assert_eq!(decoded.encode_le(), encoded);
    assert_eq!(decoded.raw.valid_imu_mask, 0b1101); // slots 0, 2, 3; not 1
    assert_eq!(decoded.raw.sequence, 7);
    // No fusion has run, so the quaternions are still zero.
    assert_eq!(decoded.quat6, [Quat6Raw::ZERO; 4]);
    // Channel 1 is IMU 0's y acceleration, 6..12 is all of IMU 1, and
    // 28.. are the gravity directions.
    let mut input = [0.0; CHANNELS];
    frame_channels(&decoded, &mut input);
    assert_eq!(input[1], 1.0); // The recorder's +/-8g scale, not the old +/-16g.
    assert_eq!(&input[6..12], &[0.0; 6]);
    assert_eq!(&input[28..], &[0.0; 12]); // Zero placeholders are not identity.
}

#[test]
fn fused_payload_preserves_raw_bytes_and_populates_only_present_gravity_channels() {
    let encoded = frame::encode(&measured_sample(), 8);
    let mut decoded = FusedFrame::decode_le(&encoded);
    FrameFusion::new(0.1).update(&mut decoded);
    let fused = decoded.encode_le();
    // Fusion may change only the status flags (14..16) and the quaternions
    // (72..120). Slot 1's quaternion is bytes 84..96.
    assert_eq!(&fused[..14], &encoded[..14]);
    assert_eq!(&fused[16..72], &encoded[16..72]);
    assert_eq!(&fused[84..96], &[0; 12]); // Fixed slot 1 remains unavailable.
    assert_eq!(
        decoded.raw.status_flags & status::ORIENTATION_SOURCE_MASK,
        status::ORIENTATION_SOFTWARE
    );
    // Channel 29 is the y component of IMU 0's "down" (gravity is along +Y);
    // 31..34 is IMU 1's "down", which stays zero.
    let mut input = [0.0; CHANNELS];
    frame_channels(&decoded, &mut input);
    assert!((input[29] - 1.0).abs() < 1e-5);
    assert_eq!(&input[31..34], &[0.0; 3]);
}

#[test]
fn one_second_history_keeps_latest_fused_frames_in_chronological_order() {
    const CAPACITY: usize = config::SAMPLE_HZ as usize;
    let mut history = FusedWindowBuffer::<CAPACITY>::new();
    let mut fusion = FrameFusion::new(0.1);
    let mut sample = measured_sample();
    for sequence in 0..CAPACITY + 3 {
        sample.read_started_us = 10_000 + sequence as u64 * 4_444;
        let bytes = frame::encode(&sample, sequence as u32);
        let mut decoded = FusedFrame::decode_le(&bytes);
        fusion.update(&mut decoded);
        history.push(decoded);
    }
    assert_eq!(history.len(), CAPACITY);
    assert_eq!(history.get(0).unwrap().raw.sequence, 3);
    assert_eq!(history.latest().unwrap().raw.sequence, CAPACITY as u32 + 2);
    for index in 0..CAPACITY {
        let frame = history.get(index).unwrap();
        assert_eq!(frame.raw.sequence, index as u32 + 3);
        assert_eq!(frame.raw.valid_imu_mask, 0b1101);
        assert_eq!(
            frame.raw.status_flags & status::ORIENTATION_SOURCE_MASK,
            status::ORIENTATION_SOFTWARE
        );
        assert_eq!(frame.encode_le().len(), 120);
    }
    // Check the actual in-memory sample size as well as the wire encoding:
    // 225 stored samples consume 27,000 bytes, plus window/mutex bookkeeping.
    assert_eq!(core::mem::size_of::<FusedFrame>(), 120);
    assert_eq!(CAPACITY * core::mem::size_of::<FusedFrame>(), 27_000);
}
