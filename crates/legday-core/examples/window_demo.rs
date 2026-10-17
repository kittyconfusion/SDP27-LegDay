use legday_core::{
    FEATURE_COUNT, IMU_COUNT, INFERENCE_HOP_SAMPLES, Imu6Raw, SAMPLE_RATE_HZ, SensorFrame,
    WINDOW_SAMPLES, WindowBuffer,
};

fn synthetic_frame(sequence: u32) -> SensorFrame {
    let phase = (sequence % 40) as i16;
    SensorFrame {
        timestamp_us: u64::from(sequence) * 1_000_000 / SAMPLE_RATE_HZ as u64,
        sequence,
        valid_imu_mask: 0b1111,
        valid_pressure_mask: 0b1111,
        status_flags: 0,
        imus: core::array::from_fn(|imu| Imu6Raw {
            accel: [phase + imu as i16, phase - 20, 1_000],
            gyro: [phase * 2, -phase, imu as i16],
        }),
        pressure_adc: [800, 1_600, 900, 1_700],
    }
}

fn main() {
    let mut window = WindowBuffer::<WINDOW_SAMPLES>::new();
    for sequence in 0..(WINDOW_SAMPLES as u32 + 3 * INFERENCE_HOP_SAMPLES as u32) {
        window.push(synthetic_frame(sequence));
        if window.is_full() && sequence % INFERENCE_HOP_SAMPLES as u32 == 0 {
            let mut features = [0_i32; FEATURE_COUNT];
            window.latest().unwrap().write_features_i32(&mut features);
            println!(
                "inference-ready: sequence={sequence}, window={} samples, latest={} features",
                window.len(),
                features.len()
            );
        }
    }
    println!("IMUs represented: {IMU_COUNT}; sample rate: {SAMPLE_RATE_HZ} Hz");
}
