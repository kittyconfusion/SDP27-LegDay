//! # Raw IMU pipeline: up to four IMUs into one 120-byte frame per sample
//!
//! Every pass of the main loop runs the same three stages:
//!
//! ```text
//!   1. acquire   drain every IMU's FIFO over I2C into that IMU's queue   (bus work only)
//!   2. pressure  read the four ADC channels                              (main loop, ~32 µs)
//!   3. assemble  for each instant every IMU has a sample for:
//!                  correct gyro bias -> update quaternion   (all IMUs together)
//!                  -> build one FusedFrame -> push it into the 1 s window
//! ```
//!
//! All bus work happens first and all maths afterwards. That keeps the bus
//! stage one contiguous block, which is what a later move to async I2C, SPI
//! with DMA, or the second core needs: start every transfer, then compute.
//!
//! **Pairing.** Each IMU has its own crystal, so over time one IMU produces
//! slightly more samples than another (1-2 % apart). A frame is built only
//! when every connected IMU has a queued sample, by taking the oldest from
//! each. If one IMU gets more than `MAX_SKEW` samples ahead, its surplus is
//! dropped and the next frame carries the `IMU_REALIGNED` flag. With one IMU
//! this never triggers; the code is here so four IMUs need no redesign.
//!
//! **Raw stays raw.** Frames store the sensor counts exactly as read. Bias
//! correction is applied only to the copy fed to the orientation filter, so
//! recorded data can always be reprocessed later.

use core::f32::consts::PI;

use esp_hal::delay::Delay;
use esp_println::println;
use heapless::Deque;

use legday_classifier::fusion::{GyroBias, ImuScale, Madgwick, accel_trust};

use legday_core::{
    IMU_COUNT, Imu6Raw,
    clock::SampleClock,
    fused::{FusedFrame, FusedWindowBuffer, Quat6Raw, status},
};

use legday_icm20948::{Config as ImuConfig, RawSample};

use super::{Imu, WINDOW_FRAMES, now_us};

/// Samples each IMU may queue between frames. One pass normally adds 1-2;
/// the headroom covers a long pause such as an SD flush or a slow
/// classification (128 samples = 0.57 s at 225 Hz).
const QUEUE_CAPACITY: usize = 128;

/// Samples read from the FIFO per I2C burst.
const DRAIN_CHUNK: usize = 32;

/// How many samples one IMU may run ahead of the slowest before its surplus
/// is dropped to bring them back into step.
const MAX_SKEW: usize = 3;

/// One raw sample and the time it was taken.
#[derive(Clone, Copy)]
struct TimedSample {
    timestamp_us: u64,
    sample: RawSample,
}

/// Running totals for one IMU, printed on the status line.
#[derive(Clone, Copy, Default)]
pub struct SlotStats {
    pub fifo_max_bytes: u16,
    pub overflows: u32,
    pub bus_errors: u32,
    pub realigns: u32,
}

/// Everything that belongs to one physical IMU.
pub struct ImuSlot {
    /// The driver. Public so the diagnostics can borrow it.
    pub imu: Imu,

    /// Orientation filter, and whether it has been started from gravity yet.
    filter: Madgwick,
    filter_seeded: bool,

    /// Learns this IMU's gyro offset whenever it is still. Kept across
    /// restarts, because the offset belongs to the sensor, not the session.
    bias: GyroBias,

    /// Turns burst drains into per-sample timestamps and measures this IMU's
    /// real sample period (its crystal is not exactly the nominal rate).
    clock: SampleClock,

    /// Sample period used to integrate the gyro, from `clock`, in seconds.
    dt_s: f32,

    /// Samples drained but not yet built into a frame.
    queue: Deque<TimedSample, QUEUE_CAPACITY>,

    /// Mark the next frame from this IMU: samples were lost before it.
    overflow_pending: bool,

    pub stats: SlotStats,
}

impl ImuSlot {
    fn new(imu: Imu, beta: f32, scale: &ImuScale, period_us: u64) -> Self {
        Self {
            imu,
            filter: Madgwick::new(beta),
            filter_seeded: false,
            bias: GyroBias::new(scale.gyro_lsb_per_dps),
            clock: SampleClock::new(period_us),
            dt_s: period_us as f32 * 1e-6,
            queue: Deque::new(),
            overflow_pending: false,
            stats: SlotStats::default(),
        }
    }

    /// Stage 1 for this IMU: empty its FIFO into the queue, in bursts of
    /// `DRAIN_CHUNK`, stopping early only if the queue is nearly full (the
    /// rest then waits in the FIFO for the next pass). Returns samples read.
    fn drain_all(&mut self) -> usize {
        let mut total = 0;

        while QUEUE_CAPACITY - self.queue.len() >= DRAIN_CHUNK {
            let read = self.drain();
            total += read;

            if read < DRAIN_CHUNK {
                break; // FIFO is empty
            }
        }

        total
    }

    /// Reads one burst of up to `DRAIN_CHUNK` samples from the FIFO and
    /// queues them with timestamps. Returns the number of samples read.
    fn drain(&mut self) -> usize {
        let mut samples = [RawSample::default(); DRAIN_CHUNK];
        let drain_us = now_us();

        let read = match self.imu.read_raw_fifo_samples(&mut samples) {
            Ok(read) => read,
            Err(error) => {
                self.stats.bus_errors += 1;

                if self.stats.bus_errors <= 5 {
                    println!("I2C error while draining IMU FIFO: {:?}", error);
                }
                return 0;
            }
        };

        self.stats.fifo_max_bytes = self.stats.fifo_max_bytes.max(read.fifo_bytes);

        if read.overflowed {
            self.stats.overflows += 1;
            self.overflow_pending = true;
            self.clock.resync(); // samples were lost; re-measure the rate
        }

        // Split the borrows: the clock hands out timestamps while the queue
        // is filled.
        let Self {
            clock,
            queue,
            overflow_pending,
            ..
        } = self;
        let mut index = 0;

        clock.burst(drain_us, read.samples, |timestamp_us| {
            if queue.is_full() {
                // Assembly fell far behind; keep the newest data.
                queue.pop_front();
                *overflow_pending = true;
            }

            let _ = queue.push_back(TimedSample {
                timestamp_us,
                sample: samples[index],
            });

            index += 1;
        });

        self.dt_s = self.clock.period_us(drain_us) as f32 * 1e-6;
        read.samples
    }

    /// Stage 3 for this IMU: bias-correct the gyro, advance the orientation
    /// filter by one sample, and return the quaternion for the frame.
    fn fuse(&mut self, sample: &RawSample, scale: &ImuScale) -> Quat6Raw {
        let accel_g = scale.accel_g(sample.accel);

        // Start the filter at the measured tilt rather than at "level", so
        // it does not spend seconds converging.
        if !self.filter_seeded {
            self.filter.reset_to_gravity(accel_g);
            self.filter_seeded = true;
        }

        // Gyro counts minus the learned offset, then counts -> rad/s.
        let gyro_counts = self.bias.correct(sample.gyro, accel_g);
        let to_rad_s = PI / 180.0 / scale.gyro_lsb_per_dps;
        let gyro_rad_s = [
            gyro_counts[0] * to_rad_s,
            gyro_counts[1] * to_rad_s,
            gyro_counts[2] * to_rad_s,
        ];

        // Trust the accelerometer less the further it is from 1 g.
        self.filter
            .update_with_gain(gyro_rad_s, accel_g, self.dt_s, accel_trust(accel_g));

        let [x, y, z] = self.filter.quat6_q30();

        Quat6Raw { x, y, z }
    }

    pub fn quaternion(&self) -> [f32; 4] {
        self.filter.quaternion_positive_w()
    }

    pub fn bias(&self) -> &GyroBias {
        &self.bias
    }

    pub fn period_us(&self) -> u64 {
        self.clock.period_us(now_us())
    }
}

/// Stage timings, summed over one status period.
#[derive(Clone, Copy, Default)]
pub struct PassTiming {
    /// Passes that ran stage 1, and the total and longest bus time.
    pub passes: u32,
    pub bus_us_total: u64,
    pub bus_us_max: u64,

    /// Frames built, and total time for fusion (all IMUs) and frame building.
    pub frames: u32,
    pub fusion_us_total: u64,
    pub frame_us_total: u64,
}

/// Up to four IMUs and everything needed to turn them into frames.
pub struct Pipeline {
    /// Slot `n` fills IMU `n` of every frame; `None` = not connected.
    pub slots: [Option<ImuSlot>; IMU_COUNT],

    config: ImuConfig,
    scale: ImuScale,
    beta: f32,
    sequence: u32,

    /// Mark the next frame: IMUs were realigned before it.
    realign_pending: bool,

    pub timing: PassTiming,
}

impl Pipeline {
    pub fn new(config: ImuConfig, beta: f32) -> Self {
        Self {
            slots: [const { None }; IMU_COUNT],
            scale: ImuScale {
                accel_lsb_per_g: config.accel_range.lsb_per_g(),
                gyro_lsb_per_dps: config.gyro_range.lsb_per_dps(),
            },
            config,
            beta,
            sequence: 0,
            realign_pending: false,
            timing: PassTiming::default(),
        }
    }

    pub const fn scale(&self) -> &ImuScale {
        &self.scale
    }

    /// Puts a detected IMU into frame slot `index`.
    pub fn add_imu(&mut self, index: usize, imu: Imu) {
        let period_us = u64::from(self.config.period_us());
        self.slots[index] = Some(ImuSlot::new(imu, self.beta, &self.scale, period_us));
    }

    /// Bit `n` set = slot `n` has an IMU.
    pub fn connected_mask(&self) -> u8 {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_some())
            .fold(0, |mask, (index, _)| mask | (1 << index))
    }

    /// The lowest-numbered connected IMU, which also sets frame timestamps.
    pub fn primary(&mut self) -> Option<&mut ImuSlot> {
        self.slots.iter_mut().flatten().next()
    }

    /// Resets, configures and starts raw FIFO streaming on every IMU, and
    /// forgets queued samples and filter state. Learned gyro biases are kept.
    pub fn start(&mut self, delay: &mut Delay) {
        let config = self.config;

        for slot in self.slots.iter_mut().flatten() {
            slot.imu.reset(delay).unwrap();
            slot.imu.configure(&config).unwrap();
            slot.imu.start_raw_fifo().unwrap();

            slot.queue.clear();
            slot.clock.reset();
            slot.filter.reset();
            slot.filter_seeded = false;
            slot.overflow_pending = false;
        }

        self.realign_pending = false;
    }

    /// Stops streaming on every IMU.
    pub fn stop(&mut self) {
        for slot in self.slots.iter_mut().flatten() {
            let _ = slot.imu.stop_raw_fifo();
        }
    }

    /// Stage 1: drain every IMU, back to back. Returns samples read.
    pub fn acquire(&mut self) -> usize {
        let started_us = now_us();
        let samples: usize = self
            .slots
            .iter_mut()
            .flatten()
            .map(ImuSlot::drain_all)
            .sum();
        let bus_us = now_us() - started_us;

        self.timing.passes += 1;
        self.timing.bus_us_total += bus_us;
        self.timing.bus_us_max = self.timing.bus_us_max.max(bus_us);

        samples
    }

    /// Stage 3: build one frame for every instant all IMUs have a sample
    /// for, and push it into the window. Returns the number of frames.
    pub fn assemble(
        &mut self,
        pressure: [u16; 4],
        pressure_mask: u8,
        window: &mut FusedWindowBuffer<WINDOW_FRAMES>,
    ) -> usize {
        let mut frames = 0;

        loop {
            // The emptiest queue decides how many complete frames exist.
            let Some(shortest) = self
                .slots
                .iter()
                .flatten()
                .map(|slot| slot.queue.len())
                .min()
            else {
                return frames; // no IMUs connected
            };

            if shortest == 0 {
                return frames;
            }

            // Bring any IMU that ran ahead back into step.
            for slot in self.slots.iter_mut().flatten() {
                while slot.queue.len() > shortest + MAX_SKEW {
                    slot.queue.pop_front();
                    slot.stats.realigns += 1;
                    self.realign_pending = true;
                }
            }

            // Take one sample from every IMU and update every quaternion.
            let fusion_started_us = now_us();
            let mut frame = FusedFrame::ZERO;
            let mut gyro_level = 3;

            for (index, slot) in self.slots.iter_mut().enumerate() {
                let Some(slot) = slot else { continue };
                let timed = slot.queue.pop_front().unwrap();

                // The first connected IMU is the time reference.
                if frame.raw.valid_imu_mask == 0 {
                    frame.raw.timestamp_us = timed.timestamp_us;
                }

                frame.raw.imus[index] = Imu6Raw {
                    accel: timed.sample.accel,
                    gyro: timed.sample.gyro,
                };

                frame.quat6[index] = slot.fuse(&timed.sample, &self.scale);
                frame.raw.valid_imu_mask |= 1 << index;

                // Report the least-calibrated IMU's level for the frame.
                gyro_level = gyro_level.min(slot.bias.level());

                if core::mem::take(&mut slot.overflow_pending) {
                    frame.raw.status_flags |= status::FIFO_OVERFLOW;
                }
            }

            let fusion_us = now_us() - fusion_started_us;

            // Fill in everything that is not per-IMU, then store the frame.
            let frame_started_us = now_us();

            frame.raw.sequence = self.sequence;
            frame.raw.pressure_adc = pressure;
            frame.raw.valid_pressure_mask = pressure_mask;
            frame.raw.status_flags |= status::TIMESTAMP_RECONSTRUCTED
                | status::ORIENTATION_SOFTWARE
                | status::RANGES_RECORDED
                | (gyro_level << status::GYRO_ACCURACY_SHIFT)
                | ((self.config.accel_range as u16) << status::ACCEL_RANGE_SHIFT)
                | ((self.config.gyro_range as u16) << status::GYRO_RANGE_SHIFT);

            if core::mem::take(&mut self.realign_pending) {
                frame.raw.status_flags |= status::IMU_REALIGNED;
            }

            window.push(frame);

            self.sequence = self.sequence.wrapping_add(1);

            self.timing.frames += 1;
            self.timing.fusion_us_total += fusion_us;
            self.timing.frame_us_total += now_us() - frame_started_us;

            frames += 1;
        }
    }
}
