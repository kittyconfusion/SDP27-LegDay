//! Timestamp reconstruction for samples that arrive in bursts.
//!
//! An IMU samples on its own crystal and queues results in a hardware FIFO.
//! The MCU drains that FIFO whenever its main loop gets around to it, so a
//! drain may return 1 sample or 10, and the drain happens somewhere between
//! 0.5 ms and 10 ms after the newest sample was taken. The MCU therefore
//! never observes the true sample instants directly; it only knows:
//!
//! 1. the nominal sample period (from the IMU's divider setting);
//! 2. that each sample was taken *before* the drain that returned it;
//! 3. that consecutive samples are one period apart on the IMU's clock.
//!
//! [`SampleClock`] turns those facts into a smooth timestamp per sample:
//!
//! - It measures the real period by dividing elapsed MCU time by the number
//!   of samples seen, which cancels the IMU crystal's error (about 1.5 % on
//!   the bench) after a few seconds.
//! - For each sample it takes the earlier of "previous timestamp + period"
//!   and "drain time minus the samples that came after it". The first keeps
//!   spacing regular; the second stops the estimate from running ahead of
//!   reality when the loop is late.
//!
//! The result tracks the IMU clock with sub-millisecond jitter instead of
//! the loop-latency jitter of naive back-calculation.

/// Fixed-point scale for the period estimate (1/256 microsecond).
const PERIOD_SCALE: u64 = 256;

/// Samples to observe before trusting the measured period over the nominal one.
const MIN_SAMPLES_FOR_MEASURED_PERIOD: u64 = 512;

/// Hard limit on how far a reconstructed timestamp may trail the drain time,
/// so a mis-estimated period can never make timestamps fall unboundedly behind.
const MAX_LAG_US: u64 = 20_000;

/// Reconstructs sample timestamps from burst drain times.
#[derive(Clone, Copy, Debug)]
pub struct SampleClock {
    /// Period from the IMU divider setting, in microseconds.
    nominal_period_us: u64,
    /// Drain time and running sample count at the first drain; the baseline
    /// for the measured period.
    anchor: Option<(u64, u64)>,
    /// Total samples passed through this clock.
    samples: u64,
    /// Timestamp handed out for the previous sample.
    last_us: Option<u64>,
}

impl SampleClock {
    pub const fn new(nominal_period_us: u64) -> Self {
        Self {
            nominal_period_us,
            anchor: None,
            samples: 0,
            last_us: None,
        }
    }

    /// Forgets all history, for example after the IMU is reconfigured.
    pub fn reset(&mut self) {
        *self = Self::new(self.nominal_period_us);
    }

    /// Restarts the rate measurement after samples were lost (a FIFO
    /// overflow or resync). The period estimate assumes every sample was
    /// seen, so counting elapsed time against missing samples would inflate
    /// it. Timestamps stay monotonic: the previous one is kept.
    pub fn resync(&mut self) {
        self.anchor = None;
        self.samples = 0;
    }

    /// Current period estimate in 1/256 µs: measured once enough samples
    /// have passed, nominal before that.
    fn period_scaled(&self, now_us: u64) -> u64 {
        match self.anchor {
            Some((anchor_us, anchor_samples))
                if self.samples - anchor_samples >= MIN_SAMPLES_FOR_MEASURED_PERIOD =>
            {
                (now_us - anchor_us) * PERIOD_SCALE / (self.samples - anchor_samples)
            }
            _ => self.nominal_period_us * PERIOD_SCALE,
        }
    }

    /// Period estimate in microseconds (for display).
    pub fn period_us(&self, now_us: u64) -> u64 {
        self.period_scaled(now_us) / PERIOD_SCALE
    }

    /// Registers a burst of `count` samples drained at `drain_us` and returns
    /// their timestamps, oldest first, through the provided callback.
    pub fn burst(&mut self, drain_us: u64, count: usize, mut emit: impl FnMut(u64)) {
        if count == 0 {
            return;
        }

        // The period is measured against the anchor *before* this burst is
        // counted, so the division never sees zero samples.
        let period_scaled = self.period_scaled(drain_us);
        if self.anchor.is_none() {
            self.anchor = Some((drain_us, self.samples));
        }

        for index in 0..count {
            // Upper bound: this sample was taken no later than the drain,
            // minus one period for every sample that arrived after it.
            let after = (count - 1 - index) as u64;
            let bound_us = drain_us.saturating_sub(after * period_scaled / PERIOD_SCALE);

            let timestamp_us = match self.last_us {
                None => bound_us,
                Some(last_us) => {
                    // Regular spacing, unless that would run ahead of the
                    // bound, and never further behind the bound than MAX_LAG.
                    let expected_us = last_us + period_scaled / PERIOD_SCALE;
                    expected_us
                        .min(bound_us)
                        .max(bound_us.saturating_sub(MAX_LAG_US))
                        .max(last_us + 1)
                }
            };

            self.last_us = Some(timestamp_us);
            self.samples += 1;
            emit(timestamp_us);
        }
    }
}

#[cfg(all(test, not(target_os = "none")))]
mod tests {
    use super::*;
    extern crate std;
    use std::vec::Vec;

    /// Simulates an IMU with a true period of 4375 µs drained by a loop with
    /// a varying latency, and checks the reconstruction against the truth.
    #[test]
    fn reconstructed_timestamps_track_true_sample_times() {
        const TRUE_PERIOD_US: u64 = 4_375;
        let mut clock = SampleClock::new(4_444);
        let mut estimates: Vec<u64> = Vec::new();
        let mut truth: Vec<u64> = Vec::new();

        let mut sample = 0_u64;
        let mut pseudo_random = 12_345_u64;
        while sample < 20_000 {
            // Burst of 1..=8 samples, drained 500..=8000 µs after the newest.
            pseudo_random = pseudo_random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let count = (pseudo_random >> 60) as usize % 8 + 1;
            let latency = 500 + (pseudo_random >> 40) % 7_500;
            let newest_true = (sample + count as u64 - 1) * TRUE_PERIOD_US + 1_000_000;
            let drain_us = newest_true + latency;

            for offset in 0..count as u64 {
                truth.push((sample + offset) * TRUE_PERIOD_US + 1_000_000);
            }
            clock.burst(drain_us, count, |timestamp| estimates.push(timestamp));
            sample += count as u64;
        }

        // Timestamps must be strictly increasing.
        assert!(estimates.windows(2).all(|pair| pair[1] > pair[0]));

        // After convergence the spacing jitter is well below the loop latency
        // jitter (7.5 ms) that naive back-calculation would show.
        let settled = &estimates[5_000..];
        let settled_truth = &truth[5_000..];
        let offsets: Vec<i64> = settled
            .iter()
            .zip(settled_truth)
            .map(|(estimate, actual)| *estimate as i64 - *actual as i64)
            .collect();
        let min_offset = *offsets.iter().min().unwrap();
        let max_offset = *offsets.iter().max().unwrap();
        assert!(
            max_offset - min_offset < 1_500,
            "offset spread {} µs (min {min_offset}, max {max_offset})",
            max_offset - min_offset
        );

        // The measured period converges on the true one.
        let period = clock.period_us(truth[truth.len() - 1] + 2_000);
        assert!(
            (period as i64 - TRUE_PERIOD_US as i64).abs() <= 2,
            "period {period}"
        );
    }

    /// A gap with lost samples must not corrupt the period estimate.
    #[test]
    fn resync_keeps_period_honest_after_lost_samples() {
        const TRUE_PERIOD_US: u64 = 4_375;
        let mut clock = SampleClock::new(4_444);
        let mut last = 0;
        // Two seconds of normal operation, one sample per drain.
        for sample in 0..600_u64 {
            clock.burst(sample * TRUE_PERIOD_US + 700, 1, |timestamp| {
                last = timestamp
            });
        }
        // Two seconds of silence in which ~460 samples were lost.
        clock.resync();
        let resume_us = 1_060 * TRUE_PERIOD_US + 700;
        for sample in 0..1_000_u64 {
            let drain_us = resume_us + sample * TRUE_PERIOD_US;
            clock.burst(drain_us, 1, |timestamp| {
                assert!(timestamp > last);
                last = timestamp;
            });
        }
        let period = clock.period_us(resume_us + 1_000 * TRUE_PERIOD_US);
        assert!(
            (period as i64 - TRUE_PERIOD_US as i64).abs() <= 2,
            "period {period}"
        );
    }

    #[test]
    fn empty_burst_is_ignored() {
        let mut clock = SampleClock::new(5_000);
        let mut emitted = 0;
        clock.burst(1_000, 0, |_| emitted += 1);
        assert_eq!(emitted, 0);
    }
}
