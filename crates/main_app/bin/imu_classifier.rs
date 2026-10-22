//! # Live classification on the board
//!
//! Every `CLASSIFY_EVERY_FRAMES` new frames (6 frames ~ 26 ms at 225 Hz),
//! the newest second of the RAM window is turned into model input and run
//! through the CNN. The result, and how long each step took, appear on the
//! status line.
//!
//! The model is `legday_classifier::cnn::DEMO_MODEL`, compiled into flash
//! from `crates/legday-classifier/models/demo.ldm`. Train a new one with
//! `Scripts/train_classifier.py train ... --install-demo` and rebuild. The
//! committed demo was trained on synthetic motion, so its answers on a real
//! IMU are meaningless until it is retrained on real recordings; its timing
//! is real either way.
//!
//! This runs in the main loop, on the same core as sensing. In the final
//! design it belongs on the other core (see `docs/imu-bringup.md`).

use esp_println::println;

use legday_classifier::{
    cnn::{DEMO_MODEL, Model, ModelError, softmax},
    preprocess::window_input,
};

use legday_core::fused::FusedWindowBuffer;
use static_cell::ConstStaticCell;

use super::{WINDOW_FRAMES, now_us};

/// Classify after this many new frames (6 at 225 Hz ~ every 26.7 ms), or
/// less often if a classification takes longer than that (see
/// `next_allowed_us`).
const CLASSIFY_EVERY_FRAMES: u32 = 6;

/// Model input: 40 channels x 75 steps.
const INPUT_FLOATS: usize = 40 * 75;

/// Activation scratch. The default build only needs room for the demo
/// model; the benchmark build needs room for the ~1M-parameter model.
#[cfg(not(feature = "model-bench"))]
const SCRATCH_FLOATS: usize = 8_192;
#[cfg(feature = "model-bench")]
const SCRATCH_FLOATS: usize = 26_624;

/// Most classes a model may have (the logits buffer).
const MAX_CLASSES: usize = 16;

static INPUT: ConstStaticCell<[f32; INPUT_FLOATS]> = ConstStaticCell::new([0.0; INPUT_FLOATS]);
static SCRATCH: ConstStaticCell<[f32; SCRATCH_FLOATS]> =
    ConstStaticCell::new([0.0; SCRATCH_FLOATS]);

/// The most recent answer.
#[derive(Clone, Copy)]
pub struct Prediction {
    pub class: usize,
    pub probability: f32,
}

/// Timings summed over one status period.
#[derive(Clone, Copy, Default)]
pub struct ClassifierTiming {
    pub runs: u32,
    pub input_us_total: u64,
    pub infer_us_total: u64,
    pub infer_us_max: u64,
}

pub struct LiveClassifier {
    pub model: Model<'static>,
    input: &'static mut [f32; INPUT_FLOATS],
    scratch: &'static mut [f32; SCRATCH_FLOATS],
    frames_since_run: u32,

    /// Earliest time the next classification may start. After each run the
    /// classifier waits at least as long as that run took, so it can never
    /// use more than half the CPU and starve sensing, however slow it is.
    next_allowed_us: u64,

    pub last: Option<Prediction>,
    pub timing: ClassifierTiming,
}

impl LiveClassifier {
    /// Parses the demo model and checks it fits the static buffers.
    pub fn new() -> Result<Self, ModelError> {
        let model = Model::parse(&DEMO_MODEL.0)?;

        if model.input_floats() != INPUT_FLOATS || model.classes > MAX_CLASSES {
            return Err(ModelError::BufferSize {
                needed: model.input_floats(),
                available: INPUT_FLOATS,
            });
        }

        if model.scratch_floats() > SCRATCH_FLOATS {
            return Err(ModelError::BufferSize {
                needed: model.scratch_floats(),
                available: SCRATCH_FLOATS,
            });
        }

        println!(
            "Classifier: demo model, {} classes, {} parameters, {} MACs per inference, every {} frames",
            model.classes,
            model.parameters(),
            model.macs(),
            CLASSIFY_EVERY_FRAMES
        );

        Ok(Self {
            model,
            input: INPUT.take(),
            scratch: SCRATCH.take(),
            frames_since_run: 0,
            next_allowed_us: 0,
            last: None,
            timing: ClassifierTiming::default(),
        })
    }

    /// Call after each assembly stage with the number of new frames. Runs at
    /// most one classification per call, once the window holds a full second.
    pub fn on_new_frames(&mut self, new_frames: usize, window: &FusedWindowBuffer<WINDOW_FRAMES>) {
        self.frames_since_run = self.frames_since_run.saturating_add(new_frames as u32);

        if self.frames_since_run < CLASSIFY_EVERY_FRAMES
            || !window.is_full()
            || now_us() < self.next_allowed_us
        {
            return;
        }

        self.frames_since_run = 0;

        // Window -> standardized model input.
        let started = now_us();
        let model = &self.model;

        if window_input(
            window,
            model.channels,
            model.steps,
            model.decimation,
            model.mean,
            model.std,
            &mut self.input[..],
        )
        .is_err()
        {
            return;
        }

        let input_done = now_us();

        // Input -> scores -> probabilities.
        let mut logits = [0.0_f32; MAX_CLASSES];
        let classes = model.classes;

        if model
            .infer(
                &self.input[..],
                &mut self.scratch[..],
                &mut logits[..classes],
                None,
            )
            .is_err()
        {
            return;
        }

        let finished = now_us();
        let infer_us = finished - input_done;
        self.next_allowed_us = finished + (finished - started);
        softmax(&mut logits[..classes]);

        let class = (0..classes)
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap_or(0);
        self.last = Some(Prediction {
            class,
            probability: logits[class],
        });

        self.timing.runs += 1;
        self.timing.input_us_total += input_done - started;
        self.timing.infer_us_total += infer_us;
        self.timing.infer_us_max = self.timing.infer_us_max.max(infer_us);
    }

    /// The input and scratch buffers, lent to the diagnostics' model benchmark.
    pub fn buffers(&mut self) -> (&mut [f32], &mut [f32]) {
        (&mut self.input[..], &mut self.scratch[..])
    }
}
