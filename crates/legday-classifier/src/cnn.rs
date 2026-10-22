//! A small `no_std` inference engine for the 1-D CNN trained by
//! `Scripts/train_classifier.py`.
//!
//! The training script exports a `.ldm` file: a short header, the class
//! names, the input standardization, and then each layer's weights. This
//! module reads that file **in place**, so the weights stay in flash and do not
//! use any RAM, then runs the layers one after another:
//!
//! ```text
//!   input (channels x steps)
//!     -> conv + ReLU  (x4, some with stride 2: fewer, richer time steps)
//!     -> [self-attention, if the model has it]
//!     -> average over time  (the "embedding": one number per feature)
//!     -> dense            (one score, or "logit", per class)
//!     -> softmax          (scores -> probabilities that sum to 1)
//! ```
//!
//! Activations live in a caller-provided scratch buffer, sized by
//! [`Model::scratch_floats`], so nothing is allocated at run time.
//!
//! The arithmetic follows PyTorch's definitions exactly (convolution with no
//! padding, `y = W x + b` for dense layers, scaled dot-product attention with
//! a residual connection). Only the order of additions differs, which moves
//! results by about one part in a million.

use crate::fusion::inv_sqrt;

#[cfg(target_endian = "big")]
compile_error!("model files store little-endian f32 and are read in place");

/// Most layers a model may have (the current design uses 7).
pub const MAX_LAYERS: usize = 12;

/// Bytes per class name in the file (ASCII, zero-padded).
pub const CLASS_NAME_BYTES: usize = 8;

/// Layer kinds, matching `LAYER_*` in `Scripts/legday_ml.py`.
const LAYER_CONV_RELU: u32 = 1;
const LAYER_GLOBAL_AVG_POOL: u32 = 2;
const LAYER_DENSE: u32 = 3;
const LAYER_SELF_ATTENTION: u32 = 4;

/// Wraps bytes so they are 16-byte aligned, which lets `f32` weights be read
/// straight out of an `include_bytes!` model, and lets the S3 load them four
/// at a time with its 128-bit vector load (which needs 16-byte alignment):
/// `static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!("model.ldm"));`
#[repr(C, align(16))]
pub struct Aligned<Bytes: ?Sized>(pub Bytes);

/// The model the firmware runs by default. Replace `models/demo.ldm` (or run
/// `train_classifier.py train ... --install-demo`) and rebuild to change it.
/// The committed one was trained on synthetic data.
pub static DEMO_MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!("../models/demo.ldm"));

/// One window exported by the training script together with the logits
/// PyTorch computed for it. [`check_against_golden`] replays it, on the Mac
/// in `cargo test` and on the board at start-up.
pub static DEMO_GOLDEN: &[u8] = include_bytes!("../models/demo_golden.bin");

/// Size of the demo model file, for callers that copy it into RAM.
pub const DEMO_MODEL_BYTES: usize = include_bytes!("../models/demo.ldm").len();

/// Default floats of scratch the convolution may use to gather input
/// columns (see `conv_relu`). Larger means each layer's weights are read
/// fewer times; 16,384 floats (64 kB) covers the demo model's first layer in
/// two passes. That matters when weights are read from flash; with weights
/// in RAM a smaller budget costs nothing (see [`Model::set_column_budget`]).
pub const COLUMN_BUDGET_FLOATS: usize = 16_384;

/// Why a model file was rejected or an inference could not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    /// The file ended in the middle of something.
    Truncated,
    /// The file does not start with `LDM1`.
    BadMagic,
    UnsupportedVersion(u32),
    /// The bytes are not 4-byte aligned (wrap them in [`Aligned`]).
    Misaligned,
    UnknownLayer(u32),
    TooManyLayers,
    /// A layer's input does not match the previous layer's output.
    ShapeMismatch {
        layer: usize,
    },
    /// The last layer must be a dense layer producing one score per class.
    BadOutput,
    /// The caller's buffer is the wrong size.
    BufferSize {
        needed: usize,
        available: usize,
    },
}

/// One layer, with its weights borrowed from the model file.
#[derive(Clone, Copy, Debug)]
enum Layer<'m> {
    /// Convolution along time followed by ReLU. Weights are
    /// `[output][input][kernel]`, as PyTorch stores them.
    ConvRelu {
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        weights: &'m [f32],
        bias: &'m [f32],
    },

    /// Single-head self-attention across time steps, plus a residual
    /// connection. Each projection is a `dim x dim` matrix and bias.
    SelfAttention {
        dim: usize,
        query: (&'m [f32], &'m [f32]),
        key: (&'m [f32], &'m [f32]),
        value: (&'m [f32], &'m [f32]),
        output: (&'m [f32], &'m [f32]),
    },

    /// Average every channel over time.
    GlobalAvgPool,

    /// Fully connected layer, weights `[output][input]`.
    Dense {
        input: usize,
        output: usize,
        weights: &'m [f32],
        bias: &'m [f32],
    },
}

/// A parsed model, borrowing its weights from the file's bytes.
pub struct Model<'m> {
    /// Input shape and how the window was reduced to it.
    pub channels: usize,
    pub steps: usize,
    pub decimation: usize,
    pub classes: usize,

    /// Per-channel standardization learned during training.
    pub mean: &'m [f32],
    pub std: &'m [f32],

    class_names: &'m [u8],
    layers: [Option<Layer<'m>>; MAX_LAYERS],

    /// Largest activation (channels x steps) any layer produces.
    max_activation: usize,
    /// Extra room the attention layer needs (Q, K, V, one score row, one vector).
    attention_floats: usize,
    /// Room for gathered input columns in the convolution, for the current
    /// budget, and the budget itself.
    column_floats: usize,
    column_budget: usize,
    /// Size of the averaged features fed to the final dense layer.
    embedding_size: usize,
    parameters: usize,
    macs: usize,
}

/// Reads little-endian values from the model file in order.
struct Reader<'m> {
    bytes: &'m [u8],
    position: usize,
    /// Version 2 files pad every float array to a 16-byte boundary.
    pad_floats_to_16: bool,
}

impl<'m> Reader<'m> {
    fn take(&mut self, count: usize) -> Result<&'m [u8], ModelError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(ModelError::Truncated)?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or(ModelError::Truncated)?;
        self.position = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, ModelError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn usize(&mut self) -> Result<usize, ModelError> {
        Ok(self.u32()? as usize)
    }

    /// `count` f32 values, borrowed in place.
    fn f32s(&mut self, count: usize) -> Result<&'m [f32], ModelError> {
        if self.pad_floats_to_16 {
            let padding = (16 - self.position % 16) % 16;
            self.take(padding)?;
        }

        let bytes = self.take(count.checked_mul(4).ok_or(ModelError::Truncated)?)?;

        if bytes.as_ptr() as usize % core::mem::align_of::<f32>() != 0 {
            return Err(ModelError::Misaligned);
        }

        // SAFETY: the pointer is aligned for f32 (checked just above), the
        // slice covers exactly `count * 4` bytes that live as long as `'m`,
        // every bit pattern is a valid f32, and the target is little-endian
        // (checked at compile time), matching how the file was written.
        Ok(unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), count) })
    }

    /// A `rows x columns` matrix followed by a bias of `rows`.
    fn linear(
        &mut self,
        rows: usize,
        columns: usize,
    ) -> Result<(&'m [f32], &'m [f32]), ModelError> {
        Ok((self.f32s(rows * columns)?, self.f32s(rows)?))
    }
}

impl<'m> Model<'m> {
    /// Parses a `.ldm` file and checks every layer's shape against the one
    /// before it, so a bad file is rejected here rather than mid-inference.
    pub fn parse(bytes: &'m [u8]) -> Result<Self, ModelError> {
        let mut reader = Reader {
            bytes,
            position: 0,
            pad_floats_to_16: false,
        };

        if reader.take(4)? != b"LDM1" {
            return Err(ModelError::BadMagic);
        }

        // Version 1: float arrays only 4-byte aligned. Version 2: every float
        // array starts on a 16-byte boundary, for the S3's vector loads.
        let version = reader.u32()?;
        if version != 1 && version != 2 {
            return Err(ModelError::UnsupportedVersion(version));
        }
        reader.pad_floats_to_16 = version >= 2;

        let channels = reader.usize()?;
        let steps = reader.usize()?;
        let decimation = reader.usize()?;
        let classes = reader.usize()?;
        let layer_count = reader.usize()?;

        if layer_count > MAX_LAYERS {
            return Err(ModelError::TooManyLayers);
        }

        let class_names = reader.take(classes * CLASS_NAME_BYTES)?;
        let mean = reader.f32s(channels)?;
        let std = reader.f32s(channels)?;

        let mut model = Self {
            channels,
            steps,
            decimation,
            classes,
            mean,
            std,
            class_names,
            layers: [None; MAX_LAYERS],
            max_activation: 0,
            attention_floats: 0,
            column_floats: 0,
            column_budget: COLUMN_BUDGET_FLOATS,
            embedding_size: 0,
            parameters: 0,
            macs: 0,
        };

        // Walk the layers, tracking the activation shape (features x steps).
        let (mut features, mut length) = (channels, steps);

        for index in 0..layer_count {
            let kind = reader.u32()?;
            let parameters = [
                reader.usize()?,
                reader.usize()?,
                reader.usize()?,
                reader.usize()?,
                reader.usize()?,
            ];

            let layer = match kind {
                LAYER_CONV_RELU => {
                    let [input, output, kernel, stride, dilation] = parameters;
                    let span = dilation * (kernel.max(1) - 1) + 1;

                    if input != features || length < span || stride == 0 || kernel == 0 {
                        return Err(ModelError::ShapeMismatch { layer: index });
                    }

                    let (weights, bias) = reader.linear(output, input * kernel)?;

                    length = (length - span) / stride + 1;
                    features = output;
                    model.parameters += weights.len() + bias.len();
                    model.macs += length * output * input * kernel;
                    model.column_floats = model
                        .column_floats
                        .max(column_tile(input * kernel, length, COLUMN_BUDGET_FLOATS).1);

                    Layer::ConvRelu {
                        input,
                        output,
                        kernel,
                        stride,
                        dilation,
                        weights,
                        bias,
                    }
                }

                LAYER_SELF_ATTENTION => {
                    let dim = parameters[0];

                    if dim != features {
                        return Err(ModelError::ShapeMismatch { layer: index });
                    }

                    let query = reader.linear(dim, dim)?;
                    let key = reader.linear(dim, dim)?;
                    let value = reader.linear(dim, dim)?;
                    let output = reader.linear(dim, dim)?;

                    model.attention_floats =
                        model.attention_floats.max(3 * dim * length + length + dim);
                    model.parameters += 4 * (dim * dim + dim);
                    model.macs += 4 * length * dim * dim + 2 * length * length * dim;

                    Layer::SelfAttention {
                        dim,
                        query,
                        key,
                        value,
                        output,
                    }
                }

                LAYER_GLOBAL_AVG_POOL => {
                    length = 1;
                    model.embedding_size = features;
                    Layer::GlobalAvgPool
                }

                LAYER_DENSE => {
                    let [input, output, ..] = parameters;

                    if input != features || length != 1 {
                        return Err(ModelError::ShapeMismatch { layer: index });
                    }

                    let (weights, bias) = reader.linear(output, input)?;

                    features = output;
                    model.parameters += weights.len() + bias.len();
                    model.macs += input * output;

                    Layer::Dense {
                        input,
                        output,
                        weights,
                        bias,
                    }
                }

                other => return Err(ModelError::UnknownLayer(other)),
            };

            model.max_activation = model.max_activation.max(features * length);
            model.layers[index] = Some(layer);
        }

        // The network must end in one score per class.
        match model.layers[..layer_count].last() {
            Some(Some(Layer::Dense { output, .. })) if *output == classes => Ok(model),
            _ => Err(ModelError::BadOutput),
        }
    }

    /// Changes how many floats the convolution may use for gathered input
    /// columns (default [`COLUMN_BUDGET_FLOATS`]). A smaller budget needs
    /// less scratch but reads each layer's weights more often, which is only
    /// expensive when the weights are in flash.
    pub fn set_column_budget(&mut self, floats: usize) {
        self.column_budget = floats.max(1);
        self.column_floats = 0;

        // Recompute the largest tile with the new budget.
        let mut length = self.steps;

        for layer in self.layers.iter().flatten() {
            match *layer {
                Layer::ConvRelu {
                    input,
                    kernel,
                    stride,
                    dilation,
                    ..
                } => {
                    length = (length - (dilation * (kernel - 1) + 1)) / stride + 1;
                    let (_, floats) = column_tile(input * kernel, length, self.column_budget);
                    self.column_floats = self.column_floats.max(floats);
                }
                Layer::GlobalAvgPool => length = 1,
                _ => {}
            }
        }
    }

    /// Floats of scratch space [`Model::infer`] needs: two activation
    /// buffers (each layer reads one and writes the other), room for
    /// attention, and room for gathered convolution columns (plus 3 floats
    /// to align those to 16 bytes).
    pub fn scratch_floats(&self) -> usize {
        2 * self.max_activation + self.attention_floats.max(self.column_floats + 3)
    }

    /// Length of the model input: `channels x steps`.
    pub fn input_floats(&self) -> usize {
        self.channels * self.steps
    }

    /// Number of features in the embedding (the input to the last layer).
    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }

    pub fn parameters(&self) -> usize {
        self.parameters
    }

    /// Multiply-accumulates per inference.
    pub fn macs(&self) -> usize {
        self.macs
    }

    /// Class name `index` as text, e.g. `"WALK"`.
    pub fn class_name(&self, index: usize) -> &'m str {
        let bytes = &self.class_names[index * CLASS_NAME_BYTES..(index + 1) * CLASS_NAME_BYTES];
        let length = bytes
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(CLASS_NAME_BYTES);
        core::str::from_utf8(&bytes[..length]).unwrap_or("?")
    }

    /// Runs the network on one standardized input (see
    /// [`crate::preprocess::window_input`]) and writes one logit per class.
    ///
    /// If `embedding` is given, it also receives the averaged features fed
    /// to the last layer. That is the hook for on-device personalization:
    /// learning a user's class centroids from these vectors needs no
    /// backpropagation.
    pub fn infer(
        &self,
        input: &[f32],
        scratch: &mut [f32],
        logits: &mut [f32],
        mut embedding: Option<&mut [f32]>,
    ) -> Result<(), ModelError> {
        check_size(input.len(), self.input_floats())?;
        check_size(logits.len(), self.classes)?;

        if scratch.len() < self.scratch_floats() {
            return Err(ModelError::BufferSize {
                needed: self.scratch_floats(),
                available: scratch.len(),
            });
        }

        if let Some(embedding) = embedding.as_deref() {
            check_size(embedding.len(), self.embedding_size)?;
        }

        // Ping-pong buffers: layer n reads `from`, writes `to`, then they swap.
        let (buffer_a, rest) = scratch.split_at_mut(self.max_activation);
        let (buffer_b, attention_scratch) = rest.split_at_mut(self.max_activation);

        let (mut features, mut length) = (self.channels, self.steps);
        let mut source = Source::Input;

        for layer in self.layers.iter().flatten() {
            let (from, to): (&[f32], &mut [f32]) = match source {
                Source::Input => (input, &mut *buffer_a),
                Source::A => (&*buffer_a, &mut *buffer_b),
                Source::B => (&*buffer_b, &mut *buffer_a),
            };

            match *layer {
                Layer::ConvRelu {
                    input,
                    output,
                    kernel,
                    stride,
                    dilation,
                    weights,
                    bias,
                } => {
                    length = conv_relu(
                        from,
                        input,
                        length,
                        weights,
                        bias,
                        output,
                        kernel,
                        stride,
                        dilation,
                        to,
                        aligned_16(attention_scratch),
                        self.column_budget,
                    );
                    features = output;
                }

                Layer::SelfAttention {
                    dim,
                    query,
                    key,
                    value,
                    output,
                } => {
                    self_attention(
                        from,
                        dim,
                        length,
                        query,
                        key,
                        value,
                        output,
                        attention_scratch,
                        to,
                    );
                }

                Layer::GlobalAvgPool => {
                    for feature in 0..features {
                        let row = &from[feature * length..(feature + 1) * length];
                        to[feature] = row.iter().sum::<f32>() / length as f32;
                    }

                    length = 1;

                    if let Some(embedding) = embedding.as_deref_mut() {
                        embedding.copy_from_slice(&to[..features]);
                    }
                }

                Layer::Dense {
                    input,
                    output,
                    weights,
                    bias,
                } => {
                    let destination = if output == self.classes {
                        &mut *logits
                    } else {
                        to
                    };

                    for (row, value) in destination.iter_mut().take(output).enumerate() {
                        *value = bias[row]
                            + dot(&weights[row * input..(row + 1) * input], &from[..input]);
                    }

                    features = output;
                }
            }

            source = match source {
                Source::Input | Source::B => Source::A,
                Source::A => Source::B,
            };
        }

        Ok(())
    }
}

/// Result of replaying a golden window (see [`check_against_golden`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoldenCheck {
    /// Largest difference from PyTorch's logits, relative to `1 + |logit|`.
    pub max_error: f32,
    /// Both picked the same class.
    pub same_answer: bool,
}

/// Replays a golden file written by the training script (frame count,
/// 120-byte frames, logit count, PyTorch's logits) through this crate's
/// preprocessing and inference, and compares the result.
///
/// Run on the Mac by `cargo test` and on the board at start-up, so both the
/// plain-Rust and the ESP32-S3 assembly paths are checked against PyTorch.
pub fn check_against_golden(
    model: &Model,
    golden: &[u8],
    input: &mut [f32],
    scratch: &mut [f32],
) -> Result<GoldenCheck, ModelError> {
    use legday_core::fused::{FUSED_FRAME_WIRE_BYTES, FusedFrame};

    let read_u32 = |at: usize| -> Result<usize, ModelError> {
        let bytes = golden.get(at..at + 4).ok_or(ModelError::Truncated)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize)
    };

    let frames = read_u32(0)?;
    let logits_at = 4 + frames * FUSED_FRAME_WIRE_BYTES;
    let classes = read_u32(logits_at)?;

    if classes != model.classes || classes > 16 {
        return Err(ModelError::BadOutput);
    }

    // Decode frames straight from the golden bytes; nothing is copied.
    let frame_at = |index: usize| {
        let start = 4 + index * FUSED_FRAME_WIRE_BYTES;
        let bytes: &[u8; FUSED_FRAME_WIRE_BYTES] = golden
            .get(start..start + FUSED_FRAME_WIRE_BYTES)?
            .try_into()
            .ok()?;
        Some(FusedFrame::decode_le(bytes))
    };

    crate::preprocess::frames_input(
        frame_at,
        frames,
        model.channels,
        model.steps,
        model.decimation,
        model.mean,
        model.std,
        &mut input[..model.input_floats()],
    )
    .map_err(|_| ModelError::BufferSize {
        needed: model.input_floats(),
        available: input.len(),
    })?;

    let mut logits = [0.0_f32; 16];
    model.infer(
        &input[..model.input_floats()],
        scratch,
        &mut logits[..classes],
        None,
    )?;

    let mut check = GoldenCheck {
        max_error: 0.0,
        same_answer: true,
    };
    let (mut best_ours, mut best_theirs) = (0, 0);

    for class in 0..classes {
        let at = logits_at + 4 + class * 4;
        let bytes = golden.get(at..at + 4).ok_or(ModelError::Truncated)?;
        let expected = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        let error = (logits[class] - expected).abs() / (1.0 + expected.abs());
        check.max_error = check.max_error.max(error);

        if logits[class] > logits[best_ours] {
            best_ours = class;
        }

        let best_expected_at = logits_at + 4 + best_theirs * 4;
        let best_expected = f32::from_le_bytes(
            golden[best_expected_at..best_expected_at + 4]
                .try_into()
                .unwrap(),
        );
        if expected > best_expected {
            best_theirs = class;
        }
    }

    check.same_answer = best_ours == best_theirs;
    Ok(check)
}

/// Which buffer holds the current activation.
#[derive(Clone, Copy)]
enum Source {
    Input,
    A,
    B,
}

fn check_size(available: usize, needed: usize) -> Result<(), ModelError> {
    if available == needed {
        Ok(())
    } else {
        Err(ModelError::BufferSize { needed, available })
    }
}

/// Dot product. On the ESP32-S3, when both slices are 16-byte aligned and
/// a multiple of 4 long, this runs Espressif's vector-load loop (1.5 cycles
/// per multiply-add, measured); everywhere else, plain Rust.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let length = a.len().min(b.len());

    #[cfg(target_arch = "xtensa")]
    {
        let aligned = (a.as_ptr() as usize | b.as_ptr() as usize) % 16 == 0;

        if aligned && length >= 4 && length % 4 == 0 {
            // SAFETY: both pointers are 16-byte aligned and valid for
            // `length` floats, `length` is a multiple of 4.
            return unsafe { xtensa::dot_vector(a.as_ptr(), b.as_ptr(), length / 4) };
        }
    }

    a[..length]
        .iter()
        .zip(&b[..length])
        .map(|(x, y)| x * y)
        .sum()
}

/// The ESP32-S3 kernels, written in assembly because the compiler never
/// produces these instructions itself.
///
/// Measured with the firmware's `k` key (cycles per multiply-add): compiled
/// Rust 9.5, hand-scheduled scalar assembly 3.5, this loop 1.5. The chip's
/// floating-point unit takes 4 cycles to produce a result but can start one
/// every cycle, so speed comes from keeping four independent multiply-adds
/// in flight. Three things the compiler does not do make that possible:
///
/// * `madd.s acc, a, b` multiplies and adds in one instruction (the
///   compiler emits a separate `mul.s` and `add.s`, because fusing them
///   changes rounding, and Rust does not allow that silently);
/// * `ee.ldf.128.ip` (an ESP32-S3 extension) loads four floats in one
///   instruction and advances the pointer;
/// * `loopnez` is a zero-overhead hardware loop.
#[cfg(target_arch = "xtensa")]
mod xtensa {
    use core::arch::asm;

    /// Sum of `a[i] * b[i]` over `groups * 4` floats: the loop from
    /// Espressif's `dsps_dotprod_f32_aes3.S`, unchanged.
    ///
    /// # Safety
    /// `a` and `b` must be 16-byte aligned and valid for `groups * 4` floats.
    #[inline(always)]
    pub unsafe fn dot_vector(a: *const f32, b: *const f32, groups: usize) -> f32 {
        let (mut sum0, mut sum1, mut sum2, mut sum3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);

        // SAFETY: the caller guarantees the alignment and length the vector
        // loads need. The FPU registers used are declared as clobbered.
        //
        // `.balign 8; nop; nop.n` puts the 3-byte `loopnez` at 5 mod 8 so the
        // loop body starts on an 8-byte boundary: the hardware loop requires
        // an aligned body, and LLVM's assembler, unlike GNU's, does not pad
        // for it (a misaligned body crashed the board).
        unsafe {
            asm!(
                ".balign 8",
                "nop",
                "nop.n",
                "loopnez {groups}, 1f",
                "ee.ldf.128.ip f11, f10, f9, f8, {a}, 16",
                "ee.ldf.128.ip f7, f6, f5, f4, {b}, 16",
                "madd.s {sum0}, f4, f8",
                "madd.s {sum1}, f5, f9",
                "madd.s {sum2}, f6, f10",
                "madd.s {sum3}, f7, f11",
                "1:",
                groups = in(reg) groups,
                a = inout(reg) a => _,
                b = inout(reg) b => _,
                sum0 = inout(freg) sum0,
                sum1 = inout(freg) sum1,
                sum2 = inout(freg) sum2,
                sum3 = inout(freg) sum3,
                out("f4") _, out("f5") _, out("f6") _, out("f7") _,
                out("f8") _, out("f9") _, out("f10") _, out("f11") _,
                options(nostack, readonly, pure)
            );
        }

        (sum0 + sum1) + (sum2 + sum3)
    }
}

/// The 16-byte aligned part of a scratch buffer (drops up to 3 floats).
fn aligned_16(buffer: &mut [f32]) -> &mut [f32] {
    let skip = (((16 - buffer.as_ptr() as usize % 16) % 16) / 4).min(buffer.len());
    &mut buffer[skip..]
}

/// How many output steps a convolution gathers at once, and the floats
/// that takes, for a column of `patch` floats, `length_out` steps and a
/// budget of `budget` floats.
fn column_tile(patch: usize, length_out: usize, budget: usize) -> (usize, usize) {
    let steps = (budget / patch.max(1)).clamp(1, length_out.max(1));
    (steps, steps * patch)
}

/// Convolution along time (no padding) followed by ReLU.
/// `input` is `[channel][time]`; returns the output length.
///
/// Written as dot products so the fast `dot` does all the arithmetic:
///
/// 1. **Gather.** For a tile of output steps, copy everything each step
///    sees into one contiguous column: `column[i * kernel + j] =
///    input[i][t * stride + j * dilation]`. That is exactly the order the
///    filter weights are stored in (`[output][input][kernel]`).
/// 2. **Multiply.** Each output value is then `bias + dot(filter, column)`,
///    a long dot product of two contiguous, 16-byte aligned arrays.
///
/// Gathering costs one copy per input value per step, about 1/32 of the
/// multiply-adds. Doing a whole tile of steps per filter means each filter
/// is read from flash once per tile, not once per step.
#[allow(clippy::too_many_arguments)]
fn conv_relu(
    input: &[f32],
    channels_in: usize,
    length_in: usize,
    weights: &[f32],
    bias: &[f32],
    channels_out: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    output: &mut [f32],
    columns: &mut [f32],
    column_budget: usize,
) -> usize {
    let length_out = (length_in - (dilation * (kernel - 1) + 1)) / stride + 1;
    let patch = channels_in * kernel;
    let (tile, _) = column_tile(patch, length_out, column_budget);

    let mut first = 0;

    while first < length_out {
        let steps = tile.min(length_out - first);

        // 1. Gather one column per output step in this tile.
        for step in 0..steps {
            let start = (first + step) * stride;
            let column = &mut columns[step * patch..(step + 1) * patch];

            for in_channel in 0..channels_in {
                let signal = &input[in_channel * length_in..(in_channel + 1) * length_in];
                let taps = &mut column[in_channel * kernel..(in_channel + 1) * kernel];

                for (tap, value) in taps.iter_mut().enumerate() {
                    *value = signal[start + tap * dilation];
                }
            }
        }

        // 2. Every filter against every column of the tile.
        for out_channel in 0..channels_out {
            let filter = &weights[out_channel * patch..(out_channel + 1) * patch];
            let row = &mut output
                [out_channel * length_out + first..out_channel * length_out + first + steps];

            for (step, value) in row.iter_mut().enumerate() {
                let total =
                    bias[out_channel] + dot(filter, &columns[step * patch..(step + 1) * patch]);
                *value = total.max(0.0); // ReLU
            }
        }

        first += steps;
    }

    length_out
}

/// Scaled dot-product self-attention over `length` time steps of `dim`
/// features, with a residual: `out = x + W_o * softmax(Q K^T / sqrt(dim)) V`.
///
/// Mirrors `SelfAttention.forward` in `Scripts/legday_ml.py`. `input` and
/// `output` are `[feature][time]`; `scratch` holds Q, K, V as `[time][feature]`
/// plus one row of scores and one mixed vector.
#[allow(clippy::too_many_arguments)]
fn self_attention(
    input: &[f32],
    dim: usize,
    length: usize,
    query: (&[f32], &[f32]),
    key: (&[f32], &[f32]),
    value: (&[f32], &[f32]),
    output_projection: (&[f32], &[f32]),
    scratch: &mut [f32],
    output: &mut [f32],
) {
    let (q, rest) = scratch.split_at_mut(dim * length);
    let (k, rest) = rest.split_at_mut(dim * length);
    let (v, rest) = rest.split_at_mut(dim * length);
    let (scores, rest) = rest.split_at_mut(length);
    let mixed = &mut rest[..dim];

    // Project every time step: Q[t] = W_q x[t] + b_q, and the same for K, V.
    for step in 0..length {
        for row in 0..dim {
            let mut totals = [query.1[row], key.1[row], value.1[row]];

            for feature in 0..dim {
                let x = input[feature * length + step];
                totals[0] += query.0[row * dim + feature] * x;
                totals[1] += key.0[row * dim + feature] * x;
                totals[2] += value.0[row * dim + feature] * x;
            }

            q[step * dim + row] = totals[0];
            k[step * dim + row] = totals[1];
            v[step * dim + row] = totals[2];
        }
    }

    let scale = inv_sqrt(dim as f32);

    for step in 0..length {
        // How much this step attends to every step, as probabilities.
        let q_row = &q[step * dim..(step + 1) * dim];

        for (other, score) in scores.iter_mut().enumerate() {
            *score = dot(q_row, &k[other * dim..(other + 1) * dim]) * scale;
        }

        softmax(scores);

        // Weighted mix of the value vectors.
        for feature in 0..dim {
            mixed[feature] = (0..length)
                .map(|other| scores[other] * v[other * dim + feature])
                .sum();
        }

        // Output projection plus the residual (the original input).
        for row in 0..dim {
            let projected = output_projection.1[row]
                + dot(&output_projection.0[row * dim..(row + 1) * dim], mixed);
            output[row * length + step] = input[row * length + step] + projected;
        }
    }
}

/// Turns scores into probabilities that sum to 1, in place. Subtracting the
/// largest score first keeps `exp` from overflowing; it does not change the
/// result.
pub fn softmax(values: &mut [f32]) {
    let largest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;

    for value in values.iter_mut() {
        *value = exp(*value - largest);
        total += *value;
    }

    for value in values.iter_mut() {
        *value /= total;
    }
}

/// `e^x` without `std` or `libm`, accurate to about 3e-7 relative.
///
/// Writes `e^x = 2^n * e^r` with `n` the integer nearest `x / ln 2` and
/// `|r| <= ln(2) / 2`. `2^n` is built directly from the float's exponent
/// bits; `e^r` is a degree-6 Taylor series, whose error is below 2e-7 on
/// that range. `r = x - n ln 2` is computed with ln 2 split into a part that
/// multiplies `n` exactly and a tiny remainder ("Cody-Waite" reduction), so
/// large `x` does not lose digits.
pub fn exp(x: f32) -> f32 {
    use core::f32::consts::LOG2_E;

    /// ln 2 = LN2_HIGH + LN2_LOW, with LN2_HIGH's last 12 bits zero so
    /// `n * LN2_HIGH` is exact for any |n| < 4096.
    const LN2_HIGH: f32 = 0.693_145_75;
    const LN2_LOW: f32 = 1.428_606_8e-6;

    if x < -87.0 {
        return 0.0;
    }

    if x > 88.0 {
        return f32::INFINITY;
    }

    let y = x * LOG2_E;
    let n = if y >= 0.0 {
        (y + 0.5) as i32
    } else {
        (y - 0.5) as i32
    };
    let r = (x - n as f32 * LN2_HIGH) - n as f32 * LN2_LOW;

    let series = 1.0
        + r * (1.0
            + r / 2.0 * (1.0 + r / 3.0 * (1.0 + r / 4.0 * (1.0 + r / 5.0 * (1.0 + r / 6.0)))));
    let power_of_two = f32::from_bits(((n + 127) as u32) << 23);

    power_of_two * series
}

#[cfg(all(test, not(target_os = "none")))]
mod tests {
    use super::*;
    extern crate std;

    #[test]
    fn exp_matches_std() {
        for step in -800..=800 {
            let x = step as f32 * 0.1;
            let expected = x.exp();
            assert!((exp(x) - expected).abs() <= expected * 5e-7, "exp({x})");
        }
    }

    #[test]
    fn softmax_sums_to_one_and_keeps_order() {
        let mut values = [1.0, 3.0, 2.0];
        softmax(&mut values);
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(values[1] > values[2] && values[2] > values[0]);
    }

    #[test]
    fn rejects_files_that_are_not_models() {
        let bytes = Aligned([0_u8; 64]);
        assert_eq!(Model::parse(&bytes.0).err(), Some(ModelError::BadMagic));
    }

    /// The parity test: one window exported by the training script, run
    /// through this crate's preprocessing and inference, must reproduce the
    /// logits PyTorch computed for it.
    #[test]
    fn demo_model_matches_pytorch() {
        let model = Model::parse(&DEMO_MODEL.0).expect("demo model parses");
        let mut input = std::vec![0.0; model.input_floats()];
        let mut scratch = std::vec![0.0; model.scratch_floats()];

        let check = check_against_golden(&model, DEMO_GOLDEN, &mut input, &mut scratch).unwrap();

        assert!(
            check.max_error < 1e-3,
            "max relative error {}",
            check.max_error
        );
        assert!(check.same_answer);
    }

    #[test]
    fn small_column_budget_gives_the_same_answer() {
        let mut model = Model::parse(&DEMO_MODEL.0).unwrap();
        model.set_column_budget(300); // one column at a time
        let mut input = std::vec![0.0; model.input_floats()];
        let mut scratch = std::vec![0.0; model.scratch_floats()];
        let check = check_against_golden(&model, DEMO_GOLDEN, &mut input, &mut scratch).unwrap();
        assert!(check.max_error < 1e-3 && check.same_answer);
    }

    #[test]
    fn dot_handles_any_length() {
        let a: std::vec::Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b = std::vec![2.0_f32; 13];
        assert_eq!(dot(&a, &b), 156.0);
        assert_eq!(dot(&a[..0], &b[..0]), 0.0);
    }
}
