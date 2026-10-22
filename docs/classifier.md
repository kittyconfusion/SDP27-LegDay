# Motion classifier: data, training, and running on the S3

This is the working guide for the classifier. It covers what the model
sees, how to collect data and train it on a Mac, how it runs on the board,
and what the board can afford, with measurements.

## The pieces

```text
  board: RECORD + CATEGORY buttons ──▶ SD card  /WALK/S0000000.Q6B ...
                                           │  (copy the folders to the Mac)
  Mac:   Scripts/train_classifier.py train <card folder> <run folder>
           -> model.ldm  report.txt  confusion.png  embedding.png  training.png
                                           │  (--install-demo, rebuild)
  board: crates/legday-classifier  (preprocess.rs + cnn.rs)
           1 s RAM window -> 40 x 75 input -> CNN -> class + probability
```

| File | Role |
| --- | --- |
| `Scripts/legday_ml.py` | Shared definitions: frame layout, preprocessing, the PyTorch model, the `.ldm` export, synthetic data |
| `Scripts/train_classifier.py` | `synth`, `train`, `bench-models` commands |
| `crates/legday-classifier/src/preprocess.rs` | The same preprocessing in Rust, for the board |
| `crates/legday-classifier/src/cnn.rs` | Reads a `.ldm` file in place from flash and runs it |
| `crates/legday-classifier/models/demo.ldm` | The model the firmware runs; currently trained on synthetic data |
| `crates/main_app/bin/imu_classifier.rs` | Runs the model on the live window and reports it on the status line |

`cargo test` in the classifier crate includes `demo_model_matches_pytorch`:
it replays a window exported by the training script through the Rust code
and checks the result against PyTorch. Any drift between the two
preprocessing implementations fails that test.

## What the model sees

One classification uses the most recent second of frames (225 at 225 Hz).

**Per frame, 40 channels** (the 72-byte raw frame is 16 bytes of
timestamp, sequence, masks and flags, which carry no motion information, then
48 bytes of IMU data and 8 bytes of pressure):

| Channels | Content | Unit |
| --- | --- | --- |
| 0-23 | 4 IMUs x accel x/y/z, gyro x/y/z | g, deg/s |
| 24-27 | 4 pressure sensors | fraction of ADC full scale |
| 28-39 | 4 IMUs x gravity direction x/y/z, from the quaternion | unit vector |

The quaternion itself is not fed in. A 6-axis quaternion has two
ambiguities: `q` and `-q` are the same rotation, and its heading (yaw) is
arbitrary and different for every IMU. The direction of gravity in each
sensor's frame has neither problem, and it is exactly what separates
sitting, standing and lying down. It is also smoother than the raw
accelerometer, because the orientation filter blends in the gyro.

**Per window**: groups of 3 frames are averaged, giving 75 steps at 75 Hz.
Gait content sits below about 20 Hz, so nothing useful is lost and compute
drops threefold. Every channel is then standardized (mean 0, standard
deviation 1) with statistics learned from the training set; those travel
inside the model file.

So the input is a **40 x 75 grid**: channels by time. A 1-D convolution
slides along the time axis, looking at all 40 channels at once. A single
frame on its own would carry posture but no motion; the window is what
lets the model see a gait cycle.

## The model

```text
conv 40 -> w,   kernel 7            75 -> 69 steps
conv w  -> w,   kernel 5, stride 2  69 -> 33
conv w  -> 2w,  kernel 5            33 -> 29
conv 2w -> 2w,  kernel 3, stride 2  29 -> 14
[optional self-attention across the 14 steps]
average over time                   -> 2w numbers (the "embedding")
dense 2w -> classes                 -> one score per class
softmax                             -> probabilities summing to 1
```

Each layer sees a short stretch of the layer below; stacked, the last layer
responds to patterns spanning most of the second. ReLU follows every
convolution. Training minimizes cross-entropy, which is the loss that
matches a softmax output; the class with the highest probability is the
answer, and the probability itself is a confidence that the mode-switching
logic should use (for example, only switch modes when the new class has
been above 80 % for several consecutive windows).

`--attention` adds one self-attention layer after the convolutions: every
time step can weigh every other one, so a moment is interpreted in the
context of the whole second. It is the same mechanism a transformer uses.
Whether it helps here is an experiment for real data; the convolutions
already cover the second, so expect gains only on transitions, if at all.
Context longer than one window (for example "standing still for minutes
during a conversation, so do not power down") is better handled outside
the CNN: by a small state machine over the classifier's outputs, or by the
pressure sensors, which show load directly.

## Collecting data

1. Two IMUs on one leg: the default address (0x69) on the thigh, one with
   its ADR jumper bridged (0x68) on the shank. They fill frame slots 0 and 1
   automatically.
2. Choose the category with CATEGORY, press RECORD, perform only that
   motion, press RECORD again. The folder is the label.
3. Record several separate sessions per category rather than one long one,
   and vary speed, footwear and surface. The training script holds out the
   last 20 % of every file, split by time, for validation.
4. Transitions: record them in `TRANS` as continuous back-and-forth
   movements (sit-stand-sit, walk-stop-walk), so the file contains only
   transition motion. Mixed files teach the model wrong labels.
5. More people beats more minutes from one person. The honest test of
   generalization is training on some people and validating on someone the
   model has never seen.

```sh
python3 Scripts/sensor_frames.py summary /Volumes/SDCARD/WALK/S0000000.Q6B   # check each file
python3 Scripts/train_classifier.py train /Volumes/SDCARD runs/first
python3 Scripts/train_classifier.py train /Volumes/SDCARD runs/first --install-demo   # deploy
```

## Reading the results

- `report.txt`: accuracy, and precision / recall for every class. Recall
  of `TRANS` and of the stair classes matter most for a prosthetic.
- `confusion.png`: which classes are mistaken for which. Walk vs. stairs
  confusion, for example, points at missing channels (pressure) or too
  little stair data.
- `embedding.png`: every validation window placed by the model's internal
  features. Tight, separate clusters are the evidence that the classes are
  separable in what the model learned; overlapping clusters show exactly
  which ones are not.
- `training.png`: if training accuracy keeps rising while validation falls,
  the model is memorizing; make it smaller or collect more data.

On the synthetic dataset the default model reaches 100 % with clean
clusters. That proves the plumbing, not the model: synthetic classes are far
easier to separate than real people.

## Model size

Parameter count is not the main lever at this data scale. Large language
models work because they train on trillions of examples; a session of
calibration motions is a few thousand windows. A model with far more
parameters than training windows memorizes them. A reasonable starting
range here is 10k-50k parameters; the data, its labels, and the choice of
channels decide accuracy far more than size.

## Measured on the S3 (October 22, 2026)

The first version of the inference engine ran at 13 million multiply-adds
per second: 95 ms for the demo model. A day of measurement traced that to
two causes, fixed both, and brought the demo model to **13 ms**.

### Why plain code was slow

Measured in CPU cycles with hand-written assembly (`k` key,
`crates/main_app/bin/imu_cycles.rs`):

| Measurement | Cycles |
| --- | --- |
| hardware loop overhead | 1.0 per iteration |
| `add.s` / `madd.s` result available after | 4.0 |
| four independent `madd.s` | 4.0 for all four (one per cycle) |
| dot product, compiled Rust | 9.5 per multiply-add |
| dot product, assembly, ordinary loads | 3.5 per multiply-add |
| dot product, Espressif's ESP-DSP loop | 1.5 per multiply-add |

The floating-point unit takes 4 cycles to produce a result but can start a
new one every cycle, so fast code keeps four independent multiply-adds in
flight. Espressif's loop does that with three things compilers do not emit:
the fused `madd.s`, the ESP32-S3's `ee.ldf.128.ip` (four floats per load),
and the zero-overhead `loopnez` hardware loop. LLVM (Rust's compiler)
instead emits `mul.s` then `add.s` and stalls on each. Espressif's own plain
C does no better at `-Os`: their published benchmark (`esp_bm_results.csv`)
shows 14 cycles per multiply-add for C at `-Os`, 5 at `-O2`, 1.5 for their
assembly. Building our firmware at `opt-level = 3` changed nothing.

Can the compiler be made to do it? Partly. Tested on our toolchain
(October 22, 2026): `core::f32::math::mul_add`, or the fast-math
`fmul_algebraic` / `fadd_algebraic`, do make LLVM emit `madd.s`; no
compiler change is needed for the fused instruction. But even with four
explicit accumulators the compiled loop is 20 instructions per 4
multiply-adds (single-float loads, extra address arithmetic, a branch),
about 5 cycles each. LLVM's Xtensa backend never generates the S3's
128-bit loads or hardware loops; teaching it to would mean writing
auto-vectorization for a vendor extension inside Espressif's LLVM fork.
Espressif itself does not do that: ESP-DSP and ESP-NN ship hand-written
assembly kernels, which is what `cnn::dot` is.

**Fix:** `cnn::dot` runs Espressif's loop, unchanged, through Rust inline
assembly on the S3, and plain Rust elsewhere. The assembly is the
`xtensa::dot_vector` function at the bottom of
`crates/legday-classifier/src/cnn.rs` (the cycle probes in `imu_cycles.rs`
are the same style). It is genuine Xtensa machine code: the `asm!` block's
strings are instructions the assembler encodes one for one, and Rust only
fills in which registers hold the pointers, count and sums. Per 4 floats:

```text
loopnez groups, end                    ; hardware loop, no branch per pass
  ee.ldf.128.ip f11,f10,f9,f8, a, 16   ; 4 floats of a, a += 16 bytes
  ee.ldf.128.ip f7,f6,f5,f4,   b, 16   ; 4 floats of b, b += 16 bytes
  madd.s sum0, f4, f8                  ; sum0 += a0*b0
  madd.s sum1, f5, f9                  ; four independent sums, so a new
  madd.s sum2, f6, f10                 ; madd.s starts every cycle while
  madd.s sum3, f7, f11                 ; earlier ones finish
end:
```

Six instructions per 4 multiply-adds, 1.5 cycles each. Every layer is
then expressed as dot products:

- **Convolutions:** gather each output step's inputs into a contiguous
  column, then `bias + dot(filter, column)`.
- **Dense:** one `dot` per output.
- **Attention:** each time step gathered into a vector, then one `dot`
  per projection row; Q and K stored by time step so a score is one `dot`;
  V stored transposed and padded to a multiple of 4 so each mixed feature
  is one `dot`. Checked against the formula in f64 and against PyTorch.

Model files pad every weight array to 16 bytes so the vector loads can
read them in place.

One trap found on the way: the hardware loop needs its body aligned in
memory. GNU's assembler pads for that automatically; LLVM's does not, and a
misaligned loop crashed the board. The kernel pads explicitly.

### Why the weights were slow

| `cnn::dot`, one operand in flash | Cycles per multiply-add |
| --- | --- |
| data already in cache | 1.57 |
| data not in cache, flash at espflash's default DIO 40 MHz | 118.6 |
| data not in cache, flash at DIO 80 MHz | 62.1 |
| data not in cache, flash at QIO 80 MHz | 37.0 |
| both operands in internal RAM | 1.58 |

Reading from flash is fine while cached and very slow on a miss.

**Fixes:**

- **80 MHz:** the flash clock is 80 MHz instead of 40.
- **QIO:** the flash is read over 4 data lines instead of 2. The chip's
  boot ROM cannot start in QIO (writing QIO into the image header
  boot-loops the board), so the header stays DIO and Espressif's own QIO
  bootloader, the one Arduino uses for this board, switches the flash
  controller to QIO before our firmware starts. Details and provenance:
  `crates/scripts/bootloader/README.md`. `cargo run` (through
  `crates/scripts/espflash-runner.sh`) and `crates/scripts/firmware.sh`
  both flash it. The `k` key prints the live mode from the controller's
  registers.
- **Weights in RAM:** the firmware copies the demo model's weights
  (149 kB) into internal RAM at start-up. Whether that is still needed
  under QIO is the open measurement below.

### Results

| Model | Parameters | Weights | First version | Vector kernel | + 80 MHz flash | + weights in RAM |
| --- | --- | --- | --- | --- | --- | --- |
| tiny | 11,686 | 46 kB | 30 ms | 6.3 ms | 6.3 ms | |
| **small (demo)** | **37,190** | **145 kB** | **95 ms** | **31 ms** | **22 ms** | **13 ms** |
| small + attention | 53,830 | 210 kB | 119 ms | 56 ms | 43 ms | |
| medium | 196,646 | 768 kB | 472 ms | 153 ms | 101 ms | |
| large | 1,031,326 | 4,029 kB | 2,391 ms | 1,728 ms | 1,339 ms | |

Not yet measured: the same table with QIO flash, with weights in PSRAM,
and with attention on `dot`. Build with `--features model-bench`, press
`b`; every row also says whether the board's answer matches PyTorch for
that model's golden window.

Every step is checked: at start-up the board replays a window exported by
the training script and compares with PyTorch (maximum error 7e-7), and
`cargo test` does the same on the Mac for the demo model and every bench
model.

Preprocessing the window adds 3.5 ms. The firmware still caps the
classifier at half the CPU, so it can never starve sensing; at 13 ms it runs
28 times per second alongside 228 Hz sensing and SD recording.

### What is left

- **Where the weights live** (see the memory budget below): internal RAM,
  flash under QIO, or PSRAM, decided by the `b` measurements.
- **int8 with the S3's integer vector instructions** (ESP-NN / ESP-DL)
  is the next large step. The S3's real SIMD (`ee.vmulas`, 16 multiplies
  per instruction) exists only for 8- and 16-bit integers, not floats.
  Espressif's benchmark shows an int8 dot product at about 0.4 cycles per
  multiply-add, four times the float kernel, and the weights shrink four
  times (the demo model to about 37 kB).
- **The second core**: the classifier belongs there, away from sensing and
  control.

## Memory budget

The ESP32-S3 has 512 kB of internal SRAM, but not all of it is ours:

| Region | Size | Used for |
| --- | --- | --- |
| SRAM0 | 32 kB | instruction cache for flash code |
| SRAM2 | 64 kB | data cache for flash and PSRAM data |
| top of SRAM1 | ~10 kB | the boot ROM's own data |
| bootloader region ("dram2") | 73 kB | free once the firmware runs: the place for a heap (Wi-Fi / BLE) |
| **main data region** | **334 kB** | everything below |

The 32 + 64 kB of caches are what make flash and PSRAM usable at all; the
8 MB PSRAM and 16 MB flash sit outside, behind those caches.

**The main data region, measured from the linked default build**
(`xtensa-esp32s3-elf-nm -S --size-sort`, October 22, 2026):

| Item | Bytes | What it is |
| --- | --- | --- |
| `MODEL_IN_RAM` | 149,360 | demo weights copied from flash |
| classifier `SCRATCH` | 50,692 | activations and convolution columns (32 kB tile) |
| `WINDOW` | 27,016 | one second of 120-byte frames |
| classifier `INPUT` | 12,004 | the 40 x 75 model input |
| `BATCH` | 7,688 | SD write batch |
| other `.bss` / `.data` | 8,160 | esp-hal driver state and tables |
| code in RAM (`.rwtext`, 10,264 + alignment) | 11,288 | interrupt handlers, start-up, PSRAM init, cycle probes |
| **stack** | **75,544** | what is left |

Everything the firmware links is code it calls: the linker drops unused
functions. Diagnostics and probes cost flash, not RAM, apart from the
buffers listed and about 1 kB of probe code that runs from RAM on purpose.
The classifier (model copy, scratch, input) is four fifths of what is
used, and the model copy alone is 149 kB. Without it the firmware would
use about 115 kB of the 334.

### Decisions and their cost

A running record of what was traded for what.

| Date | Change | RAM | Cost / reason |
| --- | --- | --- | --- |
| Oct 22 | Copy demo weights to RAM | -149 kB | 22 -> 13 ms per inference; revisit once QIO and PSRAM timings are in |
| Oct 22 | Convolution column tile 64 -> 32 kB | +32 kB | none measurable: tiles exist so each filter is read once per tile, which only matters while filters come from flash; same answer (`small_column_budget_gives_the_same_answer`) |
| Oct 22 | Removed the old prototype-classifier timing from diagnostics | +19 kB | the firmware runs the CNN, which section 7 times |
| Oct 22 | Cycle probes borrow the classifier's scratch | +8 kB | none: the probes and the classifier never run at once |
| Oct 22 | PSRAM driver enabled | -3.3 kB | its init code must run from RAM; drop it if PSRAM is not used |

**Room for what is coming:** four IMUs and four pressure channels are
already in the frame layout and the window size, so they add no RAM.
Wi-Fi or BLE need a heap, roughly 64-100 kB (an estimate from
esp-radio's examples; measure it when the radio code is added); the
bootloader region provides 73 kB of it without touching the main region. PSRAM
(8 MB) suits large, not-time-critical data: logs, a longer history, big
models if they prove fast enough there.

## On-device training

The requirement is both: train on the Mac from SD recordings (built), or
calibrate on the board after a short motion sequence (designed, not built).
Full CNN training needs backpropagation, which is slow and memory-hungry on
the S3. The practical on-device route is **personalization**: keep the
network trained on the Mac, run each of the user's calibration windows
through it, and average the embeddings per class (the `embedding` output of
`Model::infer`). New windows are classified by the nearest class average.
That is the same idea as the existing `prototype.rs` baseline, but on
learned features, and it needs only forward passes: a few minutes of
calibration data is a few thousand windows, so seconds to a minute of work.
