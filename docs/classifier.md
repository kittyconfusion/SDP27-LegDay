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

**Fix:** `cnn::dot` runs Espressif's loop, unchanged, through Rust inline
assembly on the S3, and plain Rust elsewhere. Convolutions are rewritten as
dot products: gather each output step's inputs into a contiguous column,
then `bias + dot(filter, column)`. Model files pad every weight array to 16
bytes so the vector loads can read them in place.

One trap found on the way: the hardware loop needs its body aligned in
memory. GNU's assembler pads for that automatically; LLVM's does not, and a
misaligned loop crashed the board. The kernel pads explicitly.

### Why the weights were slow

| `cnn::dot`, one operand in flash | Cycles per multiply-add |
| --- | --- |
| data already in cache | 1.57 |
| data not in cache, flash at espflash's default DIO 40 MHz | 118.6 |
| data not in cache, flash at DIO 80 MHz | 62.1 |
| both operands in internal RAM | 1.58 |

Reading from flash is fine while cached and very slow on a miss.

**Fixes:** `crates/scripts/firmware.sh` now flashes at DIO 80 MHz (twice the
bandwidth for every cache miss, code included). QIO would be faster again,
but the bootloader espflash bundles does not set the flash's quad-enable bit
and the board boot-loops; reflashing in DIO recovers it. The firmware also
copies the demo model's weights (149 kB) into internal RAM at start-up,
leaving about 51 kB for the stack.

### Results

| Model | Parameters | Weights | First version | Vector kernel | + 80 MHz flash | + weights in RAM |
| --- | --- | --- | --- | --- | --- | --- |
| tiny | 11,686 | 46 kB | 30 ms | 6.3 ms | 6.3 ms | |
| **small (demo)** | **37,190** | **145 kB** | **95 ms** | **31 ms** | **22 ms** | **13 ms** |
| small + attention | 53,830 | 210 kB | 119 ms | 56 ms | 43 ms | |
| medium | 196,646 | 768 kB | 472 ms | 153 ms | 101 ms | |
| large | 1,031,326 | 4,029 kB | 2,391 ms | 1,728 ms | 1,339 ms | |

Every step is checked: at start-up the board replays a window exported by
the training script and compares with PyTorch (maximum error 7e-7), and
`cargo test` does the same on the Mac.

Preprocessing the window adds 3.5 ms. The firmware still caps the
classifier at half the CPU, so it can never starve sensing; at 13 ms it runs
28 times per second alongside 228 Hz sensing and SD recording.

### What is left

- **Models too big for RAM** (medium and up) are limited by flash. The 8 MB
  PSRAM, once enabled, is a faster home for them than flash; int8 weights
  would cut the bytes four times.
- **Attention** still uses plain loops (about 10 cycles per multiply-add);
  it can be rewritten on `dot` the same way.
- **int8 with the S3's integer vector instructions** (ESP-NN / ESP-DL)
  is the next large step: Espressif's benchmark shows an int8 dot product at
  about 0.4 cycles per multiply-add, four times the float kernel.
- **The second core**: the classifier belongs there, away from sensing and
  control.

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
