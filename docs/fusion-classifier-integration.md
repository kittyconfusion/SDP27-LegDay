# Fusion and CNN integration handoff

Based on team commit `a6f04fa` (`data-collection-commented`). Existing algorithm
sources, model files and training scripts were recovered from personal branch
`tim/networking-playground_sensor`, commit `5cff66c`. 
* Work is local on: `tim/fusion-classifier-integration`and has NOT been committed yet! *

## Package boundaries

| Package / file | Responsibility |
| --- | --- |
| `legday-core` | Shared raw types, explicit 120-byte `FusedFrame` encoding, rolling windows and sample-clock helper |
| `legday-fusion` | Existing Madgwick, gyro bias, accelerometer trust and a new frame adapter |
| `legday-classifier` | Existing 1-D CNN, Xtensa dot-product assembly, preprocessing, Q30 helpers and the older centroid baseline |
| `Scripts/legday_ml.py`, `train_classifier.py` | Existing Python preprocessing, PyTorch training and model export |
| `record_app` | Team-owned sensor acquisition, buttons/display, publication and SD buffering |

`legday-core::lib_recording` remains available; its types are also re-exported at
the crate root for the restored classifier. There are no duplicate raw-frame
definitions. Hardware dependencies in core are optional (`esp32s3` for its legacy
SD diagnostic), allowing the algorithms and frame types to be tested on the Mac.
`legday_classifier::fusion` remains an alias for `legday_fusion` for compatibility.

## Recorder connection

The recorder retains a rolling RAM history in all builds. Software fusion is
still optional. Build the fusion integration with:

```sh
cargo check --locked --release -p quadriceps --bin quadriceps --features sensor-fusion
```

This does not flash. The optional code runs after the team's raw-frame encoder
and before both latest-frame publication and SD buffering:

```text
SensorReader -> encode 120 raw bytes -> decode FusedFrame
             -> FrameFusion::update -> encode 120 fused bytes
             -> rolling RAM history -> latest-frame publication / SD queue
```

`FrameFusion` owns one Madgwick filter and gyro-bias estimator per slot. It scales
using the recorded range flags, updates only fresh IMUs, estimates elapsed time
between their fresh MCU acquisition timestamps, and sets software-orientation
source bits only when every fresh IMU has an orientation. Missing slots remain
zero. No fresh IMUs means no orientation-source claim. An unknown range prevents
fusion; first samples and gaps longer than 250 ms re-seed tilt from gravity.
This is a new integration adapter, not a port of the old FIFO acquisition loop.
Review and hardware verification are still required. Six-axis yaw remains
unreferenced; this adapter does not synchronize sensors or recover lost samples.

## Rolling history versus SD buffering

`record.rs` now retains the newest 225 frames in a static, mutex-protected
`FusedWindowBuffer`, including while the recording switch is off. Frames enter
history after optional fusion. The samples occupy 27,000 bytes (26.37 KiB);
the linked S3 history symbol, including mutex/window bookkeeping, is 27,024 bytes.
No 27-kB task-stack allocation is introduced. `copy_sensor_history(output)` copies
the newest frames oldest-first into caller-provided storage and releases the
lock before inference can run. A future full-window snapshot needs its own
static/heap storage and must be included in the memory budget.

This is a count-based history, approximately one second at the achieved nominal
225-Hz cadence. It may cover more time during overruns, and validity flags still
matter. Startup logs report whether fusion was compiled in; once-per-second
history logs report frame count, first-to-last timestamp span and an oldest-frame
checksum. This does not enable CNN inference.

The existing SD staging/writer code is unchanged: an 8192-byte block holds 68
whole 120-byte frames, its two-byte length header and 30 padding bytes. Ten
queued blocks hold 680 frames (about three seconds at 225 Hz), independently of
the rolling history. Payload traffic is 27,000 bytes/s at 225 Hz. Actual SD
throughput still requires a board test, especially with the current 400-kHz SPI
configuration and existing write-error handling; fitting the frame format does
not establish lossless sustained recording.

## CNN status and remaining decisions

The CNN and its assembly are restored as a library, not scheduled in the recorder.
Its committed demo model is synthetic and is not an accuracy-validated gait model.
Its golden test checks the saved Python/PyTorch reference against Rust inference.

- The encoder matches the documented layout: 72 raw bytes plus 48 quaternion
  bytes, with fixed slots, little-endian fields, validity and range flags.
- With fusion disabled, quaternion bytes stay zero and source=0. Both Python and
  Rust preprocessing then produce zero gravity channels, not identity rotations.
- The restored CNN expects 40 channels (24 accel/gyro, 4 pressure, 12 gravity),
  and the existing training/window recipe uses 225 frames averaged to 75 steps.
- The latest team update sets both polling and nominal hardware output to
  225 Hz (divider 4), and both I2C buses to 400 kHz. Ranges/filters remain
  unchanged at +/-8g and +/-1000 degrees/s. Verify sustained fresh counts,
  timing and display/wiring compatibility before treating 225 polling frames
  as one second of complete 225-Hz measurements; polling can still skip data.
- `GyroBias` uses sample-count thresholds inherited from the 225-Hz prototype;
  calibration takes longer at lower fresh-sample rates.
- Agree on fresh sample rate, timestamping, gap handling, sensor mounting/order,
  filter bandwidth and range settings before training or enabling live inference.
- Keep inference outside the acquisition task. Decide window ownership, memory,
  queue limits and inference hop/budget; benchmark all work together on hardware.
- Future models can consume the same recordings and preprocessing contract;
  the existing centroid baseline remains available alongside the CNN.

The classifier crate still offers `Model::parse`, `preprocess::window_input`,
`Model::infer` and `softmax`. Applications supply windows and scratch buffers.
The original flash/QIO/PSRAM setup and board diagnostic application were not
copied over; those remain separate integration choices.

## SD and training data

The team's `.leg` files contain padded 8192-byte blocks, each with a two-byte
big-endian inclusive used length. They are not flat `.Q6B` streams. Extract the
120-byte payloads before using the restored Python tools:

```sh
python3 Scripts/unpack_leg.py recording.leg recording.Q6B
```

The extractor checks block lengths and padding, preserves flags and samples, and
refuses to overwrite an existing output. The format has no version marker, so
use only current 120-byte recordings; structural checks cannot reliably identify
every older format. Extraction does not fix timing, derive orientation, or add
activity labels. Place labeled sessions in category folders for the training
script, and resolve the sampling contract before treating them as training data.

`docs/classifier.md` is the original reference guide; its hardware entry-point
commands refer to the personal branch, not this restored library integration.

## Focused verification

```sh
cargo +stable test --locked -p legday-core -p legday-fusion -p legday-classifier --lib --target aarch64-apple-darwin
cargo +stable test --locked -p legday-classifier --test recorder_contract --target aarch64-apple-darwin
cargo build --locked --release -p legday-classifier --target xtensa-esp32s3-none-elf
python3 -m unittest discover -s Scripts -p test_unpack_leg.py
```

The contract test uses the team's actual encoder: it checks byte round-tripping,
recorded scaling, missing slots, unavailable orientation and optional fusion
feeding the restored preprocessor. Software checks do not establish the <5 ms
hardware acquisition requirement or real gait accuracy.

Verified on November 11, 2026: 38 library tests plus 10 recorder-contract tests
passed, including the saved PyTorch golden comparison. Both extractor tests
passed. The S3 classifier library and recorder with `sensor-fusion` built in
release mode; the default recorder, standalone sensor check, main application
and legacy core SD diagnostic passed targeted S3 compile checks. Existing
unused-code/profile/duplicate-target warnings remain. No device was flashed.
