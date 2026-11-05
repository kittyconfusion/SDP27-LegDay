# Tim Hoefler 

# Measured timing and latency: the track record

Every number here was measured on the bench hardware, unless it is marked
*calculated* or *published*. Conducted on an Arduino Nano ESP32
(ESP32-S3, 240 MHz) with one Adafruit ICM-20948 on I2C at 400 kHz, an SD
breakout on SPI, and the Rust firmware in `crates/main_app`.

The details and reasoning live in `docs/imu-bringup.md` (sensors and bus)
and `docs/classifier.md` (classifier and memory). This file collects the
results in one place, with what each one rules in or out. When something is
re-measured, add a dated row; don't overwrite the old one.

**Units.** 225 Hz means one sample period is 4.44 ms. The CPU runs 240 MHz,
so 1 µs is 240 cycles. A 'MAC' is one multiply-add.

---

## 1. Sensor sampling

| Measurement | Result | Date |
| --- | --- | --- |
| IMU sample rate (divider 4, nominal 225 Hz) | **228.5 Hz** actual; the IMU's crystal runs 1.5 % fast | Oct 19 |
| Measured IMU sample period | 4,372 µs (1.6 % faster than nominal) | Oct 21 |
| Exact 200 Hz | not possible: the rate is 1125 / (1 + divider), so 225 or 187.5 Hz | datasheet |
| Polling every 5 ms instead of using the FIFO | 182 Hz average, gaps up to 118 ms (SD writes stall the loop) | Oct 19 |
| Timestamp jitter: naive back-calculation vs. `SampleClock` | ~4 ms vs. under 1 ms | Oct 19 |
| IMU FIFO size, DMP mode | 1,024 bytes = ~25 packets = **~115 ms** | Oct 19 |
| IMU FIFO size, raw mode | 4,096 bytes = 341 samples = **~1.5 s** | Oct 19 |
| Recording, DMP mode | 2,053 frames in 8.98 s (228 Hz), 0 overflows | Oct 19 |
| Recording, raw mode | 1,824 frames in 7.97 s (228 Hz), 0 overflows | Oct 19 |

## 2. I2C bus cost

| Measurement | DMP | Raw |
| --- | --- | --- |
| Bytes per sample | 34 | 12 |
| Bus time per sample, drained every period | 1,041 µs | 511 µs |
| Bus busy, w/ one IMU | 23 % | 11.7 % |
| Same, with an emulated second IMU on the bus | 2,069 µs | 1,022 µs |
| Longest stall survived without losing samples | **125 ms** | **1,200 ms** |

The cost of one I2C transaction at 400 kHz is **89 µs + 23.7 µs per
byte**. Every drain from IMUs is two transactions: read the FIFO count, then read
the data. The fixed 89 µs is most of the raw-mode cost, which is why batching
helps raw more than DMP. (Measured October 21: two runs, agreeing within
1 %.)

### Projected to four IMUs (from the per-IMU numbers above)

| Layout | Path | Bus busy per 4.44 ms | Worst sample age | Verdict |
| --- | --- | --- | --- | --- |
| 1 I2C bus | DMP | 4.1 ms (93 %) | 8.6 ms | fails |
| 1 I2C bus | raw | 2.0 ms (46 %) | 6.5 ms | ok |
| 2 I2C buses, concurrent | DMP | 2.1 ms (47 %) | 6.5 ms | ok |
| 2 I2C buses, concurrent | raw | 1.0 ms (23 %) | 5.5 ms | ok |
| 2 I2C buses, blocking driver | DMP | 4.1 ms | 8.6 ms | fails |
| SPI 7 MHz x 4 (*calculated*) | DMP | 0.21 ms (5 %) | 4.7 ms | ok |
| SPI 7 MHz x 4 (*calculated*) | raw | 0.11 ms (3 %) | 4.6 ms | ok |

**What this rules in or out:**

- **The 5 ms requirement.** "Worst sample age" is the 4.44 ms sample
  interval plus the bus time. Staying under 5 ms therefore needs less than
  0.56 ms of bus time per period. **Only SPI gets there.** Raw I2C on two
  buses reaches 5.5 ms.
- **The DMP.** Four IMUs on one bus doesn't fit. Its 115 ms FIFO is also
  fragile during SD stalls of up to 50 ms (see section 4).
- **Raw mode plus fusion on the S3** halves the bus time and gives 10x the
  stall tolerance. That is why the test firmware uses it.

## 3. Wake-up and start-up

| Measurement | Result |
| --- | --- |
| DMP firmware load over I2C | 486 ms per IMU; 984 ms with verification (14,301 bytes) |
| Sleep 2 s, wake to first sample | 38 ms in DMP mode (firmware retained), 37 ms in raw mode |
| Gyro calibration at rest | DMP: 2-10 s. S3 `GyroBias`: level 3 after ~2-3 s |
| Classifier self-test against PyTorch at boot | max error **7.34e-7**, same answer; 39 → 29 → 19 ms as the kernel improved |

## 4. CPU and storage costs per operation

| Operation | Time |
| --- | --- |
| Madgwick fusion (Quat6 estimator) | **3.4 µs** (0.3 % of one core for 4 IMUs at 225 Hz) |
| Full per-IMU fusion step (bias, conversion, filter, packing) | 10-12 µs per frame |
| Four pressure sensor ADC reads | 32 µs in diagnostics; 33-95 µs in the live loop |
| Classifier preprocessing (1 s window → 40 x 75 input) | **3.5 ms** (3.3-3.7 ms across runs) |
| SD write, 16 frames (1,920 bytes) at 4 MHz SPI | 11-27 ms per pass; single writes up to 50 ms |
| Old prototype classifier (since removed/updated) | 2.2 ms features + 0.3 ms predict |

The SD stalls are why SD and radio work should probably belong on the second core, behind
a RAM queue.

## 5. Arithmetic: what the S3's CPU actually does

All of these are cycle-counter measurements of hand-written assembly
running from IRAM (`k` key), October 22.

| Probe | Cycles |
| --- | --- |
| Hardware loop overhead | 1.00 per iteration |
| `add.s` latency | 4.00 |
| `madd.s` latency (dependent chain) | 4.00 |
| Four independent `madd.s` | 4.00 for all four (one per cycle) |
| `mul.s` → `add.s` accumulate chain | 5.00 |
| `lsi` load → use | 4.00 |
| Measured CPU clock | 239 MHz (2,400,825 cycles in 10,004 µs) |

**Dot product, cycles per multiply-add:**

Added in-line assembly to enable single cycle Multiply-Adds (dot products)
This greatly speeds up Madgwick algorithm and other critical math operations.
Started at ~10 cycles per Mult-Add, and got it down to ~5 (50% speedup)

| Version | Cycles/Mult+Add |
| --- | --- |
| Compiled Rust, four running sums | 9.50 |
| Assembly, ordinary 32-bit loads | 3.51 |
| Assembly, ESP-DSP vector loads (`ee.ldf.128.ip` + `madd.s` + `loopnez`) | **1.51** |
| Espressif's C at `-Os` / `-O2` / their assembly (*published*, `esp_bm_results.csv`) | 14 / 5 / 1.5 |
| Rust with `mul_add`, four sums (*calculated* from the compiled assembly: 20 instructions per 4 MACs) | ~5 |

**Compiled loop shapes, in million MACs per second** (diagnostics
microbench, several runs):

| Loop | MMAC/s |
| --- | --- |
| `c[i] += w * a[i]` | 12.6-14.1 |
| one running sum | 14.1-16.0 |
| four running sums | 18.8-19.6 |
| four sums, from IRAM | 19.5-25.2 |
| integer version of four sums | 9.1-11.4 |

At 1.51 cycles, the vector kernel works out to about 159 MMAC/s, roughly
8x the best compiled loop.

**Things that made no difference:**

- `opt-level = 3`
- the 120 MHz QIO (Quad I/O) bootloader instead of the 80 MHz one
- per-element `madd.s` (single prec. floating point multiply-add)
inline assembly, which was slower

## 6. Memory speed: where data lives

`cnn::dot`, cycles per multiply-add, with one operand swept through 4 kB
(stays in cache) or 128 kB (keeps missing the 64 kB data cache):

| Where the operand is | Cycles/MAC | Date |
| --- | --- | --- |
| internal SRAM (both operands) | **1.57-1.58** | Oct 22 |
| flash, cached | 1.57 | Oct 22 |
| flash, cache miss, DIO (Dual I/O) 40 MHz (espflash default) | 118.6 | Oct 22 |
| flash, cache miss, DIO 80 MHz | 62.1 | Oct 22 |
| flash, cache miss, **QIO 80 MHz** | **37.0** | Oct 22 |
| PSRAM, cached / miss | *not yet measured* | |

The QIO measurement read the flash controller back as `QIO (4 lines,
address + data)`, clock divider 1 (`CTRL 0x012C2008`, `CLOCK 0x80000000`).

**What this means.** Data in internal RAM, or data that stays in cache, runs
at full speed. Weights streamed from flash cost 37 cycles per float under
QIO, 24x slower than RAM. Every model larger than the 64 kB data cache
streams its weights on every inference.

## 7. Classifier inference

One inference of the 1-D CNN. The budget is 26.7 ms, which is classifying
every 6 frames. "Budget %" is time / 26.7 ms.

### The demo model (37,190 parameters, 1.26 M MACs), step by step

| Change | Inference | Runs/s |
| --- | --- | --- |
| Earliest plain-Rust engine | 151 ms | 6 |
| Plain-Rust revisions | 146 → 98 → 95 → 93 ms | 5 |
| ESP-DSP vector kernel (`cnn::dot`), weights in flash, DIO 40 | 31.1 ms | 14 |
| + flash at DIO 80 MHz | 22.3 ms | 18 |
| + weights copied to internal RAM | **13.0 ms** | **28** |

Add 3.5 ms of preprocessing per classification. The firmware caps the
classifier at 50 % of the CPU, and at 13 ms it runs 28 times per second
alongside sensing and SD recording.

### Every model size

Weights are read from flash unless noted.

| Model | Params | MACs | Weights | Plain Rust (DIO 40) | Vector kernel (DIO 40) | Vector, DIO 80 | Vector, weights in RAM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | 11,686 | 0.47 M | 46 kB | 30.2 ms | 6.3 ms (23 %) | — | — |
| small (demo) | 37,190 | 1.26 M | 145 kB | 95.2 ms | 31.1 ms (117 %) | 22.3 ms (84 %) | **13.0 ms (49 %)** |
| small + attention | 53,830 | 1.51 M | 210 kB | 119.2 ms | 55.6 ms (208 %) | 42.7 ms (160 %) | — |
| medium | 196,646 | 5.53 M | 768 kB | 471.5 ms | 152.6 ms (572 %) | 100.7 ms (378 %) | — |
| large | 1,031,326 | 26.2 M | 4,029 kB | 2,391 ms | 1,727.5 ms | 1,339.4 ms | — |

The attention numbers are from before attention was moved onto
`cnn::dot`.

**What this rules in or out (today):**

- **tiny and small:** comfortably real time. small needs its weights in
  RAM, or will likely need only QIO (not measured quite yet).
- **small + attention:** over budget as measured. The rewrite onto `dot`
  should help a lot (my dot product function), since attention was running at 
  ~10 cycles/MAC (to be measured).
- **medium:** about 100 ms, so ~10 classifications per second at full
  CPU, or ~5 under the 50 % cap. Usable only if a slower classification
  rate is acceptable, or after int8.
- **large:** not real time on this chip in float, whatever the memory.
  26 M MACs at the kernel's best 1.5 cycles is still 164 ms.

## 8. Memory budget

Measured from the linked default build (`xtensa-esp32s3-elf-nm`),
October 22.

**General Statistics**

| Region | Size | Use |
| --- | --- | --- |
| Internal SRAM total | 512 kB | |
| Instruction cache (SRAM0) | 32 kB | flash code |
| Data cache (SRAM2) | 64 kB | flash and PSRAM data |
| Boot ROM data | ~10 kB | |
| Bootloader region ("dram2") | 73 kB | free after boot; intended for a Wi-Fi/BLE heap |
| **Main data region** | **334 kB** | the table below |

| Item | Bytes |
| --- | --- |
| Demo weights copied to RAM | 149,360 |
| Classifier scratch (32 kB column tile) | 50,692 |
| One-second window (225 x 120 bytes) | 27,016 |
| Classifier input (40 x 75 floats) | 12,004 |
| SD write batch | 7,688 |
| Other `.bss` / `.data` | 8,160 |
| Code in RAM (`.rwtext`, incl. 3.3 kB PSRAM init) | 11,288 |
| **Stack (what is left)** | **75,544** |

Before the little 'ghost-RAM' cleanup the stack was ~51 kB. In flash: 159.5 kB of
code and 225 kB of read-only data (the demo model is 149 kB of that). The
decision log, recording what each RAM change cost, is in
`docs/classifier.md` under "Memory budget".

**What this rules in or out:**

- **4 IMUs and 4 pressure sensors:** already in the frame and window
  sizes, so no extra RAM.
- **Wi-Fi/BLE heap** of roughly 64-100 kB (*estimate*, from esp-radio's
  examples): fits mostly or entirely in the 73 kB bootloader region (thankfully).
- **A second model in RAM, or medium in RAM:** No. Options are QIO flash,
  PSRAM, or int8, which would make the demo model ~37 kB... Bummer.

## 9. Data rates

| Stream | Rate |
| --- | --- |
| One frame | 120 bytes |
| Full frame stream at 225 Hz | 27 kB/s (~216 kbit/s) |
| One-second window | 27,000 bytes |
| Model input per classification | 40 x 75 = 3,000 floats |

## 10. The 120-byte frame

One frame per sample instant, defined in `crates/legday-core/src/lib.rs`
(`SensorFrame`, bytes 0-71) and `crates/legday-core/src/fused.rs`
(`FusedFrame`, which adds bytes 72-119). Everything is little-endian, with
no padding.

| Bytes | Size | Field | Type | Meaning |
| --- | --- | --- | --- | --- |
| 0-7 | 8 | `timestamp_us` | u64 | sample time in µs since boot, reconstructed by `SampleClock` |
| 8-11 | 4 | `sequence` | u32 | frame counter; a gap means a lost frame |
| 12 | 1 | `valid_imu_mask` | u8 | bit n = IMU n present and valid |
| 13 | 1 | `valid_pressure_mask` | u8 | bit n = pressure sensor n valid |
| 14-15 | 2 | `status_flags` | u16 | see below |
| 16-63 | 48 | 4 IMUs x (accel x,y,z, gyro x,y,z) | i16 x 6 each | IMU n starts at byte 16 + 12n |
| 64-71 | 8 | 4 pressure readings | u16 x 4 | raw ADC counts, A0-A3 |
| 72-119 | 48 | 4 orientations x (x, y, z) | i32 x 3 each | Quat6 in Q30; IMU n starts at byte 72 + 12n |

So the frame is 16 bytes of header, 56 bytes of raw motion data, and 48
bytes of orientation. Bytes 0-71 are the original 72-byte raw frame, so a
consumer that only wants raw data takes the first 72 bytes.

**Units:**

- **Accelerometer:** ±16 g, so 2048 counts = 1 g.
- **Gyro:** ±2000 °/s, so 16.4 counts = 1 °/s.
- **Quaternion:** x, y, z are each divided by 2^30. w is not stored; rebuild
  it as √(1 − x² − y² − z²), with w ≥ 0 by convention.
- **Pressure:** raw ADC counts. The classifier uses them as a fraction of
  full scale.

**`status_flags` bits:**

| Bits | Meaning |
| --- | --- |
| 0-1 | orientation source: 0 none, 1 IMU's DMP, 2 S3 software filter |
| 2 | IMU FIFO overflowed just before this frame |
| 3 | timestamp reconstructed by `SampleClock` |
| 4-5 / 6-7 | accelerometer / gyro calibration level (0-3) |
| 8-9 / 10-11 | accelerometer / gyro range setting |
| 12 | bits 8-11 are valid (older files predate them) |
| 13 | the IMUs drifted out of step and were realigned |
| 15 | synthetic test data |

The frame always has 4 IMU slots and 4 pressure slots. A sensor that isn't
there just clears its bit in the validity masks, so adding or dropping
sensors never changes the format.

---

## Not yet measured

| What | How | Decides |
| --- | --- | --- |
| Model inference with QIO flash | `--features model-bench`, `b` key | whether the 149 kB weight copy can leave RAM |
| PSRAM speed: `k` rows, and every model with weights in PSRAM (`b`) | same build, `k` then `b` | whether PSRAM is a home for medium-size models |
| Attention on `cnn::dot` on the board | `b`: small+attention time and the "vs PyTorch" column | whether attention fits the budget |
| Radio (Wi-Fi / BLE) heap, throughput, and stalls | when radio code exists | core split, queue sizes |
| SPI IMU bus time (the calculated rows above) | needs the SPI wiring | the 5 ms requirement |
| int8 inference (*published* ESP-NN figure: ~0.4 cycles/MAC) | needs int8 kernels | medium/large models |

Build and run the first three from `crates/main_app`:

```sh
cargo run --release --bin imu-sensor-test --features model-bench   # then press k, then b
```
