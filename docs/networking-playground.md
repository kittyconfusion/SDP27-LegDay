# Networking playground: S3 to laptop telemetry

This is an independent experimental image on `tim/networking-playground_sensor`.
It sends **simulated** classifier and control values. It does not read the IMUs,
run the CNN, command a motor, or change the sensor firmware. No commits or pushes
are needed to try it.

## How the connection works

```text
Wi-Fi: S3 station -> 2.4 GHz access point/router -> laptop IPv4:UDP port 5005
       esp-radio   -> embassy-net (DHCP + IPv4 + UDP) -> Python socket

BLE:   S3 peripheral -> direct BLE connection -> laptop central
       esp-radio + trouble-host (GATT notifications) -> Python Bleak
```

For Wi-Fi, both devices join the same reachable LAN or hotspot. DHCP gives the
board its own IP address. The board sends to the laptop's IP and port; the Python
listener binds to a local interface and receives those datagrams. Internet access
and an external server are unnecessary. The router/access point forwards local
traffic. Guest networks and campus Wi-Fi can isolate clients; a shared personal
hotspot/router is often easier for a demonstration. This first implementation
uses WPA2 Personal, not enterprise Wi-Fi or a captive portal.

The board could instead host a Wi-Fi access point, with the laptop joining it;
that needs IP configuration or a DHCP server. SoftAP is not implemented here.

For BLE, the board advertises a custom service UUID. The laptop discovers that
service, connects, and subscribes to its notification characteristic. There are
no IP addresses, router, or Python TCP server in that path. The collector refuses
to choose automatically if multiple matching boards are found; select one with
`--device`. The BLE address changes on each board reboot, so discover by service
rather than assuming a fixed MAC. This bench service has no pairing/authentication.

Application code is Rust `no_std`. `esp-radio` still uses Espressif's precompiled
radio firmware/libraries; this does not introduce an Arduino or ESP-IDF application
framework. The radio requires a heap and scheduler. See the version-matched
[Espressif driver documentation](https://docs.espressif.com/projects/rust/esp-radio/1.0.0-beta.1/esp32s3/esp_radio/index.html).

## Build and switch transports

From the repository root:

```sh
bash crates/scripts/networking.sh build offline
bash crates/scripts/networking.sh build wifi
bash crates/scripts/networking.sh build ble
```

Equivalent selection: `cargo build --release -p networking-playground
--no-default-features --features wifi` (or `ble`, `offline`). Choose exactly one.
The wrapper saves a separate ELF for each variant under
`target/xtensa-esp32s3-none-elf/release/networking-playground-{offline,wifi,ble}`.
Use `-p networking-playground`: building the whole workspace can unify features
from `main_app`, which currently enables both radios and coexistence in Cargo.

The workspace adds a driver-only optimization override (`esp-radio` at level 3),
as required by that driver; the CNN's optimization settings are unchanged.
Radio dependencies come from the existing lockfile. No assembly is needed.

Flashing **replaces the firmware running on the board**. Rediscover its port:

```sh
espflash list-ports
bash crates/scripts/networking.sh flash ble /dev/cu.usbmodem2143101
```

For Wi-Fi, first copy `networking.env.example` to `.networking.local.env` and edit
the latter locally. It is ignored by Git. Supply the SSID, WPA2 password, and the
laptop's IPv4 address on that network. They are compiled into this test image;
the password is never printed. Rebuild after changing them. The wrapper refuses
`flash wifi` without all three settings. A build without settings uses clearly
named dummy credentials and a documentation-only destination IP so that linker
size comparisons include the complete radio code.

Edit `.networking.local.env`, not the example: the example is a shareable
template. The laptop can use another Wi-Fi band/SSID on the same router when
the router allows local traffic between them; it does not have to join 2.4 GHz.
This was verified here with the S3 on the 2.4 GHz SSID and the Mac remaining on
the user's 6 GHz SSID.

```sh
python3 Scripts/telemetry.py udp --csv telemetry-captures/wifi.csv
bash crates/scripts/networking.sh flash wifi /dev/cu.usbmodem2143101
```

Run the receiver in another terminal. Permit the Python listener through the
laptop's local firewall if prompted. After learning the board's IP from serial,
`--peer BOARD_IP` filters for that sender; otherwise UDP pins to the first valid
sender. This is sender selection, not cryptographic authentication.

For BLE, use a modern Python in a virtual environment and install `bleak` there:

```sh
python3 -m venv /tmp/legday-network-venv
/tmp/legday-network-venv/bin/pip install bleak
/tmp/legday-network-venv/bin/python Scripts/telemetry.py ble --seconds 30 --csv telemetry-captures/ble.csv
```

On this Mac the tested environment uses `/opt/homebrew/bin/python3` (3.14),
Bleak 3.0.2, and PyObjC 12.2.2. The system Python 3.9 selected an incompatible
PyObjC source build. macOS may request Bluetooth permission. The host uses
[Bleak](https://bleak.readthedocs.io/en/latest/usage.html); firmware services use
[Trouble](https://embassy.dev/trouble/).

Restore the current sensor program with:

```sh
bash crates/scripts/firmware.sh imu /dev/cu.usbmodem2143101
```

## What is measured

Every image reserves the same 100 KiB internal heap: 64 KiB reclaimed bootloader
RAM and 36 KiB main RAM. That is a comparison budget, not measured radio demand.
No PSRAM is initialized in this experiment. Every image runs the scheduler and
the same synthetic 200 Hz timer task, which publishes telemetry every fourth
tick. All application tasks run on core 0; this experiment does not test dual-core
sensor/CNN integration.

Serial reports cumulative produced/transport-accepted counts, send failures,
queue-full drops, maximum wakeup lateness, missed 5 ms deadlines, current heap
use, and minimum free heap **observed every two seconds**. That sampling can miss
allocation peaks. `accepted` means queued locally, not received by the laptop.
The timer probe measures scheduling jitter, not sensor acquisition, inference,
or full CPU utilization. CPU busy percentage needs scheduler/interrupt tracing;
it cannot be inferred from packet rate or wakeup lateness.

The Python collector displays class ID, score, mock command, sequence gaps,
duplicate/old packets, invalid packets, and a rolling p95 of arrival intervals.
CSV includes the board timestamp and host monotonic arrival time. An arrival
interval is **not one-way latency**: the clocks are unsynchronized. Board reboot
requires restarting the collector; BLE disconnect also ends this first collector
implementation. Sequence arithmetic handles ordinary wrap, but gaps longer than
half the 16-bit sequence range (~11 min at 50 Hz) are ambiguous. A future packet
version should add a session ID and a model/class-map identifier.

The handoff is an allocation-free SPSC queue with capacity four. A full queue
drops a new packet instead of waiting. The sender drains to the newest queued
packet, rejects samples older than 100 ms, and bounds its async submission wait
to 10 ms. This does not bound delivery time inside the radio or operating system.
BLE's queue fills during advertising; those preconnection drops are expected.
Neither transport resends a history of stale display states after reconnecting.

## Packet version 1

20 bytes, explicit little-endian encoding; 50 Hz = **1,000 bytes/s payload**.
It fits a notification with the default 23-byte ATT MTU, so this compact stream
does not depend on negotiating a larger MTU. Framing and radio overhead are extra.

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | 2 bytes | `LD` magic |
| 2 | u8 | version = 1 |
| 3 | u8 | bit 0 = simulated data; bit 1 = mock control |
| 4 | u16 | publication sequence (wraps) |
| 6 | u8 | model-specific class ID; 255 reserved for unknown |
| 7 | u8 | quantized score 0..255; not calibrated confidence |
| 8 | u32 | low 32 bits of sample time, microseconds |
| 12 | u16 | phase 0..1000 |
| 14 | u16 | hypothetical resistance request 0..1000 |
| 16 | u16 | acquisition duration in us; 65535 = unavailable |
| 18 | u16 | inference duration in us; 65535 = unavailable |

The simulated class cycles through IDs 0..3. Phase/resistance are ramps for
transport testing, not a proposed prosthetic controller. No actuator GPIOs are
configured. A real control simulation should define units and semantics first
(for example desired damping coefficient, gait phase, and an explicit mode).

The full current 120-byte fused frame at 225 Hz would be 27,000 bytes/s payload.
For an animated leg, a separate packet can carry decimated orientations plus
validity masks and timestamps, while full-rate training data stays on SD. Four
12-byte Quat6 vectors plus this header would be 68 bytes (3.4 KB/s at 50 Hz).
That needs a larger negotiated ATT MTU and BLE pool, or explicit fragmentation;
the current 20-byte characteristic does not carry pose data. Render/interpolate
on the laptop at its display refresh rate.

## Fit and integration decisions

The existing sensor ELF was inspected, rather than treating free memory as one
pool. It reserves ~149 KB for model weights, ~51 KB for CNN scratch, ~12 KB input,
and ~27 KB for the one-second frame window. Its linked `.stack` is 75,544 bytes;
that is stack capacity, **not measured free heap**. The reclaimed bootloader
region is separate. Hardware has 8 MB PSRAM, but hot data still competes through
cache and the external-memory interface.

The isolated variants below all reserve the same heap. Main occupied is
`.rwdata_dummy + .data + .data.wifi + .bss + .noinit`; the dummy accounts for
IRAM code's reservation in the shared SRAM address space. Do not add `.rwtext`
again. Debug sections are not runtime RAM. Sizes depend on this toolchain and
configuration, and are not a measurement of a combined image.

| Variant | Main occupied incl. 36 KiB heap | Reclaimed heap | Main stack region |
| --- | ---: | ---: | ---: |
| offline | 50,700 B | 65,536 B | 291,060 B |
| Wi-Fi | 103,004 B | 65,536 B | 238,752 B |
| BLE | 86,804 B | 65,536 B | 254,956 B |

Wi-Fi adds 52,304 B and BLE adds 36,104 B relative to offline, before heap
utilization. Shared code in a combined image makes simple addition inexact.
These numbers are enough to reject the idea that a 20-byte application packet
implies a 20-byte networking cost. Size a final heap from peak measurements under
connection, loss, and reconnect, retaining stack margin; do not blindly reserve
100 KiB on top of the current sensor image.

Suggested integration order:

1. Keep acquisition/control authoritative on the board. Publish a compact copy
   of the most recent result without waiting for the PC. A laptop display may
   lag or disappear without delaying a local control update.
2. Resolve existing sensing/CNN scheduling first. `LiveClassifier::on_new_frames`
   is synchronous on the sensor core; a half-CPU duty limit does not stop one
   ~13 ms inference plus preprocessing from blocking acquisition. A FIFO preserves
   samples during a pause but does not make those samples fresh.
3. Initialize the radio on core 0, where this driver pins its background tasks.
   Keep acquisition deadlines on core 1 with appropriate priority/execution
   isolation. CNN work must be lower priority, chunked, or otherwise preemptible
   relative to acquisition; moving a single blocking loop is not sufficient.
   Account for cross-core queue synchronization, stack use, and shared cache/bus
   contention. The present crate does not implement that architecture.
4. If necessary, benchmark weights in PSRAM/QIO flash, a smaller model, or an
   int8 backend. PSRAM/quantization require timing and output checks; the current
   float implementation cannot become int8 by changing one type. Preserve the
   golden model check when moving weights or changing kernels.
5. Add the same bounded telemetry producer to the real result path. Compare
   offline, Wi-Fi, BLE during actual acquisition/CNN/SD workloads, then with the
   laptop absent and reconnecting. Record sample age, deadline misses, FIFO
   overflows, inference worst case, heap peaks, and stack high-water marks.

## Requirements that need separate evidence

At 225 Hz, six frames span 26.7 ms; four span 17.8 ms. The current CNN uses a
one-second causal window, decimates in groups of three, and schedules after six
new frames with an additional cooldown. Its preprocessing and inference time
are added to the wait for the next decision. Producing an answer every 25 ms,
finishing inference within 25 ms, and detecting a transition within 25 ms are
three different requirements. A long rolling window does not force a full
one-second delay, but old-mode data can dominate near a transition. Measure time
from a defined physical transition/event to a correct stable decision.

Likewise, <5 ms per acquisition pass differs from <5 ms sensor-to-RAM age. The
existing bus diagnostics project ~1 ms/pass for four raw IMUs on two concurrent
I2C buses, but ~5.5 ms worst sample age at a 4.44 ms polling interval. Those are
extrapolations from limited hardware, not four-IMU proof. Current wiring/code
scans two addresses on I2C0; the four-slot data layout is not implementation of
two active buses. Pressure validity is currently zero. Test the actual sensors,
filter/group delay, ADC acquisition, clock alignment, and fitted harness.

The demo CNN is trained on synthetic data. Neither a fast kernel nor a short
personal calibration establishes >95% accuracy over the requested motions.
Use held-out sessions/people, prevent leakage between overlapping train/test
windows, report per-class precision/recall and a confusion matrix, and label
transitions with event times for latency testing. A heel-strike interval is an
observed gait quantity, not a fixed timing rule for walking or a motor command.

## Verification

```sh
cargo +stable test -p networking-playground --lib --target aarch64-apple-darwin --offline
python3 -m unittest discover -s Scripts -p test_telemetry.py -v
# In separate terminals, test the host path without a board:
python3 Scripts/telemetry.py udp --bind 127.0.0.1 --port 15005 --seconds 5 --csv /tmp/legday-local.csv
python3 Scripts/telemetry.py mock --port 15005 --count 100
```

Rust/Python share an exact byte fixture. Tests cover malformed/versioned packets,
queue saturation, stale packet rejection, sequence/timestamp wrap, duplicate
packets, and reordered packets. Host tests use stable Rust to avoid rebuilding
`core` beside the host standard library under the Xtensa toolchain.

Hardware results are recorded below; they do not establish gait accuracy,
sensor latency with networking, or guaranteed RF delivery latency.

### Bench results, October 22, 2026

- **Offline:** image flashed and running; captured statistics showed zero missed
  5 ms timer deadlines, zero queue-full drops, 0 B heap used, and an observed
  maximum wakeup lateness of 2,235 us over the observed run. No I2C/CNN workload.
- **BLE to this Mac:** 600 packets received over an approximately 12-second
  subscription, one sequence gap, zero invalid/duplicate/old packets. Arrival
  gaps from the CSV: median 29.41 ms, p95 30.66 ms, maximum 90.28 ms. Multiple
  notifications can arrive together, so a ~30 ms median gap still coexists with
  ~50 packets/s. This is delivery evidence, not a 25 ms latency guarantee.
- **BLE board side:** 35,804 B heap used while advertising, 35,836 B while
  connected in sampled status lines; zero missed 5 ms deadlines, maximum wakeup
  lateness 1,562 us in the capture. Queue drops before connection were expected;
  the counter stopped growing once the subscriber was active.
- **Host UDP loopback:** all 100 simulated packets collected into CSV, zero
  sequence gaps. This checks host sockets/decoding/logging, not the Wi-Fi radio.
- **Wi-Fi, disconnected:** full image flashed; radio initialized and retried the
  build-only SSID. Sampled heap use was 37,004 B, maximum wakeup lateness 1,242 us,
  zero missed 5 ms deadlines and queue-full drops in the capture. Association,
  DHCP, and transmission were subsequently tested in the follow-up below.
- **Restoration:** the saved sensor/classifier image was flashed back and its
  serial output verified at 228–229 fps, 27–28 classifications/s, ~3,570 us
  preprocessing and ~13.0 ms inference. No FIFO overflows or I2C errors in the
  restoration capture. This was one connected IMU, not four-IMU validation.

Raw captures from this run are under `telemetry-captures/` (ignored by Git).

### Wi-Fi cross-band follow-up, October 22, 2026

With real settings in the ignored local file, the S3 joined the 2.4 GHz SSID
and obtained `10.0.0.206/24`. Telemetry reached the Mac at `10.0.0.20`, which
remained on the user's 6 GHz SSID on the same router. No router or Mac Wi-Fi
settings were changed. This verifies that this router permits the cross-band
path; separate SSIDs alone do not imply isolated LANs.

| Capture | Packets | Sequence gaps | First-to-last span | Average receive rate | Arrival gap p95 | Maximum gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial connection capture | 1,065 | 65 | 21.467 s | 49.56 Hz | 106.80 ms | 391.15 ms |
| Established connection | 998 | 3 | 19.995 s | 49.86 Hz | 90.66 ms | 202.14 ms |

Both collectors reported zero malformed, duplicate, or old packets. Sequence
gaps can originate in application coalescing/submission drops as well as network
delivery, so these counts are not measured RF packet loss. At least some gaps
in the initial capture coincided with 42 bounded send failures. That cumulative
failure count did not increase during the subsequent steady board capture.
The host delivered packets in bursts despite the board's regular publications.
These short captures show this setup's behavior; they do not establish an
inherent latency ranking between Wi-Fi and BLE, or one-way latency.

The timer probe recorded **587,303 us maximum lateness and 117 missed 5 ms
ticks during initial bring-up**. Those counters stayed constant during the
15-second established-connection serial capture, with no queue-full drops.
This is a concrete reason to finish radio bring-up before enabling critical
acquisition/control and to test reconnection separately. It is not proof that
radio operations meet real-time deadlines with sensors and the CNN active.

Sampled heap usage was 37,692 B in steady status lines; the highest sampled
usage was 37,896 B (64,504 B free of the reserved 102,400 B). Allocation peaks
between status lines and behavior under radio reconnection remain unmeasured.
Captures are `wifi-connected-test.{csv,log}` and `wifi-steady-test.{csv,log}`
inside `telemetry-captures/`. The sensor image was restored after testing.
