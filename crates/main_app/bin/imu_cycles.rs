//! # Cycle-level probes of the ESP32-S3's arithmetic (the `k` key)
//!
//! Each probe runs one hand-written instruction sequence in a hardware
//! loop (`loop`, which repeats its body with no branch cost) and reports
//! CPU cycles per iteration, read from the CPU's cycle counter. Because the
//! sequences are written in assembly, the numbers describe the chip itself,
//! not any compiler.
//!
//! Two kinds of number come out:
//!
//! * **Latency**: how long an instruction takes before its result can be
//!   used. Measured with a chain where every instruction needs the previous
//!   one's result.
//! * **Throughput**: how often an instruction can start when the results
//!   do not depend on each other.
//!
//! The last probes compare the compiled Rust loop with Espressif's ESP-DSP
//! dot-product loop (`dsps_dotprod_f32_aes3.S`), run here verbatim.
//!
//! All probe code is placed in internal RAM (`#[esp_hal::ram]`) so that
//! instruction fetch from flash does not blur the numbers.
//!
//! Every hardware loop is preceded by `.balign 8; nop; nop.n`: the 3-byte
//! `nop` and 2-byte `nop.n` put the 3-byte `loop` instruction at an address
//! of 5 mod 8, so the loop body starts on an 8-byte boundary. The Xtensa
//! loop hardware requires an aligned body; GNU's assembler pads this
//! automatically, LLVM's (used by Rust) does not, and a misaligned body
//! crashes.

use core::arch::asm;
use core::hint::black_box;
use core::sync::atomic::{AtomicUsize, Ordering};

use esp_hal::xtensa_lx::timer::get_cycle_count;
use esp_println::println;

/// Loop iterations per probe: long enough that call overhead vanishes.
const ITERATIONS: u32 = 10_000;

/// Floats in each probe operand.
const OPERAND_FLOATS: usize = 1_024;

/// Two 16-byte aligned operands of `OPERAND_FLOATS` carved out of borrowed
/// memory (the classifier's scratch, so the probes own no RAM of their
/// own), filled with fixed values. `None` if `memory` is too small.
fn operands(memory: &mut [f32]) -> Option<(&[f32], &[f32])> {
    let skip = (16 - memory.as_ptr() as usize % 16) % 16 / 4;
    let memory = memory.get_mut(skip..skip + 2 * OPERAND_FLOATS)?;
    let (x, y) = memory.split_at_mut(OPERAND_FLOATS);

    x.fill(0.5);
    y.fill(0.25);

    Some((x, y))
}

/// Cycles per iteration of `body`, measured over `ITERATIONS`.
fn per_iteration(body: impl FnOnce()) -> f32 {
    let start = get_cycle_count();
    body();
    get_cycle_count().wrapping_sub(start) as f32 / ITERATIONS as f32
}

// Each probe is one function so its assembly can be read on its own. The
// FPU registers it uses are declared as outputs so the compiler keeps
// nothing of its own in them.

/// Empty-ish loop: the cost of the loop itself (should be ~1 cycle).
#[esp_hal::ram]
fn probe_nop() -> f32 {
    per_iteration(|| unsafe {
        asm!(".balign 8", "nop", "nop.n",
            "loop {n}, 1f", "nop", "1:", n = in(reg) ITERATIONS, options(nostack, nomem));
    })
}

/// madd.s chain: each needs the previous result -> madd.s latency.
#[esp_hal::ram]
fn probe_madd_latency() -> f32 {
    per_iteration(|| unsafe {
        asm!(
            ".balign 8", "nop", "nop.n",
            "loop {n}, 1f",
            "madd.s f0, f1, f2",
            "1:",
            n = in(reg) ITERATIONS,
            inout("f0") 0.0_f32 => _, in("f1") 1.000_1_f32, in("f2") 0.999_f32,
            options(nostack, nomem)
        );
    })
}

/// Four independent madd.s per iteration -> throughput (cycles per 4).
#[esp_hal::ram]
fn probe_madd_throughput() -> f32 {
    per_iteration(|| unsafe {
        asm!(
            ".balign 8", "nop", "nop.n",
            "loop {n}, 1f",
            "madd.s f0, f4, f5",
            "madd.s f1, f4, f5",
            "madd.s f2, f4, f5",
            "madd.s f3, f4, f5",
            "1:",
            n = in(reg) ITERATIONS,
            inout("f0") 0.0_f32 => _, inout("f1") 0.0_f32 => _,
            inout("f2") 0.0_f32 => _, inout("f3") 0.0_f32 => _,
            in("f4") 1.000_1_f32, in("f5") 0.999_f32,
            options(nostack, nomem)
        );
    })
}

/// mul.s then add.s on its result, accumulating: what the compiler emits
/// for `sum += x * y` (latency of the pair).
#[esp_hal::ram]
fn probe_mul_add_chain() -> f32 {
    per_iteration(|| unsafe {
        asm!(
            ".balign 8", "nop", "nop.n",
            "loop {n}, 1f",
            "mul.s f3, f1, f2",
            "add.s f0, f0, f3",
            "1:",
            n = in(reg) ITERATIONS,
            inout("f0") 0.0_f32 => _, in("f1") 1.000_1_f32, in("f2") 0.999_f32, out("f3") _,
            options(nostack, nomem)
        );
    })
}

/// add.s chain -> add.s latency.
#[esp_hal::ram]
fn probe_add_latency() -> f32 {
    per_iteration(|| unsafe {
        asm!(
            ".balign 8", "nop", "nop.n",
            "loop {n}, 1f",
            "add.s f0, f0, f1",
            "1:",
            n = in(reg) ITERATIONS,
            inout("f0") 0.0_f32 => _, in("f1") 1.0e-6_f32,
            options(nostack, nomem)
        );
    })
}

/// Load a float from internal RAM, then use it right away (load-use).
#[esp_hal::ram]
fn probe_load_use(x: &[f32]) -> f32 {
    let pointer = x.as_ptr();
    per_iteration(|| unsafe {
        asm!(
            ".balign 8", "nop", "nop.n",
            "loop {n}, 1f",
            "lsi f1, {p}, 0",
            "add.s f0, f0, f1",
            "1:",
            n = in(reg) ITERATIONS, p = in(reg) pointer,
            inout("f0") 0.0_f32 => _, out("f1") _,
            options(nostack, readonly)
        );
    })
}

/// The scalar form of a good dot-product loop: four accumulators, ordinary
/// 32-bit loads (8 loads + 4 madd.s + 2 pointer adds per 4 multiply-adds).
/// Returns cycles per multiply-add.
#[esp_hal::ram]
fn probe_dot_scalar_loads(x: &[f32], y: &[f32]) -> f32 {
    let (x, y) = (x.as_ptr(), y.as_ptr());
    let groups = 256_u32; // 1,024 floats in groups of 4
    let start = get_cycle_count();
    for _ in 0..ITERATIONS / 256 {
        unsafe {
            asm!(
                ".balign 8", "nop", "nop.n",
            "loop {n}, 1f",
                "lsi f4, {x}, 0",
                "lsi f8, {y}, 0",
                "lsi f5, {x}, 4",
                "lsi f9, {y}, 4",
                "madd.s f0, f4, f8",
                "lsi f6, {x}, 8",
                "lsi f10, {y}, 8",
                "madd.s f1, f5, f9",
                "lsi f7, {x}, 12",
                "lsi f11, {y}, 12",
                "madd.s f2, f6, f10",
                "addi {x}, {x}, 16",
                "addi {y}, {y}, 16",
                "madd.s f3, f7, f11",
                "1:",
                n = in(reg) groups,
                x = inout(reg) x => _, y = inout(reg) y => _,
                out("f0") _, out("f1") _, out("f2") _, out("f3") _,
                out("f4") _, out("f5") _, out("f6") _, out("f7") _,
                out("f8") _, out("f9") _, out("f10") _, out("f11") _,
                options(nostack, readonly)
            );
        }
    }
    let macs = (ITERATIONS / 256) * groups * 4;
    get_cycle_count().wrapping_sub(start) as f32 / macs as f32
}

/// ESP-DSP's ESP32-S3 dot-product loop, verbatim: 128-bit vector loads
/// (four floats each) and four madd.s. Returns cycles per multiply-add.
#[esp_hal::ram]
fn probe_dot_vector_loads(x: &[f32], y: &[f32]) -> f32 {
    let (x, y) = (x.as_ptr(), y.as_ptr());
    let groups = 256_u32;
    let start = get_cycle_count();
    for _ in 0..ITERATIONS / 256 {
        unsafe {
            asm!(
                ".balign 8", "nop", "nop.n",
            "loopnez {n}, 1f",
                "ee.ldf.128.ip f11, f10, f9, f8, {x}, 16",
                "ee.ldf.128.ip f7, f6, f5, f4, {y}, 16",
                "madd.s f0, f4, f8",
                "madd.s f1, f5, f9",
                "madd.s f2, f6, f10",
                "madd.s f3, f7, f11",
                "1:",
                n = in(reg) groups,
                x = inout(reg) x => _, y = inout(reg) y => _,
                out("f0") _, out("f1") _, out("f2") _, out("f3") _,
                out("f4") _, out("f5") _, out("f6") _, out("f7") _,
                out("f8") _, out("f9") _, out("f10") _, out("f11") _,
                options(nostack, readonly)
            );
        }
    }
    let macs = (ITERATIONS / 256) * groups * 4;
    get_cycle_count().wrapping_sub(start) as f32 / macs as f32
}

/// The same four-accumulator dot product written in plain Rust, compiled by
/// the project's toolchain. Returns cycles per multiply-add.
#[esp_hal::ram]
fn probe_dot_compiled(x: &[f32], y: &[f32]) -> f32 {
    let start = get_cycle_count();
    let mut sums = [0.0_f32; 4];
    for _ in 0..ITERATIONS / 256 {
        for (a, b) in x.chunks_exact(4).zip(y.chunks_exact(4)) {
            sums[0] += a[0] * b[0];
            sums[1] += a[1] * b[1];
            sums[2] += a[2] * b[2];
            sums[3] += a[3] * b[3];
        }
    }
    black_box(sums);
    let macs = (ITERATIONS / 256) * 1_024;
    get_cycle_count().wrapping_sub(start) as f32 / macs as f32
}

/// Where PSRAM is mapped, set by `main` after `Psram::new` (0 = none).
pub static PSRAM_BASE: AtomicUsize = AtomicUsize::new(0);
pub static PSRAM_BYTES: AtomicUsize = AtomicUsize::new(0);

/// PSRAM as a float slice, if it was initialized.
pub fn psram_floats() -> Option<&'static mut [f32]> {
    let base = PSRAM_BASE.load(Ordering::Acquire);
    let bytes = PSRAM_BYTES.load(Ordering::Acquire);

    if base == 0 || bytes < 4 {
        return None;
    }

    // SAFETY: `main` stores the range `Psram::raw_parts` returned, which is
    // mapped, 4-byte aligned memory used for nothing else in this firmware.
    // Callers use it one at a time from the main loop.
    Some(unsafe { core::slice::from_raw_parts_mut(base as *mut f32, bytes / 4) })
}

/// The vector dot product (the classifier's real kernel, `cnn::dot`) with
/// one operand in external memory (flash or PSRAM), read through the data
/// cache as model weights are. `span_floats` sets how much is swept: a small
/// span stays in the cache, a large one keeps missing it. Returns cycles per
/// multiply-add.
#[esp_hal::ram]
fn probe_dot_external(external: &[f32], y: &[f32], span_floats: usize) -> f32 {
    let span = span_floats.min(external.len() / 1_024 * 1_024).max(1_024);

    let chunks = 64;
    let start = get_cycle_count();
    let mut total = 0.0_f32;
    for round in 0..chunks {
        // Walk through `span` floats in 1,024-float pieces.
        let offset = (round * 1_024) % span;
        total += legday_classifier::cnn::dot(&external[offset..offset + 1_024], y);
    }
    black_box(total);
    get_cycle_count().wrapping_sub(start) as f32 / (chunks * 1_024) as f32
}

/// The demo model's bytes in flash, viewed as floats (16-byte aligned).
fn flash_floats() -> &'static [f32] {
    let bytes = &legday_classifier::cnn::DEMO_MODEL.0;
    // SAFETY: DEMO_MODEL is 16-byte aligned (Aligned<[u8]>), at least
    // `len / 4 * 4` bytes long, and any bit pattern is a valid f32.
    unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) }
}

/// `cnn::dot` with both operands in internal RAM, through the same call
/// path as the classifier. Returns cycles per multiply-add.
#[esp_hal::ram]
fn probe_dot_sram_via_cnn(x: &[f32], y: &[f32]) -> f32 {
    let start = get_cycle_count();
    let mut total = 0.0_f32;
    for _ in 0..64 {
        total += legday_classifier::cnn::dot(black_box(x), black_box(y));
    }
    black_box(total);
    get_cycle_count().wrapping_sub(start) as f32 / (64 * 1_024) as f32
}

/// Reports how the flash controller (SPI0, which serves the cache) is set
/// up right now: its read mode and clock divider. Read straight from the
/// hardware registers, so it shows what the bootloader actually did, not
/// what the image header asked for.
fn print_flash_mode() {
    // SPI0 = 0x6000_3000. CTRL (0x08): read-mode bits. CLOCK (0x14): divider.
    // CORE_CLK_SEL (0xEC... not needed): the MSPI core clock is 80 MHz here.
    const SPI0_CTRL: *const u32 = 0x6000_3008 as *const u32;
    const SPI0_CLOCK: *const u32 = 0x6000_3014 as *const u32;

    // SAFETY: reading two always-present, read-safe hardware registers.
    let (ctrl, clock) = unsafe { (SPI0_CTRL.read_volatile(), SPI0_CLOCK.read_volatile()) };

    let mode = if ctrl & (1 << 24) != 0 {
        "QIO (4 lines, address + data)"
    } else if ctrl & (1 << 20) != 0 {
        "QOUT (4 lines, data only)"
    } else if ctrl & (1 << 23) != 0 {
        "DIO (2 lines, address + data)"
    } else if ctrl & (1 << 14) != 0 {
        "DOUT (2 lines, data only)"
    } else {
        "single line"
    };

    let divider = if clock & (1 << 31) != 0 {
        1
    } else {
        ((clock >> 16) & 0xff) + 1
    };

    println!(
        "  flash controller: {} | clock divider {} (CTRL 0x{:08X}, CLOCK 0x{:08X})",
        mode, divider, ctrl, clock
    );
}

/// Runs every probe and prints the table. `scratch` is borrowed memory for
/// the probe operands (the classifier's scratch; 8 kB of it is used).
pub fn run(scratch: &mut [f32]) {
    println!();
    println!("CPU cycle probes (hand-written assembly in IRAM, hardware loops)");

    let Some((x, y)) = operands(scratch) else {
        println!("  (needs 8 kB of borrowed scratch; the classifier is disabled)");
        return;
    };

    print_flash_mode();
    println!(
        "  loop overhead (nop)                 {:>5.2} cycles/iteration",
        probe_nop()
    );
    println!(
        "  add.s latency                       {:>5.2} cycles",
        probe_add_latency()
    );
    println!(
        "  madd.s latency (dependent chain)    {:>5.2} cycles",
        probe_madd_latency()
    );
    println!(
        "  madd.s x4 independent               {:>5.2} cycles per 4",
        probe_madd_throughput()
    );
    println!(
        "  mul.s -> add.s accumulate chain     {:>5.2} cycles",
        probe_mul_add_chain()
    );
    println!(
        "  lsi load -> add.s use               {:>5.2} cycles",
        probe_load_use(x)
    );
    println!();
    println!("  dot product, cycles per multiply-add:");
    println!(
        "    compiled Rust, four sums          {:>5.2}",
        probe_dot_compiled(x, y)
    );
    println!(
        "    assembly, 32-bit loads            {:>5.2}",
        probe_dot_scalar_loads(x, y)
    );
    println!(
        "    assembly, ESP-DSP vector loads    {:>5.2}",
        probe_dot_vector_loads(x, y)
    );
    // Warm the cache once, then time: 4 kB stays cached, 128 kB does not fit.
    let flash = flash_floats();
    let _ = probe_dot_external(flash, y, 1_024);
    println!(
        "    cnn::dot, one operand in flash, 4 kB (cached)    {:>5.2}",
        probe_dot_external(flash, y, 1_024)
    );
    println!(
        "    cnn::dot, one operand in flash, 128 kB (missing) {:>5.2}",
        probe_dot_external(flash, y, 32_768)
    );

    if let Some(psram) = psram_floats() {
        // Fill 128 kB of PSRAM, then sweep it the same way.
        for (index, value) in psram[..32_768].iter_mut().enumerate() {
            *value = (index % 13) as f32 * 0.5;
        }
        let _ = probe_dot_external(psram, y, 1_024);
        println!(
            "    cnn::dot, one operand in PSRAM, 4 kB (cached)    {:>5.2}",
            probe_dot_external(psram, y, 1_024)
        );
        println!(
            "    cnn::dot, one operand in PSRAM, 128 kB (missing) {:>5.2}",
            probe_dot_external(psram, y, 32_768)
        );
    } else {
        println!("    (PSRAM not initialized)");
    }

    println!(
        "    cnn::dot, both in SRAM                           {:>5.2}",
        probe_dot_sram_via_cnn(x, y)
    );
    print_flash_mode();
}
