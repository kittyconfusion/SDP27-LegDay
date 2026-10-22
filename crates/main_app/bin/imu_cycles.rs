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

use esp_hal::xtensa_lx::timer::get_cycle_count;
use esp_println::println;

/// Loop iterations per probe: long enough that call overhead vanishes.
const ITERATIONS: u32 = 10_000;

/// 16-byte aligned arrays for the vector-load probes.
#[repr(C, align(16))]
struct Aligned16([f32; 1_024]);

static mut X: Aligned16 = Aligned16([0.5; 1_024]);
static mut Y: Aligned16 = Aligned16([0.25; 1_024]);

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
fn probe_load_use() -> f32 {
    let pointer = unsafe { core::ptr::addr_of!(X.0).cast::<f32>() };
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
fn probe_dot_scalar_loads() -> f32 {
    let (x, y) = unsafe {
        (
            core::ptr::addr_of!(X.0).cast::<f32>(),
            core::ptr::addr_of!(Y.0).cast::<f32>(),
        )
    };
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
fn probe_dot_vector_loads() -> f32 {
    let (x, y) = unsafe {
        (
            core::ptr::addr_of!(X.0).cast::<f32>(),
            core::ptr::addr_of!(Y.0).cast::<f32>(),
        )
    };
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
fn probe_dot_compiled() -> f32 {
    let (x, y) = unsafe { (&*core::ptr::addr_of!(X.0), &*core::ptr::addr_of!(Y.0)) };
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

/// The vector dot product (the classifier's real kernel, `cnn::dot`) with
/// one operand read from flash through the data cache, as model weights are.
/// `span_floats` sets how much flash is swept: small spans stay in the
/// cache, large ones keep missing it. Returns cycles per multiply-add.
#[esp_hal::ram]
fn probe_dot_from_flash(span_floats: usize) -> f32 {
    use legday_classifier::cnn::{DEMO_MODEL, dot};

    // The demo model's bytes are 16-byte aligned in flash; read them as floats.
    let bytes = &DEMO_MODEL.0;
    let floats = bytes.len() / 4 / 1_024 * 1_024;
    // SAFETY: DEMO_MODEL is 16-byte aligned (Aligned<[u8]>), at least
    // `floats * 4` bytes long, and any bit pattern is a valid f32.
    let flash: &[f32] =
        unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), floats) };
    let span = span_floats.min(floats);
    let y = unsafe { &*core::ptr::addr_of!(Y.0) };

    let chunks = 64;
    let start = get_cycle_count();
    let mut total = 0.0_f32;
    for round in 0..chunks {
        // Walk through `span` floats of flash in 1,024-float pieces.
        let offset = (round * 1_024) % span;
        total += dot(&flash[offset..offset + 1_024], y);
    }
    black_box(total);
    get_cycle_count().wrapping_sub(start) as f32 / (chunks * 1_024) as f32
}

/// `cnn::dot` with both operands in internal RAM, through the same call
/// path as the classifier. Returns cycles per multiply-add.
#[esp_hal::ram]
fn probe_dot_sram_via_cnn() -> f32 {
    let (x, y) = unsafe { (&*core::ptr::addr_of!(X.0), &*core::ptr::addr_of!(Y.0)) };
    let start = get_cycle_count();
    let mut total = 0.0_f32;
    for _ in 0..64 {
        total += legday_classifier::cnn::dot(black_box(x), black_box(y));
    }
    black_box(total);
    get_cycle_count().wrapping_sub(start) as f32 / (64 * 1_024) as f32
}

/// Runs every probe and prints the table.
pub fn run() {
    println!();
    println!("CPU cycle probes (hand-written assembly in IRAM, hardware loops)");
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
        probe_load_use()
    );
    println!();
    println!("  dot product, cycles per multiply-add:");
    println!(
        "    compiled Rust, four sums          {:>5.2}",
        probe_dot_compiled()
    );
    println!(
        "    assembly, 32-bit loads            {:>5.2}",
        probe_dot_scalar_loads()
    );
    println!(
        "    assembly, ESP-DSP vector loads    {:>5.2}",
        probe_dot_vector_loads()
    );
    // Warm the cache once, then time: 4 kB stays cached, 128 kB does not fit.
    let _ = probe_dot_from_flash(1_024);
    println!(
        "    cnn::dot, one operand in flash, 4 kB (cached)    {:>5.2}",
        probe_dot_from_flash(1_024)
    );
    println!(
        "    cnn::dot, one operand in flash, 128 kB (missing) {:>5.2}",
        probe_dot_from_flash(32_768)
    );
    println!(
        "    cnn::dot, both in SRAM                           {:>5.2}",
        probe_dot_sram_via_cnn()
    );
}
