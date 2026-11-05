#![no_std]
#![no_main]

#[cfg(any(
    all(feature = "wifi", feature = "ble"),
    all(feature = "offline", feature = "wifi"),
    all(feature = "offline", feature = "ble"),
    not(any(feature = "offline", feature = "wifi", feature = "ble"))
))]
compile_error!("Select exactly one of offline, wifi, ble; use --no-default-features.");

#[cfg(feature = "ble")]
mod ble;
#[cfg(feature = "wifi")]
mod wifi;

use core::sync::atomic::{AtomicU32, Ordering::Relaxed};
use embassy_executor::Spawner;
use embassy_time::{Duration, Instant, Timer};
use esp_backtrace as _;
use esp_hal::{clock::CpuClock, timer::timg::TimerGroup};
use esp_println::println;
#[cfg(feature = "offline")]
use networking_playground::Consumer;
use networking_playground::{Packet, PacketQueue, Producer};
use static_cell::StaticCell;

esp_bootloader_esp_idf::esp_app_desc!();

static QUEUE: StaticCell<PacketQueue> = StaticCell::new();
static LATE_MAX_US: AtomicU32 = AtomicU32::new(0);
static MISSED_TICKS: AtomicU32 = AtomicU32::new(0);
static QUEUE_FULL: AtomicU32 = AtomicU32::new(0);
static PRODUCED: AtomicU32 = AtomicU32::new(0);
static ACCEPTED: AtomicU32 = AtomicU32::new(0);
static TX_FAILED: AtomicU32 = AtomicU32::new(0);

pub fn now_us() -> u32 {
    Instant::now().as_micros() as u32
}

/// Synthetic 200 Hz wakeup probe, publishing every fourth tick (50 Hz).
/// This measures executor wakeup jitter, NOT I2C acquisition or CNN latency.
#[embassy_executor::task]
async fn produce(mut tx: Producer) {
    let period = Duration::from_millis(5);
    let mut next = Instant::now() + period;
    let mut tick = 0_u32;
    let mut sequence = 0_u16;
    loop {
        Timer::at(next).await;
        let now = Instant::now();
        let late = now.saturating_duration_since(next).as_micros();
        LATE_MAX_US.fetch_max(late.min(u32::MAX as u64) as u32, Relaxed);
        MISSED_TICKS.fetch_add((late / 5_000) as u32, Relaxed);
        // Skip missed deadlines rather than issuing a catch-up burst.
        next = if late >= 5_000 {
            now + period
        } else {
            next + period
        };
        tick = tick.wrapping_add(1);
        if tick % 4 == 0 {
            PRODUCED.fetch_add(1, Relaxed);
            if tx.enqueue(Packet::synthetic(sequence, now_us())).is_err() {
                QUEUE_FULL.fetch_add(1, Relaxed);
            }
            sequence = sequence.wrapping_add(1);
        }
    }
}

#[cfg(feature = "offline")]
#[embassy_executor::task]
async fn discard(mut rx: Consumer) {
    loop {
        Timer::after_millis(20).await;
        if networking_playground::take_latest(&mut rx, now_us()).is_some() {
            ACCEPTED.fetch_add(1, Relaxed);
        }
    }
}

#[esp_rtos::main]
async fn main(spawner: Spawner) -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    println!("\nLegDay NETWORK PLAYGROUND: simulated data; no sensors, CNN, or actuators");
    #[cfg(feature = "wifi")]
    println!("Transport: Wi-Fi UDP (BLE excluded)");
    #[cfg(feature = "ble")]
    println!("Transport: BLE GATT (Wi-Fi excluded)");
    #[cfg(feature = "offline")]
    println!("Transport: offline baseline (no radio initialized)");

    // Same 100 KiB internal heap in every variant for comparable reservations.
    // Reclaimed bootloader memory is distinct from the sensor ELF's stack.
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 64 * 1024);
    esp_alloc::heap_allocator!(size: 36 * 1024);
    let timg0 = TimerGroup::new(peripherals.TIMG0);
    esp_rtos::start(timg0.timer0, peripherals.FROM_CPU_INTR0);
    let (tx, rx) = QUEUE.init(PacketQueue::new()).split();
    spawner.spawn(produce(tx).unwrap());

    #[cfg(feature = "wifi")]
    wifi::start(spawner, peripherals.WIFI, rx);
    #[cfg(feature = "ble")]
    spawner.spawn(ble::run(peripherals.BT, rx).unwrap());
    #[cfg(feature = "offline")]
    spawner.spawn(discard(rx).unwrap());

    println!("Packet: 20 B at 50 Hz; queue capacity 4; age cutoff 100 ms");
    println!("accepted=queued to transport (NOT proof of laptop receipt)");
    let mut min_free = esp_alloc::HEAP.free();
    loop {
        Timer::after_secs(2).await;
        let free = esp_alloc::HEAP.free();
        min_free = min_free.min(free);
        println!(
            "net: produced={} accepted={} tx_failed={} queue_full={} | wake_late_max={} us missed_5ms_ticks={} | heap used={} free={} min_observed_free={}",
            PRODUCED.load(Relaxed),
            ACCEPTED.load(Relaxed),
            TX_FAILED.load(Relaxed),
            QUEUE_FULL.load(Relaxed),
            LATE_MAX_US.load(Relaxed),
            MISSED_TICKS.load(Relaxed),
            esp_alloc::HEAP.used(),
            free,
            min_free,
        );
    }
}
