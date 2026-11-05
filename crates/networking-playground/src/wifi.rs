use super::{ACCEPTED, TX_FAILED, now_us};
use core::sync::atomic::Ordering::Relaxed;
use embassy_executor::Spawner;
use embassy_net::{
    Runner, Stack, StackResources,
    udp::{PacketMetadata, UdpSocket},
};
use embassy_time::{Duration, Timer, with_timeout};
use esp_hal::{peripherals::WIFI, rng::Rng};
use esp_println::println;
use esp_radio::wifi::{
    AuthenticationMethodConfig, Config, ControllerConfig, CountryInfo, Interface, PowerSaveMode,
    WifiController, sta::StationConfig,
};
use networking_playground::{Consumer, take_latest};
use static_cell::StaticCell;

static RESOURCES: StaticCell<StackResources<2>> = StaticCell::new();
const SSID: &str = match option_env!("LEGDAY_WIFI_SSID") {
    Some(s) => s,
    None => "LegDay-BUILD-ONLY",
};
const PASSWORD: &str = match option_env!("LEGDAY_WIFI_PASSWORD") {
    Some(s) => s,
    None => "build-only-password",
};
const TARGET: &str = match option_env!("LEGDAY_PC_IP") {
    Some(s) => s,
    None => "192.0.2.1",
};

pub fn start(spawner: Spawner, device: WIFI<'static>, rx: Consumer) {
    // Defaults keep the complete radio code linked during build-only size
    // comparisons. The flash wrapper requires real configuration. Never log it.
    assert!(
        !SSID.is_empty(),
        "Set LEGDAY_WIFI_SSID in the local config before flashing Wi-Fi"
    );
    assert!(
        !PASSWORD.is_empty(),
        "This experiment expects a WPA2 password"
    );
    let target: embassy_net::Ipv4Address = TARGET
        .parse()
        .expect("Set LEGDAY_PC_IP to the laptop IPv4 address");
    let station = Config::Station(
        StationConfig::default()
            .with_ssid(SSID.try_into().expect("SSID too long"))
            .with_authentication(AuthenticationMethodConfig::Wpa2Personal(
                PASSWORD.try_into().expect("Invalid WPA2 password"),
            )),
    );
    // Small bounded pools for status traffic, not a maximum-throughput setup.
    let config = ControllerConfig::default()
        .with_country_info(CountryInfo::from(*b"US"))
        .with_static_rx_buf_num(4)
        .with_dynamic_rx_buf_num(8)
        .with_dynamic_tx_buf_num(4)
        .with_rx_queue_size(2)
        .with_tx_queue_size(2)
        .with_ampdu_rx_enable(false)
        .with_ampdu_tx_enable(false)
        .with_rx_ba_win(2)
        .with_initial_config(station);
    let interface = Interface::station();
    let mut controller = WifiController::new(device, config).expect("Wi-Fi init failed");
    // Keep the modem awake for the latency experiment; costs more power.
    controller.set_power_saving(PowerSaveMode::None).unwrap();
    let rng = Rng::new();
    let seed = ((rng.random() as u64) << 32) | rng.random() as u64;
    let (stack, runner) = embassy_net::new(
        interface,
        embassy_net::Config::dhcpv4(Default::default()),
        RESOURCES.init(StackResources::new()),
        seed,
    );
    spawner.spawn(connection(controller).unwrap());
    spawner.spawn(network(runner).unwrap());
    spawner.spawn(transmit(stack, rx, target).unwrap());
}

#[embassy_executor::task]
async fn connection(mut controller: WifiController<'static>) {
    loop {
        match controller.connect_async().await {
            Ok(_) => {
                println!("Wi-Fi associated; waiting for DHCP");
                let _ = controller.wait_for_disconnect_async().await;
                println!("Wi-Fi disconnected");
            }
            Err(_) => println!("Wi-Fi connection failed; retry in 3 s"),
        }
        Timer::after_secs(3).await;
    }
}

#[embassy_executor::task]
async fn network(mut runner: Runner<'static, Interface>) {
    runner.run().await
}

#[embassy_executor::task]
async fn transmit(stack: Stack<'static>, mut rx: Consumer, target: embassy_net::Ipv4Address) {
    let mut rx_meta = [PacketMetadata::EMPTY; 1];
    let mut tx_meta = [PacketMetadata::EMPTY; 2];
    let mut rx_buffer = [0; 20];
    let mut tx_buffer = [0; 40];
    let mut socket = UdpSocket::new(
        stack,
        &mut rx_meta,
        &mut rx_buffer,
        &mut tx_meta,
        &mut tx_buffer,
    );
    socket.bind(5006).unwrap();
    let mut was_up = false;
    loop {
        Timer::after_millis(20).await;
        let packet = take_latest(&mut rx, now_us());
        let up = stack.is_link_up() && stack.is_config_up();
        if up && !was_up {
            println!(
                "Wi-Fi IP: {:?}; destination {}:5005",
                stack.config_v4().map(|c| c.address),
                target
            );
        }
        was_up = up;
        if !up {
            continue;
        }
        if let Some(packet) = packet {
            // A slow radio never makes the producer await. No backlog replay.
            match with_timeout(
                Duration::from_millis(10),
                socket.send_to(&packet.encode(), (target, 5005)),
            )
            .await
            {
                Ok(Ok(())) => {
                    ACCEPTED.fetch_add(1, Relaxed);
                }
                _ => {
                    TX_FAILED.fetch_add(1, Relaxed);
                }
            }
        }
    }
}
