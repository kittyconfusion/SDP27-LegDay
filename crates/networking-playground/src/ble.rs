use super::{ACCEPTED, TX_FAILED, now_us};
use core::sync::atomic::Ordering::Relaxed;
use embassy_futures::{
    join::join,
    select::{Either, select},
};
use embassy_time::{Duration, Timer, with_timeout};
use esp_hal::{peripherals::BT, rng::Rng};
use esp_println::println;
use esp_radio::ble::controller::BleConnector;
use networking_playground::{Consumer, take_latest};
use trouble_host::prelude::*;

#[gatt_server(connections_max = 1)]
struct Server {
    telemetry: TelemetryService,
}

#[gatt_service(uuid = "8ea2c100-97b5-4b99-a361-47c27d8e4250")]
struct TelemetryService {
    #[characteristic(uuid = "8ea2c101-97b5-4b99-a361-47c27d8e4250", read, notify)]
    packet: [u8; 20],
}

#[embassy_executor::task]
pub async fn run(device: BT<'static>, mut rx: Consumer) {
    let connector = BleConnector::new(device, Default::default()).expect("BLE init failed");
    let controller = ExternalController::<_, 10>::new(connector);
    let mut address = [0; 6];
    Rng::new().read(&mut address);
    address[5] |= 0xc0; // BLE static random address; identify by service UUID.
    let mut resources: HostResources<_, DefaultPacketPool, 1, 2> = HostResources::new();
    let stack = trouble_host::new(controller, &mut resources)
        .set_random_address(Address::random(address))
        .build();
    let mut runner = stack.runner();
    let mut peripheral = stack.peripheral();
    let server = Server::new_with_config(GapConfig::Peripheral(PeripheralConfig {
        name: "LegDay-Play",
        appearance: &appearance::UNKNOWN,
    }))
    .unwrap();
    let _ = join(runner.run(), async {
        // UUID bytes in BLE advertising order (least significant byte first).
        let uuid = [
            0x50, 0x42, 0x8e, 0x7d, 0xc2, 0x47, 0x61, 0xa3, 0x99, 0x4b, 0xb5, 0x97, 0x00, 0xc1,
            0xa2, 0x8e,
        ];
        let mut adv = [0; 31];
        let adv_len = AdStructure::encode_slice(
            &[
                AdStructure::Flags(LE_GENERAL_DISCOVERABLE | BR_EDR_NOT_SUPPORTED),
                AdStructure::CompleteServiceUuids128(&[uuid]),
            ],
            &mut adv,
        )
        .unwrap();
        let mut scan = [0; 31];
        let scan_len =
            AdStructure::encode_slice(&[AdStructure::CompleteLocalName(b"LegDay-Play")], &mut scan)
                .unwrap();
        loop {
            println!("BLE advertising as LegDay-Play");
            let Ok(acceptor) = peripheral
                .advertise(
                    &Default::default(),
                    Advertisement::ConnectableScannableUndirected {
                        adv_data: &adv[..adv_len],
                        scan_data: &scan[..scan_len],
                    },
                )
                .await
            else {
                Timer::after_secs(1).await;
                continue;
            };
            let Ok(connection) = acceptor.accept().await else {
                continue;
            };
            let Ok(connection) = connection.with_attribute_server(&server) else {
                continue;
            };
            // Remove anything accumulated during discovery; send fresh data only.
            let _ = take_latest(&mut rx, now_us());
            println!("BLE connected; waiting for notification subscription");
            let events = async {
                loop {
                    match connection.next().await {
                        GattConnectionEvent::Disconnected { .. } => break,
                        GattConnectionEvent::Gatt { event } => {
                            if let Ok(reply) = event.accept() {
                                reply.send().await;
                            }
                        }
                        _ => {}
                    }
                }
            };
            let notifications = async {
                loop {
                    Timer::after_millis(20).await;
                    if let Some(packet) = take_latest(&mut rx, now_us()) {
                        if !server.telemetry.packet.should_notify(&connection) {
                            continue;
                        }
                        match with_timeout(
                            Duration::from_millis(10),
                            server
                                .telemetry
                                .packet
                                .notify(&connection, &packet.encode(), false),
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
            };
            match select(events, notifications).await {
                Either::First(_) | Either::Second(_) => {}
            }
            println!("BLE disconnected; advertise again");
        }
    })
    .await;
    panic!("BLE runner exited");
}
