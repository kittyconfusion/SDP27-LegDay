// well this is frustrating to try to write
// probably want to use 'd instead of 'static because might have another device on bus in the future

use embedded_sdmmc::{Mode, RawVolume, SdCard, TimeSource, Timestamp, VolumeIdx, VolumeManager};
// In the future we could use SpiDMA for more efficient writing (but it's not usually writing much while doing other things)
use core::ops::ControlFlow;
use core::str;
use embedded_hal_bus::spi::ExclusiveDevice;
use esp_hal::gpio::Output;
use esp_hal::{Async, delay::Delay, spi::master::Spi};
use log::{error, warn};

pub struct SdController {
    time_source: DummyTimeSource,
    // By default opens with values of 4, 4, 4 respectively
    // At the very least we do not need multiple partitions open at a time
    // Reduces RAM usage 0.2 kb or so

    // ExclusiveDevice allows us to only have one CS (chip select) pin when only having one SPI device
    volume: VolumeManager<
        SdCard<ExclusiveDevice<Spi<'static, Async>, Output<'static>, Delay>, Delay>,
        DummyTimeSource,
        4,
        4,
        1,
    >,
}

// This entire controller is a whole lot of error checks because the SD card could be momentarily disconnected and we need basic recovery
// TODO: Implement logging for disconnects and other anomalies
impl SdController {
    pub fn new(spi: Spi<'static, Async>, cs: Output<'static>, delay: Delay) -> Self {
        let device =
            ExclusiveDevice::new(spi, cs, delay).expect("SD exclusive device init failed :(");
        let sdcard = SdCard::new(device, delay);
        let time_source = DummyTimeSource;
        let volume = VolumeManager::new_with_limits(sdcard, time_source, 0);

        {
            match volume.open_volume(VolumeIdx(0)) {
                Ok(volume) => {
                    log::info!("Success: SD Card Partition 0 is readable!");
                }
                Err(e) => {
                    log::info!("Initialization structural failure: {:?}", e);
                }
            }
        }

        Self {
            time_source,
            volume,
        }
    }

    // returns u16::MAX if there is any error, since only run at init can just retry this function if something has gone wrong
    pub fn find_last_numerical_file(&mut self) -> u16 {
        let vol = self.volume.open_volume(VolumeIdx(0));
        if let Err(e) = vol {
            error!("Could not open volume! {:#?}", e);
            return u16::MAX; // Will need to retry later and do NOT want this to be 0 to not overwrite files
        }

        let dir = vol.unwrap().open_root_dir();
        if let Err(e) = dir {
            error!("Could not open root directory! {:#?}", e);
            return u16::MAX;
        }

        let mut last_numerical = 0;

        let _ = dir.unwrap().iterate_dir(|entry| {
            if let Ok(num_str) = str::from_utf8(entry.name.base_name()) {
                if let Ok(num) = num_str.parse::<u16>() {
                    if last_numerical > num {
                        last_numerical = num;
                    }
                }
            }

            ControlFlow::Continue(())
        });

        return last_numerical;
    }

    // Want to catch and print every possible error for debugging purposes
    pub fn write(&mut self, data: &[u8], name: &str) -> Result<(), ()> {
        warn!("Start of function");
        let vol = self.volume.open_volume(VolumeIdx(0));
        if let Err(e) = vol {
            error!("Could not open volume! {:#?}", e);
            return Ok(());
        }

        let dir = vol.unwrap().open_root_dir();
        if let Err(e) = dir {
            error!("Could not open root directory! {:#?}", e);
            return Ok(());
        }

        let file = dir
            .unwrap()
            .open_file_in_dir(name, Mode::ReadWriteCreateOrAppend);

        if let Err(e) = file {
            error!("Could not open file {:#?}", e);
            return Ok(());
        }

        if let Err(e) = file.unwrap().write(&data) {
            error!(
                "Failed to write data but how did we even get this far? {:#?}",
                e
            );
        }

        Ok(()) // Give a good result no matter what
    }
}

// Can replace this with RTC module or internet thing if we really want to later
#[derive(Clone, Copy)] // makes SdController init happy
struct DummyTimeSource;

impl TimeSource for DummyTimeSource {
    fn get_timestamp(&self) -> Timestamp {
        Timestamp {
            year_since_1970: 56,
            zero_indexed_month: 0,
            zero_indexed_day: 0,
            hours: 01,
            minutes: 23,
            seconds: 45,
        }
    }
}
