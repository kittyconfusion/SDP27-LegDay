// In the future we could use SpiDMA for more efficient writing (but it's not usually writing much while doing other things)
use embedded_sdmmc::{Mode, SdCard, TimeSource, Timestamp, VolumeIdx, VolumeManager};
// Used to control fancy iterative loop
use core::ops::ControlFlow;
use core::str;
use embedded_hal_bus::spi::ExclusiveDevice;
use esp_hal::gpio::Output;
use esp_hal::{Async, delay::Delay, spi::master::Spi};
use log::{error, info, warn};

// All functions run synchronously
pub struct SdController {
    // By default opens with values of 4, 4, 4 respectively
    // At the very least we do not need multiple partitions open at a time
    // Reduces RAM usage 0.2 kb or so

    // ExclusiveDevice allows us to only have one CS (chip select) pin because we have one SPI device
    volume_manager: VolumeManager<
        SdCard<ExclusiveDevice<Spi<'static, Async>, Output<'static>, Delay>, Delay>,
        DummyTimeSource,
        4,
        4,
        2,
    >,
}

// This entire controller is a whole lot of error checks and logging because the SD card could be
// momentarily disconnected and we need it to not explode the entire program
// TODO: Implement basic recovery for disconnects and other anomalies
impl SdController {
    pub fn new(spi: Spi<'static, Async>, cs: Output<'static>, delay: Delay) -> Self {
        let device =
            ExclusiveDevice::new(spi, cs, delay).expect("SD exclusive device init failed :(");
        let sdcard = SdCard::new(device, delay);
        let time_source = DummyTimeSource;
        let volume_manager = VolumeManager::new_with_limits(sdcard, time_source, 0);

        // Run a boot check to log and see if we can open up an SD card
        {
            let volume = match volume_manager.open_volume(VolumeIdx(0)) {
                Ok(vol) => {
                    log::info!("SD Card Partition 0 is readable at boot");
                    Some(vol)
                }
                Err(e) => {
                    log::info!("Initialization failure: {:#?}", e);
                    None
                }
            };
            // Close the open partition (if it opened in the first place)
            if let Some(vol) = volume {
                let _ = volume_manager.close_volume(vol.to_raw_volume());
            }
        }
        Self { volume_manager }
    }

    // Finds the highest existing file number in the root
    // returns u16::MAX if there is any error. Since this is only run at initialization it is
    // retried until success in the main function.
    pub fn find_last_numerical_file(&mut self) -> u16 {
        let volume = match self.volume_manager.open_volume(VolumeIdx(0)) {
            Ok(vol) => Some(vol),
            Err(e) => {
                info!("Unable to open volume: {:#?}", e);
                None
            }
        };
        // Nothing to close if failed
        if volume.is_none() {
            return u16::MAX;
        }

        let volume = volume.unwrap();

        let dir = match volume.open_root_dir() {
            Ok(direc) => Some(direc),
            Err(e) => {
                info!("Unable to open root directory: {:#?}", e);
                None
            }
        };
        // Close the partition if failed
        if dir.is_none() {
            let __ = self.volume_manager.close_volume(volume.to_raw_volume());
            return u16::MAX;
        }

        let dir = dir.unwrap();
        let mut highest_numerical = 0;

        // Loop through all names and, if possible, treat them as numbers to compare agsinst
        let _ = dir.iterate_dir(|entry| {
            if let Ok(num_str) = str::from_utf8(entry.name.base_name()) {
                if let Ok(num) = num_str.parse::<u16>() {
                    if num > highest_numerical {
                        highest_numerical = num;
                    }
                }
            }
            // Needed to iterate through this special type of loop
            ControlFlow::Continue(())
        });
        info!("Found highest file number as {highest_numerical}.LEG");

        // Close directory and volume in reverse order of opening
        let _ = self.volume_manager.close_dir(dir.to_raw_directory());
        let _ = self.volume_manager.close_volume(volume.to_raw_volume());

        return highest_numerical;
    }

    // Want to catch and print every possible error for debugging purposes
    // Most of this is long boilerplate to handle errors and release resources
    // Currently returns Ok(()) no matter what. Result should be more useful in future.
    pub fn write(&mut self, data: &[u8], name: &str) -> Result<(), ()> {
        info!("Writing to file {name}");

        // Open Partition 0
        let volume = match self.volume_manager.open_volume(VolumeIdx(0)) {
            Ok(vol) => Some(vol),
            Err(e) => {
                info!("Unable to open volume: {:#?}", e);
                None
            }
        };
        // Nothing to close if failed
        if volume.is_none() {
            return Ok(());
        }

        let volume = volume.unwrap();

        // Open the root directory
        let dir = match volume.open_root_dir() {
            Ok(direc) => Some(direc),
            Err(e) => {
                info!("Unable to open root directory: {:#?}", e);
                None
            }
        };
        // Close the partition if failed
        if dir.is_none() {
            let __ = self.volume_manager.close_volume(volume.to_raw_volume());
            return Ok(());
        }

        let dir = dir.unwrap();

        // Open the requested file
        let file = match dir.open_file_in_dir(name, Mode::ReadWriteCreateOrAppend) {
            Ok(open_file) => Some(open_file),
            Err(e) => {
                info!("Unable to open file: {:#?}", e);
                None
            }
        };
        // Close directory and volume if failed
        if file.is_none() {
            let _ = self.volume_manager.close_dir(dir.to_raw_directory());
            let _ = self.volume_manager.close_volume(volume.to_raw_volume());
            return Ok(());
        }

        // Create the file if need be, open it, and append new data
        let file = file.unwrap();
        if let Err(e) = file.write(&data) {
            error!(
                "Failed to write data but how did we even get this far? {:#?}",
                e
            );
        }
        // Close everything in reverse order of creation
        let _ = self.volume_manager.close_file(file.to_raw_file());
        let _ = self.volume_manager.close_dir(dir.to_raw_directory());
        let _ = self.volume_manager.close_volume(volume.to_raw_volume());
        Ok(()) // Give a good result no matter what
    }
}

// SD card library needs a timekeeper for metadata
// luckily it's not picky so this crummy one works
// Can replace this with RTC module or internet thing if we really want to later
#[derive(Clone, Copy)]
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
