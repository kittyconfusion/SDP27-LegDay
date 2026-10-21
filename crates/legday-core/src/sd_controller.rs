// well this is frustrating to try to write
// probably want to use 'd instead of 'static because might have another device on bus in the future

use embedded_sdmmc::{Mode, SdCard, TimeSource, Timestamp, VolumeIdx, VolumeManager};
// In the future we could use SpiDMA for more efficient writing (but it's not usually writing much while doing other things)
use esp_hal::{Blocking, delay::Delay, spi::master::Spi};

struct SdController {
    time_source: DummyTimeSource,
    // By default opens with values of 4, 4, 4 respectively
    // At the very least we do not need multiple partitions open at a time
    // Reduces RAM usage 0.2 kb or so
    volume: VolumeManager<SdCard<Spi<'static, Blocking>, Delay>, DummyTimeSource, 4, 4, 1>,
}

impl SdController {
    pub fn new(spi: Spi<'static, Blocking>, delay: Delay) -> Self {
        let sdcard = SdCard::new(spi, delay);
        let time_source = DummyTimeSource;
        let volume = VolumeManager::new_with_limits(sdcard, time_source);

        Self {
            time_source,
            volume,
        }
    }
}
// Get the card size (this also triggers card initialisation because it's not been done yet)
//println!("Card size is {} bytes", sdcard.num_bytes()?);
// Now let's look for volumes (also known as partitions) on our block device.
// To do this we need a Volume Manager. It will take ownership of the block device.
//let volume_mgr = VolumeManager::new(sdcard, time_source);
// Try and access Volume 0 (i.e. the first partition).
// The volume object holds information about the filesystem on that volume.
//let volume0 = volume_mgr.open_volume(VolumeIdx(0))?;
//println!("Volume 0: {:?}", volume0);
// Open the root directory (mutably borrows from the volume).
//let root_dir = volume0.open_root_dir()?;

// Can replace this with RTC module or internet thing if we really want to later
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
