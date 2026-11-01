// Each instance is responsible for a single I2C controller which currently require 2 IMUs and an optional 7-segment

// These are currently written as blocking but only the 7-segment requires blocking
// TODO: convert to async and only block with information to display

use embassy_embedded_hal::shared_bus::asynch::i2c::I2cDevice;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::mutex::Mutex;
use esp_hal::i2c::master::{Config, I2c};
use esp_hal::{Async, Blocking};
use ht16k33_async::HT16K33;
use log::error;

const NUM_FONT: [u8; 10] = [0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F];

pub struct I2cController {
    current_file_num: u16,
    i2c_bus: &'static Mutex<CriticalSectionRawMutex, I2c<'static, Async>>,
    display_address: Option<u8>, // only one of the two I2cControllers can have this
}
impl I2cController {
    pub async fn new(
        bus: &'static Mutex<CriticalSectionRawMutex, I2c<'static, Async>>,
        has_display: bool,
    ) -> Self {
        let mut display_address = if has_display { Some(0x72) } else { None };

        // Initialize the 7-segment if exists. On error, set it to a none-type and don't crash.
        if display_address.is_some() {
            let device = I2cDevice::new(&bus);
            let mut display = HT16K33::new(device, display_address.unwrap());
            if display.setup().await.is_err() {
                error!("Init failure of 7-seg over I2C. Continuing to run without.");
                display_address = None;
            }
        }
        Self {
            current_file_num: 1,
            i2c_bus: bus,
            display_address: display_address,
        }
    }

    fn create_adafruit_mapping(num: u16) -> [u8; 16] {
        let mut buffer = [0u8; 16];

        // The Adafruit 7-seg backpack uses the HT16K33 LED controller like so
        buffer[0] = NUM_FONT[((num / 1000) % 10) as usize]; // Digit 1 (most significant) 
        buffer[2] = NUM_FONT[((num / 100) % 10) as usize]; // Digit 2

        buffer[6] = NUM_FONT[((num / 10) % 10) as usize]; // Digit 3
        buffer[8] = NUM_FONT[((num / 1) % 10) as usize]; // Digit 4 (least significant)

        buffer
    }

    pub async fn update_display(&mut self, num: u16) {
        if self.display_address.is_none() {
            return;
        }
        let buffer: [u8; 16] = Self::create_adafruit_mapping(num);
        // Whenever actually used, I2cDevice handles the mutex lock automatically!!!!
        let device = I2cDevice::new(self.i2c_bus);
        let mut display = HT16K33::new(device, self.display_address.unwrap());

        let _ = display.write_whole_display(&buffer).await;
    }
    /*
        let mut ht16k33 = HT16K33::new(i2c, DISP_I2C_ADDR);
    ht16k33.initialize().expect("Failed to initialize ht16k33");
    ht16k33.set_display(Display::ON).expect("Could not turn on the display!");
    ht16k33.set_dimming(Dimming::BRIGHTNESS_MIN).expect("Could not set dimming!");

    // Sending individual digits
    ht16k33.update_buffer_with_digit(Index::One, 1);
    ht16k33.write_display_buffer().unwrap();
    */
}
