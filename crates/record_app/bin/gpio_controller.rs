use esp_hal::{
    Async, Blocking,
    gpio::{self, DriveMode},
};

use esp_hal::i2c::master::{Config, I2c};
use legday_core::imu_controller;
use log::error;

#[derive(Copy, Clone)]
pub enum Status {
    RecordingOff,
    RecordingOn(u16),
}

pub struct RecordingGPIO {
    current_file_num: u16,
}
impl RecordingGPIO {
    pub fn new(i2c: I2c<'static, esp_hal::Blocking>) -> Self {
        Self {
            current_file_num: 1,
        }
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
