use embassy_embedded_hal::shared_bus::asynch::i2c::I2cDevice;
use embassy_time::Timer;
use esp_hal::{
    Async, Blocking,
    gpio::{
        self, DriveMode, Input,
        Level::{self, High, Low},
        Output,
    },
};

use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::mutex::Mutex;
use esp_hal::i2c::master::{Config, I2c};
use ht16k33_async::HT16K33;
use log::error;

//#[path = "../record.rs"]
//mod record;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Status {
    RecordingOff(u16),
    RecordingOn(u16),
    Setup,
}

const NUM_FONT: [u8; 10] = [0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F];

pub struct RecordingGPIO {
    i2c_bus: &'static Mutex<CriticalSectionRawMutex, I2c<'static, Async>>,
    display_address: Option<u8>,
    recording_light: Output<'static>,
}
impl RecordingGPIO {
    pub async fn new(
        bus: &'static Mutex<CriticalSectionRawMutex, I2c<'static, Async>>,
        mut recording_light: Output<'static>,
    ) -> Self {
        let mut display_address = Some(0x72);
        let device = I2cDevice::new(&bus);
        let mut display = HT16K33::new(device, display_address.unwrap());
        if display.setup().await.is_err() {
            error!("Init failure of 7-seg over I2C. Continuing to run without.");
            display_address = None;
        }
        recording_light.set_low();

        Self {
            i2c_bus: bus,
            display_address: display_address,
            recording_light: recording_light,
        }
    }
    pub fn set_recording_light(&mut self, level: Level) {
        self.recording_light.set_level(level);
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
}

#[embassy_executor::task]
pub async fn switch_loop(
    status: &'static Mutex<CriticalSectionRawMutex, Status>,
    mut switch: Input<'static>,
    recording_light: Output<'static>,
    bus: &'static Mutex<CriticalSectionRawMutex, I2c<'static, Async>>,
) {
    let mut gpio_func = RecordingGPIO::new(bus, recording_light).await;
    let mut display_num: u16;

    loop {
        {
            let mut current = status.lock().await;
            display_num = match *current {
                Status::RecordingOff(c) => c,
                Status::RecordingOn(c) => c,
                Status::Setup => u16::MAX,
            };
            if switch.is_high() {
                *current = Status::RecordingOn(display_num);
                gpio_func.set_recording_light(High);
            }
            if switch.is_low() {
                *current = Status::RecordingOff(display_num + 1);
                gpio_func.update_display(display_num).await;
                gpio_func.set_recording_light(Low);
            }
        }
        switch.wait_for_any_edge().await;
        // Don't allow for too rapid button switching
        Timer::after_millis(200).await;
    }
}
