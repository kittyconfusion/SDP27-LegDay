// Example for me to think (please change!)

// Each instance is responsible for a single I2C controller with an arbitrary number of connected IMUs
// I am abstracting away (in order to support differing configurations) any data recording stuff

// https://docs.rs/esp-hal/latest/esp_hal/i2c/master/struct.I2c.html
// for non-blocking these should be made async

use esp_hal::i2c::master::{Config, I2c};

struct ImuController {}

// 2. Implementation block (the behavior/methods)
impl ImuController {
    // Constructor (an "associated function")
    // By convention, this is usually named 'new'
    fn new() -> Self {
        Self {}
    }
    fn read() {}
}
