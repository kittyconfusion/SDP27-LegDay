use embassy_time::Instant;
use heapless::String;

// Do we want a global ErrorType enum for errors that we want to display / report to the end user?
// Could have a short buffer of ErrorTypes that can be read out by both GPIO and Networking threads

#[derive(PartialEq)]
pub struct GlobalConfig {
    is_sd_recording: bool,                // SD card recording enabled
    is_sd_readback: bool,                 // Reading back sensor data from the sd card
    is_serial_classifier_recording: bool, // Are we dumping classification data to the serial bus
    is_networking: bool,                  // Is the network thread running
    is_classifying: bool,                 // Is the classification thread running
}

pub const fn preset_classifying_no_network() -> GlobalConfig {
    GlobalConfig {
        is_sd_recording: false,
        is_sd_readback: false,
        is_serial_classifier_recording: true,
        is_networking: false,
        is_classifying: true,
    }
}

pub const fn preset_sensors() -> GlobalConfig {
    GlobalConfig {
        is_sd_recording: false,
        is_sd_readback: false,
        is_serial_classifier_recording: false,
        is_networking: false,
        is_classifying: false,
    }
}

pub const fn preset_demo_day() -> GlobalConfig {
    GlobalConfig {
        is_sd_recording: false,
        is_sd_readback: false,
        is_serial_classifier_recording: false,
        is_networking: true,
        is_classifying: true,
    }
}

pub const fn preset_sd_readout() -> GlobalConfig {
    GlobalConfig {
        is_sd_recording: false,
        is_sd_readback: true,
        is_serial_classifier_recording: true,
        is_networking: true,
        is_classifying: true,
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct GpioStatus {
    is_power_on: bool,          // On or in a long-term sleep state
    error_state: bool,          // Should we be warned visually about an error
    error_timeout: Option<u64>, // When should this error end?
}

pub enum NetworkStatus {
    Disabled,
    Pairing,
    Connected,
}

// Insert a generic IMU status enum / struct here?
// Or maybe move some IMU errors into here so that everything can read them
pub enum SensorStatus {}

// Insert classification status
pub enum ClassifierStatus {}
