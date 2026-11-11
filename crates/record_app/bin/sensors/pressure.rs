//! Four raw ADC1 channels using the planned 3.3 V FSR/10 kohm dividers.
//! Counts are not Newtons or calibrated voltage. A floating/disconnected ADC
//! input cannot be distinguished from a connected sensor by this reader.

use embassy_time::Instant;
use esp_hal::{
    Blocking,
    analog::adc::{Adc, AdcConfig, AdcPin, Attenuation},
    peripherals::{ADC1, GPIO1, GPIO2, GPIO3, GPIO4},
};

use super::{config::PRESSURE_COUNT, sample::Timed};

pub struct PressureReader<'d> {
    adc: Adc<'d, ADC1<'d>, Blocking>,
    a0: AdcPin<GPIO1<'d>, ADC1<'d>>,
    a1: AdcPin<GPIO2<'d>, ADC1<'d>>,
    a2: AdcPin<GPIO3<'d>, ADC1<'d>>,
    a3: AdcPin<GPIO4<'d>, ADC1<'d>>,
}

impl<'d> PressureReader<'d> {
    /// Takes only ADC1 and A0..A3; it never initializes the chip or claims
    /// any GPIOs used by Nathan's buses, recording switch, display, or lights.
    pub fn new(
        adc1: ADC1<'d>,
        a0: GPIO1<'d>,
        a1: GPIO2<'d>,
        a2: GPIO3<'d>,
        a3: GPIO4<'d>,
    ) -> Self {
        let mut config = AdcConfig::new();
        Self {
            a0: config.enable_pin(a0, Attenuation::_11dB),
            a1: config.enable_pin(a1, Attenuation::_11dB),
            a2: config.enable_pin(a2, Attenuation::_11dB),
            a3: config.enable_pin(a3, Attenuation::_11dB),
            adc: Adc::new(adc1, config),
        }
    }

    /// Always reads all four channels in A0..A3 order. Every conversion has
    /// its own MCU timestamps. HAL's blocking ADC call returns a count, not
    /// a Result; successful conversion is not physical-sensor validation.
    pub fn read(&mut self) -> [Timed<u16>; PRESSURE_COUNT] {
        macro_rules! read_channel {
            ($pin:ident) => {{
                let read_started_us = Instant::now().as_micros();
                let value = self.adc.read_blocking(&mut self.$pin);
                Timed {
                    read_started_us,
                    read_completed_us: Instant::now().as_micros(),
                    value,
                }
            }};
        }
        [
            read_channel!(a0),
            read_channel!(a1),
            read_channel!(a2),
            read_channel!(a3),
        ]
    }
}
