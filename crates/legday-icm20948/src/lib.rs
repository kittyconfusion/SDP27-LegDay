#![no_std]

//! Minimal ICM-20948 driver over `embedded-hal` 1.0 blocking I2C.
//!
//! The driver is deliberately small and explicit: every register access goes
//! through [`Icm20948::write_reg`] / [`Icm20948::read_regs`] with the user bank
//! named at the call site, so the register-level behaviour can be compared
//! line by line with the datasheet or with SparkFun's C library.
//!
//! Two acquisition paths are provided:
//!
//! - **Raw FIFO** ([`Icm20948::start_raw_fifo`] and
//!   [`Icm20948::read_raw_fifo_samples`]): the IMU queues 12-byte
//!   accel + gyro samples in its 4 kB hardware FIFO at the configured ODR, so
//!   the host can drain them in bursts without losing samples while it is
//!   busy writing to an SD card.
//! - **DMP** (module [`dmp`], feature `dmp`): the IMU's Digital Motion
//!   Processor runs InvenSense firmware that emits raw accel, raw gyro, and a
//!   6-axis quaternion per sample through the same FIFO.
//!
//! # Register access model
//!
//! The ICM-20948 has four register "banks" selected through register 0x7F.
//! [`Icm20948::set_bank`] remembers the current bank so repeated accesses in
//! the same bank cost one I2C transaction, not two. Every public method names
//! the bank and register it touches, so each call can be checked against the
//! datasheet directly.
//!
//! # Units
//!
//! Accel and gyro values are raw signed 16-bit counts. Convert with
//! [`AccelRange::lsb_per_g`] and [`GyroRange::lsb_per_dps`]; the DMP path
//! uses 4 g and 2000 dps, so 8192 counts = 1 g and 16.4 counts = 1 deg/s.

pub mod regs;

#[cfg(feature = "dmp")]
pub mod dmp;

use embedded_hal::delay::DelayNs;
use embedded_hal::i2c::I2c;
use regs::*;

/// Value of `WHO_AM_I` on every ICM-20948.
pub const WHO_AM_I_VALUE: u8 = 0xEA;
/// 7-bit I2C address when the AD0 pin is low (SparkFun default after jumper).
pub const I2C_ADDR_AD0_LOW: u8 = 0x68;
/// 7-bit I2C address when AD0 is high (Adafruit and SparkFun default).
pub const I2C_ADDR_AD0_HIGH: u8 = 0x69;
/// Internal gyro/accel base output data rate with the DLPF enabled.
pub const BASE_ODR_HZ: u32 = 1_125;
/// Hardware FIFO depth in bytes.
pub const FIFO_CAPACITY_BYTES: u16 = 4_096;
/// Size of one accel + gyro sample as queued by [`Icm20948::start_raw_fifo`].
pub const RAW_FIFO_SAMPLE_BYTES: usize = 12;
/// FIFO level at which the hardware overflow flag is checked in raw mode.
const RAW_OVERFLOW_CHECK_BYTES: u16 = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error<E> {
    /// The I2C bus returned an error.
    Bus(E),
    /// `WHO_AM_I` did not read back as `0xEA`.
    WrongId(u8),
    /// Read-back of a written register or DMP memory word did not match.
    Verify { bank: u8, address: u16 },
    /// FIFO contents did not parse; the FIFO was reset.
    FifoDesync,
}

impl<E> From<E> for Error<E> {
    fn from(error: E) -> Self {
        Error::Bus(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AccelRange {
    G2 = 0,
    G4 = 1,
    G8 = 2,
    G16 = 3,
}

impl AccelRange {
    pub const fn lsb_per_g(self) -> f32 {
        match self {
            Self::G2 => 16_384.0,
            Self::G4 => 8_192.0,
            Self::G8 => 4_096.0,
            Self::G16 => 2_048.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GyroRange {
    Dps250 = 0,
    Dps500 = 1,
    Dps1000 = 2,
    Dps2000 = 3,
}

impl GyroRange {
    pub const fn lsb_per_dps(self) -> f32 {
        match self {
            Self::Dps250 => 131.0,
            Self::Dps500 => 65.5,
            Self::Dps1000 => 32.8,
            Self::Dps2000 => 16.4,
        }
    }
}

/// Accelerometer and gyroscope configuration for raw (non-DMP) operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    pub accel_range: AccelRange,
    pub gyro_range: GyroRange,
    /// Output data rate is `1125 / (1 + sample_rate_div)` Hz for both sensors
    /// while the DLPF is enabled. `4` gives 225 Hz, `5` gives 187.5 Hz; an
    /// exact 200 Hz is not available.
    pub sample_rate_div: u8,
    /// Gyro DLPF configuration (0-7); 0 is 196.6 Hz bandwidth.
    pub gyro_dlpf: u8,
    /// Accel DLPF configuration (0-7); 0 is 246 Hz bandwidth.
    pub accel_dlpf: u8,
}

impl Config {
    /// Output data rate in millihertz.
    pub const fn odr_millihz(&self) -> u32 {
        BASE_ODR_HZ * 1_000 / (1 + self.sample_rate_div as u32)
    }

    /// Nominal sample period in microseconds (rounded down).
    pub const fn period_us(&self) -> u32 {
        1_000_000 * (1 + self.sample_rate_div as u32) / BASE_ODR_HZ
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            accel_range: AccelRange::G4,
            gyro_range: GyroRange::Dps2000,
            sample_rate_div: 4,
            gyro_dlpf: 0,
            accel_dlpf: 0,
        }
    }
}

/// One accelerometer + gyroscope sample in raw signed counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RawSample {
    pub accel: [i16; 3],
    pub gyro: [i16; 3],
}

impl RawSample {
    /// Decodes the big-endian register/FIFO layout `AX AY AZ GX GY GZ`.
    pub fn from_be_bytes(bytes: &[u8; RAW_FIFO_SAMPLE_BYTES]) -> Self {
        let word = |index: usize| i16::from_be_bytes([bytes[index], bytes[index + 1]]);
        Self {
            accel: [word(0), word(2), word(4)],
            gyro: [word(6), word(8), word(10)],
        }
    }
}

/// Result of draining the raw FIFO.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FifoRead {
    /// Samples copied into the caller's buffer.
    pub samples: usize,
    /// FIFO byte count before the read.
    pub fifo_bytes: u16,
    /// The FIFO was full or misaligned and has been reset; samples were lost.
    pub overflowed: bool,
}

pub struct Icm20948<I2C> {
    i2c: I2C,
    address: u8,
    bank: u8,
    #[cfg(feature = "dmp")]
    dmp: dmp::DmpState,
}

impl<I2C: I2c> Icm20948<I2C> {
    pub fn new(i2c: I2C, address: u8) -> Self {
        Self {
            i2c,
            address,
            bank: 0xFF,
            #[cfg(feature = "dmp")]
            dmp: dmp::DmpState::new(),
        }
    }

    pub fn release(self) -> I2C {
        self.i2c
    }

    pub const fn address(&self) -> u8 {
        self.address
    }

    /// Changes the target address, e.g. while probing 0x68 and 0x69.
    pub fn set_address(&mut self, address: u8) {
        self.address = address;
        self.bank = 0xFF;
    }

    // ----- register access -------------------------------------------------

    /// Selects a user bank, skipping the bus write if it is already active.
    pub fn set_bank(&mut self, bank: u8) -> Result<(), Error<I2C::Error>> {
        if bank == self.bank {
            return Ok(());
        }
        self.i2c
            .write(self.address, &[REG_BANK_SEL, (bank << 4) & 0x30])?;
        self.bank = bank;
        Ok(())
    }

    pub fn write_reg(&mut self, bank: u8, reg: u8, value: u8) -> Result<(), Error<I2C::Error>> {
        self.set_bank(bank)?;
        self.i2c.write(self.address, &[reg, value])?;
        Ok(())
    }

    /// Writes up to 32 consecutive bytes starting at `reg`.
    pub fn write_regs(&mut self, bank: u8, reg: u8, data: &[u8]) -> Result<(), Error<I2C::Error>> {
        assert!(data.len() <= 32, "write_regs supports at most 32 bytes");
        self.set_bank(bank)?;
        let mut buffer = [0_u8; 33];
        buffer[0] = reg;
        buffer[1..=data.len()].copy_from_slice(data);
        self.i2c.write(self.address, &buffer[..=data.len()])?;
        Ok(())
    }

    pub fn read_reg(&mut self, bank: u8, reg: u8) -> Result<u8, Error<I2C::Error>> {
        let mut value = [0_u8];
        self.read_regs(bank, reg, &mut value)?;
        Ok(value[0])
    }

    pub fn read_regs(
        &mut self,
        bank: u8,
        reg: u8,
        buffer: &mut [u8],
    ) -> Result<(), Error<I2C::Error>> {
        self.set_bank(bank)?;
        self.i2c.write_read(self.address, &[reg], buffer)?;
        Ok(())
    }

    /// Read-modify-write: clears `clear` bits, then sets `set` bits.
    pub fn modify_reg(
        &mut self,
        bank: u8,
        reg: u8,
        clear: u8,
        set: u8,
    ) -> Result<u8, Error<I2C::Error>> {
        let current = self.read_reg(bank, reg)?;
        let next = (current & !clear) | set;
        if next != current {
            self.write_reg(bank, reg, next)?;
        }
        Ok(next)
    }

    // ----- identity and power ----------------------------------------------

    pub fn who_am_i(&mut self) -> Result<u8, Error<I2C::Error>> {
        self.bank = 0xFF; // always re-select bank 0 on a probe
        self.read_reg(0, bank0::WHO_AM_I)
    }

    pub fn check_id(&mut self) -> Result<(), Error<I2C::Error>> {
        match self.who_am_i()? {
            WHO_AM_I_VALUE => Ok(()),
            other => Err(Error::WrongId(other)),
        }
    }

    /// Software reset, then wake with the auto-selected PLL clock. Leaves the
    /// device awake, out of low-power mode, with all sensors enabled.
    pub fn reset(&mut self, delay: &mut impl DelayNs) -> Result<(), Error<I2C::Error>> {
        self.write_reg(0, bank0::PWR_MGMT_1, pwr_mgmt_1::DEVICE_RESET)?;
        self.bank = 0xFF;
        #[cfg(feature = "dmp")]
        {
            self.dmp = dmp::DmpState::new();
        }
        delay.delay_ms(50);
        self.set_sleep(false)?;
        self.set_low_power(false)?;
        self.set_clock_source(pwr_mgmt_1::CLKSEL_AUTO)?;
        self.write_reg(0, bank0::PWR_MGMT_2, 0x00)?;
        delay.delay_ms(10);
        Ok(())
    }

    pub fn set_sleep(&mut self, on: bool) -> Result<(), Error<I2C::Error>> {
        let (clear, set) = if on {
            (0, pwr_mgmt_1::SLEEP)
        } else {
            (pwr_mgmt_1::SLEEP, 0)
        };
        self.modify_reg(0, bank0::PWR_MGMT_1, clear, set)?;
        Ok(())
    }

    pub fn set_low_power(&mut self, on: bool) -> Result<(), Error<I2C::Error>> {
        let (clear, set) = if on {
            (0, pwr_mgmt_1::LP_EN)
        } else {
            (pwr_mgmt_1::LP_EN, 0)
        };
        self.modify_reg(0, bank0::PWR_MGMT_1, clear, set)?;
        Ok(())
    }

    pub fn set_clock_source(&mut self, clksel: u8) -> Result<(), Error<I2C::Error>> {
        self.modify_reg(
            0,
            bank0::PWR_MGMT_1,
            pwr_mgmt_1::CLKSEL_MASK,
            clksel & pwr_mgmt_1::CLKSEL_MASK,
        )?;
        Ok(())
    }

    // ----- sensor configuration --------------------------------------------

    /// Applies full-scale, DLPF, and sample-rate settings for continuous
    /// (non-duty-cycled) accel + gyro operation.
    pub fn configure(&mut self, config: &Config) -> Result<(), Error<I2C::Error>> {
        // Continuous sampling for accel and gyro.
        self.modify_reg(
            0,
            bank0::LP_CONFIG,
            lp_config::ACCEL_CYCLE | lp_config::GYRO_CYCLE,
            0,
        )?;
        // Enable all accel and gyro axes.
        self.write_reg(0, bank0::PWR_MGMT_2, 0x00)?;

        // GYRO_CONFIG_1: [5:3] DLPFCFG, [2:1] FS_SEL, [0] FCHOICE (1 = DLPF on).
        let gyro_config =
            ((config.gyro_dlpf & 0x07) << 3) | ((config.gyro_range as u8) << 1) | 0x01;
        self.write_reg(2, bank2::GYRO_CONFIG_1, gyro_config)?;
        let readback = self.read_reg(2, bank2::GYRO_CONFIG_1)?;
        if readback != gyro_config {
            return Err(Error::Verify {
                bank: 2,
                address: bank2::GYRO_CONFIG_1 as u16,
            });
        }

        // ACCEL_CONFIG has the same layout.
        let accel_config =
            ((config.accel_dlpf & 0x07) << 3) | ((config.accel_range as u8) << 1) | 0x01;
        self.write_reg(2, bank2::ACCEL_CONFIG, accel_config)?;
        let readback = self.read_reg(2, bank2::ACCEL_CONFIG)?;
        if readback != accel_config {
            return Err(Error::Verify {
                bank: 2,
                address: bank2::ACCEL_CONFIG as u16,
            });
        }

        self.set_sample_rate_div(config.sample_rate_div)?;
        Ok(())
    }

    /// Sets both sample-rate dividers so accel and gyro share one ODR.
    pub fn set_sample_rate_div(&mut self, div: u8) -> Result<(), Error<I2C::Error>> {
        self.write_reg(2, bank2::GYRO_SMPLRT_DIV, div)?;
        self.write_reg(2, bank2::ACCEL_SMPLRT_DIV_1, 0)?;
        self.write_reg(2, bank2::ACCEL_SMPLRT_DIV_2, div)?;
        Ok(())
    }

    // ----- direct data reads -----------------------------------------------

    /// Reads the latest accel + gyro registers in one 12-byte burst.
    pub fn read_raw(&mut self) -> Result<RawSample, Error<I2C::Error>> {
        let mut bytes = [0_u8; RAW_FIFO_SAMPLE_BYTES];
        self.read_regs(0, bank0::ACCEL_XOUT_H, &mut bytes)?;
        Ok(RawSample::from_be_bytes(&bytes))
    }

    /// Raw temperature counts; degrees C = counts / 333.87 + 21.
    pub fn read_temperature_raw(&mut self) -> Result<i16, Error<I2C::Error>> {
        let mut bytes = [0_u8; 2];
        self.read_regs(0, bank0::TEMP_OUT_H, &mut bytes)?;
        Ok(i16::from_be_bytes(bytes))
    }

    // ----- FIFO ------------------------------------------------------------

    pub fn fifo_count(&mut self) -> Result<u16, Error<I2C::Error>> {
        let mut bytes = [0_u8; 2];
        self.read_regs(0, bank0::FIFO_COUNT_H, &mut bytes)?;
        Ok(u16::from_be_bytes([bytes[0] & 0x1F, bytes[1]]))
    }

    /// Reads `buffer.len()` bytes from the FIFO. The caller must have checked
    /// [`Icm20948::fifo_count`] first.
    pub fn read_fifo(&mut self, buffer: &mut [u8]) -> Result<(), Error<I2C::Error>> {
        if buffer.is_empty() {
            return Ok(());
        }
        self.read_regs(0, bank0::FIFO_R_W, buffer)
    }

    /// Resets the FIFO the way InvenSense's examples do (0x1F then 0x1E).
    pub fn reset_fifo(&mut self) -> Result<(), Error<I2C::Error>> {
        self.write_reg(0, bank0::FIFO_RST, 0x1F)?;
        self.write_reg(0, bank0::FIFO_RST, 0x1E)?;
        Ok(())
    }

    pub fn set_fifo_enabled(&mut self, on: bool) -> Result<(), Error<I2C::Error>> {
        let (clear, set) = if on {
            (0, user_ctrl::FIFO_EN)
        } else {
            (user_ctrl::FIFO_EN, 0)
        };
        self.modify_reg(0, bank0::USER_CTRL, clear, set)?;
        Ok(())
    }

    /// Streams accel + gyro samples (12 bytes each) into the hardware FIFO in
    /// snapshot mode (stops writing when full, keeping sample alignment).
    pub fn start_raw_fifo(&mut self) -> Result<(), Error<I2C::Error>> {
        self.set_fifo_enabled(false)?;
        self.write_reg(0, bank0::FIFO_EN_1, 0x00)?;
        self.write_reg(0, bank0::FIFO_EN_2, 0x00)?;
        self.write_reg(0, bank0::FIFO_MODE, 0x1F)?;
        self.reset_fifo()?;
        self.write_reg(0, bank0::FIFO_EN_2, fifo_en_2::ACCEL_AND_GYRO)?;
        self.set_fifo_enabled(true)?;
        Ok(())
    }

    pub fn stop_raw_fifo(&mut self) -> Result<(), Error<I2C::Error>> {
        self.write_reg(0, bank0::FIFO_EN_2, 0x00)?;
        self.set_fifo_enabled(false)?;
        self.reset_fifo()
    }

    /// Drains whole samples from the raw FIFO into `out`. Leaves any partial
    /// sample in the FIFO for the next call.
    pub fn read_raw_fifo_samples(
        &mut self,
        out: &mut [RawSample],
    ) -> Result<FifoRead, Error<I2C::Error>> {
        let fifo_bytes = self.fifo_count()?;
        let mut status = FifoRead {
            samples: 0,
            fifo_bytes,
            overflowed: false,
        };

        // A full FIFO stopped queuing (snapshot mode) and we have lost
        // samples; discard and realign rather than trust the tail.
        if fifo_bytes >= RAW_OVERFLOW_CHECK_BYTES && self.take_fifo_overflow()? {
            self.reset_fifo()?;
            status.overflowed = true;
            return Ok(status);
        }

        let available = (fifo_bytes as usize / RAW_FIFO_SAMPLE_BYTES).min(out.len());
        let mut chunk = [0_u8; 20 * RAW_FIFO_SAMPLE_BYTES];
        let mut done = 0;
        while done < available {
            let count = (available - done).min(chunk.len() / RAW_FIFO_SAMPLE_BYTES);
            let bytes = count * RAW_FIFO_SAMPLE_BYTES;
            self.read_fifo(&mut chunk[..bytes])?;
            for index in 0..count {
                let start = index * RAW_FIFO_SAMPLE_BYTES;
                let sample: &[u8; RAW_FIFO_SAMPLE_BYTES] = chunk
                    [start..start + RAW_FIFO_SAMPLE_BYTES]
                    .try_into()
                    .unwrap();
                out[done + index] = RawSample::from_be_bytes(sample);
            }
            done += count;
        }
        status.samples = done;
        Ok(status)
    }

    /// Reads and clears the FIFO overflow interrupt flags.
    pub fn take_fifo_overflow(&mut self) -> Result<bool, Error<I2C::Error>> {
        let flags = self.read_reg(0, bank0::INT_STATUS_2)?;
        Ok(flags & int_status_2::FIFO_OVERFLOW_MASK != 0)
    }
}
