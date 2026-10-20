//! ICM-20948 Digital Motion Processor (DMP) support.
//!
//! This is a Rust port of the DMP bring-up sequence from SparkFun's
//! `ICM_20948` Arduino library (`initializeDMP`, `enableDMPSensor`,
//! `setDMPODRrate`, `readDMPdataFromFIFO`), which SparkFun reverse-engineered
//! from InvenSense's eMD SmartMotion examples. Comments quote SparkFun where
//! the reason for a step is otherwise opaque.
//!
//! Typical use:
//!
//! ```ignore
//! imu.reset(&mut delay)?;
//! imu.dmp_initialize(&mut delay, 4, true)?;          // 225 Hz, verify firmware
//! imu.dmp_enable_outputs(DmpOutputs::ALL)?;          // raw accel, raw gyro, Quat6
//! imu.dmp_set_odr(mem::ODR_QUAT6, mem::ODR_CNTR_QUAT6, 0)?;
//! imu.dmp_set_odr(mem::ODR_ACCEL, mem::ODR_CNTR_ACCEL, 0)?;
//! imu.dmp_set_odr(mem::ODR_GYRO, mem::ODR_CNTR_GYRO, 0)?;
//! imu.dmp_start()?;
//! loop { let status = imu.dmp_read_packets(&mut packets)?; /* ... */ }
//! ```

use crate::regs::*;
use crate::{Error, Icm20948};
use embedded_hal::delay::DelayNs;
use embedded_hal::i2c::I2c;

/// InvenSense DMP3 firmware image `eMD-SmartMotion-ICM20948-1.1.0-MP`.
pub const FIRMWARE: &[u8] = include_bytes!("dmp3a.bin");
/// DMP memory address the image is loaded at.
pub const FIRMWARE_LOAD_ADDRESS: u16 = 0x90;
/// Program counter written to PRGM_START_ADDRH/L.
pub const PROGRAM_START_ADDRESS: u16 = 0x1000;
/// Largest single DMP memory burst (InvenSense's `INV_MAX_SERIAL_WRITE`).
const MAX_MEM_BURST: usize = 16;
/// Bytes drained from the FIFO per I2C transaction.
const FIFO_CHUNK_BYTES: usize = 240;
/// Longest packet our enabled outputs can produce (header, header2, accel,
/// gyro + bias, quat6, two accuracies, footer = 2+2+6+12+12+2+2+2 = 40) with
/// headroom for sensors we do not enable.
const CARRY_BYTES: usize = 96;
/// Smallest packet assumed when reserving output slots before reading a
/// chunk, so a chunk can always be parsed completely and only an incomplete
/// tail (< CARRY_BYTES) is ever carried over.
const MIN_PACKET_BYTES: usize = 12;
/// Output slots that must be free before another FIFO chunk is read.
pub const PACKETS_PER_CHUNK: usize = (CARRY_BYTES + FIFO_CHUNK_BYTES) / MIN_PACKET_BYTES + 1;
/// FIFO level at which the hardware overflow flag is checked.
const OVERFLOW_CHECK_BYTES: u16 = 512;

/// DMP memory addresses (`bank * 16 + offset`, from SparkFun's ICM_20948_DMP.h).
pub mod mem {
    pub const DATA_OUT_CTL1: u16 = 4 * 16;
    pub const DATA_OUT_CTL2: u16 = 4 * 16 + 2;
    pub const DATA_INTR_CTL: u16 = 4 * 16 + 12;
    pub const MOTION_EVENT_CTL: u16 = 4 * 16 + 14;
    pub const DATA_RDY_STATUS: u16 = 8 * 16 + 10;
    pub const ODR_ACCEL: u16 = 11 * 16 + 14;
    pub const ODR_GYRO: u16 = 11 * 16 + 10;
    pub const ODR_QUAT6: u16 = 10 * 16 + 12;
    pub const ODR_CNTR_ACCEL: u16 = 9 * 16 + 14;
    pub const ODR_CNTR_GYRO: u16 = 9 * 16 + 10;
    pub const ODR_CNTR_QUAT6: u16 = 8 * 16 + 12;
    pub const CPASS_MTX_00: u16 = 23 * 16;
    pub const B2S_MTX_00: u16 = 208 * 16;
    pub const GYRO_SF: u16 = 19 * 16;
    pub const GYRO_FULLSCALE: u16 = 72 * 16 + 12;
    pub const ACC_SCALE: u16 = 30 * 16;
    pub const ACC_SCALE2: u16 = 79 * 16 + 4;
    pub const ACCEL_ONLY_GAIN: u16 = 16 * 16 + 12;
    pub const ACCEL_ALPHA_VAR: u16 = 91 * 16;
    pub const ACCEL_A_VAR: u16 = 92 * 16;
    pub const ACCEL_CAL_RATE: u16 = 94 * 16 + 4;
    pub const CPASS_TIME_BUFFER: u16 = 112 * 16 + 14;
}

/// DMP FIFO packet header bits (`DATA_OUT_CTL1` uses the same bit layout).
pub mod header {
    pub const ACCEL: u16 = 0x8000;
    pub const GYRO: u16 = 0x4000;
    pub const COMPASS: u16 = 0x2000;
    pub const ALS: u16 = 0x1000;
    pub const QUAT6: u16 = 0x0800;
    pub const QUAT9: u16 = 0x0400;
    pub const PQUAT6: u16 = 0x0200;
    pub const GEOMAG: u16 = 0x0100;
    pub const PRESSURE: u16 = 0x0080;
    pub const GYRO_CALIBR: u16 = 0x0040;
    pub const COMPASS_CALIBR: u16 = 0x0020;
    pub const STEP_DETECTOR: u16 = 0x0010;
    pub const HEADER2: u16 = 0x0008;
    pub const STEP_INDICATOR_MASK: u16 = 0x0007;
}

/// DMP FIFO packet header-2 bits (`DATA_OUT_CTL2` uses the same bit layout).
pub mod header2 {
    pub const ACCEL_ACCURACY: u16 = 0x4000;
    pub const GYRO_ACCURACY: u16 = 0x2000;
    pub const COMPASS_ACCURACY: u16 = 0x1000;
    pub const FSYNC: u16 = 0x0800;
    pub const PICKUP: u16 = 0x0400;
    pub const ACTIVITY_RECOGNITION: u16 = 0x0080;
    pub const SECONDARY_ON_OFF: u16 = 0x0040;
}

/// `MOTION_EVENT_CTL` bits.
mod motion_event {
    pub const GYRO_CALIBR: u16 = 0x0100;
    pub const ACCEL_CALIBR: u16 = 0x0200;
}

/// `DATA_RDY_STATUS` bits.
mod data_ready {
    pub const GYRO: u16 = 0x0001;
    pub const ACCEL: u16 = 0x0002;
}

/// Which DMP outputs to enable. Mirrors SparkFun's
/// `INV_ICM20948_SENSOR_RAW_ACCELEROMETER`, `..._RAW_GYROSCOPE`, and
/// `..._GAME_ROTATION_VECTOR`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DmpOutputs {
    pub raw_accel: bool,
    pub raw_gyro: bool,
    pub quat6: bool,
}

impl DmpOutputs {
    pub const ALL: Self = Self {
        raw_accel: true,
        raw_gyro: true,
        quat6: true,
    };
    pub const QUAT6_ONLY: Self = Self {
        raw_accel: false,
        raw_gyro: false,
        quat6: true,
    };
}

/// One decoded DMP FIFO packet. Fields are `None` when the packet did not
/// carry them (the DMP only includes a sensor when its ODR counter fires, and
/// accuracy words only when they change).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DmpPacket {
    pub header: u16,
    pub header2: u16,
    pub accel: Option<[i16; 3]>,
    pub gyro: Option<[i16; 3]>,
    pub gyro_bias: Option<[i16; 3]>,
    /// Q30 vector part `[x, y, z]`; `w = sqrt(1 - x^2 - y^2 - z^2)`.
    pub quat6: Option<[i32; 3]>,
    pub accel_accuracy: Option<u16>,
    pub gyro_accuracy: Option<u16>,
    pub footer: u16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DmpReadStatus {
    /// Packets written to the caller's buffer.
    pub packets: usize,
    /// FIFO byte count before this read.
    pub fifo_bytes: u16,
    /// Unparseable data was found and the FIFO was reset.
    pub resynced: bool,
    /// The hardware FIFO overflow flag was set; the FIFO was reset.
    pub overflowed: bool,
}

/// Bookkeeping kept between FIFO reads.
#[derive(Clone, Copy, Debug)]
pub struct DmpState {
    mems_bank: u8,
    carry: [u8; CARRY_BYTES],
    carry_len: usize,
    allowed_header: u16,
    pub firmware_loaded: bool,
    pub gyro_sf: i32,
    pub pll: u8,
}

impl DmpState {
    pub const fn new() -> Self {
        Self {
            mems_bank: 0xFF,
            carry: [0; CARRY_BYTES],
            carry_len: 0,
            allowed_header: header::HEADER2 | header::STEP_INDICATOR_MASK,
            firmware_loaded: false,
            gyro_sf: 0,
            pll: 0,
        }
    }
}

impl Default for DmpState {
    fn default() -> Self {
        Self::new()
    }
}

/// Tuning constants that depend on the DMP base rate (from SparkFun DMP.md).
struct RateConstants {
    accel_only_gain: u32,
    accel_alpha_var: u32,
    accel_a_var: u32,
}

fn rate_constants(sample_rate_div: u8) -> RateConstants {
    // 225 Hz (div 4), 112 Hz (div 9), 56 Hz (div 19). Pick the nearest set.
    match sample_rate_div {
        0..=6 => RateConstants {
            accel_only_gain: 0x00E8_BA2E,
            accel_alpha_var: 0x3D27_D27D,
            accel_a_var: 0x02D8_2D83,
        },
        7..=13 => RateConstants {
            accel_only_gain: 0x01D1_745D,
            accel_alpha_var: 0x3A49_2492,
            accel_a_var: 0x05B6_DB6E,
        },
        _ => RateConstants {
            accel_only_gain: 0x03A4_9249,
            accel_alpha_var: 0x3492_4925,
            accel_a_var: 0x0B6D_B6DB,
        },
    }
}

impl<I2C: I2c> Icm20948<I2C> {
    pub const fn dmp_state(&self) -> &DmpState {
        &self.dmp
    }

    // ----- DMP memory ------------------------------------------------------

    /// Writes `data` to DMP memory at `address`, in bursts of at most 16
    /// bytes that never cross a 256-byte memory bank.
    #[inline(never)]
    pub fn dmp_write_mems(&mut self, address: u16, data: &[u8]) -> Result<(), Error<I2C::Error>> {
        // u32 arithmetic throughout: the Xtensa LLVM backend mis-selects some
        // u16 constant patterns when this is inlined into a large function.
        let mut address = u32::from(address);
        let mut written = 0;
        while written < data.len() {
            self.dmp_select_mem_bank((address >> 8) as u8)?;
            let in_bank = 0x100 - (address & 0xFF) as usize;
            let len = (data.len() - written).min(MAX_MEM_BURST).min(in_bank);
            self.write_reg(0, bank0::MEM_START_ADDR, (address & 0xFF) as u8)?;
            self.write_regs(0, bank0::MEM_R_W, &data[written..written + len])?;
            written += len;
            address += len as u32;
        }
        Ok(())
    }

    #[inline(never)]
    pub fn dmp_read_mems(
        &mut self,
        address: u16,
        buffer: &mut [u8],
    ) -> Result<(), Error<I2C::Error>> {
        let mut address = u32::from(address);
        let mut read = 0;
        while read < buffer.len() {
            self.dmp_select_mem_bank((address >> 8) as u8)?;
            let in_bank = 0x100 - (address & 0xFF) as usize;
            let len = (buffer.len() - read).min(MAX_MEM_BURST).min(in_bank);
            self.write_reg(0, bank0::MEM_START_ADDR, (address & 0xFF) as u8)?;
            self.read_regs(0, bank0::MEM_R_W, &mut buffer[read..read + len])?;
            read += len;
            address += len as u32;
        }
        Ok(())
    }

    fn dmp_select_mem_bank(&mut self, mem_bank: u8) -> Result<(), Error<I2C::Error>> {
        if self.dmp.mems_bank != mem_bank {
            self.write_reg(0, bank0::MEM_BANK_SEL, mem_bank)?;
            self.dmp.mems_bank = mem_bank;
        }
        Ok(())
    }

    #[inline(never)]
    fn dmp_write_u16(&mut self, address: u16, value: u16) -> Result<(), Error<I2C::Error>> {
        self.dmp_write_mems(address, &value.to_be_bytes())
    }

    #[inline(never)]
    fn dmp_write_u32(&mut self, address: u16, value: u32) -> Result<(), Error<I2C::Error>> {
        self.dmp_write_mems(address, &value.to_be_bytes())
    }

    /// Uploads the firmware image, optionally reading it back to verify.
    #[inline(never)]
    pub fn dmp_load_firmware(&mut self, verify: bool) -> Result<(), Error<I2C::Error>> {
        self.set_sleep(false)?;
        self.set_low_power(false)?;
        self.dmp.mems_bank = 0xFF;
        self.dmp_write_mems(FIRMWARE_LOAD_ADDRESS, FIRMWARE)?;
        if verify {
            let mut chunk = [0_u8; MAX_MEM_BURST];
            let mut offset = 0;
            while offset < FIRMWARE.len() {
                let len = (FIRMWARE.len() - offset).min(MAX_MEM_BURST);
                let address = FIRMWARE_LOAD_ADDRESS + offset as u16;
                self.dmp_read_mems(address, &mut chunk[..len])?;
                if chunk[..len] != FIRMWARE[offset..offset + len] {
                    return Err(Error::Verify {
                        bank: (address >> 8) as u8,
                        address,
                    });
                }
                offset += len;
            }
        }
        self.dmp.firmware_loaded = true;
        Ok(())
    }

    pub fn dmp_set_program_start(&mut self, address: u16) -> Result<(), Error<I2C::Error>> {
        self.write_regs(2, bank2::PRGM_START_ADDRH, &address.to_be_bytes())
    }

    // ----- configuration ---------------------------------------------------

    /// Full DMP configuration (SparkFun `initializeDMP`). Call after
    /// [`Icm20948::reset`]. `sample_rate_div` sets the DMP base rate:
    /// `1125 / (1 + div)` Hz, 4 for 225 Hz (the fastest documented rate).
    #[inline(never)]
    pub fn dmp_initialize(
        &mut self,
        delay: &mut impl DelayNs,
        sample_rate_div: u8,
        verify_firmware: bool,
    ) -> Result<(), Error<I2C::Error>> {
        self.set_sleep(false)?;
        self.set_low_power(false)?;

        // I2C master on (SparkFun startupMagnetometer): the DMP firmware
        // expects the aux bus configured even for 6-axis output.
        self.modify_reg(0, bank0::INT_PIN_CFG, int_pin_cfg::BYPASS_EN, 0)?;
        self.modify_reg(3, bank3::I2C_MST_CTRL, 0x0F, 0x07 | 0x10)?; // 345.6 kHz, stop between reads
        self.modify_reg(0, bank0::USER_CTRL, 0, user_ctrl::I2C_MST_EN)?;

        // SLV0: read ten bytes from the AK09916 starting at RSV2 (SparkFun's
        // "secret sauce" layout the DMP expects). SLV1: trigger single
        // measurements. Harmless for Quat6, required for Quat9 later.
        self.write_reg(3, bank3::I2C_SLV0_ADDR, 0x80 | ak09916::I2C_ADDR)?;
        self.write_reg(3, bank3::I2C_SLV0_REG, ak09916::REG_RSV2)?;
        self.write_reg(3, bank3::I2C_SLV0_CTRL, 0x80 | 0x40 | 0x10 | 10)?; // EN | BYTE_SW | GRP | LENG=10
        self.write_reg(3, bank3::I2C_SLV1_ADDR, ak09916::I2C_ADDR)?;
        self.write_reg(3, bank3::I2C_SLV1_DO, ak09916::MODE_SINGLE)?;
        self.write_reg(3, bank3::I2C_SLV1_REG, ak09916::REG_CNTL2)?;
        self.write_reg(3, bank3::I2C_SLV1_CTRL, 0x80 | 1)?; // EN | LENG=1
        // "It is not clear why we need to do this... But it appears to be
        // essential!" -- SparkFun. 1100 / 2^4 = 68.75 Hz magnetometer reads.
        self.write_reg(3, bank3::I2C_MST_ODR_CONFIG, 0x04)?;

        self.set_clock_source(pwr_mgmt_1::CLKSEL_AUTO)?;
        // Enable accel + gyro; bit 6 is a reserved bit the vendor code sets.
        self.write_reg(0, bank0::PWR_MGMT_2, 0x40)?;
        // Only the I2C master is duty cycled.
        self.modify_reg(
            0,
            bank0::LP_CONFIG,
            lp_config::ACCEL_CYCLE | lp_config::GYRO_CYCLE,
            lp_config::I2C_MST_CYCLE,
        )?;

        self.set_fifo_enabled(false)?;
        self.modify_reg(0, bank0::USER_CTRL, user_ctrl::DMP_EN, 0)?;

        // Gyro 2000 dps with DLPF on (config 0), accel 4 g: these ranges are
        // baked into the DMP scale constants written below.
        self.write_reg(2, bank2::GYRO_CONFIG_1, (3 << 1) | 0x01)?;
        self.modify_reg(2, bank2::ACCEL_CONFIG, 0x06, 1 << 1)?;

        self.write_reg(0, bank0::FIFO_EN_1, 0x00)?;
        self.write_reg(0, bank0::FIFO_EN_2, 0x00)?;
        self.write_reg(0, bank0::INT_ENABLE_1, 0x00)?;
        self.reset_fifo()?;

        self.set_sample_rate_div(sample_rate_div)?;

        self.dmp_set_program_start(PROGRAM_START_ADDRESS)?;
        self.dmp_load_firmware(verify_firmware)?;
        self.dmp_set_program_start(PROGRAM_START_ADDRESS)?;

        self.write_reg(0, bank0::HW_FIX_DISABLE, 0x48)?;
        self.write_reg(0, bank0::SINGLE_FIFO_PRIORITY_SEL, 0xE4)?;

        // Accel scaling: 1 g = 2^25 internally for 4 g FSR; output in hardware units.
        self.dmp_write_u32(mem::ACC_SCALE, 0x0400_0000)?;
        self.dmp_write_u32(mem::ACC_SCALE2, 0x0004_0000)?;

        // Compass mount matrix (identity scaled to uT) and B2S matrix (identity).
        const MOUNT_PLUS: u32 = 0x0999_9999;
        const MOUNT_MINUS: u32 = 0xF666_6667;
        let compass_matrix = [MOUNT_PLUS, 0, 0, 0, MOUNT_MINUS, 0, 0, 0, MOUNT_MINUS];
        self.dmp_write_matrix(mem::CPASS_MTX_00, &compass_matrix)?;
        let b2s_matrix = [0x4000_0000_u32, 0, 0, 0, 0x4000_0000, 0, 0, 0, 0x4000_0000];
        self.dmp_write_matrix(mem::B2S_MTX_00, &b2s_matrix)?;

        self.dmp_write_gyro_sf(sample_rate_div)?;
        self.dmp_write_u32(mem::GYRO_FULLSCALE, 0x1000_0000)?; // 2000 dps = 2^28

        let constants = rate_constants(sample_rate_div);
        self.dmp_write_u32(mem::ACCEL_ONLY_GAIN, constants.accel_only_gain)?;
        self.dmp_write_u32(mem::ACCEL_ALPHA_VAR, constants.accel_alpha_var)?;
        self.dmp_write_u32(mem::ACCEL_A_VAR, constants.accel_a_var)?;
        self.dmp_write_u16(mem::ACCEL_CAL_RATE, 0x0000)?;
        self.dmp_write_u16(mem::CPASS_TIME_BUFFER, 69)?;

        delay.delay_ms(10);
        Ok(())
    }

    /// Writes nine consecutive 32-bit words (a 3x3 mounting matrix).
    #[inline(never)]
    fn dmp_write_matrix(&mut self, base: u16, values: &[u32; 9]) -> Result<(), Error<I2C::Error>> {
        let mut bytes = [0_u8; 36];
        for (chunk, value) in bytes.chunks_exact_mut(4).zip(values) {
            chunk.copy_from_slice(&value.to_be_bytes());
        }
        self.dmp_write_mems(base, &bytes)
    }

    /// Computes and writes the DMP gyro scale factor from the PLL timebase
    /// correction register (SparkFun `inv_icm20948_set_gyro_sf`).
    fn dmp_write_gyro_sf(&mut self, sample_rate_div: u8) -> Result<(), Error<I2C::Error>> {
        let pll = self.read_reg(1, bank1::TIMEBASE_CORRECTION_PLL)?;
        self.dmp.pll = pll;
        const MAGIC: u64 = 264_446_880_937_391;
        const MAGIC_SCALE: u64 = 100_000;
        const GYRO_LEVEL: u64 = 4; // vendor code forces 4 regardless of FSR
        let numerator = MAGIC * (1_u64 << GYRO_LEVEL) * (1 + sample_rate_div as u64);
        let result = if pll & 0x80 != 0 {
            numerator / (1270 - (pll & 0x7F) as u64) / MAGIC_SCALE
        } else {
            numerator / (1270 + pll as u64) / MAGIC_SCALE
        };
        let gyro_sf = result.min(i32::MAX as u64) as i32;
        self.dmp.gyro_sf = gyro_sf;
        self.dmp_write_u32(mem::GYRO_SF, gyro_sf as u32)
    }

    /// Selects which sensors the DMP writes into each FIFO packet
    /// (SparkFun `enableDMPSensor` for the three sensors we use).
    pub fn dmp_enable_outputs(&mut self, outputs: DmpOutputs) -> Result<(), Error<I2C::Error>> {
        let mut ctl1 = 0_u16;
        if outputs.raw_accel {
            ctl1 |= header::ACCEL | header::HEADER2;
        }
        if outputs.raw_gyro {
            ctl1 |= header::GYRO | header::GYRO_CALIBR | header::HEADER2;
        }
        if outputs.quat6 {
            ctl1 |= header::QUAT6 | header::HEADER2;
        }

        let mut ctl2 = 0_u16;
        if ctl1 & header::ACCEL != 0 {
            ctl2 |= header2::ACCEL_ACCURACY;
        }
        if ctl1 & (header::GYRO | header::GYRO_CALIBR) != 0 {
            ctl2 |= header2::GYRO_ACCURACY;
        }

        let needs_accel = outputs.raw_accel || outputs.quat6;
        let needs_gyro = outputs.raw_gyro || outputs.quat6;
        let mut data_ready = 0_u16;
        let mut event_control = 0_u16;
        if needs_accel {
            data_ready |= data_ready::ACCEL;
            event_control |= motion_event::ACCEL_CALIBR;
        }
        if needs_gyro {
            data_ready |= data_ready::GYRO;
            event_control |= motion_event::GYRO_CALIBR;
        }

        self.set_sleep(false)?;
        self.set_low_power(false)?;
        self.dmp_write_u16(mem::DATA_OUT_CTL1, ctl1)?;
        self.dmp_write_u16(mem::DATA_OUT_CTL2, ctl2)?;
        self.dmp_write_u16(mem::DATA_RDY_STATUS, data_ready)?;
        self.dmp_write_u16(mem::MOTION_EVENT_CTL, event_control)?;
        // The vendor sequence leaves the chip in low-power mode here.
        self.set_low_power(true)?;

        self.dmp.allowed_header = ctl1 | header::HEADER2 | header::STEP_INDICATOR_MASK;
        Ok(())
    }

    /// Sets a DMP output interval: `interval` = (DMP rate / desired rate) - 1,
    /// so 0 emits that sensor on every DMP cycle.
    pub fn dmp_set_odr(
        &mut self,
        odr_register: u16,
        counter_register: u16,
        interval: u16,
    ) -> Result<(), Error<I2C::Error>> {
        self.set_sleep(false)?;
        self.set_low_power(false)?;
        self.dmp_write_u16(odr_register, interval)?;
        self.dmp_write_u16(counter_register, 0)?;
        self.set_low_power(true)?;
        Ok(())
    }

    /// Enables the FIFO and DMP, then resets both so packets start clean.
    pub fn dmp_start(&mut self) -> Result<(), Error<I2C::Error>> {
        self.set_fifo_enabled(true)?;
        self.modify_reg(0, bank0::USER_CTRL, 0, user_ctrl::DMP_EN)?;
        self.modify_reg(0, bank0::USER_CTRL, 0, user_ctrl::DMP_RST)?;
        self.reset_fifo()?;
        self.dmp.carry_len = 0;
        Ok(())
    }

    pub fn dmp_stop(&mut self) -> Result<(), Error<I2C::Error>> {
        self.modify_reg(0, bank0::USER_CTRL, user_ctrl::DMP_EN, 0)?;
        self.set_fifo_enabled(false)?;
        self.reset_fifo()?;
        self.dmp.carry_len = 0;
        Ok(())
    }

    // ----- FIFO packets ----------------------------------------------------

    /// Drains and decodes DMP packets into `out`, which should hold at least
    /// [`PACKETS_PER_CHUNK`] entries. Each FIFO chunk is read only when enough
    /// output slots remain to parse all of it, so at most an incomplete tail
    /// is carried over to the next call; call repeatedly to drain a backlog.
    /// If the stream cannot be parsed, or the hardware reports an overflow,
    /// the FIFO is reset and the status says so.
    #[inline(never)]
    pub fn dmp_read_packets(
        &mut self,
        out: &mut [DmpPacket],
    ) -> Result<DmpReadStatus, Error<I2C::Error>> {
        let fifo_bytes = self.fifo_count()?;
        let mut status = DmpReadStatus {
            packets: 0,
            fifo_bytes,
            resynced: false,
            overflowed: false,
        };
        if fifo_bytes >= OVERFLOW_CHECK_BYTES && self.take_fifo_overflow()? {
            self.reset_fifo()?;
            self.dmp.carry_len = 0;
            status.overflowed = true;
            return Ok(status);
        }
        let mut available = fifo_bytes as usize;
        let mut buffer = [0_u8; CARRY_BYTES + FIFO_CHUNK_BYTES];

        loop {
            if out.len() - status.packets < PACKETS_PER_CHUNK {
                return Ok(status);
            }
            // Assemble carry + fresh bytes.
            let carry_len = self.dmp.carry_len;
            buffer[..carry_len].copy_from_slice(&self.dmp.carry[..carry_len]);
            let to_read = available.min(FIFO_CHUNK_BYTES);
            self.read_fifo(&mut buffer[carry_len..carry_len + to_read])?;
            available -= to_read;
            let filled = carry_len + to_read;

            let mut position = 0;
            while status.packets < out.len() {
                match parse_packet(&buffer[position..filled], self.dmp.allowed_header) {
                    Ok(Some((packet, length))) => {
                        out[status.packets] = packet;
                        status.packets += 1;
                        position += length;
                    }
                    Ok(None) => break,
                    Err(()) => {
                        self.reset_fifo()?;
                        self.dmp.carry_len = 0;
                        status.resynced = true;
                        return Ok(status);
                    }
                }
            }

            let remainder = filled - position;
            if remainder > CARRY_BYTES {
                // Cannot happen while PACKETS_PER_CHUNK slots were free, but
                // never silently corrupt the stream: drop it and resync.
                self.reset_fifo()?;
                self.dmp.carry_len = 0;
                status.resynced = true;
                return Ok(status);
            }
            self.dmp.carry[..remainder].copy_from_slice(&buffer[position..filled]);
            self.dmp.carry_len = remainder;

            if available == 0 || status.packets == out.len() {
                return Ok(status);
            }
        }
    }
}

/// Parses one packet from the front of `bytes`. Returns `Ok(None)` when more
/// bytes are needed and `Err(())` when the header is not plausible.
pub fn parse_packet(bytes: &[u8], allowed_header: u16) -> Result<Option<(DmpPacket, usize)>, ()> {
    if bytes.len() < 2 {
        return Ok(None);
    }
    let header = u16::from_be_bytes([bytes[0], bytes[1]]);
    if header == 0 || header & !allowed_header != 0 {
        return Err(());
    }

    let mut packet = DmpPacket {
        header,
        ..DmpPacket::default()
    };
    let mut position = 2;

    // Returns the next `len` bytes or signals that the packet is incomplete.
    fn take<'a>(bytes: &'a [u8], position: &mut usize, len: usize) -> Option<&'a [u8]> {
        let slice = bytes.get(*position..*position + len)?;
        *position += len;
        Some(slice)
    }
    macro_rules! take_or_wait {
        ($len:expr) => {
            match take(bytes, &mut position, $len) {
                Some(slice) => slice,
                None => return Ok(None),
            }
        };
    }

    if header & header::HEADER2 != 0 {
        let slice = take_or_wait!(2);
        packet.header2 = u16::from_be_bytes([slice[0], slice[1]]);
    }
    if header & header::ACCEL != 0 {
        let slice = take_or_wait!(6);
        packet.accel = Some(be_i16x3(slice));
    }
    if header & header::GYRO != 0 {
        let slice = take_or_wait!(12);
        packet.gyro = Some(be_i16x3(&slice[..6]));
        packet.gyro_bias = Some(be_i16x3(&slice[6..]));
    }
    if header & header::COMPASS != 0 {
        take_or_wait!(6);
    }
    if header & header::ALS != 0 {
        take_or_wait!(8);
    }
    if header & header::QUAT6 != 0 {
        let slice = take_or_wait!(12);
        packet.quat6 = Some([
            be_i32(&slice[0..4]),
            be_i32(&slice[4..8]),
            be_i32(&slice[8..12]),
        ]);
    }
    if header & header::QUAT9 != 0 {
        take_or_wait!(14);
    }
    if header & header::PQUAT6 != 0 {
        take_or_wait!(6);
    }
    if header & header::GEOMAG != 0 {
        take_or_wait!(14);
    }
    if header & header::PRESSURE != 0 {
        take_or_wait!(6);
    }
    // GYRO_CALIBR (0x0040): the firmware sets the bit but emits no bytes.
    if header & header::COMPASS_CALIBR != 0 {
        take_or_wait!(12);
    }
    if header & header::STEP_DETECTOR != 0 {
        take_or_wait!(4);
    }
    if packet.header2 & header2::ACCEL_ACCURACY != 0 {
        let slice = take_or_wait!(2);
        packet.accel_accuracy = Some(u16::from_be_bytes([slice[0], slice[1]]));
    }
    if packet.header2 & header2::GYRO_ACCURACY != 0 {
        let slice = take_or_wait!(2);
        packet.gyro_accuracy = Some(u16::from_be_bytes([slice[0], slice[1]]));
    }
    if packet.header2 & header2::COMPASS_ACCURACY != 0 {
        take_or_wait!(2);
    }
    if packet.header2 & header2::FSYNC != 0 {
        take_or_wait!(2);
    }
    if packet.header2 & header2::PICKUP != 0 {
        take_or_wait!(2);
    }
    if packet.header2 & header2::ACTIVITY_RECOGNITION != 0 {
        take_or_wait!(6);
    }
    if packet.header2 & header2::SECONDARY_ON_OFF != 0 {
        take_or_wait!(2);
    }
    let footer = take_or_wait!(2);
    packet.footer = u16::from_be_bytes([footer[0], footer[1]]);

    Ok(Some((packet, position)))
}

fn be_i16x3(slice: &[u8]) -> [i16; 3] {
    [
        i16::from_be_bytes([slice[0], slice[1]]),
        i16::from_be_bytes([slice[2], slice[3]]),
        i16::from_be_bytes([slice[4], slice[5]]),
    ]
}

fn be_i32(slice: &[u8]) -> i32 {
    i32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]])
}

#[cfg(all(test, not(target_os = "none")))]
mod tests {
    use super::*;
    extern crate std;
    use std::vec::Vec;

    #[test]
    fn firmware_image_has_expected_size() {
        assert_eq!(FIRMWARE.len(), 14_301);
    }

    #[test]
    fn parses_full_packet_with_accuracies() {
        let allowed = header::ACCEL
            | header::GYRO
            | header::GYRO_CALIBR
            | header::QUAT6
            | header::HEADER2
            | 0x7;
        let mut bytes = alloc_packet();
        let (packet, length) = parse_packet(&bytes, allowed).unwrap().unwrap();
        assert_eq!(length, bytes.len());
        assert_eq!(packet.accel, Some([1, -2, 3]));
        assert_eq!(packet.gyro, Some([4, 5, -6]));
        assert_eq!(packet.gyro_bias, Some([7, 8, 9]));
        assert_eq!(packet.quat6, Some([0x1000_0000, -0x0800_0000, 5]));
        assert_eq!(packet.accel_accuracy, Some(3));
        assert_eq!(packet.gyro_accuracy, Some(2));
        assert_eq!(packet.footer, 0xBEEF);

        // Truncated input waits for more bytes instead of failing.
        bytes.truncate(bytes.len() - 1);
        assert_eq!(parse_packet(&bytes, allowed), Ok(None));
        // A header with disallowed bits is a desync.
        assert_eq!(parse_packet(&[0x20, 0x00, 0, 0], allowed), Err(()));
    }

    fn alloc_packet() -> Vec<u8> {
        let mut bytes = Vec::new();
        let header =
            header::ACCEL | header::GYRO | header::GYRO_CALIBR | header::QUAT6 | header::HEADER2;
        bytes.extend_from_slice(&header.to_be_bytes());
        bytes.extend_from_slice(&(header2::ACCEL_ACCURACY | header2::GYRO_ACCURACY).to_be_bytes());
        for value in [1_i16, -2, 3, 4, 5, -6, 7, 8, 9] {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        for value in [0x1000_0000_i32, -0x0800_0000, 5] {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        bytes.extend_from_slice(&3_u16.to_be_bytes());
        bytes.extend_from_slice(&2_u16.to_be_bytes());
        bytes.extend_from_slice(&0xBEEF_u16.to_be_bytes());
        bytes
    }
}
