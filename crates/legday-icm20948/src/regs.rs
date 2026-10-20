//! ICM-20948 register map (user banks 0-3). Addresses from the TDK datasheet
//! as transcribed in SparkFun's `ICM_20948_REGISTERS.h`.

pub const REG_BANK_SEL: u8 = 0x7F;

pub mod bank0 {
    pub const WHO_AM_I: u8 = 0x00;
    pub const USER_CTRL: u8 = 0x03;
    pub const LP_CONFIG: u8 = 0x05;
    pub const PWR_MGMT_1: u8 = 0x06;
    pub const PWR_MGMT_2: u8 = 0x07;
    pub const INT_PIN_CFG: u8 = 0x0F;
    pub const INT_ENABLE: u8 = 0x10;
    pub const INT_ENABLE_1: u8 = 0x11;
    pub const INT_ENABLE_2: u8 = 0x12;
    pub const INT_ENABLE_3: u8 = 0x13;
    pub const I2C_MST_STATUS: u8 = 0x17;
    pub const DMP_INT_STATUS: u8 = 0x18;
    pub const INT_STATUS: u8 = 0x19;
    pub const INT_STATUS_1: u8 = 0x1A;
    pub const INT_STATUS_2: u8 = 0x1B;
    pub const INT_STATUS_3: u8 = 0x1C;
    pub const SINGLE_FIFO_PRIORITY_SEL: u8 = 0x26;
    pub const ACCEL_XOUT_H: u8 = 0x2D;
    pub const GYRO_XOUT_H: u8 = 0x33;
    pub const TEMP_OUT_H: u8 = 0x39;
    pub const EXT_SLV_SENS_DATA_00: u8 = 0x3B;
    pub const FIFO_EN_1: u8 = 0x66;
    pub const FIFO_EN_2: u8 = 0x67;
    pub const FIFO_RST: u8 = 0x68;
    pub const FIFO_MODE: u8 = 0x69;
    pub const FIFO_COUNT_H: u8 = 0x70;
    pub const FIFO_COUNT_L: u8 = 0x71;
    pub const FIFO_R_W: u8 = 0x72;
    pub const DATA_RDY_STATUS: u8 = 0x74;
    pub const HW_FIX_DISABLE: u8 = 0x75;
    pub const FIFO_CFG: u8 = 0x76;
    pub const MEM_START_ADDR: u8 = 0x7C;
    pub const MEM_R_W: u8 = 0x7D;
    pub const MEM_BANK_SEL: u8 = 0x7E;
}

pub mod bank1 {
    pub const TIMEBASE_CORRECTION_PLL: u8 = 0x28;
}

pub mod bank2 {
    pub const GYRO_SMPLRT_DIV: u8 = 0x00;
    pub const GYRO_CONFIG_1: u8 = 0x01;
    pub const GYRO_CONFIG_2: u8 = 0x02;
    pub const ODR_ALIGN_EN: u8 = 0x09;
    pub const ACCEL_SMPLRT_DIV_1: u8 = 0x10;
    pub const ACCEL_SMPLRT_DIV_2: u8 = 0x11;
    pub const ACCEL_CONFIG: u8 = 0x14;
    pub const ACCEL_CONFIG_2: u8 = 0x15;
    pub const PRGM_START_ADDRH: u8 = 0x50;
    pub const PRGM_START_ADDRL: u8 = 0x51;
}

pub mod bank3 {
    pub const I2C_MST_ODR_CONFIG: u8 = 0x00;
    pub const I2C_MST_CTRL: u8 = 0x01;
    pub const I2C_MST_DELAY_CTRL: u8 = 0x02;
    pub const I2C_SLV0_ADDR: u8 = 0x03;
    pub const I2C_SLV0_REG: u8 = 0x04;
    pub const I2C_SLV0_CTRL: u8 = 0x05;
    pub const I2C_SLV0_DO: u8 = 0x06;
    pub const I2C_SLV1_ADDR: u8 = 0x07;
    pub const I2C_SLV1_REG: u8 = 0x08;
    pub const I2C_SLV1_CTRL: u8 = 0x09;
    pub const I2C_SLV1_DO: u8 = 0x0A;
}

/// USER_CTRL bits.
pub mod user_ctrl {
    pub const DMP_EN: u8 = 0x80;
    pub const FIFO_EN: u8 = 0x40;
    pub const I2C_MST_EN: u8 = 0x20;
    pub const I2C_IF_DIS: u8 = 0x10;
    pub const DMP_RST: u8 = 0x08;
    pub const SRAM_RST: u8 = 0x04;
    pub const I2C_MST_RST: u8 = 0x02;
}

/// PWR_MGMT_1 bits.
pub mod pwr_mgmt_1 {
    pub const DEVICE_RESET: u8 = 0x80;
    pub const SLEEP: u8 = 0x40;
    pub const LP_EN: u8 = 0x20;
    pub const TEMP_DIS: u8 = 0x08;
    pub const CLKSEL_MASK: u8 = 0x07;
    pub const CLKSEL_AUTO: u8 = 0x01;
}

/// LP_CONFIG bits.
pub mod lp_config {
    pub const I2C_MST_CYCLE: u8 = 0x40;
    pub const ACCEL_CYCLE: u8 = 0x20;
    pub const GYRO_CYCLE: u8 = 0x10;
}

/// FIFO_EN_2 bits.
pub mod fifo_en_2 {
    pub const ACCEL: u8 = 0x10;
    pub const GYRO_Z: u8 = 0x08;
    pub const GYRO_Y: u8 = 0x04;
    pub const GYRO_X: u8 = 0x02;
    pub const TEMP: u8 = 0x01;
    pub const ACCEL_AND_GYRO: u8 = ACCEL | GYRO_X | GYRO_Y | GYRO_Z;
}

/// INT_PIN_CFG bits.
pub mod int_pin_cfg {
    pub const BYPASS_EN: u8 = 0x02;
}

/// INT_STATUS_2: FIFO overflow flags (one per FIFO, only FIFO 0 is used).
pub mod int_status_2 {
    pub const FIFO_OVERFLOW_MASK: u8 = 0x1F;
}

/// Magnetometer (AK09916) constants used by the DMP I2C-master setup.
pub mod ak09916 {
    pub const I2C_ADDR: u8 = 0x0C;
    pub const REG_RSV2: u8 = 0x03;
    pub const REG_CNTL2: u8 = 0x31;
    pub const MODE_SINGLE: u8 = 0x01;
}
