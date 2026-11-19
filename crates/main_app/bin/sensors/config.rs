//! Sensor assignments from pinoutplan.png. Numbers below are ESP32 GPIOs,
//! not Arduino D-pin numbers. The caller retains ownership of bus setup.

pub const IMU_COUNT: usize = 4;
pub const PRESSURE_COUNT: usize = 4;
pub const I2C0_SDA_GPIO: u8 = 11; // A4
pub const I2C0_SCL_GPIO: u8 = 12; // A5
pub const I2C1_SDA_GPIO: u8 = 13; // A6
pub const I2C1_SCL_GPIO: u8 = 14; // A7
pub const PRESSURE_GPIOS: [u8; PRESSURE_COUNT] = [1, 2, 3, 4]; // A0..A3

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BusId {
    I2c0,
    I2c1,
}

/// Slots never move when the training reference is removed. Slot 1 is the
/// fourth physical IMU (bottom-left in the diagram), despite its array index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum ImuSlot {
    NonProstheticUpper = 0,
    NonProstheticLower = 1,
    ProstheticUpper = 2,
    ProstheticLower = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImuAssignment {
    pub slot: ImuSlot,
    pub bus: BusId,
    pub address: u8,
}

pub const IMU_LAYOUT: [ImuAssignment; IMU_COUNT] = [
    ImuAssignment {
        slot: ImuSlot::NonProstheticUpper,
        bus: BusId::I2c0,
        address: 0x69,
    },
    ImuAssignment {
        slot: ImuSlot::NonProstheticLower,
        bus: BusId::I2c0,
        address: 0x68,
    },
    ImuAssignment {
        slot: ImuSlot::ProstheticUpper,
        bus: BusId::I2c1,
        address: 0x69,
    },
    ImuAssignment {
        slot: ImuSlot::ProstheticLower,
        bus: BusId::I2c1,
        address: 0x68,
    },
];

/// Pressure array order is A0, A1 (non-prosthetic), A2, A3 (prosthetic).
/// Heel/ball placement has not been specified, so it is not assumed here.
pub const PRESSURE_NON_PROSTHETIC_MASK: u8 = 0b0011;
pub const PRESSURE_PROSTHETIC_MASK: u8 = 0b1100;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquisitionMode {
    Training,
    /// Never reads the removable reference, even when it is plugged in.
    Demo,
}

impl AcquisitionMode {
    pub const fn enabled_imu_mask(self) -> u8 {
        match self {
            Self::Training => 0b1111,
            Self::Demo => 0b1101,
        }
    }
}

// Shared by the recorder and hardware check to prevent mismatched settings.
// Changed from Nathan's divider 19 at the user's request for 225-Hz acquisition.
// 1125 / (4 + 1) = 225 Hz nominal IMU output; ranges/filters stay unchanged.
pub const SAMPLE_HZ: u64 = 225;
pub const IMU_SAMPLE_RATE_DIV: u8 = (1125 / SAMPLE_HZ - 1) as u8;
const _: () =
    assert!(SAMPLE_HZ > 0 && SAMPLE_HZ <= 1125 && 1125 % SAMPLE_HZ == 0 && 1125 / SAMPLE_HZ <= 256);
// Faster transfers give the two IMUs per bus headroom within each ~4.44 ms poll.
// TODO(Nathan, Tim): verify wiring/pull-ups/display at 400 kHz and measured fresh
// counts at 225 Hz. Equal nominal rates do not synchronize physical sample times.
pub const I2C_BUS_KHZ: u32 = 400;
pub const IMU_GYRO_CONFIG: u8 = 0x25; // +/-1000 deg/s, DLPF 4 enabled
pub const IMU_ACCEL_CONFIG: u8 = 0x25; // +/-8 g, DLPF 4 enabled
pub const ACCEL_COUNTS_PER_G: f32 = 4096.0;
pub const GYRO_COUNTS_PER_DEG_PER_SECOND: f32 = 32.8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_removes_only_the_lower_non_prosthetic_slot() {
        let removed =
            AcquisitionMode::Training.enabled_imu_mask() ^ AcquisitionMode::Demo.enabled_imu_mask();
        assert_eq!(removed, 1 << ImuSlot::NonProstheticLower as usize);
        assert_eq!(
            IMU_LAYOUT[ImuSlot::NonProstheticLower as usize].address,
            0x68
        );
        assert_eq!(
            IMU_LAYOUT[ImuSlot::NonProstheticLower as usize].bus,
            BusId::I2c0
        );
    }

    #[test]
    fn addresses_are_distinct_within_each_bus() {
        for (index, assignment) in IMU_LAYOUT.iter().enumerate() {
            assert_eq!(assignment.slot as usize, index);
            for other in &IMU_LAYOUT[index + 1..] {
                assert!(assignment.bus != other.bus || assignment.address != other.address);
            }
        }
    }
}
