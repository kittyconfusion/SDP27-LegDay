//! Raw polling reader using Nathan's existing async I2C mutex type.
//! No FIFO, tasks, timers for sample scheduling, or recording logic is added.

use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, mutex::Mutex};
use embassy_time::{Instant, Timer};
use esp_hal::{
    Async,
    i2c::master::{Error, I2c},
};

use super::{
    config::{
        AcquisitionMode, BusId, IMU_ACCEL_CONFIG, IMU_COUNT, IMU_GYRO_CONFIG, IMU_LAYOUT,
        IMU_SAMPLE_RATE_DIV,
    },
    sample::{ImuError, ImuInitialization, ImuReading, RawImu, Timed},
};

pub type SharedI2c = Mutex<CriticalSectionRawMutex, I2c<'static, Async>>;

const BANK_SELECT: u8 = 0x7f;
const WHO_AM_I: u8 = 0x00;
const EXPECTED_ID: u8 = 0xea;
const POWER_1: u8 = 0x06;
const POWER_2: u8 = 0x07;
const INTERRUPT_ENABLE_1: u8 = 0x11;
const INTERRUPT_STATUS_1: u8 = 0x1a;
const ACCEL_X_HIGH: u8 = 0x2d;

pub struct ImuReader {
    buses: [&'static SharedI2c; 2],
    mode: AcquisitionMode,
    initialized_mask: u8,
}

impl ImuReader {
    /// Pass the already-created I2C0 and I2C1 mutexes from Nathan's branch.
    /// This reader never reconfigures their pins or speed and permits sharing
    /// I2C0 with the display. Use one reader as owner of these IMU addresses.
    pub fn new(i2c0: &'static SharedI2c, i2c1: &'static SharedI2c, mode: AcquisitionMode) -> Self {
        Self {
            buses: [i2c0, i2c1],
            mode,
            initialized_mask: 0,
        }
    }

    pub const fn initialized_mask(&self) -> u8 {
        self.initialized_mask
    }

    /// Reports initialization success only, not current connection health.
    pub fn all_required_initialized(&self) -> bool {
        self.initialized_mask == self.mode.enabled_imu_mask()
    }

    fn bus(&self, slot: usize) -> &'static SharedI2c {
        self.buses[match IMU_LAYOUT[slot].bus {
            BusId::I2c0 => 0,
            BusId::I2c1 => 1,
        }]
    }

    /// Tries every enabled IMU and returns every result; one disconnected
    /// sensor cannot trap initialization in an infinite retry loop. Calling
    /// this again explicitly resets/reinitializes all enabled IMUs.
    pub async fn initialize(&mut self) -> [ImuInitialization<Error>; IMU_COUNT] {
        self.initialized_mask = 0;
        let mut report = core::array::from_fn(|_| ImuInitialization::Disabled);
        for (slot, result) in report.iter_mut().enumerate() {
            if self.mode.enabled_imu_mask() & (1 << slot) == 0 {
                continue;
            }
            *result = match self.initialize_one(slot).await {
                Ok(()) => {
                    self.initialized_mask |= 1 << slot;
                    ImuInitialization::Ready
                }
                Err(error) => ImuInitialization::Failed(error),
            };
        }
        report
    }

    async fn initialize_one(&self, slot: usize) -> Result<(), ImuError<Error>> {
        let bus = self.bus(slot);
        let address = IMU_LAYOUT[slot].address;
        {
            let mut i2c = bus.lock().await;
            select_bank(&mut i2c, address, 0).await?;
            let actual = read_register(&mut i2c, address, WHO_AM_I).await?;
            if actual != EXPECTED_ID {
                return Err(ImuError::UnexpectedIdentity { actual });
            }
            write_register(&mut i2c, address, POWER_1, 0x80).await?;
        }
        // Yield without holding the shared bus during the device reset.
        Timer::after_millis(100).await;
        {
            let mut i2c = bus.lock().await;
            select_bank(&mut i2c, address, 0).await?;
            write_register(&mut i2c, address, POWER_1, 0x01).await?;
            write_register(&mut i2c, address, POWER_2, 0x00).await?;
        }
        Timer::after_millis(100).await;
        {
            let mut i2c = bus.lock().await;
            select_bank(&mut i2c, address, 2).await?;
            // Retain Nathan's ranges/filters; use config.rs's shared sample-rate
            // divider so hardware output matches the recorder's nominal cadence.
            for (register, value) in [
                (0x00, IMU_SAMPLE_RATE_DIV),
                (0x01, IMU_GYRO_CONFIG),
                (0x10, 0x00),
                (0x11, IMU_SAMPLE_RATE_DIV),
                (0x14, IMU_ACCEL_CONFIG),
            ] {
                write_register(&mut i2c, address, register, value).await?;
                let actual = read_register(&mut i2c, address, register).await?;
                if actual != value {
                    return Err(ImuError::ConfigurationMismatch {
                        bank: 2,
                        register,
                        expected: value,
                        actual,
                    });
                }
            }
            select_bank(&mut i2c, address, 0).await?;
            write_register(&mut i2c, address, INTERRUPT_ENABLE_1, 0x01).await?;
        }
        Timer::after_millis(100).await;
        // Discard the startup indication. A subsequent poll must see new data.
        let mut i2c = bus.lock().await;
        select_bank(&mut i2c, address, 0).await?;
        read_register(&mut i2c, address, INTERRUPT_STATUS_1).await?;
        Ok(())
    }

    /// Poll both independent buses concurrently, preserving fixed output slots.
    /// Devices sharing a bus remain sequential, and the display keeps its mutex.
    /// TODO(Nathan, Tim): this overlaps bus transfers, not physical sample times;
    /// verify ODR/FIFO/synchronization before promising 225 fresh samples/sec.
    /// This polls latest registers, so intermediate hardware samples may be
    /// skipped if the caller is late. It does not promise lossless capture.
    pub async fn read(&mut self) -> [ImuReading<Error>; IMU_COUNT] {
        let (mut left, mut right) =
            embassy_futures::join::join(self.read_bus(BusId::I2c0), self.read_bus(BusId::I2c1))
                .await;
        core::array::from_fn(|slot| {
            // Move each result once without requiring hardware errors to be Copy.
            if IMU_LAYOUT[slot].bus == BusId::I2c0 {
                left[slot].take().unwrap()
            } else {
                right[slot].take().unwrap()
            }
        })
    }

    async fn read_bus(&self, bus: BusId) -> [Option<ImuReading<Error>>; IMU_COUNT] {
        let mut results = core::array::from_fn(|_| None);
        for (slot, result) in results.iter_mut().enumerate() {
            if IMU_LAYOUT[slot].bus != bus {
                continue;
            }
            *result = Some(if self.mode.enabled_imu_mask() & (1 << slot) == 0 {
                ImuReading::Disabled
            } else if self.initialized_mask & (1 << slot) == 0 {
                ImuReading::NotInitialized
            } else {
                match self.read_one(slot).await {
                    Ok(Some(sample)) => ImuReading::Fresh(sample),
                    Ok(None) => ImuReading::NotReady,
                    Err(error) => ImuReading::Failed(error),
                }
            });
        }
        results
    }

    async fn read_one(&self, slot: usize) -> Result<Option<Timed<RawImu>>, ImuError<Error>> {
        let address = IMU_LAYOUT[slot].address;
        let mut i2c = self.bus(slot).lock().await;
        let read_started_us = Instant::now().as_micros();
        // Never cache bank state across releasing the mutex: another bus user
        // might have selected a different bank on this device.
        select_bank(&mut i2c, address, 0).await?;
        if read_register(&mut i2c, address, INTERRUPT_STATUS_1).await? & 1 == 0 {
            return Ok(None);
        }
        let mut bytes = [0; 12];
        i2c.write_read_async(address, &[ACCEL_X_HIGH], &mut bytes)
            .await
            .map_err(ImuError::Bus)?;
        Ok(Some(Timed {
            read_started_us,
            read_completed_us: Instant::now().as_micros(),
            value: RawImu::from_register_bytes(bytes),
        }))
    }
}

async fn select_bank(
    i2c: &mut I2c<'static, Async>,
    address: u8,
    bank: u8,
) -> Result<(), ImuError<Error>> {
    write_register(i2c, address, BANK_SELECT, bank << 4).await
}

async fn write_register(
    i2c: &mut I2c<'static, Async>,
    address: u8,
    register: u8,
    value: u8,
) -> Result<(), ImuError<Error>> {
    i2c.write_async(address, &[register, value])
        .await
        .map_err(ImuError::Bus)
}

async fn read_register(
    i2c: &mut I2c<'static, Async>,
    address: u8,
    register: u8,
) -> Result<u8, ImuError<Error>> {
    let mut byte = [0];
    i2c.write_read_async(address, &[register], &mut byte)
        .await
        .map_err(ImuError::Bus)?;
    Ok(byte[0])
}
