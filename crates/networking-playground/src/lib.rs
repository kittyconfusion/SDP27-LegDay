#![no_std]

//! Experimental telemetry contract, independent of either radio and of the CNN.
//! Integers have explicit units and encoding; Rust struct layout is never sent.

pub const WIRE_BYTES: usize = 20;
pub const SYNTHETIC: u8 = 1;
pub const MOCK_CONTROL: u8 = 2;
pub const SERVICE_UUID: &str = "8ea2c100-97b5-4b99-a361-47c27d8e4250";
pub const CHARACTERISTIC_UUID: &str = "8ea2c101-97b5-4b99-a361-47c27d8e4250";
pub const MAX_AGE_US: u32 = 100_000;

/// Four waiting packets, bounded regardless of disconnect duration.
pub type PacketQueue = heapless::spsc::Queue<Packet, 5>;
pub type Producer = heapless::spsc::Producer<'static, Packet>;
pub type Consumer = heapless::spsc::Consumer<'static, Packet>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packet {
    pub flags: u8,
    pub sequence: u16,
    /// Model-specific class ID. 255 is reserved for unknown/unavailable.
    pub class: u8,
    /// Quantized model score, 0..255. This is not a calibrated probability.
    pub score: u8,
    /// Low 32 bits of sample time in microseconds; wraps after ~71.6 minutes.
    pub sample_us: u32,
    pub phase_permille: u16,
    /// Hypothetical resistance request, 0..1000; no actuator connection.
    pub mock_resistance_permille: u16,
    /// 65535 means unavailable; genuine durations saturate at 65534 us.
    pub acquire_us: u16,
    pub infer_us: u16,
}

impl Packet {
    pub fn encode(self) -> [u8; WIRE_BYTES] {
        let mut b = [0; WIRE_BYTES];
        b[..2].copy_from_slice(b"LD");
        b[2] = 1;
        b[3] = self.flags;
        b[4..6].copy_from_slice(&self.sequence.to_le_bytes());
        b[6] = self.class;
        b[7] = self.score;
        b[8..12].copy_from_slice(&self.sample_us.to_le_bytes());
        b[12..14].copy_from_slice(&self.phase_permille.to_le_bytes());
        b[14..16].copy_from_slice(&self.mock_resistance_permille.to_le_bytes());
        b[16..18].copy_from_slice(&self.acquire_us.to_le_bytes());
        b[18..20].copy_from_slice(&self.infer_us.to_le_bytes());
        b
    }

    pub fn is_fresh(self, now_us: u32) -> bool {
        now_us.wrapping_sub(self.sample_us) <= MAX_AGE_US
    }

    /// A deterministic transport test. Does not read a sensor or execute a model.
    pub fn synthetic(sequence: u16, now_us: u32) -> Self {
        Self {
            flags: SYNTHETIC | MOCK_CONTROL,
            sequence,
            class: ((now_us / 5_000_000) % 4) as u8,
            score: 230,
            sample_us: now_us,
            phase_permille: ((now_us % 1_000_000) / 1_000) as u16,
            mock_resistance_permille: ((now_us % 2_000_000) / 2_000) as u16,
            acquire_us: u16::MAX,
            infer_us: u16::MAX,
        }
    }
}

/// Drain a bounded queue and keep its newest entry, discarding expired entries.
pub fn take_latest(
    consumer: &mut heapless::spsc::Consumer<'_, Packet>,
    now_us: u32,
) -> Option<Packet> {
    let mut newest = None;
    while let Some(packet) = consumer.dequeue() {
        newest = Some(packet);
    }
    newest.filter(|packet| packet.is_fresh(now_us))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_contract_matches_python_fixture() {
        let p = Packet {
            flags: 3,
            sequence: 0x1234,
            class: 2,
            score: 230,
            sample_us: 0x12345678,
            phase_permille: 500,
            mock_resistance_permille: 750,
            acquire_us: 1234,
            infer_us: 13000,
        };
        assert_eq!(
            p.encode(),
            [
                76, 68, 1, 3, 52, 18, 2, 230, 120, 86, 52, 18, 244, 1, 238, 2, 210, 4, 200, 50
            ]
        );
    }

    #[test]
    fn disconnected_consumer_cannot_grow_queue_and_resume_discards_old_data() {
        let mut q = PacketQueue::new();
        let (mut tx, mut rx) = q.split();
        for sequence in 0..4 {
            assert!(tx.enqueue(Packet::synthetic(sequence, 0)).is_ok());
        }
        assert!(tx.enqueue(Packet::synthetic(4, 0)).is_err());
        assert_eq!(take_latest(&mut rx, MAX_AGE_US + 1), None);
        assert!(tx.enqueue(Packet::synthetic(5, MAX_AGE_US + 1)).is_ok());
        assert_eq!(take_latest(&mut rx, MAX_AGE_US + 1).unwrap().sequence, 5);
    }

    #[test]
    fn newest_packet_and_timestamp_wrap() {
        let mut q = PacketQueue::new();
        let (mut tx, mut rx) = q.split();
        tx.enqueue(Packet::synthetic(1, u32::MAX - 9)).unwrap();
        tx.enqueue(Packet::synthetic(2, u32::MAX - 4)).unwrap();
        assert_eq!(take_latest(&mut rx, 10).unwrap().sequence, 2);
        assert!(take_latest(&mut rx, 10).is_none());
    }
}
