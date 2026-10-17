use crate::{FRAME_WIRE_BYTES, SensorFrame};

pub const FRAMES_PER_SD_BATCH: usize = 64;
pub const SD_BATCH_BYTES: usize = FRAMES_PER_SD_BATCH * FRAME_WIRE_BYTES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionCategory {
    Walk,
    Run,
    Stairs,
    Sit,
    Stand,
    Transition,
}

impl MotionCategory {
    pub const ALL: [Self; 6] = [
        Self::Walk,
        Self::Run,
        Self::Stairs,
        Self::Sit,
        Self::Stand,
        Self::Transition,
    ];

    pub const fn directory(self) -> &'static str {
        match self {
            Self::Walk => "WALK",
            Self::Run => "RUN",
            Self::Stairs => "STAIRS",
            Self::Sit => "SIT",
            Self::Stand => "STAND",
            Self::Transition => "TRANS",
        }
    }

    pub const fn next(self) -> Self {
        match self {
            Self::Walk => Self::Run,
            Self::Run => Self::Stairs,
            Self::Stairs => Self::Sit,
            Self::Sit => Self::Stand,
            Self::Stand => Self::Transition,
            Self::Transition => Self::Walk,
        }
    }
}

pub struct FrameBatch {
    bytes: [u8; SD_BATCH_BYTES],
    frames: usize,
}

impl FrameBatch {
    pub const fn new() -> Self {
        Self {
            bytes: [0; SD_BATCH_BYTES],
            frames: 0,
        }
    }

    pub fn push(&mut self, frame: &SensorFrame) -> bool {
        assert!(!self.is_full(), "clear a full batch before pushing");
        let start = self.frames * FRAME_WIRE_BYTES;
        self.bytes[start..start + FRAME_WIRE_BYTES].copy_from_slice(&frame.encode_le());
        self.frames += 1;
        self.is_full()
    }

    pub const fn frames(&self) -> usize {
        self.frames
    }

    pub const fn is_empty(&self) -> bool {
        self.frames == 0
    }

    pub const fn is_full(&self) -> bool {
        self.frames == FRAMES_PER_SD_BATCH
    }

    pub fn encoded(&self) -> &[u8] {
        &self.bytes[..self.frames * FRAME_WIRE_BYTES]
    }

    pub fn clear(&mut self) {
        self.frames = 0;
    }
}

impl Default for FrameBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, not(target_os = "none")))]
mod tests {
    use super::*;

    #[test]
    fn batch_contains_concatenated_wire_frames() {
        let mut batch = FrameBatch::new();
        let first = SensorFrame {
            sequence: 1,
            ..SensorFrame::ZERO
        };
        let second = SensorFrame {
            sequence: 2,
            ..SensorFrame::ZERO
        };

        assert!(!batch.push(&first));
        assert!(!batch.push(&second));
        assert_eq!(batch.frames(), 2);
        assert_eq!(&batch.encoded()[..FRAME_WIRE_BYTES], &first.encode_le());
        assert_eq!(&batch.encoded()[FRAME_WIRE_BYTES..], &second.encode_le());
    }
}
