import unittest

from unpack_leg import BLOCK_BYTES, FRAME_BYTES, unpack


class UnpackTests(unittest.TestCase):
    def block(self, payload):
        used = len(payload) + 2
        return used.to_bytes(2, "big") + payload + bytes(BLOCK_BYTES - used)

    def test_preserves_frames_across_block_boundaries_and_removes_padding(self):
        first = bytes(range(FRAME_BYTES)) * 68
        last = bytes(reversed(range(FRAME_BYTES)))
        self.assertEqual(unpack(self.block(first) + self.block(last)), first + last)

    def test_rejects_truncation_misalignment_and_wrong_padding(self):
        for data in [bytes(1), self.block(bytes(40)), self.block(bytes(120))[:-1] + b"x"]:
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                unpack(data)


if __name__ == "__main__":
    unittest.main()
