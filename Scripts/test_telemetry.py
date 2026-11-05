import dataclasses
import unittest
from telemetry import Packet, Tracker, WIRE, decode


class TelemetryTests(unittest.TestCase):
    def test_rust_wire_fixture(self):
        raw = bytes([76, 68, 1, 3, 52, 18, 2, 230, 120, 86, 52, 18, 244, 1, 238, 2, 210, 4, 200, 50])
        self.assertEqual(decode(raw), Packet(3, 0x1234, 2, 230, 0x12345678, 500, 750, 1234, 13000))
        self.assertEqual(WIRE.size, 20)

    def test_reject_truncated_unknown_and_out_of_range(self):
        raw = WIRE.pack(b"LD", 1, 3, 1, 0, 230, 100, 0, 500, 65535, 65535)
        for bad in (raw[:-1], b"XX" + raw[2:], raw[:2] + b"\x02" + raw[3:],
                    WIRE.pack(b"LD", 1, 3, 1, 0, 230, 100, 1001, 500, 65535, 65535)):
            with self.assertRaises(ValueError):
                decode(bad)

    def test_wrap_gaps_duplicates_and_reordering(self):
        t = Tracker()
        p = Packet(3, 65534, 0, 230, 0xfffffff0, 0, 0, 65535, 65535)
        self.assertTrue(t.accept(p, 1))
        p = dataclasses.replace(p, sequence=1, sample_us=20)
        self.assertTrue(t.accept(p, 1.06))
        self.assertEqual(t.gaps, 2)
        self.assertEqual(t.board_elapsed_us, 36)
        self.assertFalse(t.accept(p, 1.07))
        self.assertFalse(t.accept(dataclasses.replace(p, sequence=0), 1.08))
        self.assertEqual((t.received, t.old, t.gaps), (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
