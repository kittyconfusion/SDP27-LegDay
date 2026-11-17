import tempfile
import unittest
from pathlib import Path

from sensors import FIELDS, parse_sample, record, synthetic_lines, validate


class SensorDataTests(unittest.TestCase):
    def test_rejects_bad_data(self):
        for line in ['1,2,3', '0,0,0,imu0,nan,0,0,0,0,0', '-1,0,0,imu0,0,0,0,0,0,0']:
            with self.assertRaises(ValueError):
                parse_sample(line)

    def test_ignores_diagnostics_and_header(self):
        self.assertIsNone(parse_sample('# boot'))
        self.assertIsNone(parse_sample(','.join(FIELDS)))

    def test_synthetic_roundtrip_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'demo.csv'
            result = record(synthetic_lines(1, 50), path, 'synthetic', 'bench', 'synthetic')
            self.assertEqual(result['rows'], 50)
            self.assertEqual(validate(path)['mean_interval_us'], 20_000)
            self.assertEqual(result['source'], 'synthetic')
            with self.assertRaises(ValueError):
                record([], path, 'synthetic', 'bench', 'synthetic')

    def test_tracks_corruption_gaps_and_resets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.csv'
            lines = ['0,100,1,imu0,0,0,9.8,0,0,0', 'bad data',
                     '2,20100,1,imu0,0,0,9.8,0,0,0', '0,10,1,imu0,0,0,9.8,0,0,0']
            result = record(lines, path, 'bench', 'bench', 'measured')
            self.assertEqual((result['rejected_rows'], result['sequence_gaps'], result['resets']), (1, 1, 1))
            with self.assertRaises(ValueError):
                validate(path)

    def test_rejects_empty_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.csv'
            record([], path, 'bench', 'bench', 'measured')
            with self.assertRaises(ValueError):
                validate(path)


if __name__ == '__main__':
    unittest.main()
