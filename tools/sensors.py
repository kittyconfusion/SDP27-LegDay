"""Sensor CSV capture, validation and explicitly synthetic offline demo data."""
import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

FIELDS = 'seq,t_us,read_us,sensor_id,ax_m_s2,ay_m_s2,az_m_s2,gx_rad_s,gy_rad_s,gz_rad_s'.split(',')


def parse_sample(line):
    line = line.strip()
    if not line or line.startswith('#') or line == ','.join(FIELDS):
        return None
    row = next(csv.reader([line]))
    if len(row) != len(FIELDS) or row[3] != 'imu0':
        raise ValueError('Wrong field count or sensor ID')
    for value in row[:3]:
        if int(value) < 0:
            raise ValueError('Negative sequence/timestamp/duration')
    if not all(math.isfinite(float(value)) for value in row[4:]):
        raise ValueError('Nonfinite sensor reading')
    return row


def record(lines, destination, label, placement, source):
    """Stream rows without buffering a session; retain diagnostics separately."""
    destination = Path(destination)
    metadata_path = destination.with_suffix('.metadata.json')
    if destination.exists() or metadata_path.exists():
        raise ValueError('Output exists; choose a new filename')
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(schema='legday_imu6_v1', source=source, label=label,
                    placement=placement, started_utc=datetime.now(timezone.utc).isoformat(),
                    rows=0, rejected_rows=0, sequence_gaps=0, resets=0, diagnostics=[])
    previous = None
    created = False
    try:
        with destination.open('x', newline='', encoding='utf-8') as output:
            created = True
            writer = csv.writer(output)
            writer.writerow(FIELDS)
            for line in lines:
                if line.strip().startswith('#'):
                    metadata['diagnostics'].append(line.strip())
                    continue
                try:
                    row = parse_sample(line)
                except (ValueError, csv.Error):
                    metadata['rejected_rows'] += 1
                    continue
                if row is None:
                    continue
                seq, timestamp = map(int, row[:2])
                if previous is not None:
                    if seq <= previous[0] or timestamp <= previous[1]:
                        metadata['resets'] += 1
                    else:
                        metadata['sequence_gaps'] += max(0, seq - previous[0] - 1)
                previous = seq, timestamp
                writer.writerow(row)
                metadata['rows'] += 1
    except KeyboardInterrupt:
        metadata['interrupted'] = True
    finally:
        if created:
            metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
            with metadata_path.open('x', encoding='utf-8') as sidecar:
                json.dump(metadata, sidecar, indent=2)
    return metadata


def synthetic_lines(seconds, rate):
    yield '# SYNTHETIC SOFTWARE TEST DATA; not human motion measurements'
    for seq in range(int(seconds * rate)):
        phase = 2 * math.pi * seq / rate
        values = [0.3 * math.sin(phase), 0, 9.80665, 0, 0.2 * math.cos(phase), 0]
        yield ','.join(map(str, [seq, round(seq * 1_000_000 / rate), 0, 'imu0', *values]))


def validate(path):
    count = 0
    previous = None
    intervals = []
    with Path(path).open(encoding='utf-8', newline='') as source:
        if source.readline().strip() != ','.join(FIELDS):
            raise ValueError('CSV header does not match legday_imu6_v1')
        for line_number, line in enumerate(source, 2):
            row = parse_sample(line)
            if row is None:
                raise ValueError(f'Unexpected empty/header/comment row at line {line_number}')
            seq, timestamp = map(int, row[:2])
            if previous is not None:
                if seq <= previous[0] or timestamp <= previous[1]:
                    raise ValueError(f'Device reset or non-increasing sequence/time at line {line_number}')
                intervals.append(timestamp - previous[1])
            previous = seq, timestamp
            count += 1
    if not count:
        raise ValueError('No samples')
    return dict(rows=count, mean_interval_us=sum(intervals) / len(intervals) if intervals else None,
                min_interval_us=min(intervals) if intervals else None,
                max_interval_us=max(intervals) if intervals else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('ports')
    check = commands.add_parser('validate')
    check.add_argument('path', type=Path)
    for command in ('demo', 'capture'):
        sub = commands.add_parser(command)
        sub.add_argument('--out', type=Path, required=True)
        sub.add_argument('--seconds', type=float, default=10)
        if command == 'capture':
            sub.add_argument('--port', required=True)
            sub.add_argument('--label', required=True)
            sub.add_argument('--placement', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'validate':
            print(json.dumps(validate(args.path), indent=2))
            return
        if args.command != 'ports' and (not math.isfinite(args.seconds) or args.seconds <= 0):
            parser.error('--seconds must be positive and finite')
        if args.command == 'demo':
            result = record(synthetic_lines(args.seconds, 50), args.out,
                            'synthetic', 'synthetic', 'synthetic')
        else:
            import serial
            from serial.tools import list_ports
            if args.command == 'ports':
                for port in list_ports.comports():
                    print(f'{port.device}: {port.description}')
                return
            with serial.Serial(args.port, 115200, timeout=1) as device:
                deadline = time.monotonic() + args.seconds

                def lines():
                    while time.monotonic() < deadline:
                        yield device.readline().decode('utf-8', errors='replace')

                result = record(lines(), args.out, args.label, args.placement, 'measured')
        print(json.dumps(result, indent=2))
        if result['rows'] == 0:
            raise ValueError('No valid samples recorded')
    except ImportError:
        parser.error('Live capture requires: python -m pip install pyserial')
    except (OSError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
