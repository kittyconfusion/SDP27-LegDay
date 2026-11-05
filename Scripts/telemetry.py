#!/usr/bin/env python3
"""LegDay playground collector: Wi-Fi UDP or BLE, with the same 20-byte records.

UDP and mock use only the standard library. BLE additionally needs bleak.
Arrival gaps and relative delay variation are observable; one-way latency is
not measurable from two unsynchronized clocks.
"""
import argparse
import asyncio
import csv
from dataclasses import asdict, dataclass
from collections import deque
from pathlib import Path
import socket
import struct
import time

SERVICE_UUID = "8ea2c100-97b5-4b99-a361-47c27d8e4250"
CHARACTERISTIC_UUID = "8ea2c101-97b5-4b99-a361-47c27d8e4250"
WIRE = struct.Struct("<2sBBHBBIHHHH")


@dataclass(frozen=True)
class Packet:
    flags: int
    sequence: int
    class_id: int
    score: int
    sample_us: int
    phase_permille: int
    mock_resistance_permille: int
    acquire_us: int
    infer_us: int


def decode(data):
    if len(data) != WIRE.size:
        raise ValueError(f"expected {WIRE.size} bytes, got {len(data)}")
    magic, version, *fields = WIRE.unpack(data)
    if magic != b"LD" or version != 1:
        raise ValueError("unknown telemetry magic/version")
    packet = Packet(*fields)
    if packet.phase_permille > 1000 or packet.mock_resistance_permille > 1000:
        raise ValueError("invalid permille value")
    return packet


class Tracker:
    def __init__(self):
        self.last = None
        self.received = self.gaps = self.old = 0
        self.board_elapsed_us = 0
        self.last_arrival = None
        self.intervals_ms = deque(maxlen=500)

    def accept(self, packet, arrival):
        if self.last is not None:
            step = (packet.sequence - self.last.sequence) & 0xffff
            board_step = (packet.sample_us - self.last.sample_us) & 0xffffffff
            # Duplicate/out-of-order, or a board restart. Restart the receiver
            # after a board reboot. No session ID exists in this small v1 format.
            if step == 0 or step >= 0x8000 or board_step >= 0x80000000:
                self.old += 1
                return False
            self.gaps += step - 1
            self.board_elapsed_us += board_step
            self.intervals_ms.append((arrival - self.last_arrival) * 1000)
        self.last, self.last_arrival = packet, arrival
        self.received += 1
        return True


class Collector:
    def __init__(self, csv_path=None):
        self.tracker = Tracker()
        self.invalid = 0
        self.next_report = 0
        self.file = None
        self.writer = None
        if csv_path:
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w", newline="")

    def receive(self, data, source):
        try:
            packet = decode(data)
        except ValueError:
            self.invalid += 1
            return False
        now = time.monotonic()
        if not self.tracker.accept(packet, now):
            return False
        row = {"arrival_monotonic_ns": time.monotonic_ns(), "source": source, **asdict(packet)}
        if self.file:
            if self.writer is None:
                self.writer = csv.DictWriter(self.file, fieldnames=row.keys())
                self.writer.writeheader()
            self.writer.writerow(row)
        if now >= self.next_report:
            self.report()
            if self.file:
                self.file.flush()
            self.next_report = now + 1
        return True

    def report(self):
        t = self.tracker
        if t.last is None:
            print(f"No valid packets received. invalid={self.invalid}", flush=True)
            return
        intervals = sorted(t.intervals_ms)
        p95 = intervals[min(len(intervals) - 1, int(len(intervals) * .95))] if intervals else 0
        p = t.last
        print(f"{'SIMULATED' if p.flags & 1 else 'LIVE'} "
              f"class_id={p.class_id} score={p.score}/255 phase={p.phase_permille}/1000 "
              f"mock_resistance={p.mock_resistance_permille}/1000 "
              f"received={t.received} seq_gaps={t.gaps} old={t.old} invalid={self.invalid} "
              f"arrival_gap_p95={p95:.1f}ms", flush=True)

    def close(self):
        if self.file:
            self.file.close()


def udp(args, collector):
    peer = args.peer
    deadline = time.monotonic() + args.seconds if args.seconds else float("inf")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((args.bind, args.port))
        sock.settimeout(.25)
        print(f"Listening on {args.bind}:{args.port}; peer={peer or 'first valid sender (then pinned)'}", flush=True)
        while time.monotonic() < deadline:
            try:
                data, source = sock.recvfrom(2048)
            except socket.timeout:
                continue
            if peer is not None and source[0] != peer:
                continue
            if collector.receive(data, f"{source[0]}:{source[1]}"):
                peer = source[0]


async def ble(args, collector):
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError:
        raise SystemExit("BLE needs bleak: create a venv, then pip install bleak")
    print("Scanning for LegDay service (10 seconds)...", flush=True)
    discovered = await BleakScanner.discover(timeout=10, return_adv=True)
    matches = [(device, adv) for device, adv in discovered.values()
               if SERVICE_UUID in [u.lower() for u in adv.service_uuids]
               and (not args.device or args.device in (device.address, device.name, adv.local_name))]
    if len(matches) != 1:
        for device, adv in matches:
            print(device.address, adv.local_name)
        raise SystemExit(f"Found {len(matches)} matching devices; use --device ADDRESS for one board")
    device, _ = matches[0]
    async with BleakClient(device) as client:
        print(f"Connected: {device.address}; subscribing", flush=True)
        await client.start_notify(CHARACTERISTIC_UUID,
                                  lambda _char, data: collector.receive(bytes(data), device.address))
        started = time.monotonic()
        while client.is_connected and (not args.seconds or time.monotonic() - started < args.seconds):
            await asyncio.sleep(.1)
        if client.is_connected:
            await client.stop_notify(CHARACTERISTIC_UUID)


def mock(args):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for sequence in range(args.count):
            us = (sequence * 20_000) & 0xffffffff
            data = WIRE.pack(b"LD", 1, 3, sequence & 0xffff, (sequence // 250) % 4, 230,
                             us, (us % 1_000_000) // 1000, (us % 2_000_000) // 2000, 65535, 65535)
            sock.sendto(data, (args.host, args.port))
            time.sleep(.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("udp", "ble"):
        p = sub.add_parser(name)
        p.add_argument("--csv")
        p.add_argument("--seconds", type=float, default=0, help="0 runs until Ctrl+C")
        if name == "udp":
            p.add_argument("--bind", default="0.0.0.0")
            p.add_argument("--port", type=int, default=5005)
            p.add_argument("--peer", help="only accept this board IPv4 address")
        else:
            p.add_argument("--device", help="address/UUID from a scan, or exact device name")
    p = sub.add_parser("mock")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "mock":
        return mock(args)
    collector = Collector(args.csv)
    try:
        if args.mode == "udp":
            udp(args, collector)
        else:
            asyncio.run(ble(args, collector))
    except KeyboardInterrupt:
        pass
    finally:
        collector.report()
        collector.close()


if __name__ == "__main__":
    main()
