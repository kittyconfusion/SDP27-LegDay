#!/usr/bin/env python3
"""Capture ESP32 serial output without an interactive monitor.

Uses only the Python standard library (no pyserial). The Nano ESP32 exposes a
native USB CDC port, so the baud rate is irrelevant, but we still put the tty
into raw mode. The port can briefly disappear while espflash resets the board,
so opening is retried for a few seconds.

    python3 Scripts/serial_capture.py /dev/cu.usbmodem2143101 20 capture.log
    python3 Scripts/serial_capture.py /dev/cu.usbmodem2143101 20 --send r@3 --send r@13
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
from pathlib import Path


def open_port(path: str, retry_seconds: float) -> int:
    deadline = time.monotonic() + retry_seconds
    while True:
        try:
            fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            break
        except OSError as error:
            if time.monotonic() >= deadline:
                raise SystemExit(f"could not open {path}: {error}")
            time.sleep(0.25)
    attributes = termios.tcgetattr(fd)
    attributes[0] = 0  # iflag
    attributes[1] = 0  # oflag
    attributes[2] = termios.CS8 | termios.CREAD | termios.CLOCAL  # cflag
    attributes[3] = 0  # lflag: raw
    attributes[4] = termios.B115200
    attributes[5] = termios.B115200
    attributes[6][termios.VMIN] = 0
    attributes[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attributes)
    return fd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port")
    parser.add_argument("seconds", type=float, help="capture duration")
    parser.add_argument("output", nargs="?", type=Path, help="also append output here")
    parser.add_argument("--retry", type=float, default=10.0, help="seconds to wait for the port")
    parser.add_argument("--send", action="append", default=[], metavar="TEXT@SECONDS",
                        help="write TEXT to the port SECONDS after start (repeatable), e.g. --send r@3")
    args = parser.parse_args()

    sends: list[tuple[float, bytes]] = []
    for item in args.send:
        text, _, when = item.rpartition("@")
        if not when:
            raise SystemExit(f"--send needs TEXT@SECONDS, got {item!r}")
        sends.append((float(when), text.encode()))
    sends.sort()

    fd = open_port(args.port, args.retry)
    output = args.output.open("ab") if args.output else None
    started = time.monotonic()
    deadline = started + args.seconds
    pending = b""
    try:
        while time.monotonic() < deadline:
            while sends and time.monotonic() - started >= sends[0][0]:
                _, payload = sends.pop(0)
                os.write(fd, payload)
                print(f"[serial_capture] sent {payload!r}")
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                continue
            except OSError as error:
                print(f"\n[serial_capture] read failed: {error}", file=sys.stderr)
                break
            if not chunk:
                continue
            if output:
                output.write(chunk)
                output.flush()
            pending += chunk
            lines = pending.split(b"\n")
            pending = lines.pop()
            for line in lines:
                print(line.decode("utf-8", errors="replace").rstrip("\r"))
            sys.stdout.flush()
        if pending:
            print(pending.decode("utf-8", errors="replace"))
    finally:
        if output:
            output.close()
        os.close(fd)


if __name__ == "__main__":
    main()
