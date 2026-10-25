"""Latch preopen before PICK_TARGET; ACK is not a movement-completed event."""
from __future__ import annotations
import math
import time
from communication.sorting_protocol import Mode, SortingFrame


def latch_preopen(link, opening_mm: float, timeout_s: float = 3.0) -> float:
    opening = float(opening_mm)
    if not math.isfinite(opening) or not 0 < opening <= 120:
        raise ValueError("Invalid preopen command")
    command = math.ceil(opening * 10) / 10
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("preopen ACK timeout must be positive")
    # Do not accept an ACK left over from an earlier latch transaction.
    for frame in link.poll():
        if frame.mode in (Mode.MCU_STOP_WORK, Mode.MCU_STOP_ACK, Mode.MCU_ERROR, Mode.MCU_BUSY):
            raise RuntimeError(f"Cannot start preopen transaction: {link.describe(frame)}")
    raw = link.send_frame(SortingFrame(mode=Mode.PICK_PREOPEN, x_mm=command))
    print(f"[TX] PICK_PREOPEN {command:.1f}mm | {raw.hex(' ').upper()}")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ack = False
        for frame in link.poll():
            print(f"[RX] {link.describe(frame)}")
            if frame.mode in (Mode.MCU_STOP_WORK, Mode.MCU_STOP_ACK, Mode.MCU_ERROR, Mode.MCU_BUSY):
                raise RuntimeError(f"Preopen aborted: {link.describe(frame)}")
            if frame.mode == Mode.MCU_PICK_PREOPEN_ACK:
                if abs(float(frame.x_mm) - command) > 0.051 or abs(float(frame.y_mm)) > 0.051:
                    raise RuntimeError("Preopen ACK echo mismatch")
                ack = True
        if ack:
            return command
        time.sleep(0.003)
    raise TimeoutError("No MCU_PICK_PREOPEN_ACK; PICK_TARGET was NOT sent")
