from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from communication.sorting_protocol import (
    Mode,
    make_heartbeat,
    make_pick_target,
    make_stop,
)
from communication.stm32_link import (
    STM32Link,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Jetson <-> STM32 UART live test"
        )
    )

    parser.add_argument(
        "--device",
        default="/dev/ttyCH341USB0",
    )

    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
    )

    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    link = STM32Link(
        device=args.device,
        baudrate=args.baudrate,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "STM32 UART LIVE TEST"
    )
    print(
        "=" * 72
    )
    print(
        f"Device : {args.device}"
    )
    print(
        f"Baud   : {args.baudrate}"
    )
    print()
    print(
        "1 : 发送 HEARTBEAT 并等待 MCU_HEARTBEAT_ACK"
    )
    print(
        "2 : 仅发送测试 PICK_TARGET(-56.2, 25.1)"
    )
    print(
        "3 : 发送 STOP"
    )
    print(
        "P : 轮询并打印当前收到的全部帧"
    )
    print(
        "Q : 退出"
    )
    print()
    print(
        "注意：2 会真正向 STM32 发送 PICK_TARGET。"
    )
    print(
        "如果机构当前不适合动作，不要按 2。"
    )
    print(
        "=" * 72
    )

    try:
        link.open()

        while True:
            # 平时持续接收，避免缓冲区堆积
            frames = link.poll()

            for frame in frames:
                print(
                    "[RX]",
                    link.describe(
                        frame
                    ),
                )

            command = input(
                "\ncommand> "
            ).strip().lower()

            if command == "q":
                break

            if command == "1":
                frame = make_heartbeat()

                raw = link.send_frame(
                    frame
                )

                print(
                    "[TX] HEARTBEAT:",
                    raw.hex(
                        " "
                    ).upper(),
                )

                ack = link.wait_for_mode(
                    Mode.MCU_HEARTBEAT_ACK,
                    timeout=
                        args.heartbeat_timeout,
                )

                if ack is None:
                    print(
                        "[TIMEOUT] "
                        "未收到 MCU_HEARTBEAT_ACK"
                    )

                else:
                    print(
                        "[ACK]",
                        link.describe(
                            ack
                        ),
                    )

                continue

            if command == "2":
                frame = make_pick_target(
                    x_mm=-56.2,
                    y_mm=25.1,
                )

                raw = link.send_frame(
                    frame
                )

                print(
                    "[TX] PICK_TARGET:",
                    raw.hex(
                        " "
                    ).upper(),
                )

                continue

            if command == "3":
                raw = link.send_frame(
                    make_stop()
                )

                print(
                    "[TX] STOP:",
                    raw.hex(
                        " "
                    ).upper(),
                )

                continue

            if command == "p":
                frames = link.poll()

                if not frames:
                    print(
                        "[RX] no complete frame"
                    )

                for frame in frames:
                    print(
                        "[RX]",
                        link.describe(
                            frame
                        ),
                    )

                print(
                    "[STATS]",
                    link.stats,
                )

                continue

            print(
                "未知命令。"
            )

    finally:
        link.close()

        print()
        print(
            "[STM32] closed"
        )


if __name__ == "__main__":
    main()
