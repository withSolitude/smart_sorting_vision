from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from communication.sorting_protocol import (
    Mode,
    SortingFrame,
    make_calib_yaw,
)
from communication.stm32_link import STM32Link
from runtime.app_config import load_runtime_config


class UserStopRequested(RuntimeError):
    pass


def parse_targets(text: str) -> list[float]:
    values = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        value = float(item)

        # 当前舵机驱动的绝对安全测试边界
        if not -135.0 <= value <= 135.0:
            raise argparse.ArgumentTypeError(
                f"servo target {value} outside [-135, 135]"
            )

        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError(
            "targets cannot be empty"
        )

    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Yaw servo physical-range test. "
            "Only sends STOP + CALIB_YAW. "
            "No XY motion, no lift descent, no claw closing."
        )
    )

    parser.add_argument(
        "--config",
        default=str(
            ROOT / "config" / "runtime_config.yaml"
        ),
    )

    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=[
            -60.0,
            -70.0,
            -80.0,
            -90.0,
            -100.0,
            -110.0,
            -120.0,
            -130.0,
            -135.0,
        ],
        help=(
            'comma-separated direct servo commands, '
            'e.g. "-60,-70,-80,-90"'
        ),
    )

    parser.add_argument(
        "--ack-timeout",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--reach-timeout",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--settle",
        type=float,
        default=0.8,
    )

    return parser.parse_args()


def drain(link: STM32Link) -> None:
    for frame in link.poll():
        print(
            f"[OLD RX] {link.describe(frame)}"
        )

        if frame.mode == Mode.MCU_ERROR:
            raise RuntimeError(
                f"STM32 MCU_ERROR: "
                f"code={frame.box_id}, "
                f"x={frame.x_mm:.1f}, "
                f"y={frame.y_mm:.1f}"
            )


def ensure_stopped(
    link: STM32Link,
    timeout_s: float = 1.5,
) -> None:

    drain(link)

    raw = link.send_frame(
        SortingFrame(
            mode=Mode.STOP,
        )
    )

    print(
        f"[TX] STOP | "
        f"{raw.hex(' ').upper()}"
    )

    deadline = (
        time.monotonic()
        + timeout_s
    )

    while (
        time.monotonic()
        < deadline
    ):

        for frame in link.poll():

            print(
                f"[RX] "
                f"{link.describe(frame)}"
            )

            if frame.mode == Mode.MCU_ERROR:
                raise RuntimeError(
                    f"STM32 MCU_ERROR: "
                    f"code={frame.box_id}"
                )

            if frame.mode in (
                Mode.MCU_STOP_ACK,
                Mode.MCU_STOP_WORK,
            ):
                print(
                    "[OK] MCU is STOPPED"
                )
                return

        time.sleep(0.003)

    raise TimeoutError(
        "STOP_ACK not received"
    )


def send_yaw_and_wait(
    link: STM32Link,
    servo_deg: float,
    *,
    ack_timeout: float,
    reach_timeout: float,
) -> None:

    drain(link)

    raw = link.send_frame(
        make_calib_yaw(
            float(servo_deg)
        )
    )

    print(
        f"[TX] CALIB_YAW "
        f"servo={servo_deg:+.2f} deg | "
        f"{raw.hex(' ').upper()}"
    )

    start = time.monotonic()

    ack_seen = False
    reached_seen = False

    while True:

        now = time.monotonic()

        for frame in link.poll():

            print(
                f"[RX] "
                f"{link.describe(frame)}"
            )

            if frame.mode == Mode.MCU_ERROR:
                raise RuntimeError(
                    f"STM32 MCU_ERROR: "
                    f"code={frame.box_id}"
                )

            if frame.mode == Mode.MCU_STOP_WORK:
                raise UserStopRequested(
                    "physical STOP pressed"
                )

            if (
                frame.mode
                == Mode.MCU_BUSY
            ):
                raise RuntimeError(
                    "MCU_BUSY: "
                    "CALIB_YAW requires STOPPED state"
                )

            if (
                frame.mode
                == Mode.MCU_CALIB_YAW_ACK
                and
                abs(
                    frame.x_mm
                    - servo_deg
                )
                <= 0.2
            ):
                ack_seen = True

            if (
                frame.mode
                == Mode.MCU_CALIB_YAW_REACHED
                and
                abs(
                    frame.x_mm
                    - servo_deg
                )
                <= 0.2
            ):
                reached_seen = True

        if (
            ack_seen
            and
            reached_seen
        ):
            return

        if (
            not ack_seen
            and
            now - start
            >= ack_timeout
        ):
            raise TimeoutError(
                "timeout waiting "
                "MCU_CALIB_YAW_ACK"
            )

        if (
            now - start
            >= reach_timeout
        ):
            raise TimeoutError(
                "timeout waiting "
                "MCU_CALIB_YAW_REACHED"
            )

        time.sleep(0.002)


def main() -> int:

    args = parse_args()

    cfg = load_runtime_config(
        ROOT,
        Path(args.config),
    )

    link = STM32Link(
        device=cfg["stm32"]["device"],
        baudrate=cfg["stm32"]["baud"],
    )

    print()
    print("=" * 68)
    print(
        " YAW SERVO PHYSICAL RANGE TEST"
    )
    print("=" * 68)
    print(
        "[SAFETY] Only STOP + CALIB_YAW are sent."
    )
    print(
        "[SAFETY] No XY move, no Lift descent, no Claw motion."
    )
    print(
        "[IMPORTANT] Remove objects and make sure "
        "the gripper/USB wires cannot hit anything."
    )
    print()
    print(
        "Targets:",
        ", ".join(
            f"{v:+.1f}"
            for v in args.targets
        ),
    )
    print()

    last_safe = None

    try:

        link.open()

        ensure_stopped(link)

        for index, target in enumerate(
            args.targets,
            start=1,
        ):

            print()
            print(
                "-" * 68
            )

            print(
                f"[{index}/"
                f"{len(args.targets)}] "
                f"next servo target "
                f"= {target:+.1f} deg"
            )

            answer = input(
                "Press ENTER to move, "
                "or type q then ENTER to stop: "
            ).strip().lower()

            if answer == "q":
                break

            send_yaw_and_wait(
                link,
                target,
                ack_timeout=
                    args.ack_timeout,
                reach_timeout=
                    args.reach_timeout,
            )

            time.sleep(
                args.settle
            )

            print(
                f"\n[REACHED] "
                f"servo={target:+.1f} deg"
            )

            print(
                "Check physically:"
            )
            print(
                "  - no mechanical collision"
            )
            print(
                "  - no hard stop / abnormal force"
            )
            print(
                "  - no wire pulling"
            )
            print(
                "  - yaw still rotates normally"
            )

            answer = input(
                "Safe? "
                "[y = safe / q = stop test]: "
            ).strip().lower()

            if answer != "y":

                print(
                    "[STOP] Do not continue "
                    "toward a larger angle."
                )

                break

            last_safe = target

        print()
        print("=" * 68)

        if last_safe is None:
            print(
                "[RESULT] No new safe "
                "target confirmed."
            )
        else:
            print(
                "[RESULT] Last confirmed "
                f"safe servo target: "
                f"{last_safe:+.1f} deg"
            )

        print("=" * 68)

    except KeyboardInterrupt:

        print(
            "\n[ABORT] Ctrl-C"
        )

    except UserStopRequested as exc:

        print(
            f"\n[ABORT] {exc}"
        )

    finally:

        link.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
