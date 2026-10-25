from __future__ import annotations

import sys
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
    SortingFrame,
    SortingProtocol,
    make_heartbeat,
    make_pick_target,
    make_sort_result,
    make_stop,
)


def round_trip(
    frame: SortingFrame,
) -> None:
    raw = SortingProtocol.pack(
        frame
    )

    decoded = SortingProtocol.unpack(
        raw
    )

    print()
    print(
        "-" * 64
    )

    print(
        "TX frame :",
        frame,
    )

    print(
        "TX bytes :",
        SortingProtocol.to_hex(
            raw
        ),
    )

    print(
        "RX frame :",
        decoded,
    )

    assert (
        decoded.mode
        ==
        frame.mode
    )

    assert (
        abs(
            decoded.x_mm
            -
            frame.x_mm
        )
        <=
        0.051
    )

    assert (
        abs(
            decoded.y_mm
            -
            frame.y_mm
        )
        <=
        0.051
    )

    if frame.mode in (Mode.PICK_TARGET, Mode.MCU_PICK_TARGET_ACK):
        # yaw is transmitted as signed int8, 1 degree/LSB.
        assert abs(decoded.yaw_deg - round(frame.yaw_deg)) <= 1e-9
        assert decoded.box_id == 0
    else:
        assert decoded.box_id == frame.box_id


def main() -> None:
    print()
    print(
        "=" * 64
    )

    print(
        "SORTING PROTOCOL LOCAL TEST"
    )

    print(
        "=" * 64
    )

    # ------------------------------------------------------------
    # 1. 心跳
    # ------------------------------------------------------------
    round_trip(
        make_heartbeat()
    )

    # ------------------------------------------------------------
    # 2. 顶部视觉 -> 抓取坐标
    #
    # 对应以后 TopPickPipeline READY 的输出。
    # ------------------------------------------------------------
    round_trip(
        make_pick_target(
            x_mm=-56.2,
            y_mm=25.1,
            yaw_deg=-37.4,
        )
    )

    # ------------------------------------------------------------
    # 3. 侧面识别 -> 箱号
    # ------------------------------------------------------------
    round_trip(
        make_sort_result(
            box_id=4
        )
    )

    # ------------------------------------------------------------
    # 4. STOP
    # ------------------------------------------------------------
    round_trip(
        make_stop()
    )

    # ------------------------------------------------------------
    # 5. 模拟 STM32 -> Jetson 状态
    # ------------------------------------------------------------
    for mode in (
        Mode.MCU_HEARTBEAT_ACK,
        Mode.MCU_PICK_TARGET_ACK,
        Mode.MCU_RECOGNITION_READY,
        Mode.MCU_SORT_RESULT_ACK,
        Mode.MCU_SORT_DONE,
        Mode.MCU_BUSY,
        Mode.MCU_START_WORK,
        Mode.MCU_STOP_WORK,
        Mode.MCU_ERROR,
    ):
        round_trip(
            SortingFrame(
                mode=mode,
                box_id=(
                    4
                    if mode
                    ==
                    Mode.MCU_SORT_DONE
                    else
                    0
                ),
            )
        )

    print()
    print(
        "=" * 64
    )

    print(
        "ALL PROTOCOL TESTS PASSED"
    )

    print(
        "=" * 64
    )


if __name__ == "__main__":
    main()
