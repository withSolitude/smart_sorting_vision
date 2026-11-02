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
)
from task.sorting_state_machine import (
    SideSortReady,
    SortingState,
    SortingStateMachine,
    TopPickReady,
)


def show(
    title: str,
    step,
) -> None:
    tx = [
        frame.mode.name
        for frame
        in step.tx_frames
    ]

    print(
        f"{title:<28} "
        f"state={step.state.name:<24} "
        f"event={step.event:<28} "
        f"tx={tx}"
    )


def main() -> None:
    machine = SortingStateMachine(
        pick_ack_timeout_s=1.0,
        sort_ack_timeout_s=1.0,
        recognition_ready_timeout_s=10.0,
        sort_done_timeout_s=10.0,
        max_ack_retries=2,
    )

    t = 100.0

    print()
    print(
        "=" * 120
    )
    print(
        "SORTING STATE MACHINE OFFLINE TEST"
    )
    print(
        "=" * 120
    )

    # ------------------------------------------------------------
    # 1. 上电默认 STOPPED
    # ------------------------------------------------------------
    step = machine.update(
        now=t
    )

    show(
        "power on",
        step,
    )

    assert (
        step.state
        ==
        SortingState.STOPPED
    )

    # ------------------------------------------------------------
    # 2. STM32 物理按钮启动
    # ------------------------------------------------------------
    t += 0.01

    step = machine.update(
        rx_frames=[
            SortingFrame(
                mode=
                    Mode.MCU_START_WORK
            )
        ],
        now=t,
    )

    show(
        "MCU_START_WORK",
        step,
    )

    assert (
        step.state
        ==
        SortingState.WAIT_TOP_TARGET
    )

    # ------------------------------------------------------------
    # 3. Top 视觉第1帧，尚未 READY
    # ------------------------------------------------------------
    t += 0.01

    step = machine.update(
        top_pick=TopPickReady(
            ready=False
        ),
        now=t,
    )

    show(
        "top WAIT 1/2",
        step,
    )

    assert not (
        step.tx_frames
    )

    # ------------------------------------------------------------
    # 4. Top 视觉 2/2 READY
    # ------------------------------------------------------------
    t += 0.01

    step = machine.update(
        top_pick=TopPickReady(
            ready=True,
            x_mm=-56.2,
            y_mm=25.1,
        ),
        now=t,
    )

    show(
        "top READY 2/2",
        step,
    )

    assert (
        step.state
        ==
        SortingState.WAIT_PICK_ACK
    )

    assert len(
        step.tx_frames
    ) == 1

    assert (
        step.tx_frames[0].mode
        ==
        Mode.PICK_TARGET
    )

    # ------------------------------------------------------------
    # 5. 即使视觉继续 READY，也不能重复发送
    # ------------------------------------------------------------
    t += 0.02

    step = machine.update(
        top_pick=TopPickReady(
            ready=True,
            x_mm=-56.2,
            y_mm=25.1,
        ),
        now=t,
    )

    show(
        "top still READY",
        step,
    )

    assert not (
        step.tx_frames
    )

    # ------------------------------------------------------------
    # 6. 下位机 ACK
    # ------------------------------------------------------------
    t += 0.02

    step = machine.update(
        rx_frames=[
            SortingFrame(
                mode=
                    Mode.MCU_PICK_TARGET_ACK
            )
        ],
        now=t,
    )

    show(
        "PICK ACK",
        step,
    )

    assert (
        step.state
        ==
        SortingState.WAIT_RECOGNITION_READY
    )

    # ------------------------------------------------------------
    # 7. 机械完成抓取、放到侧面识别位
    # ------------------------------------------------------------
    t += 1.0

    step = machine.update(
        rx_frames=[
            SortingFrame(
                mode=
                    Mode.MCU_RECOGNITION_READY
            )
        ],
        now=t,
    )

    show(
        "RECOGNITION_READY",
        step,
    )

    assert (
        step.state
        ==
        SortingState.SIDE_RECOGNIZING
    )

    # ------------------------------------------------------------
    # 8. 侧面识别 READY -> Box 4
    # ------------------------------------------------------------
    t += 0.1

    step = machine.update(
        side_sort=SideSortReady(
            ready=True,
            box_id=4,
        ),
        now=t,
    )

    show(
        "side READY box=4",
        step,
    )

    assert (
        step.state
        ==
        SortingState.WAIT_SORT_ACK
    )

    assert len(
        step.tx_frames
    ) == 1

    assert (
        step.tx_frames[0].mode
        ==
        Mode.SORT_RESULT
    )

    assert (
        step.tx_frames[0].box_id
        ==
        4
    )

    # ------------------------------------------------------------
    # 9. SORT ACK
    # ------------------------------------------------------------
    t += 0.02

    step = machine.update(
        rx_frames=[
            SortingFrame(
                mode=
                    Mode.MCU_SORT_RESULT_ACK
            )
        ],
        now=t,
    )

    show(
        "SORT ACK",
        step,
    )

    assert (
        step.state
        ==
        SortingState.WAIT_SORT_DONE
    )

    # ------------------------------------------------------------
    # 10. SORT DONE
    # ------------------------------------------------------------
    t += 1.0

    step = machine.update(
        rx_frames=[
            SortingFrame(
                mode=
                    Mode.MCU_SORT_DONE,
                box_id=4,
            )
        ],
        now=t,
    )

    show(
        "SORT DONE",
        step,
    )

    assert (
        step.state
        ==
        SortingState.WAIT_TOP_TARGET
    )

    assert (
        step.cycle_index
        ==
        1
    )

    # ------------------------------------------------------------
    # 11. 任意状态 MCU_STOP_WORK 都应强制 STOPPED
    # ------------------------------------------------------------
    t += 0.1

    step = machine.update(
        top_pick=TopPickReady(
            ready=True,
            x_mm=-40.0,
            y_mm=10.0,
        ),
        now=t,
    )

    assert (
        step.state
        ==
        SortingState.WAIT_PICK_ACK
    )

    t += 0.01

    step = machine.update(
        rx_frames=[
            SortingFrame(
                mode=
                    Mode.MCU_STOP_WORK
            )
        ],
        now=t,
    )

    show(
        "MCU_STOP_WORK",
        step,
    )

    assert (
        step.state
        ==
        SortingState.STOPPED
    )

    print()
    print(
        "=" * 120
    )

    print(
        "SORTING STATE MACHINE TEST PASSED"
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
