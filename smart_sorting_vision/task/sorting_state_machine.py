from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import time
from typing import Iterable

from communication.sorting_protocol import (
    Mode,
    SortingFrame,
    make_pick_target,
    make_sort_result,
    make_stop,
)


class SortingState(Enum):
    STOPPED = auto()

    WAIT_TOP_TARGET = auto()
    WAIT_PICK_ACK = auto()
    WAIT_RECOGNITION_READY = auto()

    SIDE_RECOGNIZING = auto()
    WAIT_SORT_ACK = auto()
    WAIT_SORT_DONE = auto()

    ERROR = auto()


@dataclass(frozen=True)
class TopPickReady:
    ready: bool

    x_mm: float | None = None
    y_mm: float | None = None


@dataclass(frozen=True)
class SideSortReady:
    ready: bool

    box_id: int | None = None


@dataclass
class StateMachineStep:
    state: SortingState

    tx_frames: list[SortingFrame] = field(
        default_factory=list
    )

    event: str = ""

    cycle_index: int = 0

    error: str | None = None


class SortingStateMachine:
    """
    智能分拣整机任务状态机。

    本类只负责：
        - 状态切换
        - ACK / READY / DONE 处理
        - 超时
        - 重发
        - 防止重复发送 PICK_TARGET / SORT_RESULT

    本类不直接操作：
        - 相机
        - YOLO
        - 串口
        - 电机

    主程序每一轮调用：

        step = machine.update(
            rx_frames=...,
            top_pick=...,
            side_sort=...,
        )

    然后把 step.tx_frames 逐帧交给 STM32Link.send_frame()。

    状态：

        STOPPED
          └─ MCU_START_WORK
               ↓
        WAIT_TOP_TARGET
          └─ Top READY 2/2
               ↓ send PICK_TARGET
        WAIT_PICK_ACK
          └─ MCU_PICK_TARGET_ACK
               ↓
        WAIT_RECOGNITION_READY
          └─ MCU_RECOGNITION_READY
               ↓
        SIDE_RECOGNIZING
          └─ Side READY
               ↓ send SORT_RESULT
        WAIT_SORT_ACK
          └─ MCU_SORT_RESULT_ACK
               ↓
        WAIT_SORT_DONE
          └─ MCU_SORT_DONE
               ↓
        WAIT_TOP_TARGET

    全局：
        MCU_STOP_WORK -> STOPPED
        MCU_ERROR     -> ERROR
    """

    def __init__(
        self,
        *,
        pick_ack_timeout_s: float = 1.0,
        sort_ack_timeout_s: float = 1.0,
        recognition_ready_timeout_s: float = 15.0,
        sort_done_timeout_s: float = 15.0,
        max_ack_retries: int = 2,
    ) -> None:
        if pick_ack_timeout_s <= 0:
            raise ValueError(
                "pick_ack_timeout_s 必须 > 0"
            )

        if sort_ack_timeout_s <= 0:
            raise ValueError(
                "sort_ack_timeout_s 必须 > 0"
            )

        if recognition_ready_timeout_s <= 0:
            raise ValueError(
                "recognition_ready_timeout_s 必须 > 0"
            )

        if sort_done_timeout_s <= 0:
            raise ValueError(
                "sort_done_timeout_s 必须 > 0"
            )

        if max_ack_retries < 0:
            raise ValueError(
                "max_ack_retries 不能 < 0"
            )

        self.pick_ack_timeout_s = float(
            pick_ack_timeout_s
        )

        self.sort_ack_timeout_s = float(
            sort_ack_timeout_s
        )

        self.recognition_ready_timeout_s = float(
            recognition_ready_timeout_s
        )

        self.sort_done_timeout_s = float(
            sort_done_timeout_s
        )

        self.max_ack_retries = int(
            max_ack_retries
        )

        self.state = SortingState.STOPPED

        self.state_entered_at = time.monotonic()

        self.cycle_index = 0

        self.pick_retry_count = 0
        self.sort_retry_count = 0

        self.latched_pick_x_mm: float | None = None
        self.latched_pick_y_mm: float | None = None

        self.latched_box_id: int | None = None

        self.error_message: str | None = None

    # ============================================================
    # 基础状态操作
    # ============================================================

    def _now(
        self,
        now: float | None,
    ) -> float:
        if now is None:
            return time.monotonic()

        return float(
            now
        )

    def _enter(
        self,
        state: SortingState,
        now: float,
    ) -> None:
        self.state = state
        self.state_entered_at = now

    def _elapsed(
        self,
        now: float,
    ) -> float:
        return (
            now
            -
            self.state_entered_at
        )

    def _clear_cycle_data(
        self,
    ) -> None:
        self.pick_retry_count = 0
        self.sort_retry_count = 0

        self.latched_pick_x_mm = None
        self.latched_pick_y_mm = None

        self.latched_box_id = None

    def reset_to_stopped(
        self,
        *,
        now: float | None = None,
    ) -> None:
        current = self._now(
            now
        )

        self._clear_cycle_data()

        self.error_message = None

        self._enter(
            SortingState.STOPPED,
            current,
        )

    def _fail(
        self,
        message: str,
        *,
        now: float,
    ) -> StateMachineStep:
        self.error_message = str(
            message
        )

        self._enter(
            SortingState.ERROR,
            now,
        )

        return StateMachineStep(
            state=self.state,
            tx_frames=[
                make_stop()
            ],
            event="ERROR",
            cycle_index=self.cycle_index,
            error=self.error_message,
        )

    # ============================================================
    # RX 全局事件
    # ============================================================

    @staticmethod
    def _find_mode(
        frames: Iterable[SortingFrame],
        mode: Mode,
    ) -> SortingFrame | None:
        for frame in frames:
            if frame.mode == mode:
                return frame

        return None

    def _handle_global_rx(
        self,
        rx_frames: list[SortingFrame],
        *,
        now: float,
    ) -> StateMachineStep | None:
        stop_frame = self._find_mode(
            rx_frames,
            Mode.MCU_STOP_WORK,
        )

        if stop_frame is not None:
            self._clear_cycle_data()

            self.error_message = None

            self._enter(
                SortingState.STOPPED,
                now,
            )

            return StateMachineStep(
                state=self.state,
                event="MCU_STOP_WORK",
                cycle_index=self.cycle_index,
            )

        error_frame = self._find_mode(
            rx_frames,
            Mode.MCU_ERROR,
        )

        if error_frame is not None:
            return self._fail(
                "MCU_ERROR",
                now=now,
            )

        return None

    # ============================================================
    # 主更新
    # ============================================================

    def update(
        self,
        *,
        rx_frames: Iterable[SortingFrame] = (),
        top_pick: TopPickReady | None = None,
        side_sort: SideSortReady | None = None,
        now: float | None = None,
    ) -> StateMachineStep:
        current = self._now(
            now
        )

        frames = list(
            rx_frames
        )

        global_result = (
            self._handle_global_rx(
                frames,
                now=current,
            )
        )

        if global_result is not None:
            return global_result

        # ========================================================
        # STOPPED
        # ========================================================

        if self.state == SortingState.STOPPED:
            start_frame = self._find_mode(
                frames,
                Mode.MCU_START_WORK,
            )

            if start_frame is not None:
                self._clear_cycle_data()

                self.error_message = None

                self._enter(
                    SortingState.WAIT_TOP_TARGET,
                    current,
                )

                return StateMachineStep(
                    state=self.state,
                    event="MCU_START_WORK",
                    cycle_index=self.cycle_index,
                )

            return StateMachineStep(
                state=self.state,
                event="STOPPED",
                cycle_index=self.cycle_index,
            )

        # ========================================================
        # WAIT_TOP_TARGET
        # ========================================================

        if (
            self.state
            ==
            SortingState.WAIT_TOP_TARGET
        ):
            if (
                top_pick is not None
                and
                top_pick.ready
                and
                top_pick.x_mm is not None
                and
                top_pick.y_mm is not None
            ):
                self.latched_pick_x_mm = float(
                    top_pick.x_mm
                )

                self.latched_pick_y_mm = float(
                    top_pick.y_mm
                )

                self.pick_retry_count = 0

                tx = make_pick_target(
                    x_mm=
                        self.latched_pick_x_mm,
                    y_mm=
                        self.latched_pick_y_mm,
                )

                self._enter(
                    SortingState.WAIT_PICK_ACK,
                    current,
                )

                return StateMachineStep(
                    state=self.state,
                    tx_frames=[
                        tx
                    ],
                    event="SEND_PICK_TARGET",
                    cycle_index=self.cycle_index,
                )

            return StateMachineStep(
                state=self.state,
                event="WAIT_TOP_TARGET",
                cycle_index=self.cycle_index,
            )

        # ========================================================
        # WAIT_PICK_ACK
        # ========================================================

        if (
            self.state
            ==
            SortingState.WAIT_PICK_ACK
        ):
            ack = self._find_mode(
                frames,
                Mode.MCU_PICK_TARGET_ACK,
            )

            if ack is not None:
                self._enter(
                    SortingState.WAIT_RECOGNITION_READY,
                    current,
                )

                return StateMachineStep(
                    state=self.state,
                    event="MCU_PICK_TARGET_ACK",
                    cycle_index=self.cycle_index,
                )

            busy = self._find_mode(
                frames,
                Mode.MCU_BUSY,
            )

            if busy is not None:
                # MCU 明确回复 BUSY 时，
                # 重新计时，不立即重复发送。
                self.state_entered_at = current

                return StateMachineStep(
                    state=self.state,
                    event="MCU_BUSY",
                    cycle_index=self.cycle_index,
                )

            if (
                self._elapsed(
                    current
                )
                >=
                self.pick_ack_timeout_s
            ):
                if (
                    self.pick_retry_count
                    <
                    self.max_ack_retries
                ):
                    self.pick_retry_count += 1

                    assert (
                        self.latched_pick_x_mm
                        is not None
                    )

                    assert (
                        self.latched_pick_y_mm
                        is not None
                    )

                    tx = make_pick_target(
                        x_mm=
                            self.latched_pick_x_mm,
                        y_mm=
                            self.latched_pick_y_mm,
                    )

                    self.state_entered_at = current

                    return StateMachineStep(
                        state=self.state,
                        tx_frames=[
                            tx
                        ],
                        event=(
                            "RETRY_PICK_TARGET_"
                            f"{self.pick_retry_count}"
                        ),
                        cycle_index=self.cycle_index,
                    )

                return self._fail(
                    "PICK_TARGET ACK timeout",
                    now=current,
                )

            return StateMachineStep(
                state=self.state,
                event="WAIT_PICK_ACK",
                cycle_index=self.cycle_index,
            )

        # ========================================================
        # WAIT_RECOGNITION_READY
        # ========================================================

        if (
            self.state
            ==
            SortingState.WAIT_RECOGNITION_READY
        ):
            ready = self._find_mode(
                frames,
                Mode.MCU_RECOGNITION_READY,
            )

            if ready is not None:
                self._enter(
                    SortingState.SIDE_RECOGNIZING,
                    current,
                )

                return StateMachineStep(
                    state=self.state,
                    event="MCU_RECOGNITION_READY",
                    cycle_index=self.cycle_index,
                )

            if (
                self._elapsed(
                    current
                )
                >=
                self.recognition_ready_timeout_s
            ):
                return self._fail(
                    "RECOGNITION_READY timeout",
                    now=current,
                )

            return StateMachineStep(
                state=self.state,
                event="WAIT_RECOGNITION_READY",
                cycle_index=self.cycle_index,
            )

        # ========================================================
        # SIDE_RECOGNIZING
        # ========================================================

        if (
            self.state
            ==
            SortingState.SIDE_RECOGNIZING
        ):
            if (
                side_sort is not None
                and
                side_sort.ready
                and
                side_sort.box_id is not None
            ):
                box_id = int(
                    side_sort.box_id
                )

                if not (
                    1
                    <=
                    box_id
                    <=
                    6
                ):
                    return self._fail(
                        "非法 Box_ID："
                        f"{box_id}",
                        now=current,
                    )

                self.latched_box_id = (
                    box_id
                )

                self.sort_retry_count = 0

                tx = make_sort_result(
                    box_id=
                        self.latched_box_id
                )

                self._enter(
                    SortingState.WAIT_SORT_ACK,
                    current,
                )

                return StateMachineStep(
                    state=self.state,
                    tx_frames=[
                        tx
                    ],
                    event="SEND_SORT_RESULT",
                    cycle_index=self.cycle_index,
                )

            return StateMachineStep(
                state=self.state,
                event="SIDE_RECOGNIZING",
                cycle_index=self.cycle_index,
            )

        # ========================================================
        # WAIT_SORT_ACK
        # ========================================================

        if (
            self.state
            ==
            SortingState.WAIT_SORT_ACK
        ):
            ack = self._find_mode(
                frames,
                Mode.MCU_SORT_RESULT_ACK,
            )

            if ack is not None:
                self._enter(
                    SortingState.WAIT_SORT_DONE,
                    current,
                )

                return StateMachineStep(
                    state=self.state,
                    event="MCU_SORT_RESULT_ACK",
                    cycle_index=self.cycle_index,
                )

            busy = self._find_mode(
                frames,
                Mode.MCU_BUSY,
            )

            if busy is not None:
                self.state_entered_at = current

                return StateMachineStep(
                    state=self.state,
                    event="MCU_BUSY",
                    cycle_index=self.cycle_index,
                )

            if (
                self._elapsed(
                    current
                )
                >=
                self.sort_ack_timeout_s
            ):
                if (
                    self.sort_retry_count
                    <
                    self.max_ack_retries
                ):
                    self.sort_retry_count += 1

                    assert (
                        self.latched_box_id
                        is not None
                    )

                    tx = make_sort_result(
                        box_id=
                            self.latched_box_id
                    )

                    self.state_entered_at = current

                    return StateMachineStep(
                        state=self.state,
                        tx_frames=[
                            tx
                        ],
                        event=(
                            "RETRY_SORT_RESULT_"
                            f"{self.sort_retry_count}"
                        ),
                        cycle_index=self.cycle_index,
                    )

                return self._fail(
                    "SORT_RESULT ACK timeout",
                    now=current,
                )

            return StateMachineStep(
                state=self.state,
                event="WAIT_SORT_ACK",
                cycle_index=self.cycle_index,
            )

        # ========================================================
        # WAIT_SORT_DONE
        # ========================================================

        if (
            self.state
            ==
            SortingState.WAIT_SORT_DONE
        ):
            done = self._find_mode(
                frames,
                Mode.MCU_SORT_DONE,
            )

            if done is not None:
                if (
                    done.box_id != 0
                    and
                    self.latched_box_id is not None
                    and
                    done.box_id
                    !=
                    self.latched_box_id
                ):
                    return self._fail(
                        "SORT_DONE Box_ID mismatch: "
                        f"expected="
                        f"{self.latched_box_id}, "
                        f"received="
                        f"{done.box_id}",
                        now=current,
                    )

                self.cycle_index += 1

                self._clear_cycle_data()

                self._enter(
                    SortingState.WAIT_TOP_TARGET,
                    current,
                )

                return StateMachineStep(
                    state=self.state,
                    event="MCU_SORT_DONE",
                    cycle_index=self.cycle_index,
                )

            if (
                self._elapsed(
                    current
                )
                >=
                self.sort_done_timeout_s
            ):
                return self._fail(
                    "SORT_DONE timeout",
                    now=current,
                )

            return StateMachineStep(
                state=self.state,
                event="WAIT_SORT_DONE",
                cycle_index=self.cycle_index,
            )

        # ========================================================
        # ERROR
        # ========================================================

        if self.state == SortingState.ERROR:
            return StateMachineStep(
                state=self.state,
                event="ERROR",
                cycle_index=self.cycle_index,
                error=self.error_message,
            )

        return self._fail(
            "Unknown state",
            now=current,
        )
