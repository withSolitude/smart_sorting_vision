from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Optional

import serial

from communication.sorting_protocol import (
    FRAME_LEN,
    HEADER_0,
    HEADER_1,
    TAIL_0,
    TAIL_1,
    Mode,
    SortingFrame,
    SortingProtocol,
)


@dataclass
class LinkStats:
    rx_bytes: int = 0
    tx_bytes: int = 0

    rx_frames: int = 0
    tx_frames: int = 0

    rx_bad_frames: int = 0

    last_rx_time: float | None = None
    last_tx_time: float | None = None


class STM32Link:
    """
    Jetson <-> STM32 串口通信层。

    功能：
        1. 打开/关闭串口
        2. 发送 SortingFrame
        3. 非阻塞读取串口
        4. 自动处理粘包/拆包
        5. 自动寻找帧头并重新同步
        6. 解析固定 10 字节协议
        7. 统计收发状态
        8. 提供 wait_for_mode() 给后续状态机使用

    注意：
        本类只负责通信，不负责业务状态机。
    """

    def __init__(
        self,
        device: str = "/dev/ttyCH341USB0",
        baudrate: int = 115200,
        read_timeout: float = 0.0,
        write_timeout: float = 0.2,
    ) -> None:
        self.device = str(
            device
        )

        self.baudrate = int(
            baudrate
        )

        self.read_timeout = float(
            read_timeout
        )

        self.write_timeout = float(
            write_timeout
        )

        self.serial: serial.Serial | None = None

        self.rx_buffer = bytearray()

        self.stats = LinkStats()

    # ============================================================
    # 生命周期
    # ============================================================

    @property
    def is_open(self) -> bool:
        return bool(
            self.serial is not None
            and self.serial.is_open
        )

    def open(self) -> None:
        if self.is_open:
            return

        self.serial = serial.Serial(
            port=self.device,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.read_timeout,
            write_timeout=self.write_timeout,
        )

        # 清掉程序启动前残留的数据
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

        self.rx_buffer.clear()

        print(
            f"[STM32] opened: "
            f"{self.device} @ {self.baudrate}"
        )

    def close(self) -> None:
        if self.serial is None:
            return

        try:
            if self.serial.is_open:
                self.serial.close()

        finally:
            self.serial = None
            self.rx_buffer.clear()

    def __enter__(self):
        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()

    # ============================================================
    # 发送
    # ============================================================

    def send_frame(
        self,
        frame: SortingFrame,
    ) -> bytes:
        if not self.is_open:
            raise RuntimeError(
                "STM32Link 尚未打开"
            )

        assert self.serial is not None

        raw = SortingProtocol.pack(
            frame
        )

        written = self.serial.write(
            raw
        )

        if written != len(raw):
            raise RuntimeError(
                "串口发送长度异常："
                f"{written}/{len(raw)}"
            )

        self.serial.flush()

        now = time.monotonic()

        self.stats.tx_bytes += written
        self.stats.tx_frames += 1
        self.stats.last_tx_time = now

        return raw

    # ============================================================
    # 接收
    # ============================================================

    def _read_available_bytes(
        self,
    ) -> bytes:
        if not self.is_open:
            raise RuntimeError(
                "STM32Link 尚未打开"
            )

        assert self.serial is not None

        waiting = int(
            self.serial.in_waiting
        )

        if waiting <= 0:
            return b""

        data = self.serial.read(
            waiting
        )

        if data:
            self.stats.rx_bytes += len(
                data
            )

            self.stats.last_rx_time = (
                time.monotonic()
            )

        return data

    def _drop_until_header(
        self,
    ) -> None:
        """
        丢弃无效前导数据，直到：
            A5 5A
        出现在 buffer[0:2]。
        """

        while len(
            self.rx_buffer
        ) >= 2:
            if (
                self.rx_buffer[0]
                ==
                HEADER_0
                and
                self.rx_buffer[1]
                ==
                HEADER_1
            ):
                return

            del self.rx_buffer[0]

    def _extract_frames(
        self,
    ) -> list[SortingFrame]:
        frames: list[
            SortingFrame
        ] = []

        while True:
            self._drop_until_header()

            if len(
                self.rx_buffer
            ) < FRAME_LEN:
                break

            candidate = bytes(
                self.rx_buffer[
                    :FRAME_LEN
                ]
            )

            # 固定帧尾检查。
            # 如果帧尾不对，说明可能是错位帧，
            # 丢1字节后重新搜索 A5 5A。
            if (
                candidate[8]
                !=
                TAIL_0
                or
                candidate[9]
                !=
                TAIL_1
            ):
                del self.rx_buffer[0]

                self.stats.rx_bad_frames += 1

                continue

            try:
                frame = (
                    SortingProtocol.unpack(
                        candidate
                    )
                )

            except ValueError:
                del self.rx_buffer[0]

                self.stats.rx_bad_frames += 1

                continue

            del self.rx_buffer[
                :FRAME_LEN
            ]

            frames.append(
                frame
            )

            self.stats.rx_frames += 1

        return frames

    def poll(
        self,
    ) -> list[SortingFrame]:
        """
        非阻塞轮询。

        每次主循环调用一次即可。

        返回本轮完整收到的所有帧。
        """

        data = self._read_available_bytes()

        if data:
            self.rx_buffer.extend(
                data
            )

        return self._extract_frames()

    # ============================================================
    # 等待某类帧
    # ============================================================

    def wait_for_mode(
        self,
        expected_mode: Mode,
        *,
        timeout: float = 1.0,
        poll_interval: float = 0.002,
    ) -> Optional[SortingFrame]:
        """
        阻塞等待指定 Mode。

        适合调试或简单握手。
        正式 SortingStateMachine 后面会改成完全非阻塞。
        """

        deadline = (
            time.monotonic()
            +
            float(timeout)
        )

        while time.monotonic() < deadline:
            frames = self.poll()

            for frame in frames:
                if (
                    frame.mode
                    ==
                    expected_mode
                ):
                    return frame

            time.sleep(
                poll_interval
            )

        return None

    # ============================================================
    # Debug
    # ============================================================

    @staticmethod
    def describe(
        frame: SortingFrame,
    ) -> str:
        extra = (
            f"yaw={frame.yaw_deg:.1f}deg"
            if frame.mode in (Mode.PICK_TARGET, Mode.MCU_PICK_TARGET_ACK)
            else f"box={frame.box_id}"
        )
        return (
            f"mode={frame.mode.name}"
            f"(0x{int(frame.mode):02X}) "
            f"x={frame.x_mm:.1f}mm "
            f"y={frame.y_mm:.1f}mm "
            f"{extra}"
        )
