from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

try:
    import serial
except ModuleNotFoundError:  # Windows 离线测试机可能没有安装 pyserial
    serial = None

_SERIAL_ERRORS = (
    (serial.SerialException,) if serial is not None else ()
) + (OSError, IOError)

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
    reconnect_count: int = 0


class LinkOfflineError(ConnectionError):
    """串口设备暂时不可用。

    该异常表示通信链路故障，不表示分拣业务故障。上层运行时应进入
    离线恢复状态并继续尝试重连，而不是退出主程序。
    """


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
        self._lock = threading.RLock()
        self._online = False
        self.last_error: str | None = None
        self.offline_since: float | None = None
        self.connection_generation = 0

    # ============================================================
    # 生命周期
    # ============================================================

    @property
    def is_open(self) -> bool:
        return bool(
            self.serial is not None
            and self.serial.is_open
        )

    @property
    def is_online(self) -> bool:
        """返回串口当前是否已经打开且没有记录通信异常。"""
        return self.is_open and self._online

    @property
    def last_rx_time(self) -> float | None:
        return self.stats.last_rx_time

    def _mark_offline(self, exc: BaseException | str) -> None:
        """统一记录掉线，避免把串口异常传播成业务 ERROR。"""
        self._online = False
        self.last_error = str(exc)
        if self.offline_since is None:
            self.offline_since = time.monotonic()

    def _serial_failure(self, action: str, exc: BaseException) -> LinkOfflineError:
        self._mark_offline(exc)
        try:
            if self.serial is not None and self.serial.is_open:
                self.serial.close()
        except Exception:
            pass
        self.serial = None
        return LinkOfflineError(f"STM32 {action} failed: {exc}")

    def open(self) -> None:
        with self._lock:
            if self.is_open:
                self._online = True
                return

            if serial is None:
                raise LinkOfflineError(
                    "当前 Python 环境缺少 pyserial，请在 Jetson 运行环境安装"
                )

            try:
                self.serial = serial.Serial(
                    port=self.device,
                    baudrate=self.baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=self.read_timeout,
                    write_timeout=self.write_timeout,
                )

                # 清掉程序启动前残留的数据；重连后也必须丢弃半帧，
                # 让协议解析从一个完整帧重新开始。
                self.serial.reset_input_buffer()
                self.serial.reset_output_buffer()
                self.rx_buffer.clear()
            except _SERIAL_ERRORS as exc:
                self.serial = None
                raise self._serial_failure("open", exc) from exc

            self._online = True
            self.last_error = None
            self.offline_since = None
            self.connection_generation += 1
            self.stats.reconnect_count += 1
            print(f"[STM32] opened: {self.device} @ {self.baudrate}")

    def reconnect_once(self) -> bool:
        """尝试一次重连；失败只返回 False，不抛出业务异常。"""
        with self._lock:
            self.close()
            try:
                self.open()
            except LinkOfflineError as exc:
                print(f"[STM32][OFFLINE] reconnect failed: {exc}")
                return False
            print("[STM32][ONLINE] serial link recovered")
            return True

    def close(self) -> None:
        with self._lock:
            if self.serial is None:
                self._online = False
                return

            try:
                if self.serial.is_open:
                    self.serial.close()
            finally:
                self.serial = None
                self._online = False
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
        with self._lock:
            if not self.is_online:
                raise LinkOfflineError("STM32Link 尚未连接")

            assert self.serial is not None
            raw = SortingProtocol.pack(frame)

            try:
                written = self.serial.write(raw)
                self.serial.flush()
            except _SERIAL_ERRORS as exc:
                raise self._serial_failure("send", exc) from exc

            if written != len(raw):
                exc = LinkOfflineError(
                    "串口发送长度异常："
                    f"{written}/{len(raw)}"
                )
                self._mark_offline(exc)
                raise exc

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
        with self._lock:
            if not self.is_online:
                raise LinkOfflineError("STM32Link 尚未连接")

            assert self.serial is not None
            try:
                waiting = int(self.serial.in_waiting)
            except _SERIAL_ERRORS as exc:
                raise self._serial_failure("read", exc) from exc

            if waiting <= 0:
                return b""

            try:
                data = self.serial.read(waiting)
            except _SERIAL_ERRORS as exc:
                raise self._serial_failure("read", exc) from exc

            if data:
                self.stats.rx_bytes += len(data)
                self.stats.last_rx_time = time.monotonic()
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
