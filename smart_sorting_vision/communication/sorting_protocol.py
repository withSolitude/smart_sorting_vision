from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct


FRAME_LEN = 10

HEADER_0 = 0xA5
HEADER_1 = 0x5A

TAIL_0 = 0x0D
TAIL_1 = 0x0A

# 坐标协议：
# 0.1 mm / LSB
XY_SCALE = 10.0

# PICK_TARGET / MCU_PICK_TARGET_ACK 的 byte[7] 不再是 Box_ID，
# 而是机械 XY 坐标系中的“夹爪闭合轴角度”，signed int8，1 deg / LSB。
# 该轴具有 180° 周期，因此正式视觉输出统一归一到 [-90, 90)。
YAW_SCALE_DEG = 1.0
YAW_MIN_DEG = -90.0
YAW_MAX_DEG = 90.0


class Mode(IntEnum):
    HEARTBEAT = 0x00

    PICK_TARGET = 0x01
    SORT_RESULT = 0x02

    # XY 标定专用：只移动 XY，不执行抓取/侧面识别/分箱流程。
    CALIB_MOVE = 0x03

    # Yaw 标定专用：只直接控制 Yaw 舵机，不下降、不夹取。
    # 该模式复用 X 字段传“舵机命令角度”，仍为 0.1/LSB；
    # Y 和 AUX 必须为 0。
    CALIB_YAW = 0x04

    # 储物机构独立测试：仅在下位机 STOPPED 状态使用。
    # AUX = Box_ID (1~6)。不进入正式 PICK/SORT 握手链。
    STORAGE_TEST = 0x05

    # 侧面识别台连续确认没有物体时，放弃本次失败夹取，
    # 要求下位机仅将夹爪从 Recognition 姿态安全返回 Disable，
    # 然后继续等待新的 PICK_TARGET。该命令不是 STOP，不结束比赛。
    RETRY_PICK = 0x06

    # 动态夹爪预开口。X字段承载目标预开口(mm)，Y/AUX=0。
    # 该命令只锁存参数，不立即驱动夹爪；真正预收缩仍在 PICK 流程内执行。
    PICK_PREOPEN = 0x07

    STOP = 0x7F

    MCU_HEARTBEAT_ACK = 0x80
    MCU_PICK_TARGET_ACK = 0x81
    MCU_RECOGNITION_READY = 0x82
    MCU_SORT_RESULT_ACK = 0x83
    MCU_SORT_DONE = 0x84
    MCU_STOP_ACK = 0x85
    MCU_BUSY = 0x86
    MCU_START_WORK = 0x87
    MCU_STOP_WORK = 0x88

    # XY 标定握手。
    # ACK: 下位机已接受目标；REACHED: 机械 XY 已稳定到位。
    MCU_CALIB_MOVE_ACK = 0x89
    MCU_CALIB_REACHED = 0x8A

    # Yaw 标定握手。X 字段回显目标舵机命令角度（deg）。
    MCU_CALIB_YAW_ACK = 0x8B
    MCU_CALIB_YAW_REACHED = 0x8C

    # 储物机构独立测试 ACK / DONE，AUX = Box_ID (1~6)。
    MCU_STORAGE_TEST_ACK = 0x8D
    MCU_STORAGE_TEST_DONE = 0x8E

    # RETRY_PICK 的完成通知。不是“刚收到命令”的 ACK：
    # 只有夹爪已经真正回到 Disable、下位机重新进入 WAIT_PICK_TARGET
    # 后才发送，保证上位机此时重新打开顶部识别不会被机构遮挡。
    MCU_RETRY_PICK_READY = 0x8F

    # PICK_PREOPEN 参数已锁存。X字段回显量化后的预开口(mm)。
    MCU_PICK_PREOPEN_ACK = 0x90

    MCU_ERROR = 0xE0


@dataclass(frozen=True)
class SortingFrame:
    mode: Mode

    x_mm: float = 0.0
    y_mm: float = 0.0

    box_id: int = 0

    # 仅 PICK_TARGET / MCU_PICK_TARGET_ACK 使用。
    # 含义：机械 XY 坐标系中的夹爪闭合轴方向（deg），不是舵机原始角度。
    yaw_deg: float = 0.0


class SortingProtocol:
    """
    智能分拣 Jetson <-> STM32 固定 10 字节协议。

    Frame:
        [0] 0xA5
        [1] 0x5A
        [2] Mode
        [3] X_L
        [4] X_H
        [5] Y_L
        [6] Y_H
        [7] AUX
        [8] 0x0D
        [9] 0x0A

    AUX:
        PICK_TARGET / MCU_PICK_TARGET_ACK -> signed int8 yaw_deg, 1°/LSB
        SORT_RESULT / SORT_ACK / SORT_DONE -> Box_ID
        STORAGE_TEST / STORAGE_TEST_ACK / STORAGE_TEST_DONE -> Box_ID
        RETRY_PICK / MCU_RETRY_PICK_READY -> 0
        PICK_PREOPEN / MCU_PICK_PREOPEN_ACK -> 0（X字段为预开口mm）
        MCU_ERROR -> Error_Code
        其他模式 -> 0

    X/Y:
        常规/XY标定模式：int16 little-endian，0.1 mm / LSB
        CALIB_YAW / 其 ACK、REACHED：X 复用为舵机命令角度，0.1 deg / LSB；Y=0
    """

    @staticmethod
    def _mm_to_raw(
        value_mm: float,
    ) -> int:
        raw = int(
            round(
                float(value_mm)
                *
                XY_SCALE
            )
        )

        if not (
            -32768
            <=
            raw
            <=
            32767
        ):
            raise ValueError(
                "坐标超出 int16 协议范围："
                f"{value_mm} mm -> {raw}"
            )

        return raw

    @staticmethod
    def _raw_to_mm(
        raw: int,
    ) -> float:
        return float(raw) / XY_SCALE

    @staticmethod
    def _yaw_to_raw(value_deg: float) -> int:
        value = float(value_deg)
        if not (YAW_MIN_DEG <= value <= YAW_MAX_DEG):
            raise ValueError(
                f"PICK yaw must be within [{YAW_MIN_DEG}, {YAW_MAX_DEG}] deg, got {value}"
            )
        raw = int(round(value / YAW_SCALE_DEG))
        if not (-128 <= raw <= 127):
            raise ValueError(f"yaw cannot fit int8: {value} deg -> {raw}")
        return raw

    @staticmethod
    def _raw_to_yaw(raw: int) -> float:
        return float(raw) * YAW_SCALE_DEG

    @classmethod
    def pack(
        cls,
        frame: SortingFrame,
    ) -> bytes:
        mode_value = int(
            frame.mode
        )

        if not (
            0
            <=
            mode_value
            <=
            255
        ):
            raise ValueError(
                f"非法 Mode: {mode_value}"
            )

        if not (0 <= int(frame.box_id) <= 255):
            raise ValueError("Box_ID 必须为 0~255")

        yaw_modes = {Mode.PICK_TARGET, Mode.MCU_PICK_TARGET_ACK}
        if frame.mode in yaw_modes:
            aux_raw = cls._yaw_to_raw(frame.yaw_deg) & 0xFF
        else:
            aux_raw = int(frame.box_id)

        x_raw = cls._mm_to_raw(
            frame.x_mm
        )

        y_raw = cls._mm_to_raw(
            frame.y_mm
        )

        payload = bytearray(
            FRAME_LEN
        )

        payload[0] = HEADER_0
        payload[1] = HEADER_1
        payload[2] = mode_value

        struct.pack_into(
            "<h",
            payload,
            3,
            x_raw,
        )

        struct.pack_into(
            "<h",
            payload,
            5,
            y_raw,
        )

        payload[7] = aux_raw

        payload[8] = TAIL_0
        payload[9] = TAIL_1

        return bytes(
            payload
        )

    @classmethod
    def unpack(
        cls,
        data: bytes | bytearray,
    ) -> SortingFrame:
        if len(data) != FRAME_LEN:
            raise ValueError(
                "帧长度错误："
                f"{len(data)} != {FRAME_LEN}"
            )

        if (
            data[0] != HEADER_0
            or
            data[1] != HEADER_1
        ):
            raise ValueError(
                "帧头错误"
            )

        if (
            data[8] != TAIL_0
            or
            data[9] != TAIL_1
        ):
            raise ValueError(
                "帧尾错误"
            )

        try:
            mode = Mode(
                data[2]
            )

        except ValueError as exc:
            raise ValueError(
                "未知 Mode："
                f"0x{data[2]:02X}"
            ) from exc

        x_raw = struct.unpack_from(
            "<h",
            data,
            3,
        )[0]

        y_raw = struct.unpack_from(
            "<h",
            data,
            5,
        )[0]

        yaw_modes = {Mode.PICK_TARGET, Mode.MCU_PICK_TARGET_ACK}
        if mode in yaw_modes:
            yaw_raw = struct.unpack_from("<b", data, 7)[0]
            yaw_deg = cls._raw_to_yaw(yaw_raw)
            box_id = 0
        else:
            yaw_deg = 0.0
            box_id = int(data[7])

        return SortingFrame(
            mode=mode,
            x_mm=cls._raw_to_mm(
                x_raw
            ),
            y_mm=cls._raw_to_mm(
                y_raw
            ),
            box_id=box_id,
            yaw_deg=yaw_deg,
        )

    @staticmethod
    def to_hex(
        data: bytes | bytearray,
    ) -> str:
        return " ".join(
            f"{byte:02X}"
            for byte
            in data
        )


def make_heartbeat() -> SortingFrame:
    return SortingFrame(
        mode=Mode.HEARTBEAT
    )


def make_pick_target(
    x_mm: float,
    y_mm: float,
    yaw_deg: float = 0.0,
) -> SortingFrame:
    return SortingFrame(
        mode=Mode.PICK_TARGET,
        x_mm=x_mm,
        y_mm=y_mm,
        box_id=0,
        yaw_deg=yaw_deg,
    )



def make_sort_result(
    box_id: int,
) -> SortingFrame:
    if not (
        1
        <=
        int(box_id)
        <=
        6
    ):
        raise ValueError(
            "分拣 Box_ID 必须为 1~6"
        )

    return SortingFrame(
        mode=Mode.SORT_RESULT,
        box_id=int(
            box_id
        ),
    )


def make_calib_yaw(servo_command_deg: float) -> SortingFrame:
    value = float(servo_command_deg)
    if not (-135.0 <= value <= 135.0):
        raise ValueError(
            f"CALIB_YAW servo command must be within [-135, 135] deg, got {value}"
        )
    return SortingFrame(
        mode=Mode.CALIB_YAW,
        # CALIB_YAW 协议复用 X 字段承载舵机命令角度，单位 deg。
        x_mm=value,
        y_mm=0.0,
        box_id=0,
    )


def make_storage_test(box_id: int) -> SortingFrame:
    value = int(box_id)
    if not (1 <= value <= 6):
        raise ValueError("储物测试 Box_ID 必须为 1~6")
    return SortingFrame(
        mode=Mode.STORAGE_TEST,
        box_id=value,
    )


def make_stop() -> SortingFrame:
    return SortingFrame(
        mode=Mode.STOP
    )
