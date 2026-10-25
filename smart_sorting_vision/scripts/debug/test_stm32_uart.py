import argparse
import math
import queue
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum

import serial


# ============================================================
# 串口协议
# ============================================================

FRAME_HEAD_0 = 0xA5
FRAME_HEAD_1 = 0x5A

FRAME_TAIL_0 = 0x0D
FRAME_TAIL_1 = 0x0A

FRAME_LENGTH = 10

COORDINATE_SCALE_MM = 0.1


# ============================================================
# MODE
# ============================================================

class Mode(IntEnum):

    # --------------------------------------------------------
    # Jetson -> STM32
    # --------------------------------------------------------

    HEARTBEAT = 0x00

    PICK_TARGET = 0x01

    SORT_RESULT = 0x02

    STOP = 0x7F


    # --------------------------------------------------------
    # STM32 -> Jetson
    # --------------------------------------------------------

    MCU_HEARTBEAT_ACK = 0x80

    MCU_PICK_TARGET_ACK = 0x81

    MCU_RECOGNITION_READY = 0x82

    MCU_SORT_RESULT_ACK = 0x83

    MCU_SORT_DONE = 0x84

    MCU_STOP_ACK = 0x85

    MCU_BUSY = 0x86

    MCU_START_WORK = 0x87

    MCU_STOP_WORK = 0x88

    MCU_ERROR = 0xE0


# ============================================================
# MODE名称
# ============================================================

MODE_NAME = {

    Mode.HEARTBEAT:
        "HEARTBEAT",

    Mode.PICK_TARGET:
        "PICK_TARGET",

    Mode.SORT_RESULT:
        "SORT_RESULT",

    Mode.STOP:
        "STOP",

    Mode.MCU_HEARTBEAT_ACK:
        "MCU_HEARTBEAT_ACK",

    Mode.MCU_PICK_TARGET_ACK:
        "MCU_PICK_TARGET_ACK",

    Mode.MCU_RECOGNITION_READY:
        "MCU_RECOGNITION_READY",

    Mode.MCU_SORT_RESULT_ACK:
        "MCU_SORT_RESULT_ACK",

    Mode.MCU_SORT_DONE:
        "MCU_SORT_DONE",

    Mode.MCU_STOP_ACK:
        "MCU_STOP_ACK",

    Mode.MCU_BUSY:
        "MCU_BUSY",

    Mode.MCU_START_WORK:
        "MCU_START_WORK",

    Mode.MCU_STOP_WORK:
        "MCU_STOP_WORK",

    Mode.MCU_ERROR:
        "MCU_ERROR",
}


# ============================================================
# 接收帧
# ============================================================

@dataclass
class SortingFrame:

    mode: int

    x_mm: float

    y_mm: float

    box_id: int

    raw: bytes


# ============================================================
# C/C++ lround
#
# Python round() 是 bankers rounding，
# 为了尽量和 STM32 std::lround 一致，
# 自己处理。
# ============================================================

def cpp_lround(value: float) -> int:

    if value >= 0.0:

        return math.floor(
            value + 0.5
        )

    return math.ceil(
        value - 0.5
    )


# ============================================================
# mm -> int16
# ============================================================

def mm_to_raw(value_mm: float) -> int:

    raw = cpp_lround(
        value_mm /
        COORDINATE_SCALE_MM
    )

    raw = max(
        -32768,
        min(
            32767,
            raw
        )
    )

    return raw


# ============================================================
# 构造10字节协议帧
# ============================================================

def build_frame(
    mode: int,
    x_mm: float = 0.0,
    y_mm: float = 0.0,
    box_id: int = 0,
) -> bytes:

    mode = int(mode)

    # --------------------------------------------------------
    # 参数检查
    # --------------------------------------------------------

    if mode == Mode.SORT_RESULT:

        if not 1 <= box_id <= 6:

            raise ValueError(
                "SORT_RESULT 的 box_id 必须为 1~6"
            )

    else:

        # 除SORT_RESULT外，
        # Jetson发送的其他帧Box_ID都应该是0

        if mode in (
            Mode.HEARTBEAT,
            Mode.PICK_TARGET,
            Mode.STOP,
        ):

            box_id = 0


    x_raw = mm_to_raw(
        x_mm
    )

    y_raw = mm_to_raw(
        y_mm
    )


    # --------------------------------------------------------
    # < ：小端
    #
    # B B B
    # h
    # h
    # B B B
    #
    # 共10字节
    # --------------------------------------------------------

    frame = struct.pack(
        "<BBBhhBBB",

        FRAME_HEAD_0,

        FRAME_HEAD_1,

        mode,

        x_raw,

        y_raw,

        box_id,

        FRAME_TAIL_0,

        FRAME_TAIL_1,
    )


    assert len(frame) == FRAME_LENGTH

    return frame


# ============================================================
# 解码10字节帧
# ============================================================

def decode_frame(
    frame: bytes
) -> SortingFrame:

    if len(frame) != FRAME_LENGTH:

        raise ValueError(
            f"帧长度错误: {len(frame)}"
        )


    if (
        frame[0] != FRAME_HEAD_0
        or
        frame[1] != FRAME_HEAD_1
    ):

        raise ValueError(
            "帧头错误"
        )


    if (
        frame[8] != FRAME_TAIL_0
        or
        frame[9] != FRAME_TAIL_1
    ):

        raise ValueError(
            "帧尾错误"
        )


    (
        _head0,
        _head1,
        mode,
        x_raw,
        y_raw,
        box_id,
        _tail0,
        _tail1,

    ) = struct.unpack(
        "<BBBhhBBB",
        frame
    )


    return SortingFrame(

        mode=mode,

        x_mm=
            x_raw *
            COORDINATE_SCALE_MM,

        y_mm=
            y_raw *
            COORDINATE_SCALE_MM,

        box_id=box_id,

        raw=frame,
    )


# ============================================================
# UART通信类
# ============================================================

class SortingUART:

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
    ):

        self.port = port

        self.baudrate = baudrate


        self.serial = serial.Serial(

            port=self.port,

            baudrate=self.baudrate,

            bytesize=serial.EIGHTBITS,

            parity=serial.PARITY_NONE,

            stopbits=serial.STOPBITS_ONE,

            timeout=0.05,

            write_timeout=0.5,
        )


        # 接收缓存

        self.rx_buffer = bytearray()


        # 已解析帧队列

        self.rx_queue = queue.Queue()


        # 线程控制

        self.running = True


        self.rx_thread = threading.Thread(

            target=self._rx_loop,

            daemon=True,
        )


        self.rx_thread.start()


        print(
            f"[UART] opened "
            f"{self.port} @ "
            f"{self.baudrate}"
        )


    # ========================================================
    # Close
    # ========================================================

    def close(self):

        self.running = False


        if self.rx_thread.is_alive():

            self.rx_thread.join(
                timeout=1.0
            )


        if self.serial.is_open:

            self.serial.close()


        print(
            "[UART] closed"
        )


    # ========================================================
    # 发送原始协议帧
    # ========================================================

    def send_frame(
        self,
        mode: int,
        x_mm: float = 0.0,
        y_mm: float = 0.0,
        box_id: int = 0,
    ):

        frame = build_frame(

            mode=mode,

            x_mm=x_mm,

            y_mm=y_mm,

            box_id=box_id,
        )


        self.serial.write(
            frame
        )

        self.serial.flush()


        mode_name = MODE_NAME.get(
            mode,
            f"UNKNOWN_0x{mode:02X}"
        )


        print(
            "\n[TX]",
            mode_name
        )

        print(
            "     HEX:",
            frame.hex(" ").upper()
        )

        print(
            f"     X={x_mm:.1f} mm, "
            f"Y={y_mm:.1f} mm, "
            f"BOX={box_id}"
        )


    # ========================================================
    # 心跳
    # ========================================================

    def send_heartbeat(self):

        self.send_frame(
            Mode.HEARTBEAT
        )


    # ========================================================
    # 顶部识别坐标
    # ========================================================

    def send_pick_target(
        self,
        x_mm: float,
        y_mm: float,
    ):

        self.send_frame(

            Mode.PICK_TARGET,

            x_mm=x_mm,

            y_mm=y_mm,

            box_id=0,
        )


    # ========================================================
    # 侧面识别结果
    # ========================================================

    def send_sort_result(
        self,
        box_id: int,
    ):

        self.send_frame(

            Mode.SORT_RESULT,

            x_mm=0.0,

            y_mm=0.0,

            box_id=box_id,
        )


    # ========================================================
    # STOP
    # ========================================================

    def send_stop(self):

        self.send_frame(
            Mode.STOP
        )


    # ========================================================
    # 接收线程
    # ========================================================

    def _rx_loop(self):

        while self.running:

            try:

                count = (
                    self.serial.in_waiting
                )


                if count <= 0:

                    data = self.serial.read(
                        1
                    )

                else:

                    data = self.serial.read(
                        count
                    )


                if not data:

                    continue


                self.rx_buffer.extend(
                    data
                )


                self._parse_rx_buffer()


            except serial.SerialException as e:

                print(
                    "[UART ERROR]",
                    e
                )

                self.running = False

                break


    # ========================================================
    # 流式解析
    # ========================================================

    def _parse_rx_buffer(self):

        while True:

            # ------------------------------------------------
            # 数据不足
            # ------------------------------------------------

            if len(self.rx_buffer) < 2:

                return


            # ------------------------------------------------
            # 找A5 5A
            # ------------------------------------------------

            head_index = self.rx_buffer.find(
                bytes(
                    [
                        FRAME_HEAD_0,
                        FRAME_HEAD_1
                    ]
                )
            )


            if head_index < 0:

                # 如果最后一个字节是A5，
                # 留下来等下一次5A

                if (
                    len(self.rx_buffer) > 0
                    and
                    self.rx_buffer[-1]
                    ==
                    FRAME_HEAD_0
                ):

                    self.rx_buffer[:] = bytes(
                        [FRAME_HEAD_0]
                    )

                else:

                    self.rx_buffer.clear()


                return


            # ------------------------------------------------
            # 丢弃帧头前垃圾数据
            # ------------------------------------------------

            if head_index > 0:

                del self.rx_buffer[
                    :head_index
                ]


            # ------------------------------------------------
            # 等完整10字节
            # ------------------------------------------------

            if (
                len(self.rx_buffer)
                <
                FRAME_LENGTH
            ):

                return


            candidate = bytes(
                self.rx_buffer[
                    :FRAME_LENGTH
                ]
            )


            # ------------------------------------------------
            # 帧尾正确
            # ------------------------------------------------

            if (
                candidate[8]
                ==
                FRAME_TAIL_0
                and
                candidate[9]
                ==
                FRAME_TAIL_1
            ):

                try:

                    parsed = decode_frame(
                        candidate
                    )

                except ValueError as e:

                    print(
                        "[RX PARSE ERROR]",
                        e
                    )

                    del self.rx_buffer[0]

                    continue


                del self.rx_buffer[
                    :FRAME_LENGTH
                ]


                self.rx_queue.put(
                    parsed
                )


                self._print_rx_frame(
                    parsed
                )


                continue


            # ------------------------------------------------
            # 帧尾错误
            #
            # 丢一个字节重新找帧头
            # ------------------------------------------------

            del self.rx_buffer[0]


    # ========================================================
    # 打印接收
    # ========================================================

    @staticmethod
    def _print_rx_frame(
        frame: SortingFrame
    ):

        try:

            mode = Mode(
                frame.mode
            )

            name = MODE_NAME.get(
                mode,
                f"0x{frame.mode:02X}"
            )

        except ValueError:

            name = (
                f"UNKNOWN_"
                f"0x{frame.mode:02X}"
            )


        print(
            "\n[RX]",
            name
        )

        print(
            "     HEX:",
            frame.raw.hex(" ").upper()
        )

        print(
            f"     X="
            f"{frame.x_mm:.1f} mm, "
            f"Y="
            f"{frame.y_mm:.1f} mm, "
            f"BOX="
            f"{frame.box_id}"
        )


    # ========================================================
    # 等待指定响应
    # ========================================================

    def wait_for(
        self,
        expected_mode: int,
        timeout: float = 2.0,
    ):

        end_time = (
            time.monotonic()
            +
            timeout
        )


        while (
            time.monotonic()
            <
            end_time
        ):

            remaining = (
                end_time
                -
                time.monotonic()
            )


            try:

                frame = self.rx_queue.get(

                    timeout=max(
                        0.01,
                        remaining
                    )
                )

            except queue.Empty:

                break


            if (
                frame.mode
                ==
                int(expected_mode)
            ):

                return frame


        return None


# ============================================================
# 心跳测试
# ============================================================

def heartbeat_test(
    uart: SortingUART
):

    print(
        "\n========== HEARTBEAT TEST =========="
    )


    uart.send_heartbeat()


    frame = uart.wait_for(

        Mode.MCU_HEARTBEAT_ACK,

        timeout=2.0,
    )


    if frame is None:

        print(
            "\n[FAIL] "
            "没有收到 STM32 HEARTBEAT_ACK"
        )

        return False


    print(
        "\n[PASS] "
        "Jetson -> STM32 -> Jetson "
        "双向通信正常"
    )

    return True


# ============================================================
# 完整握手测试
# ============================================================

def full_handshake_test(
    uart: SortingUART
):

    print(
        "\n======================================"
    )

    print(
        "FULL SORTING HANDSHAKE TEST"
    )

    print(
        "======================================"
    )


    print(
        "\n等待 STM32 发送 MCU_START_WORK (0x87)"
    )

    print(
        "请按下 STM32 启动按键，"
        "或让测试程序主动 Request_Start_Work()"
    )


    start_frame = uart.wait_for(

        Mode.MCU_START_WORK,

        timeout=30.0,
    )


    if start_frame is None:

        print(
            "[FAIL] 30秒内没有收到 START_WORK"
        )

        return


    print(
        "\n[OK] 收到 START_WORK"
    )


    # --------------------------------------------------------
    # 模拟顶部相机
    # --------------------------------------------------------

    x_mm = float(
        input(
            "\n输入测试 X(mm)，例如 50: "
        )
    )

    y_mm = float(
        input(
            "输入测试 Y(mm)，例如 80: "
        )
    )


    uart.send_pick_target(
        x_mm,
        y_mm
    )


    pick_ack = uart.wait_for(

        Mode.MCU_PICK_TARGET_ACK,

        timeout=2.0,
    )


    if pick_ack is None:

        print(
            "[FAIL] 没收到 PICK_TARGET_ACK"
        )

        return


    print(
        "[OK] STM32已经收到X/Y"
    )


    # --------------------------------------------------------
    # 等机械第一阶段完成
    # --------------------------------------------------------

    print(
        "\n等待 RECOGNITION_READY (0x82)..."
    )


    ready = uart.wait_for(

        Mode.MCU_RECOGNITION_READY,

        timeout=30.0,
    )


    if ready is None:

        print(
            "[FAIL] 没收到 RECOGNITION_READY"
        )

        return


    print(
        "[OK] STM32通知可以进行侧面识别"
    )


    # --------------------------------------------------------
    # 模拟侧面识别
    # --------------------------------------------------------

    box_id = int(
        input(
            "\n输入测试储物盒编号 1~6: "
        )
    )


    uart.send_sort_result(
        box_id
    )


    sort_ack = uart.wait_for(

        Mode.MCU_SORT_RESULT_ACK,

        timeout=2.0,
    )


    if sort_ack is None:

        print(
            "[FAIL] 没收到 SORT_RESULT_ACK"
        )

        return


    print(
        "[OK] STM32已经收到Box_ID"
    )


    # --------------------------------------------------------
    # 等最终投放完成
    # --------------------------------------------------------

    print(
        "\n等待 SORT_DONE (0x84)..."
    )


    done = uart.wait_for(

        Mode.MCU_SORT_DONE,

        timeout=30.0,
    )


    if done is None:

        print(
            "[FAIL] 没收到 SORT_DONE"
        )

        return


    print(
        "\n======================================"
    )

    print(
        "[PASS] 一整轮上下位机握手测试完成"
    )

    print(
        "======================================"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--port",

        required=True,

        help=(
            "串口，例如 "
            "/dev/ttyUSB0 "
            "或 COM5"
        ),
    )


    parser.add_argument(

        "--baud",

        type=int,

        default=115200,
    )


    args = parser.parse_args()


    uart = SortingUART(

        port=args.port,

        baudrate=args.baud,
    )


    try:

        while True:

            print(
                "\n"
                "===============================\n"
                "1. 心跳测试\n"
                "2. 发送 PICK_TARGET\n"
                "3. 发送 SORT_RESULT\n"
                "4. 发送 STOP\n"
                "5. 完整握手测试\n"
                "q. 退出\n"
                "==============================="
            )


            cmd = input(
                "请选择: "
            ).strip().lower()


            if cmd == "1":

                heartbeat_test(
                    uart
                )


            elif cmd == "2":

                x = float(
                    input(
                        "X(mm): "
                    )
                )

                y = float(
                    input(
                        "Y(mm): "
                    )
                )


                uart.send_pick_target(
                    x,
                    y
                )


            elif cmd == "3":

                box = int(
                    input(
                        "Box ID(1~6): "
                    )
                )


                uart.send_sort_result(
                    box
                )


            elif cmd == "4":

                uart.send_stop()


            elif cmd == "5":

                full_handshake_test(
                    uart
                )


            elif cmd == "q":

                break


            else:

                print(
                    "无效输入"
                )


    except KeyboardInterrupt:

        pass


    finally:

        uart.close()


if __name__ == "__main__":

    main()