"""心跳线程回归：主循环被长推理阻塞时，心跳仍要按周期发出。

背景：顶部单帧 `process()` 在物体多/开 grasp_search 时可以到 2 s 以上。
心跳原来只在 `run_once()` 里发，这段时间发不出去；MCU 侧连续 3 s 没有
任何帧就会 `Request_Stop_Work()`，上位机收到 `MCU_STOP_WORK` 整场停下。
"""
from __future__ import annotations

import threading
import time
import unittest

from communication.sorting_protocol import Mode
from runtime.sorting_runtime import RuntimeState, SortingRuntime


class FakeSTM:
    def __init__(self) -> None:
        self.is_online = True
        self.frames = []

    def send_frame(self, frame):
        self.frames.append(frame)
        return b""


def make_runtime(*, link_offline: bool, state: RuntimeState) -> SortingRuntime:
    rt = SortingRuntime.__new__(SortingRuntime)
    rt._link_offline = link_offline
    rt.state = state
    rt.stm32 = FakeSTM()
    rt._last_heartbeat_sent_at = 0.0
    rt.heartbeat_interval_s = 0.2
    rt._heartbeat_stop_event = threading.Event()
    rt._heartbeat_thread = None
    return rt


class HeartbeatThreadTests(unittest.TestCase):
    def test_heartbeat_sent_while_main_thread_is_busy(self):
        rt = make_runtime(link_offline=False, state=RuntimeState.WAIT_TOP_OBJECT)
        rt._start_heartbeat_thread()
        try:
            # 模拟主循环卡在一次 1.1s 的顶部推理里（不调用 run_once）。
            time.sleep(1.1)
        finally:
            rt._stop_heartbeat_thread()
        sent = [frame for frame in rt.stm32.frames if frame.mode == Mode.HEARTBEAT]
        self.assertGreaterEqual(len(sent), 4)

    def test_no_heartbeat_while_link_offline(self):
        rt = make_runtime(link_offline=True, state=RuntimeState.LINK_OFFLINE)
        rt.heartbeat_interval_s = 0.05
        rt._start_heartbeat_thread()
        try:
            time.sleep(0.3)
        finally:
            rt._stop_heartbeat_thread()
        self.assertEqual(rt.stm32.frames, [])

    def test_thread_stops_cleanly(self):
        rt = make_runtime(link_offline=False, state=RuntimeState.IDLE)
        rt._start_heartbeat_thread()
        time.sleep(0.25)
        rt._stop_heartbeat_thread()
        count = len(rt.stm32.frames)
        time.sleep(0.3)
        self.assertEqual(len(rt.stm32.frames), count)


if __name__ == "__main__":
    unittest.main()
