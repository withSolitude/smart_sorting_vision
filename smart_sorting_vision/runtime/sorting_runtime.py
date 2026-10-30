"""Sorting Runtime

整机正式比赛运行流程。

关键约束：
- PICK_TARGET 发出后必须先等 PICK_ACK，再等 RECOGNITION_READY。
- 只有 READY 后才开始新的侧面识别采样窗口。
- SORT_RESULT 发出后必须先等 SORT_ACK，再等匹配箱号的 SORT_DONE。
- 视觉缓存按周期显式 reset。
- 库存采用 reserve -> commit/cancel，DONE 前不计数。
- MCU_ERROR 仍作为真实下位机故障记录；动作等待超时只触发重发，
  心跳/串口掉线进入可恢复的 LINK_OFFLINE。
- 侧面识别超时不再直接 ERROR：有有效候选时按整个识别时段的多数票结果继续分拣。
- 侧面形状分类确认 no_object 时，不 STOP；走 RETRY_PICK 恢复链后自动重新顶部夹取。
- PICK/SORT 命令周期重发，利用下位机幂等握手恢复 ACK/READY/DONE 丢包。
- 顶部连续无物体且已达到最小件数后自动结束比赛。
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum

from communication.sorting_protocol import SortingFrame, SortingProtocol, Mode, make_heartbeat
from communication.stm32_link import LinkOfflineError
from vision.storage.object_info import ObjectInfo


class RuntimeState(Enum):
    IDLE = 0
    WAIT_TOP_OBJECT = 1
    # 动态预开口参数先单独握手，ACK后才允许真正下发PICK_TARGET。
    WAIT_PICK_PREOPEN_ACK = 2
    WAIT_PICK_ACK = 3
    # 兼容旧代码命名；现在语义明确为“等待 ACK”，不是“抓取完成”。
    WAIT_PICK_DONE = 3
    WAIT_RECOGNITION_READY = 4
    WAIT_SIDE_OBJECT = 5
    WAIT_SORT_ACK = 6
    WAIT_SORT_DONE = 7
    # 侧面识别台确认空载后，等待下位机将夹爪安全回到 Disable。
    WAIT_RETRY_PICK_READY = 8
    ERROR = 9
    # 串口暂时断开。该状态只暂停当前动作，不结束主程序或比赛会话。
    LINK_OFFLINE = 11
    # 串口已恢复，等待下位机确认 STOP 安全态后再继续分拣。
    LINK_RECOVERY = 12


@dataclass(frozen=True)
class _LinkOfflineEvent:
    """接收线程投递的链路掉线事件。"""

    reason: str


@dataclass(frozen=True)
class _LinkOnlineEvent:
    """接收线程投递的链路恢复事件。"""

    generation: int


class SortingRuntime:
    def __init__(
        self,
        top_camera,
        side_camera,
        top_pipeline,
        side_pipeline,
        storage_manager,
        stm32,
        *,
        pick_ack_timeout_s: float = 1.5,
        recognition_ready_timeout_s: float = 18.0,
        side_recognition_timeout_s: float = 4.0,
        retry_pick_ready_timeout_s: float = 4.0,
        sort_ack_timeout_s: float = 1.5,
        sort_done_timeout_s: float = 16.0,
        top_no_frame_timeout_s: float = 2.0,
        side_no_frame_timeout_s: float = 2.0,
        pick_x_limits_mm: tuple[float, float] | list[float] | None = None,
        pick_y_limits_mm: tuple[float, float] | list[float] | None = None,
        descent_drift_compensator=None,
        yaw_servo_calibration=None,
        retry_interval_s: float = 0.25,
        link_reconnect_interval_s: float = 1.0,
        heartbeat_interval_s: float = 0.5,
        link_dead_timeout_s: float = 3.0,
        minimum_items_before_empty_finish: int = 12,
        empty_finish_hold_s: float = 1.5,
        max_items: int = 0,
        reject_unknown: bool = True,
        on_finished=None,
        on_session_start=None,
        loop_sleep_idle_s: float = 0.010,
        loop_sleep_wait_s: float = 0.002,
        loop_sleep_vision_s: float = 0.001,
    ):
        self.top_camera = top_camera
        self.side_camera = side_camera
        self.top_pipeline = top_pipeline
        self.side_pipeline = side_pipeline
        self.storage_manager = storage_manager
        self.stm32 = stm32
        self.descent_drift_compensator = descent_drift_compensator
        self.yaw_servo_calibration = yaw_servo_calibration

        self.retry_interval_s = float(retry_interval_s)
        self.link_reconnect_interval_s = max(0.2, float(link_reconnect_interval_s))
        self.heartbeat_interval_s = max(0.1, float(heartbeat_interval_s))
        self.link_dead_timeout_s = max(1.0, float(link_dead_timeout_s))
        self.minimum_items_before_empty_finish = max(0, int(minimum_items_before_empty_finish))
        self.empty_finish_hold_s = float(empty_finish_hold_s)
        self.max_items = max(0, int(max_items))
        self.reject_unknown = bool(reject_unknown)
        self.on_finished = on_finished
        self.on_session_start = on_session_start

        # 状态感知的主循环调度：
        # - IDLE 不需要高频轮询，保持低 CPU；
        # - ACK/READY/DONE 等待阶段提高串口事件响应速度；
        # - 视觉阶段主要由相机/推理本身节拍限制，只保留极短让步避免空帧忙等。
        self.loop_sleep_idle_s = max(0.0, float(loop_sleep_idle_s))
        self.loop_sleep_wait_s = max(0.0, float(loop_sleep_wait_s))
        self.loop_sleep_vision_s = max(0.0, float(loop_sleep_vision_s))

        self.pick_ack_timeout_s = float(pick_ack_timeout_s)
        self.recognition_ready_timeout_s = float(recognition_ready_timeout_s)
        self.side_recognition_timeout_s = float(side_recognition_timeout_s)
        self.retry_pick_ready_timeout_s = float(retry_pick_ready_timeout_s)
        self.sort_ack_timeout_s = float(sort_ack_timeout_s)
        self.sort_done_timeout_s = float(sort_done_timeout_s)
        self.top_no_frame_timeout_s = float(top_no_frame_timeout_s)
        self.side_no_frame_timeout_s = float(side_no_frame_timeout_s)

        self.pick_x_limits_mm = self._normalize_limits(
            pick_x_limits_mm, name="pick_x_limits_mm"
        )
        self.pick_y_limits_mm = self._normalize_limits(
            pick_y_limits_mm, name="pick_y_limits_mm"
        )

        for name, value in (
            ("pick_ack_timeout_s", self.pick_ack_timeout_s),
            ("recognition_ready_timeout_s", self.recognition_ready_timeout_s),
            ("side_recognition_timeout_s", self.side_recognition_timeout_s),
            ("retry_pick_ready_timeout_s", self.retry_pick_ready_timeout_s),
            ("sort_ack_timeout_s", self.sort_ack_timeout_s),
            ("sort_done_timeout_s", self.sort_done_timeout_s),
            ("top_no_frame_timeout_s", self.top_no_frame_timeout_s),
            ("side_no_frame_timeout_s", self.side_no_frame_timeout_s),
            ("retry_interval_s", self.retry_interval_s),
            ("link_reconnect_interval_s", self.link_reconnect_interval_s),
            ("heartbeat_interval_s", self.heartbeat_interval_s),
            ("link_dead_timeout_s", self.link_dead_timeout_s),
            ("empty_finish_hold_s", self.empty_finish_hold_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} 必须 > 0")

        self.state = RuntimeState.IDLE
        self.state_entered_at = time.monotonic()
        # 相机 watchdog 使用“最近一次成功取得图像”的单调时钟。进入对应
        # 视觉状态时会重置为状态进入时刻，避免沿用上一周期的旧时间戳。
        self._top_last_frame_at = self.state_entered_at
        self._side_last_frame_at = self.state_entered_at
        self.running = False
        self.cycle_id = 0
        self.fault_reason: str | None = None
        self._link_offline = False
        self._offline_cause: str | None = None
        self._camera_offline_name: str | None = None
        self._resume_after_link_recovery = False
        self._next_reconnect_stop_at = 0.0
        self._recovery_grace_until = 0.0
        self._last_heartbeat_sent_at = 0.0
        self._last_peer_seen_at: float | None = None
        self._timeout_notice_states: set[RuntimeState] = set()

        self.current_pick = None
        # 保存真正经过协议量化后的PICK命令，用于严格核对ACK/READY回显，
        # 防止迟到/旧周期回包被误认为当前动作。
        self.current_pick_command: SortingFrame | None = None
        self.current_preopen_command: SortingFrame | None = None
        self.current_object: ObjectInfo | None = None
        self.current_box: int | None = None
        self.current_reservation_id: int | None = None
        self.current_sort_command: SortingFrame | None = None
        self.current_retry_pick_command: SortingFrame | None = None
        self.current_side_image = None

        self.completed_count = 0
        # 本场比赛内“夹取后识别台为空”的自动恢复次数，仅用于日志，不计成功件数。
        self.drop_recovery_count = 0
        self._empty_scene_since: float | None = None

        # 侧面识别超时兜底统计。正常情况下仍由 SideRecognitionConsensus
        # 按 4/5（配置值）立即确认；只有超过 side_recognition_timeout_s
        # 仍未 READY 时，才从“整个本次侧面识别阶段”累计的有效完整结果中
        # 选择出现次数最多的 (color, qr_present, shape) 继续分拣。
        self._side_vote_stats: dict[tuple[str, bool, str], dict] = {}
        self._side_vote_seq = 0
        self._side_frames_seen = 0
        self._side_object_frames_seen = 0

        self._last_command_send_at = 0.0
        self._session_started_at: float | None = None
        self._shutdown_event = threading.Event()

        # 串口接收与视觉计算解耦。接收线程只负责 poll + 入队，
        # 正式状态变更仍由主 Runtime 线程按帧顺序执行。
        self._rx_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._rx_stop_event = threading.Event()
        self._urgent_stop_event = threading.Event()
        self._rx_thread: threading.Thread | None = None

    @staticmethod
    def _normalize_limits(value, *, name: str):
        if value is None:
            return None
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"{name} 必须是 [min, max]")
        lo, hi = map(float, value)
        if lo > hi:
            raise ValueError(f"{name} 必须满足 min <= max")
        return (lo, hi)

    def _pick_xy_inside_limits(self, x_mm: float, y_mm: float) -> bool:
        if self.pick_x_limits_mm is not None:
            x_min, x_max = self.pick_x_limits_mm
            if not (x_min <= x_mm <= x_max):
                return False
        if self.pick_y_limits_mm is not None:
            y_min, y_max = self.pick_y_limits_mm
            if not (y_min <= y_mm <= y_max):
                return False
        return True

    # =====================================================
    # 串口接收线程
    # =====================================================

    def _rx_loop(self) -> None:
        while not self._rx_stop_event.is_set():
            try:
                frames = self.stm32.poll()
            except LinkOfflineError as exc:
                # 串口设备拔出、USB-UART 芯片复位或 MCU 掉电都走同一条
                # 恢复路径：通知主线程后在后台周期重连，绝不结束 Runtime。
                self._rx_queue.put(_LinkOfflineEvent(str(exc)))
                while not self._rx_stop_event.is_set():
                    if self.stm32.reconnect_once():
                        self._rx_queue.put(
                            _LinkOnlineEvent(self.stm32.connection_generation)
                        )
                        break
                    self._rx_stop_event.wait(self.link_reconnect_interval_s)
                continue
            except Exception as exc:
                # 未知的串口读异常也按链路掉线处理；业务线程仍然存活，
                # 后续重连成功后可以继续比赛。
                self._rx_queue.put(_LinkOfflineEvent(str(exc)))
                while not self._rx_stop_event.is_set():
                    if self.stm32.reconnect_once():
                        self._rx_queue.put(
                            _LinkOnlineEvent(self.stm32.connection_generation)
                        )
                        break
                    self._rx_stop_event.wait(self.link_reconnect_interval_s)
                continue

            for frame in frames:
                if frame.mode in (Mode.MCU_ERROR, Mode.MCU_STOP_WORK):
                    self._urgent_stop_event.set()
                self._rx_queue.put(frame)

            time.sleep(0.002)

    def _start_rx_thread(self) -> None:
        if self._rx_thread is not None and self._rx_thread.is_alive():
            return

        self._rx_stop_event.clear()
        self._rx_thread = threading.Thread(
            target=self._rx_loop,
            name="stm32-rx",
            daemon=True,
        )
        self._rx_thread.start()

    def _stop_rx_thread(self) -> None:
        self._rx_stop_event.set()
        thread = self._rx_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        self._rx_thread = None

    def _poll_control_events(self):
        # 单元测试/独立 run_once 调用时没有接收线程，保持原来的同步 poll。
        if self._rx_thread is None or not self._rx_thread.is_alive():
            try:
                return self.stm32.poll()
            except (LinkOfflineError, OSError) as exc:
                self._mark_link_offline(str(exc))
                return []

        frames = []
        while True:
            try:
                item = self._rx_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(item, _LinkOfflineEvent):
                self._mark_link_offline(item.reason)
                continue
            if isinstance(item, _LinkOnlineEvent):
                self._mark_link_online(item.generation)
                continue
            if isinstance(item, Exception):
                self._mark_link_offline(str(item))
                continue
            frames.append(item)

        return frames

    def _mark_link_offline(self, reason: str) -> None:
        """进入可恢复的离线状态，并清理当前未完成事务。"""
        if self._link_offline:
            # 相机已经离线时，串口也可能随后掉线。保留相机故障来源，
            # 等两类设备都恢复后再进入新的分拣周期。
            if self._offline_cause == "camera":
                self._offline_cause = "stm32"
            return

        was_recovering = self.state in (
            RuntimeState.LINK_OFFLINE,
            RuntimeState.LINK_RECOVERY,
        )
        self._link_offline = True
        self._offline_cause = "stm32"
        self._camera_offline_name = None
        if not was_recovering:
            self._resume_after_link_recovery = bool(self.running)
        self.running = False
        self.fault_reason = None
        self._urgent_stop_event.clear()
        self._clear_cycle_data(cancel_reservation=True)
        self._safe_reset(self.top_pipeline)
        self._safe_reset(self.side_pipeline)
        self._enter_state(RuntimeState.LINK_OFFLINE)
        print(f"[LINK][OFFLINE] {reason}; keep process alive and retry reconnect")

    def _mark_link_online(self, generation: int) -> None:
        """串口恢复后先让下位机回到安全 STOP，再恢复新周期。"""
        if not self._link_offline:
            return

        camera_name = self._camera_offline_name
        if camera_name is not None:
            camera = self.top_camera if camera_name == "top" else self.side_camera
            if not bool(getattr(camera, "is_online", False)):
                # 串口恢复了，但相机仍未恢复；继续保持离线等待。
                self._offline_cause = "camera"
                return

        self._link_offline = False
        self._offline_cause = None
        self._camera_offline_name = None
        self._last_peer_seen_at = time.monotonic()
        self._next_reconnect_stop_at = 0.0
        # 下位机链路恢复后会在机构安全回位时主动发 MCU_START_WORK。
        # 先留出短暂窗口接收该通知，避免刚重连就发送 STOP 把自动恢复打断。
        self._recovery_grace_until = time.monotonic() + 2.0
        self._enter_state(RuntimeState.LINK_RECOVERY)
        print(
            f"[LINK][ONLINE] generation={generation}; "
            "synchronize STOP before resuming sorting"
        )

    def _service_link_recovery(self) -> None:
        """非阻塞地重发 STOP，等待 MCU_STOP_ACK。"""
        if self.state == RuntimeState.LINK_OFFLINE and self._offline_cause == "camera":
            camera = (
                self.top_camera
                if self._camera_offline_name == "top"
                else self.side_camera
            )
            # 相机驱动的 reconnect_once() 内部负责节流重连；离线暂停期间
            # 仍需调用它，否则驱动没有机会重新打开已经恢复的 USB 设备。
            try:
                reconnect = getattr(camera, "reconnect_once", None)
                if callable(reconnect):
                    reconnect()
                else:
                    # 兼容旧相机驱动：read() 失败时只返回 None，不进入业务。
                    camera.read()
            except Exception as exc:
                print(f"[CAMERA][RECOVERY] read failed: {exc}")
            if bool(getattr(camera, "is_online", False)):
                self._link_offline = False
                self._offline_cause = None
                self._camera_offline_name = None
                self._next_reconnect_stop_at = 0.0
                self._recovery_grace_until = time.monotonic() + 2.0
                self._enter_state(RuntimeState.LINK_RECOVERY)
                print("[CAMERA][ONLINE] camera recovered; synchronize MCU STOP")
            else:
                return

        if self.state != RuntimeState.LINK_RECOVERY:
            return
        if not bool(getattr(self.stm32, "is_online", True)):
            self._mark_link_offline("serial link lost during recovery")
            return
        now = time.monotonic()
        if now < self._recovery_grace_until:
            return
        if now < self._next_reconnect_stop_at:
            return
        try:
            self.stm32.send_frame(SortingFrame(mode=Mode.STOP))
            print("[LINK][RECOVERY] STOP sent")
            self._next_reconnect_stop_at = now + self.retry_interval_s
        except (LinkOfflineError, OSError) as exc:
            self._mark_link_offline(str(exc))

    def _send_heartbeat_if_due(self) -> None:
        """周期发送心跳，用于区分“串口仍开着”和“MCU真正在线”。"""
        if self._link_offline or self.state == RuntimeState.LINK_OFFLINE:
            return
        if not bool(getattr(self.stm32, "is_online", True)):
            self._mark_link_offline("serial link is not open")
            return
        now = time.monotonic()
        if now - self._last_heartbeat_sent_at < self.heartbeat_interval_s:
            return
        try:
            self.stm32.send_frame(make_heartbeat())
            self._last_heartbeat_sent_at = now
        except (LinkOfflineError, OSError) as exc:
            self._mark_link_offline(str(exc))

    def _check_link_watchdog(self) -> None:
        """检测 MCU 无回包的掉线情况，避免把它误报为动作超时。"""
        if self._link_offline or self.state == RuntimeState.LINK_OFFLINE:
            return
        if self._last_peer_seen_at is None:
            return
        elapsed = time.monotonic() - self._last_peer_seen_at
        if elapsed > self.link_dead_timeout_s:
            self._mark_link_offline(
                f"no valid MCU frame for {elapsed:.2f}s "
                f"> {self.link_dead_timeout_s:.2f}s"
            )

    def _action_still_allowed(self, expected_state: RuntimeState) -> bool:
        return (
            self.running
            and self.state == expected_state
            and not self._urgent_stop_event.is_set()
        )

    # =====================================================
    # 状态 / 周期基础操作
    # =====================================================

    def _enter_state(self, state: RuntimeState) -> None:
        self.state = state
        self.state_entered_at = time.monotonic()
        self._timeout_notice_states.discard(state)
        if state == RuntimeState.WAIT_TOP_OBJECT:
            self._top_last_frame_at = self.state_entered_at
        elif state == RuntimeState.WAIT_SIDE_OBJECT:
            self._side_last_frame_at = self.state_entered_at
        print(f"[Runtime] state -> {state.name} cycle={self.cycle_id}")

    @staticmethod
    def _safe_reset(pipeline) -> None:
        reset = getattr(pipeline, "reset", None)
        if callable(reset):
            reset()

    def _reset_side_vote_stats(self) -> None:
        """清空本次侧面识别阶段的超时多数票统计。"""
        self._side_vote_stats.clear()
        self._side_vote_seq = 0
        self._side_frames_seen = 0
        self._side_object_frames_seen = 0

    @staticmethod
    def _side_vote_key(result):
        """
        提取可用于分箱的完整侧面识别结果。

        unknown / 空颜色 / 空形状不参与超时兜底投票，避免为了不中断而
        把一个本身不可分箱的结果强行送给 StorageManager。
        """
        color = str(getattr(result, "color", "unknown")).strip()
        shape = str(getattr(result, "shape", "unknown")).strip()
        if color.lower() in ("", "unknown", "none"):
            return None
        if shape.lower() in ("", "unknown", "none"):
            return None
        qr_present = bool(getattr(result, "qr_present", False))
        key = [color, qr_present, shape]

        # SideSortPipeline 仅在 YAML 开启二维码内容共识时提供 qr_content。
        # 这样超时多数票不会绕过“内容必须一致”的设置。
        qr_content = getattr(result, "qr_content", None)
        if qr_present and qr_content is not None:
            qr_content = str(qr_content)
            if not qr_content:
                return None
            key.append(qr_content)

        return tuple(key)

    def _record_side_vote(self, result) -> None:
        """累计整个 WAIT_SIDE_OBJECT 阶段的完整结果票数。"""
        self._side_object_frames_seen += 1
        key = self._side_vote_key(result)
        if key is None:
            return

        self._side_vote_seq += 1
        confidence = float(getattr(result, "color_confidence", 0.0)) + float(
            getattr(result, "shape_confidence", 0.0)
        )
        stats = self._side_vote_stats.get(key)
        if stats is None:
            stats = {
                "count": 0,
                "confidence_sum": 0.0,
                "last_seq": 0,
                "result": None,
            }
            self._side_vote_stats[key] = stats

        stats["count"] += 1
        stats["confidence_sum"] += confidence
        stats["last_seq"] = self._side_vote_seq
        stats["result"] = result

    def _best_side_timeout_result(self):
        """
        返回超时兜底的多数票候选。

        首要判据是出现帧数；仅在票数相同的情况下，才依次用平均置信度
        和最近一次出现顺序打破平票。
        """
        if not self._side_vote_stats:
            return None, None

        def rank(item):
            _key, stats = item
            count = int(stats["count"])
            mean_conf = float(stats["confidence_sum"]) / max(count, 1)
            return (count, mean_conf, int(stats["last_seq"]))

        key, stats = max(self._side_vote_stats.items(), key=rank)
        return key, stats

    def _clear_cycle_data(self, *, cancel_reservation: bool) -> None:
        if cancel_reservation and self.current_reservation_id is not None:
            self.storage_manager.cancel(self.current_reservation_id)

        self.current_pick = None
        self.current_pick_command = None
        self.current_preopen_command = None
        self.current_object = None
        self.current_box = None
        self.current_reservation_id = None
        self.current_sort_command = None
        self.current_retry_pick_command = None
        self.current_side_image = None
        self._reset_side_vote_stats()
        self._last_command_send_at = 0.0

    def _begin_cycle(self) -> None:
        self._clear_cycle_data(cancel_reservation=True)
        self._safe_reset(self.top_pipeline)
        self._safe_reset(self.side_pipeline)
        self._empty_scene_since = None
        self.cycle_id += 1
        self._enter_state(RuntimeState.WAIT_TOP_OBJECT)

    def _send_command(self, frame: SortingFrame, *, label: str, retry: bool = False) -> bool:
        try:
            self.stm32.send_frame(frame)
        except (LinkOfflineError, OSError) as exc:
            # 发送失败属于链路掉线，保留主循环并转入自动恢复状态。
            self._mark_link_offline(f"{label} send failed: {exc}")
            return False
        except Exception as exc:
            self._fail(f"{label} send failed: {exc}")
            return False
        self._last_command_send_at = time.monotonic()
        prefix = "[RETRY]" if retry else "[TX]"
        print(prefix, self.stm32.describe(frame))
        return True

    def _send_pick_target(self) -> bool:
        """预开口确认后直接发送抓取目标，不再发送分支选择帧。"""
        if self.current_pick_command is None:
            self._fail("PICK_TARGET command missing")
            return False
        if not self._send_command(self.current_pick_command, label="PICK_TARGET", retry=False):
            return False
        self._enter_state(RuntimeState.WAIT_PICK_ACK)
        return True

    def _retry_pending_command(self) -> None:
        if not self.running:
            return
        now = time.monotonic()
        if (now - self._last_command_send_at) < self.retry_interval_s:
            return

        frame = None
        label = ""
        if self.state == RuntimeState.WAIT_PICK_PREOPEN_ACK:
            frame = self.current_preopen_command
            label = "PICK_PREOPEN retry"
        elif self.state in (RuntimeState.WAIT_PICK_ACK, RuntimeState.WAIT_RECOGNITION_READY):
            frame = self.current_pick_command
            label = "PICK_TARGET retry"
        elif self.state in (RuntimeState.WAIT_SORT_ACK, RuntimeState.WAIT_SORT_DONE):
            frame = self.current_sort_command
            label = "SORT_RESULT retry"
        elif self.state == RuntimeState.WAIT_RETRY_PICK_READY:
            frame = self.current_retry_pick_command
            label = "RETRY_PICK retry"

        if frame is not None:
            self._send_command(frame, label=label, retry=True)

    def _request_retry_pick(self, *, reason: str) -> None:
        """放弃本次未成功送达识别台的夹取，但保持比赛运行。

        注意：不能直接回 WAIT_TOP_OBJECT。下位机此时还停在 Recognition
        姿态，必须先通过 RETRY_PICK 让机构安全 Return -> Disable；只有收到
        MCU_RETRY_PICK_READY 后才重新打开顶部识别。
        """
        if not self._action_still_allowed(RuntimeState.WAIT_SIDE_OBJECT):
            return

        # 理论上 WAIT_SIDE_OBJECT 尚未 reserve；这里仍做防御性取消，
        # 保证恢复路径不会留下“软件占箱但机械没投放”的脏事务。
        if self.current_reservation_id is not None:
            self.storage_manager.cancel(self.current_reservation_id)
            self.current_reservation_id = None

        self.current_object = None
        self.current_box = None
        self.current_sort_command = None
        self.current_side_image = None
        self._safe_reset(self.side_pipeline)

        self.current_retry_pick_command = SortingFrame(mode=Mode.RETRY_PICK)
        self.drop_recovery_count += 1
        print(
            f"[RECOVERY] side tray empty -> RETRY_PICK "
            f"count={self.drop_recovery_count} reason={reason}"
        )

        if not self._send_command(
            self.current_retry_pick_command,
            label="RETRY_PICK",
            retry=False,
        ):
            return

        self._enter_state(RuntimeState.WAIT_RETRY_PICK_READY)

    def _finish_competition(self, reason: str) -> None:
        print(
            f"[COMPETITION] FINISH reason={reason} "
            f"completed={self.completed_count}"
        )
        self.running = False
        self._clear_cycle_data(cancel_reservation=True)
        self._safe_reset(self.top_pipeline)
        self._safe_reset(self.side_pipeline)
        self._enter_state(RuntimeState.IDLE)
        self._send_stop_best_effort()

    def _send_stop_best_effort(self) -> None:
        if self.stm32 is None:
            return
        try:
            self.stm32.send_frame(SortingFrame(mode=Mode.STOP))
        except (LinkOfflineError, OSError) as exc:
            self._mark_link_offline(f"STOP send failed: {exc}")
        except Exception as exc:
            print(f"[Runtime] STOP send failed: {exc}")

    def _fail(self, reason: str) -> None:
        if self.state == RuntimeState.ERROR and self.fault_reason == reason:
            return

        self.running = False
        self.fault_reason = str(reason)
        self._clear_cycle_data(cancel_reservation=True)
        self._safe_reset(self.top_pipeline)
        self._safe_reset(self.side_pipeline)
        self._enter_state(RuntimeState.ERROR)
        self._send_stop_best_effort()
        print(f"[Runtime][ERROR] {self.fault_reason}")

    # =====================================================
    # Jetson <-> STM32 上电同步
    #
    # 在正式 WAIT START 之前由 main.initialize() 同步调用。
    # 语义：
    # 1) Jetson 主动周期发送 STOP；
    # 2) STM32 收到后先把所有机械机构安全回到 Disable；
    # 3) 只有真正达到“可启动安全态”后才回 MCU_STOP_ACK；
    # 4) 同步期间收到的旧 START/STOP_WORK/ACK 等帧全部丢弃，
    #    防止上电残留帧导致一进入 Runtime 就误启动。
    #
    # 注意：这一版本要求下位机同步补丁将 MCU_STOP_ACK 改为
    # “安全回位完成 ACK”，而不是“刚收到 STOP 就立刻 ACK”。
    # =====================================================

    def synchronize_startup(
        self,
        *,
        timeout_s: float = 20.0,
        retry_interval_s: float = 0.40,
        poll_sleep_s: float = 0.010,
    ) -> None:
        if self.stm32 is None:
            raise RuntimeError("startup sync requires STM32 link")

        if self._rx_thread is not None and self._rx_thread.is_alive():
            raise RuntimeError("startup sync must run before RX thread starts")

        timeout_s = max(1.0, float(timeout_s))
        retry_interval_s = max(0.10, float(retry_interval_s))
        poll_sleep_s = max(0.001, float(poll_sleep_s))

        self.running = False
        self.fault_reason = None
        self._urgent_stop_event.clear()
        self._clear_cycle_data(cancel_reservation=True)
        self._safe_reset(self.top_pipeline)
        self._safe_reset(self.side_pipeline)
        self._enter_state(RuntimeState.IDLE)

        print(
            "[SYNC] MCU startup synchronization: "
            "force STOP -> wait safe MCU_STOP_ACK"
        )

        # 丢弃Warmup期间可能积累的旧START/STOP_ACK/READY等帧。
        # 最终同步只接受“本次重新发送STOP之后”返回的安全ACK。
        stale_count = 0
        for _ in range(5):
            stale_frames = self.stm32.poll()
            if not stale_frames:
                break
            stale_count += len(stale_frames)
            for stale in stale_frames:
                print(f"[SYNC][DRAIN] {stale.mode.name}")
            time.sleep(0.005)

        if stale_count:
            print(f"[SYNC] drained {stale_count} stale frame(s)")

        deadline = time.monotonic() + timeout_s
        next_send_at = 0.0
        tx_count = 0

        while True:
            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError(
                    "MCU startup sync timeout: "
                    f"no safe MCU_STOP_ACK within {timeout_s:.1f}s"
                )

            if now >= next_send_at:
                self.stm32.send_frame(SortingFrame(mode=Mode.STOP))
                tx_count += 1
                if tx_count == 1:
                    print("[SYNC][TX] STOP(0x7F)")
                else:
                    print(f"[SYNC][RETRY] STOP(0x7F) #{tx_count}")
                next_send_at = now + retry_interval_s

            try:
                frames = self.stm32.poll()
            except Exception as exc:
                raise RuntimeError(
                    f"MCU startup sync serial failure: {exc}"
                ) from exc

            for frame in frames:
                mode = frame.mode

                if mode == Mode.MCU_STOP_ACK:
                    # 下位机新版只有在夹爪/XY/储物机构全部安全回位并且
                    # 达到可START条件后才发送该ACK。
                    self._urgent_stop_event.clear()
                    self._clear_cycle_data(cancel_reservation=True)
                    self._safe_reset(self.top_pipeline)
                    self._safe_reset(self.side_pipeline)
                    self._enter_state(RuntimeState.IDLE)
                    print(
                        "[SYNC] READY: MCU safe STOP confirmed; "
                        "waiting for physical START button"
                    )
                    return

                if mode == Mode.MCU_ERROR:
                    # 同步阶段的目的就是强制错误/旧动作回安全态。
                    # 继续发STOP，不让旧ERROR直接把PC Runtime带入ERROR。
                    print(
                        "[SYNC][RX] MCU_ERROR "
                        f"code={frame.box_id}; keep forcing STOP"
                    )
                    continue

                if mode == Mode.MCU_START_WORK:
                    print(
                        "[SYNC][DROP] stale/early MCU_START_WORK "
                        "ignored until synchronization completes"
                    )
                    continue

                if mode == Mode.MCU_STOP_WORK:
                    print(
                        "[SYNC][RX] MCU_STOP_WORK; "
                        "waiting for safe MCU_STOP_ACK"
                    )
                    continue

                # 其他旧ACK/BUSY/READY帧在同步边界前一律丢弃。
                print(f"[SYNC][DROP] {mode.name}")

            time.sleep(poll_sleep_s)


    # =====================================================
    # 启动 / 停止
    # =====================================================

    def start(self):
        self._urgent_stop_event.clear()
        self.running = True
        self.fault_reason = None
        self._session_started_at = time.monotonic()
        self.drop_recovery_count = 0
        self._begin_cycle()
        if callable(self.on_session_start):
            try:
                self.on_session_start(self)
            except Exception as exc:
                print(f"[Runtime][WARN] on_session_start failed: {exc}")
        print("[Runtime] START COMMAND")

    def stop(self, *, send_stop: bool = True):
        self.running = False
        self.fault_reason = None
        self._clear_cycle_data(cancel_reservation=True)
        self._safe_reset(self.top_pipeline)
        self._safe_reset(self.side_pipeline)
        self._enter_state(RuntimeState.IDLE)
        if send_stop:
            self._send_stop_best_effort()
        print("[Runtime] STOP")

    def shutdown(self, *, send_stop: bool = True) -> None:
        if self.running:
            self.stop(send_stop=send_stop)
        self._shutdown_event.set()

    # =====================================================
    # 超时 / 相机 watchdog
    # =====================================================

    def _check_camera_watchdog(self) -> None:
        """视觉状态下连续无新图像时暂停，等待相机驱动自行重连。

        这与“有画面但识别结果不稳定”是两种不同故障：
        - 相机无帧：超过 watchdog 后暂停当前周期并等待重连；
        - Side 有帧但未达共识：到 side_recognition_timeout_s 后多数票兜底。
        """
        if not self.running:
            return

        now = time.monotonic()
        if self.state == RuntimeState.WAIT_TOP_OBJECT:
            no_frame_for = now - self._top_last_frame_at
            if no_frame_for > self.top_no_frame_timeout_s:
                camera = self.top_camera
                if not bool(getattr(camera, "is_online", True)):
                    self._pause_for_camera("top", no_frame_for)
                return

        if self.state == RuntimeState.WAIT_SIDE_OBJECT:
            no_frame_for = now - self._side_last_frame_at
            if no_frame_for > self.side_no_frame_timeout_s:
                camera = self.side_camera
                if not bool(getattr(camera, "is_online", True)):
                    self._pause_for_camera("side", no_frame_for)

    def _pause_for_camera(self, name: str, no_frame_for: float) -> None:
        """相机掉线时暂停比赛，恢复后从新的顶部周期继续。"""
        if not self.running:
            return
        self._resume_after_link_recovery = True
        self._link_offline = True
        self._offline_cause = "camera"
        self._camera_offline_name = name
        self.running = False
        self._clear_cycle_data(cancel_reservation=True)
        self._safe_reset(self.top_pipeline)
        self._safe_reset(self.side_pipeline)
        self._enter_state(RuntimeState.LINK_OFFLINE)
        print(
            f"[CAMERA][OFFLINE] {name} no frame for {no_frame_for:.2f}s; "
            "pause sorting and wait for camera reconnect"
        )

    def _check_timeout(self) -> None:
        if not self.running:
            return

        # 必须先检查相机新鲜度，再做 Side 多数票兜底，避免相机冻结后
        # 拿冻结前的旧票数继续分箱。
        self._check_camera_watchdog()
        if not self.running:
            return

        self._check_link_watchdog()
        if not self.running:
            return

        limits = {
            RuntimeState.WAIT_PICK_PREOPEN_ACK: (
                self.pick_ack_timeout_s,
                "PICK_PREOPEN_ACK timeout",
            ),
            RuntimeState.WAIT_PICK_ACK: (self.pick_ack_timeout_s, "PICK_ACK timeout"),
            RuntimeState.WAIT_RECOGNITION_READY: (
                self.recognition_ready_timeout_s,
                "RECOGNITION_READY timeout",
            ),
            RuntimeState.WAIT_RETRY_PICK_READY: (
                self.retry_pick_ready_timeout_s,
                "RETRY_PICK return-to-disable timeout",
            ),
            RuntimeState.WAIT_SORT_ACK: (self.sort_ack_timeout_s, "SORT_ACK timeout"),
            RuntimeState.WAIT_SORT_DONE: (self.sort_done_timeout_s, "SORT_DONE timeout"),
        }

        # 侧面视觉超时采用“多数票兜底”，不再直接把整机置为 ERROR。
        # 其他 ACK/READY/DONE 控制链超时同样只记录提示并持续重发；
        # 真正的通信掉线由心跳和串口状态单独判定。
        if self.state == RuntimeState.WAIT_SIDE_OBJECT:
            elapsed = time.monotonic() - self.state_entered_at
            if elapsed <= self.side_recognition_timeout_s:
                return

            key, stats = self._best_side_timeout_result()
            if stats is not None and stats.get("result") is not None:
                count = int(stats["count"])
                total_valid = sum(int(x["count"]) for x in self._side_vote_stats.values())
                print(
                    "[SIDE TIMEOUT FALLBACK] "
                    f"{elapsed:.2f}s without {getattr(self.side_pipeline, 'required_hits', '?')}/"
                    f"{getattr(self.side_pipeline, 'window_size', '?')} consensus; "
                    f"use majority={key} votes={count}/{total_valid}"
                )
                self._accept_side_result(stats["result"], source="timeout_majority")
                return

            # 有相机帧且见过物体，但颜色/形状一直都是 unknown 时没有可用
            # 多数票。此时不终止比赛，也不强行错误分箱；重新给一个识别窗口。
            if self._side_frames_seen > 0 and self._side_object_frames_seen > 0:
                print(
                    "[SIDE TIMEOUT FALLBACK] no valid complete candidate yet; "
                    "continue sampling instead of stopping"
                )
                self.state_entered_at = time.monotonic()
                return

            # 完全没有取得相机帧属于相机离线/采集异常，也走可恢复的暂停，
            # 不把一次采集超时永久锁死为 ERROR。
            if self._side_frames_seen == 0:
                self._pause_for_camera("side", elapsed)
                return

            # 有帧但始终没有物体：优先走原有掉落重抓恢复链，而不是终止。
            self._request_retry_pick(
                reason=f"side recognition timeout with no object ({elapsed:.2f}s)"
            )
            return

        item = limits.get(self.state)
        if item is None:
            return

        timeout_s, reason = item
        if (time.monotonic() - self.state_entered_at) > timeout_s:
            # ACK/READY/DONE 的截止时间只用于提示和加快重发，不能把整场
            # 比赛锁死为 ERROR。若心跳也消失，链路 watchdog 会负责进入
            # LINK_OFFLINE；若心跳正常，继续等待下位机完成动作。
            if self.state not in self._timeout_notice_states:
                self._timeout_notice_states.add(self.state)
                print(f"[WAIT][RETRY] {reason}; MCU heartbeat is alive")
            self.state_entered_at = time.monotonic()

    # =====================================================
    # 主循环
    # =====================================================

    def run_once(self):
        frames = self._poll_control_events()
        for msg in frames:
            print("[RX]", self.stm32.describe(msg))
            self.process_message(msg)

        self._service_link_recovery()
        self._send_heartbeat_if_due()
        self._check_link_watchdog()

        if self.state in (RuntimeState.LINK_OFFLINE, RuntimeState.LINK_RECOVERY):
            return

        if not self.running:
            return

        self._retry_pending_command()
        if not self.running:
            return

        self._check_timeout()
        if not self.running:
            return

        if self.state == RuntimeState.WAIT_TOP_OBJECT:
            self.process_top()
        elif self.state == RuntimeState.WAIT_SIDE_OBJECT:
            self.process_side()

    # =====================================================
    # 顶部视觉
    # =====================================================

    def process_top(self):
        # 防止调试代码/回调在非顶部识别状态重复下发 PICK_TARGET。
        if not self.running or self.state != RuntimeState.WAIT_TOP_OBJECT:
            return

        image = self.top_camera.read()
        if image is None:
            return

        self._top_last_frame_at = time.monotonic()
        result = self.top_pipeline.process(image)

        # 正式比赛自动结束：只有顶部模型连续确认“0 个物体”，并且已经
        # 完成不少于配置的最小件数，才判定场上货物清空。
        if int(getattr(result, "object_count", 0)) == 0:
            now = time.monotonic()
            if self._empty_scene_since is None:
                self._empty_scene_since = now
            elif (
                self.completed_count >= self.minimum_items_before_empty_finish
                and (now - self._empty_scene_since) >= self.empty_finish_hold_s
            ):
                self._finish_competition("top_scene_empty")
                return
        else:
            self._empty_scene_since = None

        if not result.ready:
            return

        if result.x_mm is None or result.y_mm is None:
            self._fail("top pipeline returned READY without XY")
            return

        x_mm = float(result.x_mm)
        y_mm = float(result.y_mm)
        # 兼容旧测试/调试Pipeline；正式TopPickPipeline会始终给出动态yaw_deg。
        yaw_deg = float(getattr(result, "yaw_deg", 0.0))
        if not math.isfinite(yaw_deg) or not (-90.0 <= yaw_deg <= 90.0):
            print(f"[Runtime] 跳过无效抓取角度: {yaw_deg}")
            self._safe_reset(self.top_pipeline)
            return
        # 先采用协议实际发送的整度轴向角，尤其避免跨180°区间边界后TCP相反。
        yaw_deg = ((float(round(yaw_deg)) + 90.0) % 180.0) - 90.0
        try:
            if self.yaw_servo_calibration is None:
                raise ValueError("缺少 calibration.yaw_servo")
            servo_yaw_deg = self.yaw_servo_calibration.command_deg(yaw_deg)
            physical_yaw_deg = self.yaw_servo_calibration.physical_axis_deg(yaw_deg)
        except ValueError as exc:
            print(f"[Runtime] 跳过当前Yaw不可执行候选: {exc}")
            self._safe_reset(self.top_pipeline)
            return

        # 下降横向漂移补偿只在真正执行 PICK_TARGET 前应用。
        # TopPickPipeline/空抓对心保持“下降前真实TCP应在物体中心”的几何语义，
        # 避免把下降机构误差污染到H、Yaw-TCP和视觉抓取点标定中。
        if self.descent_drift_compensator is not None:
            try:
                drift_comp = self.descent_drift_compensator.apply(x_mm, y_mm, physical_yaw_deg)
            except Exception as exc:
                self._fail(f"descent drift compensation failed: {exc}")
                return
            print(
                "[Runtime] descent precomp "
                f"predicted_drift=({drift_comp.predicted_drift_x_mm:+.3f},"
                f"{drift_comp.predicted_drift_y_mm:+.3f}) mm, "
                f"command_delta=({drift_comp.command_delta_x_mm:+.3f},"
                f"{drift_comp.command_delta_y_mm:+.3f}) mm"
            )
            x_mm = float(drift_comp.x_mm)
            y_mm = float(drift_comp.y_mm)

        # R06 defense-in-depth：Pipeline 已过滤一次，真正发送机械动作前
        # Runtime 再按正式机械工作区复核。注意这里检查的是加入下降预补偿后的
        # 最终PICK命令，避免为了补偿把机构推到工作区之外。
        if not self._pick_xy_inside_limits(x_mm, y_mm):
            self._fail(
                "PICK_TARGET outside workspace: "
                f"x={x_mm:.3f}, y={y_mm:.3f}, "
                f"x_limits={self.pick_x_limits_mm}, "
                f"y_limits={self.pick_y_limits_mm}"
            )
            return

        # 推理期间可能已经收到 MCU_ERROR / MCU_STOP_WORK。
        # 接收线程会立即置位 urgent 标志；此处发送动作前必须重查。
        if not self._action_still_allowed(RuntimeState.WAIT_TOP_OBJECT):
            return

        self.current_pick = result
        tx = SortingFrame(
            mode=Mode.PICK_TARGET,
            x_mm=x_mm,
            y_mm=y_mm,
            box_id=0,
            yaw_deg=yaw_deg,
        )

        # MCU只能看到协议量化后的0.1mm / 1deg数值。保存同样的线值，
        # 后续ACK/READY必须与当前周期严格匹配。
        try:
            self.current_pick_command = SortingProtocol.unpack(SortingProtocol.pack(tx))
        except Exception as exc:
            self._fail(f"PICK_TARGET pack failed: {exc}")
            return

        print(
            "[Runtime] single-yaw selected "
            f"axis={self.current_pick_command.yaw_deg:+.1f}deg "
            f"physical={physical_yaw_deg:+.1f}deg servo={servo_yaw_deg:+.2f}deg"
        )

        preopen_mm = getattr(result, "preopen_mm", None)
        grasp_width_mm = getattr(result, "grasp_width_mm", None)
        if preopen_mm is not None:
            try:
                preopen_mm = float(preopen_mm)
            except (TypeError, ValueError):
                preopen_mm = None

        if preopen_mm is not None and preopen_mm > 0.0:
            preopen_tx = SortingFrame(
                mode=Mode.PICK_PREOPEN,
                x_mm=preopen_mm,
                y_mm=0.0,
                box_id=0,
            )
            try:
                self.current_preopen_command = SortingProtocol.unpack(
                    SortingProtocol.pack(preopen_tx)
                )
            except Exception as exc:
                self._fail(f"PICK_PREOPEN pack failed: {exc}")
                return

            print(
                "[Runtime] dynamic preopen "
                f"object_width={grasp_width_mm if grasp_width_mm is not None else 'n/a'}mm "
                f"command={self.current_preopen_command.x_mm:.1f}mm"
            )
            if not self._send_command(
                self.current_preopen_command, label="PICK_PREOPEN", retry=False
            ):
                return
            self._enter_state(RuntimeState.WAIT_PICK_PREOPEN_ACK)
            return

        # 没有可靠动态预开口时保持原有固定预收缩流程，直接发送抓取目标。
        self.current_preopen_command = None
        print("[Runtime] dynamic preopen unavailable -> legacy fixed pre-clip")
        if not self._send_pick_target():
            return


    # =====================================================
    # 侧面识别
    # =====================================================

    def _accept_side_result(self, result, *, source: str = "consensus", fallback_image=None):
        """将已经决定采用的侧面识别结果正式送入分箱流程。"""
        if not self._action_still_allowed(RuntimeState.WAIT_SIDE_OBJECT):
            return

        color = str(getattr(result, "color", "unknown"))
        shape = str(getattr(result, "shape", "unknown"))
        qr_present = bool(getattr(result, "qr_present", False))

        if self.reject_unknown and (
            color.strip().lower() in ("", "unknown", "none")
            or shape.strip().lower() in ("", "unknown", "none")
        ):
            print(
                f"[SIDE] unstable/unknown result rejected: "
                f"color={color}, shape={shape}, qr={int(qr_present)}"
            )
            self._safe_reset(self.side_pipeline)
            self._reset_side_vote_stats()
            # 拒绝 unknown 后重新计时，避免下一轮 run_once 立刻再次触发超时。
            self.state_entered_at = time.monotonic()
            return

        obj = ObjectInfo(
            color=color,
            shape=shape,
            qr_present=qr_present,
        )
        crop = getattr(result, "crop", None)
        if crop is not None:
            self.current_side_image = crop.copy()
        elif fallback_image is not None:
            try:
                self.current_side_image = fallback_image.copy()
            except AttributeError:
                self.current_side_image = fallback_image
        else:
            self.current_side_image = None

        if source == "timeout_majority":
            print(f"[SIDE RESULT][TIMEOUT MAJORITY] {obj}")
        else:
            print("[SIDE RESULT]", obj)

        reservation = self.storage_manager.reserve(obj)
        if not reservation.success or reservation.reservation_id is None:
            self._fail(f"storage reserve failed: {reservation.reason}")
            return

        self.current_object = obj
        self.current_box = reservation.box_id
        self.current_reservation_id = reservation.reservation_id

        if not self._action_still_allowed(RuntimeState.WAIT_SIDE_OBJECT):
            self.storage_manager.cancel(self.current_reservation_id)
            self.current_reservation_id = None
            self.current_box = None
            self.current_object = None
            return

        tx = SortingFrame(
            mode=Mode.SORT_RESULT,
            x_mm=0.0,
            y_mm=0.0,
            box_id=self.current_box,
        )

        self.current_sort_command = tx
        if not self._send_command(tx, label="SORT_RESULT", retry=False):
            if self.current_reservation_id is not None:
                self.storage_manager.cancel(self.current_reservation_id)
            self.current_reservation_id = None
            self.current_box = None
            return

        self._enter_state(RuntimeState.WAIT_SORT_ACK)

    def process_side(self):
        # SideRecognitionConsensus READY 后会锁存，因此这里必须用状态门控，
        # 避免重复 reserve / 重复发送 SORT_RESULT。
        if not self.running or self.state != RuntimeState.WAIT_SIDE_OBJECT:
            return

        image = self.side_camera.read()
        if image is None:
            return

        self._side_last_frame_at = time.monotonic()
        self._side_frames_seen += 1

        result = self.side_pipeline.process(image)

        # 侧面流水线现在先运行形状分类器。只有形状分类器确认不是
        # no_object，才会继续颜色、二维码等属性识别。
        side_reason = str(getattr(result, "reason", ""))
        if side_reason.startswith("shape_pending:"):
            return

        # -----------------------------------------------------
        # 无物体恢复
        # -----------------------------------------------------
        # no_object 已由形状 YOLO 直接确认。此时不再进入颜色/QR/Shape
        # 后续流程，也不等待背景差分的时间/帧数门槛，马上请求下位机
        # Return -> Disable；收到 MCU_RETRY_PICK_READY 后重新开始顶部识别。
        if not bool(getattr(result, "found", True)):
            self._request_retry_pick(
                reason=f"no_object from shape classifier: {side_reason or 'no_object'}"
            )
            return

        # 无论当前是否达到 4/5 共识，只要本帧给出了完整有效结果，就累计到
        # “本次识别阶段”的超时兜底票池。这样超时后选择的是整个时段
        # 票数最多的结果，而不是仅仅最后 5 帧中的多数。
        self._record_side_vote(result)

        if not bool(getattr(result, "ready", False)):
            return

        self._accept_side_result(result, source="consensus", fallback_image=image)

    # =====================================================
    # STM32 事件
    # =====================================================

    def process_message(self, msg):
        mode = msg.mode

        # 心跳 ACK 是链路健康证据，不参与具体分拣状态机。
        if mode == Mode.MCU_HEARTBEAT_ACK:
            self._last_peer_seen_at = time.monotonic()
            if self.state in (RuntimeState.LINK_OFFLINE, RuntimeState.LINK_RECOVERY):
                return

        # 重连阶段只接受安全 STOP ACK；掉线期间 MCU 可能上报一次
        # MCU_STOP_WORK，这是下位机的安全收敛通知，不应再次把上位机置错。
        if self.state == RuntimeState.LINK_RECOVERY:
            if mode == Mode.MCU_STOP_ACK:
                self._last_peer_seen_at = time.monotonic()
                self._timeout_notice_states.clear()
                self._resume_after_link_recovery = False
                self.running = False
                self._enter_state(RuntimeState.IDLE)
                print("[LINK][RECOVERY] MCU safe STOP confirmed; waiting START")
                return
            if mode == Mode.MCU_START_WORK:
                # 下位机已经完成掉线安全回位并主动重新启动。此时它处于
                # WAIT_PICK_TARGET，不能再发送 STOP，否则会打断自动恢复。
                if self._resume_after_link_recovery:
                    self._resume_after_link_recovery = False
                    self.running = True
                    self._begin_cycle()
                    print("[LINK][RECOVERY] MCU START received; sorting resumed")
                return
            if mode in (Mode.MCU_STOP_WORK, Mode.MCU_ERROR, Mode.MCU_BUSY):
                print(f"[LINK][RECOVERY] ignore {mode.name} until safe STOP ACK")
                return
            # 丢弃恢复前残留的 ACK/READY/DONE，避免旧周期污染新周期。
            if mode != Mode.MCU_HEARTBEAT_ACK:
                return

        if self.state == RuntimeState.LINK_OFFLINE:
            # 离线期间收到的旧动作帧全部丢弃；重连后必须以新的 STOP
            # 安全同步边界开始，防止迟到 ACK 驱动错误动作。
            return

        # 全局高优先级事件先处理。
        if mode == Mode.MCU_ERROR:
            self._fail(
                f"MCU_ERROR x={msg.x_mm:.1f} y={msg.y_mm:.1f} "
                f"error_code={msg.box_id}"
            )
            return

        if mode == Mode.MCU_STOP_WORK:
            if self.state == RuntimeState.LINK_OFFLINE:
                print("[LINK][OFFLINE] MCU_STOP_WORK received; waiting reconnect")
                return
            # MCU 已发出停止请求，不再回发一条 STOP 形成无意义回环。
            self.stop(send_stop=False)
            return

        if mode == Mode.MCU_START_WORK:
            if not self.running:
                self.start()
            return

        if not self.running:
            return

        if mode == Mode.MCU_BUSY:
            # BUSY 仅作为状态信息，不刷新绝对截止时间。
            print(f"[Runtime] MCU_BUSY while {self.state.name}")
            return

        if mode == Mode.MCU_PICK_PREOPEN_ACK:
            if self.state == RuntimeState.WAIT_PICK_PREOPEN_ACK:
                expected = self.current_preopen_command
                if expected is None:
                    self._fail("PICK_PREOPEN_ACK received without active preopen command")
                    return
                if abs(float(msg.x_mm) - float(expected.x_mm)) > 0.051:
                    self._fail(
                        "PICK_PREOPEN_ACK echo mismatch: "
                        f"expected={expected.x_mm:.1f}mm, got={msg.x_mm:.1f}mm"
                    )
                    return
                print(f"[Runtime] PICK PREOPEN ACK {msg.x_mm:.1f}mm")
                if self.current_pick_command is None:
                    self._fail("PICK_PREOPEN_ACK received without prepared PICK_TARGET")
                    return
                if not self._send_pick_target():
                    return
            return

        if mode == Mode.MCU_PICK_TARGET_ACK:
            if self.state == RuntimeState.WAIT_PICK_ACK:
                expected = self.current_pick_command
                if expected is None:
                    self._fail("PICK_ACK received without active PICK command")
                    return
                if (
                    abs(float(msg.x_mm) - float(expected.x_mm)) > 0.051
                    or abs(float(msg.y_mm) - float(expected.y_mm)) > 0.051
                    or abs(float(msg.yaw_deg) - float(expected.yaw_deg)) > 0.51
                ):
                    self._fail(
                        "PICK_ACK echo mismatch: "
                        f"expected=({expected.x_mm:.1f},{expected.y_mm:.1f},"
                        f"{expected.yaw_deg:.0f}deg), "
                        f"got=({msg.x_mm:.1f},{msg.y_mm:.1f},{msg.yaw_deg:.0f}deg)"
                    )
                    return
                print("[Runtime] PICK ACK")
                self._enter_state(RuntimeState.WAIT_RECOGNITION_READY)
            return

        if mode == Mode.MCU_RECOGNITION_READY:
            if self.state == RuntimeState.WAIT_RECOGNITION_READY:
                expected = self.current_pick_command
                if expected is None:
                    self._fail("RECOGNITION_READY received without active PICK command")
                    return
                # READY没有yaw字段，但下位机会回显本周期PICK的XY。
                if (
                    abs(float(msg.x_mm) - float(expected.x_mm)) > 0.051
                    or abs(float(msg.y_mm) - float(expected.y_mm)) > 0.051
                ):
                    self._fail(
                        "RECOGNITION_READY echo mismatch: "
                        f"expected=({expected.x_mm:.1f},{expected.y_mm:.1f}), "
                        f"got=({msg.x_mm:.1f},{msg.y_mm:.1f})"
                    )
                    return
                # 机械已经把新物体放到识别台之后，才打开新的侧面采样窗口。
                self._safe_reset(self.side_pipeline)
                self._reset_side_vote_stats()
                print("[Runtime] RECOGNITION READY")
                self._enter_state(RuntimeState.WAIT_SIDE_OBJECT)
            return

        if mode == Mode.MCU_RETRY_PICK_READY:
            if self.state == RuntimeState.WAIT_RETRY_PICK_READY:
                print(
                    f"[RECOVERY] RETRY_PICK READY -> resume top vision "
                    f"recoveries={self.drop_recovery_count}"
                )
                # 下位机已确认夹爪真正回到 Disable 且重新等待 PICK_TARGET。
                # 当前失败夹取不计完成、不占库存，直接开始新的顶部周期。
                self._begin_cycle()
            return

        if mode == Mode.MCU_SORT_RESULT_ACK:
            if self.state == RuntimeState.WAIT_SORT_ACK:
                if self.current_box is None:
                    self._fail("SORT_ACK received without active reservation")
                    return
                if int(msg.box_id) != int(self.current_box):
                    self._fail(
                        f"SORT_ACK box mismatch: expected={self.current_box}, got={msg.box_id}"
                    )
                    return
                print("[Runtime] SORT ACK")
                self._enter_state(RuntimeState.WAIT_SORT_DONE)
            return

        if mode == Mode.MCU_SORT_DONE:
            if self.state != RuntimeState.WAIT_SORT_DONE:
                return

            if self.current_box is None or self.current_reservation_id is None:
                self._fail("SORT_DONE received without active reservation")
                return

            if int(msg.box_id) != int(self.current_box):
                self._fail(
                    f"SORT_DONE box mismatch: expected={self.current_box}, got={msg.box_id}"
                )
                return

            committed = self.storage_manager.commit(self.current_reservation_id)
            if not committed.success:
                self._fail(f"storage commit failed: {committed.reason}")
                return

            self.completed_count += 1
            print(
                f"[Runtime] SORT DONE box={committed.box_id} "
                f"box_count={committed.count} total={self.completed_count} "
                f"cycle={self.cycle_id}"
            )

            record = {
                "sequence": self.completed_count,
                "cycle_id": self.cycle_id,
                "color": getattr(self.current_object, "color", "unknown"),
                "shape": getattr(self.current_object, "shape", "unknown"),
                "qr_present": bool(getattr(self.current_object, "qr_present", False)),
                "box_id": int(committed.box_id),
                "box_count": int(committed.count),
                "success_count": int(self.completed_count),
                "image_bgr": self.current_side_image,
                "storage_status": self.storage_manager.get_status(),
            }
            if callable(self.on_finished):
                try:
                    self.on_finished(record)
                except Exception as exc:
                    print(f"[Runtime][WARN] on_finished failed: {exc}")

            self.current_reservation_id = None

            if self.max_items > 0 and self.completed_count >= self.max_items:
                self._finish_competition("max_items_reached")
            else:
                self._begin_cycle()

    # =====================================================
    # 无限循环 / 状态感知调度
    # =====================================================

    def _loop_delay_s(self) -> float:
        if not self.running:
            return self.loop_sleep_idle_s

        if self.state in (
            RuntimeState.WAIT_TOP_OBJECT,
            RuntimeState.WAIT_SIDE_OBJECT,
        ):
            return self.loop_sleep_vision_s

        # ACK / READY / DONE / ERROR 等状态不做视觉推理，
        # 用更短周期及时消费 RX 线程已经入队的控制帧。
        return self.loop_sleep_wait_s

    def spin(self):
        print(
            "[Runtime] WAIT START "
            f"scheduler(idle={self.loop_sleep_idle_s * 1000:.0f}ms, "
            f"wait={self.loop_sleep_wait_s * 1000:.0f}ms, "
            f"vision={self.loop_sleep_vision_s * 1000:.0f}ms)"
        )
        self._shutdown_event.clear()
        self._start_rx_thread()
        try:
            while not self._shutdown_event.is_set():
                self.run_once()

                # Event.wait() 比固定 sleep 更适合关闭流程：shutdown 时可立即唤醒。
                delay_s = self._loop_delay_s()
                if delay_s > 0.0:
                    self._shutdown_event.wait(delay_s)
        except KeyboardInterrupt:
            self.shutdown(send_stop=True)
        finally:
            self._stop_rx_thread()
