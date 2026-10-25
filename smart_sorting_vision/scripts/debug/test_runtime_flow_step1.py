from __future__ import annotations

from types import SimpleNamespace

from communication.sorting_protocol import Mode, SortingFrame
from runtime.sorting_runtime import RuntimeState, SortingRuntime
from vision.pipeline.side_recognition_consensus import SideRecognitionConsensus
from vision.storage.storage_manager import StorageManager


class FakeLink:
    def __init__(self):
        self.sent = []
        self.rx = []
        self.fail_send = False

    def send_frame(self, frame):
        if self.fail_send:
            raise OSError("simulated UART failure")
        self.sent.append(frame)
        return b"ok"

    def poll(self):
        frames = list(self.rx)
        self.rx.clear()
        return frames

    @staticmethod
    def describe(frame):
        return frame.mode.name


class FakeCamera:
    def read(self):
        return object()


class FakeTopPipeline:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def process(self, frame):
        return SimpleNamespace(ready=True, x_mm=-50.0, y_mm=20.0)


class FakeSidePipeline:
    def __init__(self):
        self.consensus = SideRecognitionConsensus(window_size=5, required_hits=4)
        self.reset_count = 0
        self.value = {
            "object_found": True,
            "color": "red",
            "shape": "cube",
            "qr_present": True,
            "color_confidence": 0.99,
            "shape_confidence": 0.99,
        }

    def reset(self):
        self.reset_count += 1
        self.consensus.reset()

    def process(self, frame):
        state = self.consensus.update(self.value)
        if not state["ready"]:
            return SimpleNamespace(ready=False)
        result = state["result"]
        return SimpleNamespace(
            ready=True,
            color=result["color"],
            shape=result["shape"],
            qr_present=result["qr_present"],
        )


def make_runtime():
    link = FakeLink()
    top = FakeTopPipeline()
    side = FakeSidePipeline()
    storage = StorageManager(6)
    runtime = SortingRuntime(
        FakeCamera(),
        FakeCamera(),
        top,
        side,
        storage,
        link,
    )
    return runtime, link, top, side, storage


def pick_ack(x=-50.0, y=20.0, yaw=0.0):
    return SortingFrame(Mode.MCU_PICK_TARGET_ACK, x_mm=x, y_mm=y, yaw_deg=yaw)


def pick_yaw_branch_ack(branch=0):
    return SortingFrame(Mode.MCU_PICK_YAW_BRANCH_ACK, box_id=branch)


def process_top_to_pick_ack(runtime):
    """完成当前协议的 Yaw 分支握手，推进到等待 PICK_ACK。"""
    runtime.process_top()
    if runtime.state == RuntimeState.WAIT_PICK_YAW_BRANCH_ACK:
        runtime.process_message(pick_yaw_branch_ack())
    assert runtime.state == RuntimeState.WAIT_PICK_ACK


def recognition_ready(x=-50.0, y=20.0):
    return SortingFrame(Mode.MCU_RECOGNITION_READY, x_mm=x, y_mm=y)


def sort_ack(box_id=1):
    return SortingFrame(Mode.MCU_SORT_RESULT_ACK, box_id=box_id)


def recognize_side(runtime, frames=4):
    for _ in range(frames):
        runtime.process_side()


def test_happy_path_and_cycle_reset():
    r, link, top, side, storage = make_runtime()
    r.start()
    assert r.state == RuntimeState.WAIT_TOP_OBJECT

    process_top_to_pick_ack(r)

    r.process_message(pick_ack())
    assert r.state == RuntimeState.WAIT_RECOGNITION_READY

    # R01: ACK 不能直接开始侧面识别。
    assert r.current_object is None

    r.process_message(recognition_ready())
    assert r.state == RuntimeState.WAIT_SIDE_OBJECT

    recognize_side(r)
    assert r.state == RuntimeState.WAIT_SORT_ACK
    assert r.current_object is not None
    assert r.current_object.qr_present is True
    assert "qr_present=1" in r.current_object.key()

    # R05: SORT_DONE 前库存不得增加。
    assert storage.boxes[0].count == 0

    r.process_message(sort_ack())
    assert r.state == RuntimeState.WAIT_SORT_DONE
    assert storage.boxes[0].count == 0

    r.process_message(SortingFrame(Mode.MCU_SORT_DONE, box_id=1))
    assert storage.boxes[0].count == 1
    assert r.cycle_id == 2
    # run_once 在处理 DONE 后可以立即读取下一帧顶部图像，因此 FakeTopPipeline
    # 可能已经把状态推进到 WAIT_PICK_ACK；关键是 DONE 已提交且新周期已建立。
    assert r.state in (
        RuntimeState.WAIT_TOP_OBJECT,
        RuntimeState.WAIT_PICK_YAW_BRANCH_ACK,
        RuntimeState.WAIT_PICK_ACK,
    )
    assert r.cycle_id == 2

    # R02: 新周期必须清空侧面锁存，不能一帧返回上一物体。
    side.value.update(color="blue", shape="ball", qr_present=False)
    process_top_to_pick_ack(r)
    r.process_message(pick_ack())
    r.process_message(recognition_ready())
    r.process_side()
    assert r.current_object is None
    assert r.state == RuntimeState.WAIT_SIDE_OBJECT


def test_wrong_done_box_and_mcu_error():
    r, _, _, _, storage = make_runtime()
    r.start()
    process_top_to_pick_ack(r)
    r.process_message(pick_ack())
    r.process_message(recognition_ready())
    recognize_side(r)
    r.process_message(sort_ack())

    r.process_message(SortingFrame(Mode.MCU_SORT_DONE, box_id=6))
    assert r.state == RuntimeState.ERROR
    assert not r.running
    assert storage.boxes[0].count == 0

    r.start()
    r.process_message(SortingFrame(Mode.MCU_ERROR, box_id=3))
    assert r.state == RuntimeState.ERROR
    assert not r.running


def test_send_failure_does_not_commit_inventory():
    r, link, _, _, storage = make_runtime()
    r.start()
    process_top_to_pick_ack(r)
    r.process_message(pick_ack())
    r.process_message(recognition_ready())

    # 只让 SORT_RESULT 发送失败。
    link.fail_send = True
    recognize_side(r)
    assert r.state == RuntimeState.ERROR
    assert storage.boxes[0].count == 0
    assert storage.get_pending_reservations() == []


def test_batched_ack_ready_and_ack_done_are_consumed_in_order():
    r, link, _, _, storage = make_runtime()
    r.start()
    r.process_top()

    # 同一次 poll 收到 ACK + READY，必须连续推进，不丢 READY。
    link.rx = [
        pick_yaw_branch_ack(),
        pick_ack(),
        recognition_ready(),
    ]
    r.run_once()
    assert r.state == RuntimeState.WAIT_SIDE_OBJECT

    # 再完成侧面识别，然后同批收到 SORT_ACK + DONE。
    recognize_side(r)
    link.rx = [
        sort_ack(),
        SortingFrame(Mode.MCU_SORT_DONE, box_id=1),
    ]
    r.run_once()
    assert storage.boxes[0].count == 1
    assert r.cycle_id == 2
    assert r.state in (
        RuntimeState.WAIT_TOP_OBJECT,
        RuntimeState.WAIT_PICK_YAW_BRANCH_ACK,
        RuntimeState.WAIT_PICK_ACK,
    )


def test_urgent_event_blocks_action_after_inference():
    r, link, _, _, _ = make_runtime()
    r.start()

    def process_and_receive_stop(frame):
        # 模拟视觉推理执行期间，接收线程已经收到 STOP/ERROR。
        r._urgent_stop_event.set()
        return SimpleNamespace(ready=True, x_mm=-50.0, y_mm=20.0)

    r.top_pipeline.process = process_and_receive_stop
    r.process_top()

    # 关键安全性质：推理结束后不会再发新的 PICK_TARGET。
    assert not any(frame.mode == Mode.PICK_TARGET for frame in link.sent)
    assert r.state == RuntimeState.WAIT_TOP_OBJECT



def test_mismatched_echoes_are_rejected():
    # 旧/串周期PICK ACK不得推进当前动作。
    r, _, _, _, _ = make_runtime()
    r.start()
    process_top_to_pick_ack(r)
    r.process_message(pick_ack(x=-49.0))
    assert r.state == RuntimeState.ERROR
    assert "PICK_ACK echo mismatch" in r.fault_reason

    # 正确PICK/READY后，错误箱号SORT ACK也必须拒绝。
    r, _, _, _, storage = make_runtime()
    r.start()
    process_top_to_pick_ack(r)
    r.process_message(pick_ack())
    r.process_message(recognition_ready())
    recognize_side(r)
    assert r.current_box == 1
    r.process_message(sort_ack(box_id=2))
    assert r.state == RuntimeState.ERROR
    assert storage.boxes[0].count == 0
    assert "SORT_ACK box mismatch" in r.fault_reason


def test_action_timeout_keeps_runtime_running():
    r, _, _, _, _ = make_runtime()
    r.start()
    process_top_to_pick_ack(r)

    # 不需要真实等待：把状态进入时间向前移动，模拟 ACK 等待超时。
    # 新逻辑只记录提示并继续等待，链路掉线由心跳 watchdog 单独判定。
    r.state_entered_at -= (r.pick_ack_timeout_s + 0.5)
    r._check_timeout()
    assert r.state == RuntimeState.WAIT_PICK_ACK
    assert r.running
    assert r.fault_reason is None


def main():
    tests = [
        test_happy_path_and_cycle_reset,
        test_wrong_done_box_and_mcu_error,
        test_send_failure_does_not_commit_inventory,
        test_batched_ack_ready_and_ack_done_are_consumed_in_order,
        test_urgent_event_blocks_action_after_inference,
        test_mismatched_echoes_are_rejected,
        test_action_timeout_keeps_runtime_running,
    ]

    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")

    print("STEP1 RUNTIME FLOW TESTS PASSED")


if __name__ == "__main__":
    main()
