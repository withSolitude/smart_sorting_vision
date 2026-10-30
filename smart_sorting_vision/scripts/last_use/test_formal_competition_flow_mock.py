from pathlib import Path
from runtime.app_config import load_runtime_config
from vision.calibration.yaw_servo import YawServoCalibration
from types import SimpleNamespace
import numpy as np

from runtime.sorting_runtime import SortingRuntime, RuntimeState
from communication.sorting_protocol import SortingFrame, Mode
from vision.storage.storage_manager import StorageManager


class FakeCam:
    def read(self):
        return np.zeros((32, 32, 3), dtype=np.uint8)

class FakeTop:
    def __init__(self): self.used=False
    def reset(self): pass
    def process(self, image):
        if not self.used:
            self.used=True
            return SimpleNamespace(ready=True, object_count=1, x_mm=-50.0, y_mm=20.0, yaw_deg=15.0, preopen_mm=20.0, grasp_width_mm=15.0)
        return SimpleNamespace(ready=False, object_count=1)

class FakeSide:
    def __init__(self): self.used=False
    def reset(self): self.used=False
    def process(self, image):
        if not self.used:
            self.used=True
            return SimpleNamespace(ready=True, color='yellow', shape='三棱柱', qr_present=False, crop=image)
        return SimpleNamespace(ready=False)

class FakeSTM:
    def __init__(self): self.sent=[]; self.rx=[]
    def poll(self):
        out=self.rx[:]; self.rx.clear(); return out
    def send_frame(self, f): self.sent.append(f); return b''
    def describe(self, f): return f'{f.mode.name} x={f.x_mm} y={f.y_mm} box={f.box_id} yaw={f.yaw_deg}'

root=Path(__file__).resolve().parents[2]
cfg=load_runtime_config(root, root / "config/runtime_config.yaml")
records=[]
stm=FakeSTM()
r=SortingRuntime(FakeCam(), FakeCam(), FakeTop(), FakeSide(), StorageManager(6), stm,
    yaw_servo_calibration=YawServoCalibration.from_config(cfg["calibration"]["yaw_servo"]),
    minimum_items_before_empty_finish=12, empty_finish_hold_s=3.0,
    on_finished=records.append, pick_ack_timeout_s=3, sort_ack_timeout_s=3,
    recognition_ready_timeout_s=30, side_recognition_timeout_s=10, sort_done_timeout_s=30)
r.start()
r.run_once()
assert r.state == RuntimeState.WAIT_PICK_PREOPEN_ACK
pick=r.current_pick_command
assert pick is not None
r.process_message(SortingFrame(mode=Mode.MCU_PICK_PREOPEN_ACK, x_mm=20.0))
assert r.state == RuntimeState.WAIT_PICK_ACK
r.process_message(SortingFrame(mode=Mode.MCU_PICK_TARGET_ACK, x_mm=pick.x_mm, y_mm=pick.y_mm, yaw_deg=pick.yaw_deg))
assert r.state == RuntimeState.WAIT_RECOGNITION_READY
r.process_message(SortingFrame(mode=Mode.MCU_RECOGNITION_READY, x_mm=pick.x_mm, y_mm=pick.y_mm))
assert r.state == RuntimeState.WAIT_SIDE_OBJECT
r.run_once()
assert r.state == RuntimeState.WAIT_SORT_ACK
box=r.current_box
assert box == 1
r.process_message(SortingFrame(mode=Mode.MCU_SORT_RESULT_ACK, box_id=box))
assert r.state == RuntimeState.WAIT_SORT_DONE
r.process_message(SortingFrame(mode=Mode.MCU_SORT_DONE, box_id=box))
assert r.completed_count == 1
assert len(records)==1 and records[0]['box_id']==1
assert r.state == RuntimeState.WAIT_TOP_OBJECT
print('formal competition mock flow OK')
