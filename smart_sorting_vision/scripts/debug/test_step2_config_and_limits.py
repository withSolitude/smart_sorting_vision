from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from communication.sorting_protocol import Mode, SortingFrame
from runtime.app_config import load_runtime_config
from vision.calibration.yaw_servo import YawServoCalibration

from runtime.sorting_runtime import RuntimeState, SortingRuntime
from vision.cameras.side_camera import SideCamera
from vision.cameras.top_camera import TopCamera
from vision.storage.storage_manager import StorageManager


# 当前文件位于 smart_sorting_vision/scripts/debug/，项目根是 parents[2]。
ROOT = Path(__file__).resolve().parents[2]


class FakeLink:
    def __init__(self):
        self.sent = []

    def send_frame(self, frame):
        self.sent.append(frame)

    def poll(self):
        return []

    @staticmethod
    def describe(frame):
        return frame.mode.name


class FakeCamera:
    def read(self):
        return object()


class FakeTopPipeline:
    def __init__(self, x_mm, y_mm):
        self.x_mm = x_mm
        self.y_mm = y_mm

    def reset(self):
        pass

    def process(self, frame):
        return SimpleNamespace(ready=True, x_mm=self.x_mm, y_mm=self.y_mm, yaw_deg=0.0, preopen_mm=20.0, grasp_width_mm=15.0)


class FakeSidePipeline:
    def reset(self):
        pass


def make_runtime(x, y, *, limits_x=(-77.0, 0.0), limits_y=(0.0, 55.0)):
    link = FakeLink()
    cfg = load_runtime_config(ROOT, ROOT / "config/runtime_config.yaml")
    runtime = SortingRuntime(
        FakeCamera(),
        FakeCamera(),
        FakeTopPipeline(x, y),
        FakeSidePipeline(),
        StorageManager(6),
        link,
        yaw_servo_calibration=YawServoCalibration.from_config(cfg["calibration"]["yaw_servo"]),
        pick_x_limits_mm=limits_x,
        pick_y_limits_mm=limits_y,
    )
    return runtime, link


def test_config_and_relative_paths():
    cfg = load_runtime_config(ROOT, ROOT / "config" / "runtime_config.yaml")
    x_limits = cfg["mechanics"]["workspace_mm"]["x"]
    y_limits = cfg["mechanics"]["workspace_mm"]["y"]
    assert x_limits[0] < x_limits[1]
    assert y_limits[0] < y_limits[1]
    # 配置单元测试只检查路径已按项目根解析为绝对路径；
    # 模型和标定文件是否已经部署由 check_runtime_config.py 单独检查。
    assert Path(cfg["model"]["top"]["path"]).is_absolute()
    assert Path(cfg["model"]["side"]["path"]).is_absolute()
    assert Path(cfg["calibration"]["pixel_to_robot"]).is_absolute()


def test_camera_profile_is_loaded_from_yaml():
    cfg = load_runtime_config(ROOT, ROOT / "config" / "runtime_config.yaml")
    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )
    SideCamera.configure(cfg["camera"]["side"])

    assert TopCamera.DEVICE == cfg["camera"]["top"]["device"]
    assert [
        TopCamera.SAFE_ROI_X1,
        TopCamera.SAFE_ROI_Y1,
        TopCamera.SAFE_ROI_X2,
        TopCamera.SAFE_ROI_Y2,
    ] == cfg["camera"]["top"]["safe_roi"]

    assert SideCamera.DEVICE == cfg["camera"]["side"]["device"]
    assert [
        SideCamera.ROI_X1,
        SideCamera.ROI_Y1,
        SideCamera.ROI_X2,
        SideCamera.ROI_Y2,
    ] == cfg["camera"]["side"]["roi"]


def test_runtime_blocks_out_of_workspace_pick():
    runtime, link = make_runtime(1.0, 20.0)
    runtime.start()
    runtime.process_top()

    assert runtime.state == RuntimeState.ERROR
    assert not runtime.running
    assert "outside workspace" in runtime.fault_reason
    assert not any(frame.mode == Mode.PICK_TARGET for frame in link.sent)


def test_workspace_boundaries_are_valid():
    cfg = load_runtime_config(ROOT, ROOT / "config" / "runtime_config.yaml")
    x_min, x_max = cfg["mechanics"]["workspace_mm"]["x"]
    y_min, y_max = cfg["mechanics"]["workspace_mm"]["y"]
    limits_x = (x_min, x_max)
    limits_y = (y_min, y_max)
    for x, y in ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)):
        runtime, link = make_runtime(x, y, limits_x=limits_x, limits_y=limits_y)
        runtime.start()
        runtime.process_top()
        runtime.process_message(SortingFrame(Mode.MCU_PICK_PREOPEN_ACK, x_mm=20.0))
        assert runtime.state == RuntimeState.WAIT_PICK_ACK
        assert any(frame.mode == Mode.PICK_TARGET for frame in link.sent)


def main():
    tests = [
        test_config_and_relative_paths,
        test_camera_profile_is_loaded_from_yaml,
        test_runtime_blocks_out_of_workspace_pick,
        test_workspace_boundaries_are_valid,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print("STEP2 CONFIG/LIMIT TESTS PASSED")


if __name__ == "__main__":
    main()
