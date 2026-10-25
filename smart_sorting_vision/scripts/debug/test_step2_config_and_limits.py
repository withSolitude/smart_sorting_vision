from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from communication.sorting_protocol import Mode
from runtime.app_config import load_runtime_config
from runtime.sorting_runtime import RuntimeState, SortingRuntime
from vision.cameras.side_camera import SideCamera
from vision.cameras.top_camera import TopCamera
from vision.storage.storage_manager import StorageManager


ROOT = Path(__file__).resolve().parents[1]


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
        return SimpleNamespace(ready=True, x_mm=self.x_mm, y_mm=self.y_mm)


class FakeSidePipeline:
    def reset(self):
        pass


def make_runtime(x, y):
    link = FakeLink()
    runtime = SortingRuntime(
        FakeCamera(),
        FakeCamera(),
        FakeTopPipeline(x, y),
        FakeSidePipeline(),
        StorageManager(6),
        link,
        pick_x_limits_mm=(-77.0, 0.0),
        pick_y_limits_mm=(0.0, 55.0),
    )
    return runtime, link


def test_config_and_relative_paths():
    cfg = load_runtime_config(ROOT, ROOT / "config" / "runtime_config.yaml")
    assert cfg["mechanics"]["workspace_mm"]["x"] == [-77.0, 0.0]
    assert cfg["mechanics"]["workspace_mm"]["y"] == [0.0, 55.0]
    assert Path(cfg["model"]["top"]["path"]).exists()
    assert Path(cfg["model"]["side"]["path"]).exists()
    assert Path(cfg["calibration"]["pixel_to_robot"]).exists()


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
    for x, y in ((-77.0, 0.0), (-77.0, 55.0), (0.0, 0.0), (0.0, 55.0)):
        runtime, link = make_runtime(x, y)
        runtime.start()
        runtime.process_top()
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
