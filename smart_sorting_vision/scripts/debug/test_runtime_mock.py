from pathlib import Path
from runtime.app_config import load_runtime_config
from vision.calibration.yaw_servo import YawServoCalibration
from runtime.sorting_runtime import SortingRuntime


def main():
    root = Path(__file__).resolve().parents[2]
    cfg = load_runtime_config(root, root / "config/runtime_config.yaml")
    runtime = SortingRuntime(
        None,
        None,
        None,
        None,
        None,
        None,
        yaw_servo_calibration=YawServoCalibration.from_config(cfg["calibration"]["yaw_servo"]),
    )

    runtime.start()
    print('SortingRuntime structure OK')
    print(runtime.state)


if __name__ == '__main__':
    main()
