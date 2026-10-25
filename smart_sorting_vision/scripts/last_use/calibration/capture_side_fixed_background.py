from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def find_project_root() -> Path:
    """
    自动寻找 smart_sorting_vision 工程根目录。

    这样无论从哪个工作目录启动：
        python -m scripts.last_use.calibration.capture_side_fixed_background

    都能稳定找到：
        config/runtime_config.yaml
        vision/cameras/side_camera.py
    """
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (
            (parent / "config" / "runtime_config.yaml").is_file()
            and (parent / "vision" / "cameras" / "side_camera.py").is_file()
            and (parent / "runtime" / "app_config.py").is_file()
        ):
            return parent

    raise RuntimeError(
        "无法自动找到 smart_sorting_vision 工程根目录"
    )


ROOT = find_project_root()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.app_config import load_runtime_config  # noqa: E402
from vision.cameras.side_camera import SideCamera  # noqa: E402
from vision.processors.side_object_extractor import SideObjectExtractor  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "runtime_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用最终比赛 runtime_config.yaml 参数，"
            "重新采集侧面相机固定空背景。"
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="正式 runtime_config.yaml 路径",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="背景平均帧数，默认 60",
    )

    parser.add_argument(
        "--settle",
        type=int,
        default=30,
        help="正式采集前丢弃的稳定帧数，默认 30",
    )

    return parser.parse_args()


def draw_roi_preview(
    full_frame,
) -> None:
    """
    在完整画面中画出正式比赛 mechanical ROI。
    """
    x1 = int(SideCamera.ROI_X1)
    y1 = int(SideCamera.ROI_Y1)
    x2 = int(SideCamera.ROI_X2)
    y2 = int(SideCamera.ROI_Y2)

    cv2.rectangle(
        full_frame,
        (x1, y1),
        (x2 - 1, y2 - 1),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        full_frame,
        "FINAL SIDE ROI",
        (x1 + 8, max(25, y1 + 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()

    config_path = args.config.expanduser().resolve()

    # ==========================================================
    # 1. 加载与 main.py 完全相同的最终比赛配置
    # ==========================================================
    cfg = load_runtime_config(
        ROOT,
        config_path,
    )

    side_cfg = cfg["camera"]["side"]

    # calibration.side_background 已由 load_runtime_config()
    # 解析成工程内绝对路径。
    background_path = Path(
        cfg["calibration"]["side_background"]
    )

    # 关键：
    # 不使用 SideCamera 类内部默认值，
    # 强制套用 runtime_config.yaml 的正式比赛参数。
    SideCamera.configure(
        side_cfg
    )

    print("=" * 72)
    print(" SIDE FIXED BACKGROUND CAPTURE - FINAL COMPETITION CONFIG ")
    print("=" * 72)

    print(f"[CONFIG] file       : {config_path}")
    print(f"[CONFIG] device     : {SideCamera.DEVICE}")
    print(
        "[CONFIG] stream     : "
        f"{SideCamera.WIDTH}x{SideCamera.HEIGHT} "
        f"@ {SideCamera.FPS} fps "
        f"{SideCamera.FOURCC}"
    )
    print(
        "[CONFIG] ROI        : "
        f"[{SideCamera.ROI_X1}, "
        f"{SideCamera.ROI_Y1}, "
        f"{SideCamera.ROI_X2}, "
        f"{SideCamera.ROI_Y2}] "
        f"-> {SideCamera.ROI_WIDTH}x{SideCamera.ROI_HEIGHT}"
    )

    print("[CONFIG] controls:")
    controls = side_cfg.get("controls", {}) or {}

    for name in (
        "auto_exposure",
        "exposure_time_absolute",
        "gain",
        "white_balance_automatic",
        "white_balance_temperature",
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "gamma",
        "power_line_frequency",
        "sharpness",
        "backlight_compensation",
    ):
        if name in controls:
            print(
                f"         {name:28s} = "
                f"{controls[name]}"
            )

    print(f"[CONFIG] output     : {background_path}")
    print(f"[CAPTURE] frames     : {max(1, args.frames)}")
    print(f"[CAPTURE] settle     : {max(0, args.settle)}")
    print()
    print("操作：")
    print("  1. 清空侧面识别平台，不能放任何物体")
    print("  2. 保持比赛时的灯光、机械位置和遮光环境")
    print("  3. C : 采集并覆盖正式背景")
    print("  4. Q / ESC : 退出，不保存")
    print()

    camera = SideCamera()
    extractor = SideObjectExtractor()

    camera.open()

    print("[OK] SideCamera opened with final runtime config")

    window_full = "SIDE BACKGROUND - FULL"
    window_roi = "SIDE BACKGROUND - FINAL ROI"

    cv2.namedWindow(
        window_full,
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        window_roi,
        cv2.WINDOW_NORMAL,
    )

    saved = False

    try:
        while True:
            # 为了同时显示完整画面和正式 ROI，
            # 一次只读一帧，避免额外调用 camera.read()
            # 导致两个窗口不是同一时刻。
            full = camera.read_full()

            x1 = int(SideCamera.ROI_X1)
            y1 = int(SideCamera.ROI_Y1)
            x2 = int(SideCamera.ROI_X2)
            y2 = int(SideCamera.ROI_Y2)

            roi = full[
                y1:y2,
                x1:x2,
            ].copy()

            preview = full.copy()
            draw_roi_preview(
                preview
            )

            cv2.putText(
                preview,
                "EMPTY PLATFORM -> press C to capture",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                window_full,
                preview,
            )

            cv2.imshow(
                window_roi,
                roi,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                print("[EXIT] no background changed")
                break

            if key in (
                ord("c"),
                ord("C"),
            ):
                print()
                print(
                    "[CAPTURE] collecting empty-platform background..."
                )

                # capture_background() 内部调用 camera.read()，
                # 因此采集的是与正式 main.py 完全相同的 mechanical ROI，
                # 而不是完整 1280x720 画面。
                background = extractor.capture_background(
                    camera,
                    frame_count=max(
                        1,
                        int(args.frames),
                    ),
                    settle_frames=max(
                        0,
                        int(args.settle),
                    ),
                )

                expected_shape = (
                    SideCamera.ROI_HEIGHT,
                    SideCamera.ROI_WIDTH,
                    3,
                )

                if background.shape != expected_shape:
                    raise RuntimeError(
                        "背景尺寸与正式 ROI 不一致：\n"
                        f"background = {background.shape}\n"
                        f"expected   = {expected_shape}"
                    )

                background_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                saved_path = extractor.save_background(
                    background_path
                )

                print(
                    f"[OK] background saved: {saved_path}"
                )
                print(
                    f"[OK] shape: {background.shape}"
                )
                print(
                    "[OK] main.py / SideObjectExtractor "
                    "will use this file via "
                    "calibration.side_background"
                )

                saved = True

                # 保存之后继续显示新背景，方便肉眼确认。
                show = background.copy()

                cv2.putText(
                    show,
                    "SAVED - press Q to exit / C to recapture",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    window_roi,
                    show,
                )

                cv2.waitKey(300)

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if saved:
        print()
        print("[DONE] 正式侧面背景已更新。")
        print(
            "[NEXT] 可直接运行："
            "python -m scripts.last_use.calibration.calibrate_side_hsv"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
