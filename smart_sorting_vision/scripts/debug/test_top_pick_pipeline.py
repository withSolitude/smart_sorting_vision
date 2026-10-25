from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from vision.cameras.top_camera import TopCamera
from vision.pipeline.top_pick_pipeline import (
    TopPickPipeline,
)


MODEL_PATH = (
    ROOT
    / "outputs"
    / "training"
    / "top_seg"
    / "v3"
    / "v3_interim_aug"
    / "weights"
    / "best.pt"
)

CALIBRATION_PATH = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "robot_mapping"
    / "pixel_to_robot.json"
)

SAVE_DIR = (
    ROOT
    / "outputs"
    / "debug"
    / "top_pick_pipeline"
)


DEVICE = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


def safe_release_camera(
    camera,
) -> None:
    if hasattr(
        camera,
        "release",
    ):
        try:
            camera.release()
            return
        except Exception:
            pass

    if hasattr(
        camera,
        "close",
    ):
        try:
            camera.close()
        except Exception:
            pass


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    x, y = origin

    (
        width,
        height,
    ), baseline = (
        cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )
    )

    cv2.rectangle(
        image,
        (
            x - 3,
            y - height - 5,
        ),
        (
            x + width + 4,
            y + baseline + 4,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (
            x,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline = TopPickPipeline(
        model_path=
            MODEL_PATH,
        calibration_path=
            CALIBRATION_PATH,
        conf=
            0.35,
        imgsz=
            832,
        device=
            DEVICE,
        obstacle_margin=
            12,
        border_margin=
            12,
        min_safe_area=
            50,
        min_clearance=
            3.0,
        consensus_window=
            5,
        consensus_required=
            4,
        max_track_center_shift_px=
            55.0,
        max_grasp_spread_mm=
            3.0,
        reset_after_misses=
            3,

        # 当前暂不设置机械极限。
        # 后面明确机构极限后再填，例如：
        #
        # x_limits_mm=(-120.0, -20.0),
        # y_limits_mm=(-30.0, 80.0),
        #
        x_limits_mm=None,
        y_limits_mm=None,
    )

    camera = TopCamera()

    window = (
        "Formal TopPickPipeline Test"
    )

    previous_time = (
        time.perf_counter()
    )

    smooth_fps = 0.0

    last_ready_key = None

    print()
    print(
        "=" * 76
    )
    print(
        "FORMAL TOP PICK PIPELINE TEST"
    )
    print(
        "=" * 76
    )
    print(
        "TopCamera"
        " -> YOLO"
        " -> Collision Planner"
        " -> Homography"
        " -> 4/5 Consensus"
        " -> READY"
    )
    print(
        "P1~P4 不作为抓取区域限制。"
    )
    print(
        "当前不向 STM32 发送任何数据。"
    )
    print(
        "=" * 76
    )
    print(
        "R : reset consensus"
    )
    print(
        "S : save debug image"
    )
    print(
        "Q / ESC : quit"
    )
    print(
        "=" * 76
    )

    try:
        camera.open()

        cv2.namedWindow(
            window,
            cv2.WINDOW_NORMAL,
        )

        while True:
            frame = camera.read()

            result = (
                pipeline.process(
                    frame
                )
            )

            display = (
                frame.copy()
            )

            # --------------------------------------------
            # 当前帧所有合法候选
            # --------------------------------------------
            for candidate in (
                result.candidates
            ):
                contour = (
                    np.round(
                        candidate.polygon
                    )
                    .astype(
                        np.int32
                    )
                    .reshape(
                        -1,
                        1,
                        2,
                    )
                )

                cv2.drawContours(
                    display,
                    [contour],
                    -1,
                    (
                        0,
                        200,
                        0,
                    ),
                    2,
                )

                gx, gy = (
                    candidate.grasp_pixel
                )

                cv2.circle(
                    display,
                    (
                        gx,
                        gy,
                    ),
                    5,
                    (
                        0,
                        255,
                        0,
                    ),
                    -1,
                )

                draw_text(
                    display,
                    (
                        f"ID={candidate.object_index} "
                        f"XY="
                        f"({candidate.x_mm:.1f},"
                        f"{candidate.y_mm:.1f}) "
                        f"C="
                        f"{candidate.clearance_px:.1f}"
                    ),
                    (
                        gx + 8,
                        gy - 8,
                    ),
                )

            # --------------------------------------------
            # 稳定 READY
            # --------------------------------------------
            if (
                result.ready
                and
                result.grasp_pixel
                is not None
                and
                result.x_mm
                is not None
                and
                result.y_mm
                is not None
            ):
                gx, gy = (
                    result.grasp_pixel
                )

                cv2.circle(
                    display,
                    (
                        gx,
                        gy,
                    ),
                    16,
                    (
                        0,
                        255,
                        255,
                    ),
                    3,
                )

                cv2.drawMarker(
                    display,
                    (
                        gx,
                        gy,
                    ),
                    (
                        0,
                        255,
                        255,
                    ),
                    cv2.MARKER_CROSS,
                    30,
                    2,
                )

                draw_text(
                    display,
                    (
                        "READY "
                        f"{result.votes}/"
                        f"{result.window_size}  "
                        f"pixel="
                        f"{result.grasp_pixel}  "
                        f"robot="
                        f"({result.x_mm:.2f},"
                        f"{result.y_mm:.2f})mm"
                    ),
                    (
                        18,
                        80,
                    ),
                    scale=0.60,
                    thickness=2,
                )

                ready_key = (
                    round(
                        result.x_mm,
                        1,
                    ),
                    round(
                        result.y_mm,
                        1,
                    ),
                )

                if (
                    ready_key
                    !=
                    last_ready_key
                ):
                    print(
                        "[READY] "
                        f"pixel="
                        f"{result.grasp_pixel} "
                        f"robot="
                        f"({result.x_mm:.3f},"
                        f"{result.y_mm:.3f}) mm "
                        f"class="
                        f"{result.class_name} "
                        f"conf="
                        f"{result.confidence:.3f} "
                        f"clear="
                        f"{result.clearance_px:.2f}px"
                    )

                    last_ready_key = (
                        ready_key
                    )

            else:
                draw_text(
                    display,
                    (
                        "WAIT "
                        f"{result.votes}/"
                        f"{result.window_size} "
                        f"{result.reason}"
                    ),
                    (
                        18,
                        80,
                    ),
                    scale=0.60,
                    thickness=2,
                )

                last_ready_key = None

            now = (
                time.perf_counter()
            )

            dt = (
                now
                -
                previous_time
            )

            previous_time = now

            if dt > 0:
                fps = 1.0 / dt

                if smooth_fps == 0.0:
                    smooth_fps = fps
                else:
                    smooth_fps = (
                        0.9
                        *
                        smooth_fps
                        +
                        0.1
                        *
                        fps
                    )

            draw_text(
                display,
                (
                    f"FPS={smooth_fps:.1f} "
                    f"Infer="
                    f"{result.inference_ms:.1f}ms "
                    f"Objects="
                    f"{result.object_count} "
                    f"Candidates="
                    f"{result.candidate_count} "
                    "NO STM32"
                ),
                (
                    18,
                    34,
                ),
                scale=0.54,
                thickness=2,
            )

            cv2.imshow(
                window,
                display,
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            if key in (
                ord("r"),
                ord("R"),
            ):
                pipeline.reset()

                print(
                    "[RESET] "
                    "TopPickPipeline reset"
                )

            if key in (
                ord("s"),
                ord("S"),
            ):
                timestamp = (
                    datetime.now()
                    .strftime(
                        "%Y%m%d_%H%M%S_%f"
                    )
                )

                save_path = (
                    SAVE_DIR
                    /
                    f"top_pick_pipeline_"
                    f"{timestamp}.jpg"
                )

                cv2.imwrite(
                    str(save_path),
                    display,
                )

                print(
                    f"[SAVE] "
                    f"{save_path}"
                )

    finally:
        safe_release_camera(
            camera
        )

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
