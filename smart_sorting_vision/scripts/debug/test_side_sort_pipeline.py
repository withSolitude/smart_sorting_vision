from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import torch


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from vision.cameras.side_camera import (
    SideCamera,
)
from vision.pipeline.side_sort_pipeline import (
    SideSortPipeline,
)


BACKGROUND_PATH = (
    ROOT
    / "data"
    / "calibration"
    / "side_camera"
    / "side_mechanical_roi_background.png"
)

SHAPE_MODEL_PATH = (
    ROOT
    / "outputs"
    / "training"
    / "side_cls"
    / "v2_yolo26n_cls"
    / "weights"
    / "best.pt"
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


def main() -> None:
    pipeline = SideSortPipeline(
        background_path=
            BACKGROUND_PATH,
        shape_model_path=
            SHAPE_MODEL_PATH,
        shape_device=
            DEVICE,

        # 当前侧面继续保持原来的 4/5
        consensus_window=
            5,
        consensus_required=
            4,
    )

    camera = SideCamera()

    print()
    print(
        "=" * 72
    )
    print(
        "FORMAL SIDE SORT PIPELINE TEST"
    )
    print(
        "=" * 72
    )
    print(
        "SideCamera"
        " -> ObjectExtractor"
        " -> Color/QR/Shape"
        " -> 4/5 Consensus"
        " -> READY"
    )
    print(
        "本步骤只输出识别属性，暂不映射 Box_ID。"
    )
    print(
        "=" * 72
    )
    print(
        "R : reset"
    )
    print(
        "Q / ESC : quit"
    )
    print(
        "=" * 72
    )

    last_ready_signature = None

    try:
        camera.open()

        cv2.namedWindow(
            "SideSortPipeline",
            cv2.WINDOW_NORMAL,
        )

        while True:
            frame = (
                camera.read()
            )

            result = (
                pipeline.process(
                    frame
                )
            )

            display = (
                frame.copy()
            )

            if (
                result.crop_bbox
                is not None
            ):
                (
                    x1,
                    y1,
                    x2,
                    y2,
                ) = (
                    result.crop_bbox
                )

                cv2.rectangle(
                    display,
                    (
                        int(x1),
                        int(y1),
                    ),
                    (
                        int(x2),
                        int(y2),
                    ),
                    (
                        0,
                        255,
                        0,
                    ),
                    2,
                )

            if not result.found:
                line1 = "NO OBJECT"
                line2 = (
                    f"{result.reason}"
                )

            elif result.ready:
                line1 = (
                    "READY "
                    f"{result.votes}/"
                    f"{result.window}"
                )

                line2 = (
                    f"{result.color} | "
                    f"QR="
                    f"{'Y' if result.qr_present else 'N'} | "
                    f"{result.shape}"
                )

                signature = (
                    result.color,
                    result.qr_present,
                    result.shape,
                )

                if (
                    signature
                    !=
                    last_ready_signature
                ):
                    print()
                    print(
                        "[SIDE READY]"
                    )
                    print(
                        f"Color : "
                        f"{result.color}"
                    )
                    print(
                        f"QR    : "
                        f"{result.qr_present}"
                    )
                    print(
                        f"Shape : "
                        f"{result.shape}"
                    )
                    print(
                        f"ColorConf : "
                        f"{result.color_confidence:.3f}"
                    )
                    print(
                        f"ShapeConf : "
                        f"{result.shape_confidence:.3f}"
                    )

                    last_ready_signature = (
                        signature
                    )

            else:
                line1 = (
                    "WAIT "
                    f"{result.votes}/"
                    f"{result.window} "
                    f"(need "
                    f"{result.required_hits})"
                )

                line2 = (
                    f"{result.color} | "
                    f"QR="
                    f"{'Y' if result.qr_present else 'N'} | "
                    f"{result.shape}"
                )

                last_ready_signature = (
                    None
                )

            cv2.putText(
                display,
                line1,
                (
                    16,
                    34,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    0,
                    255,
                    0,
                )
                if result.ready
                else (
                    0,
                    165,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                line2,
                (
                    16,
                    64,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (
                    255,
                    255,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "SideSortPipeline",
                display,
            )

            if (
                result.crop
                is not None
            ):
                cv2.imshow(
                    "SideSortPipeline Crop",
                    result.crop,
                )

            key = (
                cv2.waitKey(
                    1
                )
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

                last_ready_signature = (
                    None
                )

                print(
                    "[SIDE] pipeline reset"
                )

    finally:
        safe_release_camera(
            camera
        )

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
