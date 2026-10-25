from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]


DATA_V2_TEST = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "prepared"
    / "v3"
    / "data.yaml"
)


DATA_LEGACY_TEST = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "prepared"
    / "v3"
    / "data_legacy_test.yaml"
)


V2_MODEL = (
    ROOT
    / "weights"
    / "top_seg"
    / "v2"
    / "best.pt"
)


V3_MODEL = (
    ROOT
    / "outputs"
    / "training"
    / "top_seg"
    / "v3"
    / "v3_interim_aug"
    / "weights"
    / "best.pt"
)


OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "evaluation"
    / "top_seg"
    / "v3"
)


IMGSZ = 832


def print_metrics(
    title,
    metrics,
):

    print()
    print(
        "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )

    print(
        "BOX"
    )

    print(
        f"P         : "
        f"{metrics.box.mp:.5f}"
    )

    print(
        f"R         : "
        f"{metrics.box.mr:.5f}"
    )

    print(
        f"mAP50     : "
        f"{metrics.box.map50:.5f}"
    )

    print(
        f"mAP50-95  : "
        f"{metrics.box.map:.5f}"
    )

    print()

    print(
        "MASK"
    )

    print(
        f"P         : "
        f"{metrics.seg.mp:.5f}"
    )

    print(
        f"R         : "
        f"{metrics.seg.mr:.5f}"
    )

    print(
        f"mAP50     : "
        f"{metrics.seg.map50:.5f}"
    )

    print(
        f"mAP50-95  : "
        f"{metrics.seg.map:.5f}"
    )


def evaluate(
    model_path,
    data_path,
    run_name,
    device,
):

    print()
    print(
        "-" * 70
    )

    print(
        f"MODEL : {model_path}"
    )

    print(
        f"DATA  : {data_path}"
    )

    print(
        "-" * 70
    )

    model = YOLO(
        str(
            model_path
        )
    )

    metrics = model.val(

        data=str(
            data_path
        ),

        split="test",

        imgsz=IMGSZ,

        # ----------------------------------------------------
        # mAP评估使用低conf。
        #
        # 不要使用部署时的0.50。
        # ----------------------------------------------------

        conf=0.001,

        device=device,

        batch=8,

        workers=4,

        plots=True,

        project=str(
            OUTPUT_ROOT
        ),

        name=run_name,

        exist_ok=True,

        verbose=True,
    )

    print_metrics(
        run_name,
        metrics,
    )

    return metrics


def main():

    if torch.cuda.is_available():

        device = 0

    else:

        device = "cpu"

    for path in (
        DATA_V2_TEST,
        DATA_LEGACY_TEST,
        V2_MODEL,
        V3_MODEL,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"找不到：{path}"
            )

    print(
        "=" * 70
    )

    print(
        "V2 vs V3 FIXED TEST"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 1. V2 -> 原V2 Test
    # ========================================================

    evaluate(
        model_path=V2_MODEL,
        data_path=DATA_V2_TEST,
        run_name="v2_on_fixed_v2_test",
        device=device,
    )

    # ========================================================
    # 2. V3 -> 完全相同的Test
    # ========================================================

    evaluate(
        model_path=V3_MODEL,
        data_path=DATA_V2_TEST,
        run_name="v3_on_fixed_v2_test",
        device=device,
    )

    # ========================================================
    # 3. V2 -> legacy
    # ========================================================

    evaluate(
        model_path=V2_MODEL,
        data_path=DATA_LEGACY_TEST,
        run_name="v2_on_legacy_test",
        device=device,
    )

    # ========================================================
    # 4. V3 -> legacy
    # ========================================================

    evaluate(
        model_path=V3_MODEL,
        data_path=DATA_LEGACY_TEST,
        run_name="v3_on_legacy_test",
        device=device,
    )


if __name__ == "__main__":

    main()