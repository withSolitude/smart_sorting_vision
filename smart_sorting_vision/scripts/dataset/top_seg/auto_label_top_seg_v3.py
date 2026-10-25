from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# 工程根目录
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# V3图片
# ============================================================

IMAGE_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "new_images_v3"
)


# ============================================================
# V3自动标签
# ============================================================

LABEL_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "auto_labels_v3"
)


# ============================================================
# 自动标注预览
# ============================================================

PREVIEW_ROOT = (
    ROOT
    / "outputs"
    / "debug"
    / "top_seg_auto_label"
    / "v3"
)


# ============================================================
# V2当前最佳模型
# ============================================================

MODEL_PATH = (
    ROOT
    / "outputs"
    / "training"
    / "top_seg"
    / "v2"
    / "v2_aug"
    / "weights"
    / "best.pt"
)


# ============================================================
# 自动预标注参数
# ============================================================

DEFAULT_CONF = 0.25

DEFAULT_IMGSZ = 832


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
}


# ============================================================
# 中文路径安全读取
# ============================================================

def imread_unicode(
    path: Path,
) -> np.ndarray:

    data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    if data.size == 0:

        raise RuntimeError(
            f"图片为空：{path}"
        )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise RuntimeError(
            f"无法读取图片：{path}"
        )

    return image


# ============================================================
# 中文路径安全保存
# ============================================================

def imwrite_unicode(
    path: Path,
    image: np.ndarray,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92,
        ],
    )

    if not success:

        raise RuntimeError(
            f"图片编码失败：{path}"
        )

    encoded.tofile(
        str(path)
    )


# ============================================================
# 查找V3全部图片
# ============================================================

def find_images() -> list[Path]:

    images = []

    for path in IMAGE_ROOT.rglob(
        "*"
    ):

        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_SUFFIXES
        ):

            images.append(
                path
            )

    images.sort(
        key=lambda path:
        path.as_posix()
    )

    return images


# ============================================================
# YOLO结果 -> Segmentation TXT
# ============================================================

def result_to_yolo_seg(
    result,
) -> str:

    if (
        result.boxes is None
        or
        result.masks is None
        or
        len(result.boxes) == 0
    ):

        return ""

    lines = []

    for polygon in result.masks.xyn:

        polygon = np.asarray(
            polygon,
            dtype=np.float32,
        )

        if (
            polygon.ndim != 2
            or
            polygon.shape[1] != 2
            or
            len(polygon) < 3
        ):

            continue

        polygon = np.clip(
            polygon,
            0.0,
            1.0,
        )

        values = []

        for x, y in polygon:

            values.append(
                f"{float(x):.6f}"
            )

            values.append(
                f"{float(y):.6f}"
            )

        lines.append(
            "0 "
            +
            " ".join(
                values
            )
        )

    if not lines:

        return ""

    return (
        "\n".join(
            lines
        )
        +
        "\n"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # 检查
    # ========================================================

    if not IMAGE_ROOT.exists():

        raise FileNotFoundError(
            "找不到V3图片目录：\n"
            f"{IMAGE_ROOT}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "找不到V2模型：\n"
            f"{MODEL_PATH}"
        )

    images = find_images()

    if not images:

        raise RuntimeError(
            "V3目录中没有图片"
        )

    # ========================================================
    # GPU / CPU
    # ========================================================

    if args.device is None:

        device = (
            0
            if torch.cuda.is_available()
            else
            "cpu"
        )

    else:

        device = (
            args.device
        )

    print(
        "============================================================"
    )

    print(
        "V2 -> V3 AUTO PRE-ANNOTATION"
    )

    print(
        "============================================================"
    )

    print(
        f"Model : {MODEL_PATH}"
    )

    print(
        f"Images: {len(images)}"
    )

    print(
        f"conf  : {args.conf}"
    )

    print(
        f"imgsz : {args.imgsz}"
    )

    print(
        f"device: {device}"
    )

    print(
        "============================================================"
    )

    # ========================================================
    # 模型
    # ========================================================

    model = YOLO(
        str(MODEL_PATH)
    )

    total_instances = 0

    miss_count = 0

    empty_false_positive_count = 0

    skipped_count = 0

    # ========================================================
    # 图片循环
    # ========================================================

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        relative = (
            image_path.relative_to(
                IMAGE_ROOT
            )
        )

        # ====================================================
        # 保留完整目录结构
        # ====================================================

        label_path = (
            LABEL_ROOT
            /
            relative
        ).with_suffix(
            ".txt"
        )

        preview_path = (
            PREVIEW_ROOT
            /
            relative
        ).with_suffix(
            ".jpg"
        )

        label_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ====================================================
        # 已标注则跳过
        # ====================================================

        if (
            label_path.exists()
            and
            not args.overwrite
        ):

            print(
                f"[SKIP] "
                f"{relative.as_posix()}"
            )

            skipped_count += 1

            continue

        # ====================================================
        # 读取
        # ====================================================

        image = imread_unicode(
            image_path
        )

        # ====================================================
        # V2模型预标注
        # ====================================================

        results = model.predict(
            source=image,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )

        if len(results) != 1:

            raise RuntimeError(
                "YOLO预测结果数量异常："
                f"{relative}"
            )

        result = results[0]

        predicted_count = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        # ====================================================
        # 一级目录就是scene
        #
        # single/black/圆柱/xxx.jpg
        # scene依然是single
        # ====================================================

        scene = (
            relative.parts[0]
        )

        # ====================================================
        # empty强制为空标签
        #
        # 但preview仍保存模型预测，
        # 便于发现新托盘斜坡是否产生假阳性。
        # ====================================================

        if scene == "empty":

            label_text = ""

            if predicted_count > 0:

                empty_false_positive_count += 1

                print(
                    "[EMPTY FALSE POSITIVE] "
                    f"{relative.as_posix()} "
                    f"pred={predicted_count}"
                )

        else:

            label_text = (
                result_to_yolo_seg(
                    result
                )
            )

        # ====================================================
        # 保存Label
        # ====================================================

        label_path.write_text(
            label_text,
            encoding="utf-8",
        )

        # ====================================================
        # 数实例
        # ====================================================

        instance_count = sum(
            1
            for line
            in label_text.splitlines()
            if line.strip()
        )

        total_instances += (
            instance_count
        )

        if (
            scene != "empty"
            and
            instance_count == 0
        ):

            miss_count += 1

            print(
                "[MISS WARNING] "
                f"{relative.as_posix()}"
            )

        else:

            print(
                f"[{index:04d}/{len(images):04d}] "
                f"{relative.as_posix()} "
                f"instances={instance_count}"
            )

        # ====================================================
        # Preview
        # ====================================================

        if not args.no_preview:

            preview = (
                result.plot()
            )

            imwrite_unicode(
                preview_path,
                preview,
            )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "============================================================"
    )

    print(
        "V3 AUTO PRE-ANNOTATION FINISHED"
    )

    print(
        "============================================================"
    )

    print(
        f"Images                : {len(images)}"
    )

    print(
        f"Skipped               : {skipped_count}"
    )

    print(
        f"Auto instances        : {total_instances}"
    )

    print(
        f"Non-empty misses      : {miss_count}"
    )

    print(
        f"Empty false positives : "
        f"{empty_false_positive_count}"
    )

    print()

    print(
        f"Labels : {LABEL_ROOT}"
    )

    print(
        f"Preview: {PREVIEW_ROOT}"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":

    main()