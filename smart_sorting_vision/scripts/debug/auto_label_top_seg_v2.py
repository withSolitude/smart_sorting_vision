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

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# V1正式模型
# ============================================================

DEFAULT_MODEL_PATH = (
    ROOT
    / "smart_sorting_vision"
    / "weights"
    / "top_seg"
    / "v1"
    / "best.pt"
)


# ============================================================
# V2新采图片
# ============================================================

DEFAULT_IMAGE_ROOT = (
    ROOT
    / "smart_sorting_vision"
    / "data"
    / "datasets"
    / "top_seg"
    / "new_images"
)


# ============================================================
# V1自动生成的预标注
#
# 注意：
# 这里只是机器预标注。
# 还不是最终Ground Truth。
# ============================================================

DEFAULT_LABEL_ROOT = (
    ROOT
    / "smart_sorting_vision"
    / "data"
    / "datasets"
    / "top_seg"
    / "auto_labels"
)


# ============================================================
# 自动标注预览图
# ============================================================

DEFAULT_PREVIEW_ROOT = (
    ROOT
    / "smart_sorting_vision"
    / "outputs"
    / "debug"
    / "top_seg_auto_label"
    / "v2"
)


# ============================================================
# 正式图像尺寸
#
# TopCamera.read()
# -> 870 x 840
# ============================================================

EXPECTED_WIDTH = 870
EXPECTED_HEIGHT = 840


# ============================================================
# 推理参数
#
# 自动预标注阶段：
#
# conf可以比正式部署稍低。
#
# 原因：
#
# 漏检一个物体
# -> 需要人工重新画整个Polygon
#
# 多出来一个误检
# -> CVAT里面删掉即可
#
# 所以预标注更倾向于提高Recall。
# ============================================================

DEFAULT_CONF = 0.35

DEFAULT_IMGSZ = 832


# ============================================================
# 支持的图片格式
# ============================================================

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


# ============================================================
# 查找全部图片
#
# 支持：
#
# new_images/
# ├── single/
# │   └── black/
# │       └── 圆柱/
# │           └── top_v2_000001.jpg
# ├── contact/
# └── ...
#
# 中文目录完全没问题。
# ============================================================

def find_images(
    image_root: Path,
) -> list[Path]:

    images = []

    for path in image_root.rglob("*"):

        if not path.is_file():

            continue

        if (
            path.suffix.lower()
            not in IMAGE_SUFFIXES
        ):

            continue

        images.append(
            path
        )

    images.sort(
        key=lambda path:
            path.as_posix()
    )

    return images


# ============================================================
# 镜像输出路径
#
# 例如：
#
# new_images/
# single/black/圆柱/top_v2_000001.jpg
#
# ->
#
# auto_labels/
# single/black/圆柱/top_v2_000001.txt
# ============================================================

def get_label_path(
    image_path: Path,
    image_root: Path,
    label_root: Path,
) -> Path:

    relative_path = (
        image_path.relative_to(
            image_root
        )
    )

    return (
        label_root
        / relative_path
    ).with_suffix(
        ".txt"
    )


# ============================================================
# 预览输出路径
# ============================================================

def get_preview_path(
    image_path: Path,
    image_root: Path,
    preview_root: Path,
) -> Path:

    relative_path = (
        image_path.relative_to(
            image_root
        )
    )

    return (
        preview_root
        / relative_path
    ).with_suffix(
        ".jpg"
    )


# ============================================================
# 检查输入图片
# ============================================================

def validate_image(
    image_path: Path,
) -> tuple[int, int]:

    image = cv2.imread(
        str(
            image_path
        ),
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise RuntimeError(
            f"OpenCV无法读取：\n"
            f"{image_path}"
        )

    height, width = (
        image.shape[:2]
    )

    if (
        width != EXPECTED_WIDTH
        or
        height != EXPECTED_HEIGHT
    ):

        raise RuntimeError(
            f"图片尺寸错误：\n"
            f"{image_path}\n"
            f"实际：{width}x{height}\n"
            f"期望："
            f"{EXPECTED_WIDTH}x"
            f"{EXPECTED_HEIGHT}"
        )

    return (
        width,
        height,
    )


# ============================================================
# 将一个YOLO-seg预测结果写成标签
#
# YOLO segmentation格式：
#
# class_id x1 y1 x2 y2 x3 y3 ...
#
# 所有坐标归一化到0~1。
#
# 这里不保存confidence。
#
# 最终仍然只有一个类别：
#
# 0 = object
# ============================================================

def result_to_yolo_seg_lines(
    result,
) -> tuple[
    list[str],
    list[str],
]:

    lines = []

    warnings = []

    # ========================================================
    # 一个目标都没检测到
    # ========================================================

    if (
        result.masks is None
        or
        result.boxes is None
        or
        len(result.boxes) == 0
    ):

        return (
            lines,
            warnings,
        )

    # ========================================================
    # normalized Polygon
    #
    # 每个实例一个 ndarray:
    #
    # [
    #   [x1, y1],
    #   [x2, y2],
    #   ...
    # ]
    #
    # 坐标已经是0~1。
    # ========================================================

    polygons = (
        result.masks.xyn
    )

    class_ids = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.int32
        )
    )

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    if (
        len(polygons)
        !=
        len(class_ids)
    ):

        raise RuntimeError(
            "YOLO返回的Mask数量和Box数量不一致。"
        )

    # ========================================================
    # 每个实例
    # ========================================================

    for index, (
        polygon,
        class_id,
        confidence,
    ) in enumerate(
        zip(
            polygons,
            class_ids,
            confidences,
        ),
        start=1,
    ):

        # ----------------------------------------------------
        # 当前模型理论上只有class 0
        # ----------------------------------------------------

        if class_id != 0:

            warnings.append(
                f"instance#{index}: "
                f"出现非0类别 "
                f"class={class_id}"
            )

        polygon = np.asarray(
            polygon,
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Polygon必须是 Nx2
        # ----------------------------------------------------

        if (
            polygon.ndim != 2
            or
            polygon.shape[1] != 2
        ):

            warnings.append(
                f"instance#{index}: "
                "Polygon形状异常，跳过。"
            )

            continue

        # ----------------------------------------------------
        # 至少三个点
        # ----------------------------------------------------

        if len(polygon) < 3:

            warnings.append(
                f"instance#{index}: "
                "Polygon少于3个点，跳过。"
            )

            continue

        # ----------------------------------------------------
        # NaN / Inf检查
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(
                polygon
            )
        ):

            warnings.append(
                f"instance#{index}: "
                "存在NaN/Inf，跳过。"
            )

            continue

        # ----------------------------------------------------
        # 理论上xyn已经在0~1。
        #
        # 做一次clip避免极小浮点误差。
        # ----------------------------------------------------

        polygon = np.clip(
            polygon,
            0.0,
            1.0,
        )

        # ====================================================
        # 所有目标统一写class=0
        #
        # 因为顶部相机任务只有：
        #
        # object
        # ====================================================

        values = [
            "0"
        ]

        for x, y in polygon:

            values.append(
                f"{x:.6f}"
            )

            values.append(
                f"{y:.6f}"
            )

        line = " ".join(
            values
        )

        lines.append(
            line
        )

        print(
            f"      instance#{index}: "
            f"conf={confidence:.3f}, "
            f"points={len(polygon)}"
        )

    return (
        lines,
        warnings,
    )


# ============================================================
# 写Label
#
# 即使没有预测到物体，
# 也建立一个空txt。
#
# 这样以后能区分：
#
# 空label
# =
# 该图片已经被程序处理过
#
# 而不是：
#
# 没有txt
# =
# 程序漏处理
# ============================================================

def write_label(
    label_path: Path,
    lines: list[str],
) -> None:

    label_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if lines:

        content = (
            "\n".join(
                lines
            )
            +
            "\n"
        )

    else:

        content = ""

    label_path.write_text(
        content,
        encoding="utf-8",
    )


# ============================================================
# 保存预测可视化
# ============================================================

def save_preview(
    result,
    preview_path: Path,
) -> None:

    preview_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Ultralytics自动把：
    #
    # mask
    # box
    # label
    # confidence
    #
    # 画回原图
    annotated = (
        result.plot()
    )

    success = cv2.imwrite(
        str(
            preview_path
        ),
        annotated,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92,
        ],
    )

    if not success:

        raise RuntimeError(
            f"预览图保存失败：\n"
            f"{preview_path}"
        )


# ============================================================
# 自动选择设备
# ============================================================

def select_device(
    requested_device: str | None,
):

    if requested_device:

        return requested_device

    if torch.cuda.is_available():

        return 0

    return "cpu"


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Use V1 YOLO-seg best.pt "
            "to pre-annotate V2 top-camera images."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABEL_ROOT,
    )

    parser.add_argument(
        "--previews",
        type=Path,
        default=DEFAULT_PREVIEW_ROOT,
    )

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
        type=str,
        default=None,
        help=(
            "例如 0 或 cpu。"
            "默认自动检测CUDA。"
        ),
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="不保存预测预览图。",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的自动标签。",
    )

    args = parser.parse_args()

    model_path = (
        args.model
        .expanduser()
        .resolve()
    )

    image_root = (
        args.images
        .expanduser()
        .resolve()
    )

    label_root = (
        args.labels
        .expanduser()
        .resolve()
    )

    preview_root = (
        args.previews
        .expanduser()
        .resolve()
    )

    # ========================================================
    # 基础检查
    # ========================================================

    if not model_path.exists():

        raise FileNotFoundError(
            f"找不到V1模型：\n"
            f"{model_path}"
        )

    if not image_root.exists():

        raise FileNotFoundError(
            f"找不到new_images：\n"
            f"{image_root}"
        )

    # ========================================================
    # 查找图片
    # ========================================================

    image_paths = (
        find_images(
            image_root
        )
    )

    if not image_paths:

        raise RuntimeError(
            f"没有找到任何图片：\n"
            f"{image_root}"
        )

    # ========================================================
    # 设备
    # ========================================================

    device = (
        select_device(
            args.device
        )
    )

    # ========================================================
    # 加载模型
    # ========================================================

    print(
        "============================================================"
    )

    print(
        "TOP SEG V2 AUTO ANNOTATION"
    )

    print(
        "============================================================"
    )

    print(
        f"Model:\n"
        f"{model_path}"
    )

    print()

    print(
        f"Images:\n"
        f"{image_root}"
    )

    print()

    print(
        f"Auto labels:\n"
        f"{label_root}"
    )

    if not args.no_preview:

        print()

        print(
            f"Previews:\n"
            f"{preview_root}"
        )

    print()

    print(
        f"Images found : "
        f"{len(image_paths)}"
    )

    print(
        f"conf         : "
        f"{args.conf}"
    )

    print(
        f"imgsz        : "
        f"{args.imgsz}"
    )

    print(
        f"device       : "
        f"{device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "============================================================"
    )

    model = YOLO(
        str(
            model_path
        )
    )

    # ========================================================
    # 统计
    # ========================================================

    processed_count = 0

    skipped_count = 0

    empty_prediction_count = 0

    total_instances = 0

    warning_count = 0

    # ========================================================
    # 逐张处理
    #
    # 不使用result.save_txt()。
    #
    # 原因：
    # Ultralytics当前save_txt对已有文件采用追加行为，
    # 重复执行时有可能产生重复标注。
    #
    # 这里由我们自己明确覆盖/跳过，更安全。
    # ========================================================

    for image_number, image_path in enumerate(
        image_paths,
        start=1,
    ):

        relative_path = (
            image_path.relative_to(
                image_root
            )
        )

        label_path = (
            get_label_path(
                image_path=
                    image_path,

                image_root=
                    image_root,

                label_root=
                    label_root,
            )
        )

        preview_path = (
            get_preview_path(
                image_path=
                    image_path,

                image_root=
                    image_root,

                preview_root=
                    preview_root,
            )
        )

        print()

        print(
            "------------------------------------------------------------"
        )

        print(
            f"[{image_number}/"
            f"{len(image_paths)}] "
            f"{relative_path}"
        )

        # ====================================================
        # 已有Label默认跳过
        # ====================================================

        if (
            label_path.exists()
            and
            not args.overwrite
        ):

            print(
                "    [SKIP] "
                "auto label已经存在"
            )

            skipped_count += 1

            continue

        # ====================================================
        # 图片验证
        # ====================================================

        validate_image(
            image_path
        )

        # ====================================================
        # YOLO预测
        # ====================================================

        results = model.predict(

            source=str(
                image_path
            ),

            imgsz=args.imgsz,

            conf=args.conf,

            device=device,

            verbose=False,
        )

        if len(results) != 1:

            raise RuntimeError(
                f"{image_path.name}: "
                "YOLO返回结果数量异常。"
            )

        result = (
            results[0]
        )

        # ====================================================
        # 转YOLO segmentation label
        # ====================================================

        (
            lines,
            warnings,
        ) = (
            result_to_yolo_seg_lines(
                result
            )
        )

        # ====================================================
        # 保存标签
        # ====================================================

        write_label(
            label_path=
                label_path,

            lines=
                lines,
        )

        instance_count = (
            len(lines)
        )

        total_instances += (
            instance_count
        )

        if instance_count == 0:

            empty_prediction_count += 1

        print(
            f"    predicted objects: "
            f"{instance_count}"
        )

        print(
            f"    label: "
            f"{label_path}"
        )

        # ====================================================
        # Warning
        # ====================================================

        for warning in warnings:

            print(
                f"    [WARNING] "
                f"{warning}"
            )

            warning_count += 1

        # ====================================================
        # 保存预览
        # ====================================================

        if not args.no_preview:

            save_preview(
                result=
                    result,

                preview_path=
                    preview_path,
            )

            print(
                f"    preview: "
                f"{preview_path}"
            )

        processed_count += 1

    # ========================================================
    # 完成
    # ========================================================

    print()

    print(
        "============================================================"
    )

    print(
        "AUTO ANNOTATION FINISHED"
    )

    print(
        "============================================================"
    )

    print(
        f"Total images        : "
        f"{len(image_paths)}"
    )

    print(
        f"Processed           : "
        f"{processed_count}"
    )

    print(
        f"Skipped             : "
        f"{skipped_count}"
    )

    print(
        f"Predicted instances : "
        f"{total_instances}"
    )

    print(
        f"Empty predictions   : "
        f"{empty_prediction_count}"
    )

    print(
        f"Warnings            : "
        f"{warning_count}"
    )

    print()

    print(
        "Auto labels:"
    )

    print(
        label_root
    )

    if not args.no_preview:

        print()

        print(
            "Prediction previews:"
        )

        print(
            preview_root
        )

    print(
        "============================================================"
    )

    print()

    print(
        "注意：auto_labels只是V1模型预标注，"
        "不能直接作为V2最终Ground Truth。"
    )

    print(
        "下一步必须人工检查/修正。"
    )


if __name__ == "__main__":

    main()