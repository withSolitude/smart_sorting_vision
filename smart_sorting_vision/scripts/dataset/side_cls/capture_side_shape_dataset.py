from __future__ import annotations

import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from vision.cameras.side_camera import SideCamera
from vision.processors.side_object_extractor import (
    SideObjectExtractor,
)


DATASET_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "side_cls"
    / "v1"
)


BACKGROUND_PATH = (
    ROOT
    / "data"
    / "calibration"
    / "side_camera"
    / "side_mechanical_roi_background.png"
)


def choose_directory(
    parent: Path,
    title: str,
) -> Path:
    """
    在终端中选择 parent 下的一个子目录。
    """

    directories = sorted(
        [
            path
            for path in parent.iterdir()
            if path.is_dir()
        ],
        key=lambda path: path.name,
    )

    if not directories:

        raise RuntimeError(
            f"目录下没有可选子目录：{parent}"
        )

    print()
    print(
        "=" * 64
    )

    print(
        title
    )

    print(
        "=" * 64
    )

    for index, directory in enumerate(
        directories,
        start=1,
    ):

        print(
            f"{index:>2}. {directory.name}"
        )

    while True:

        value = input(
            "请输入序号："
        ).strip()

        try:

            index = int(
                value
            )

        except ValueError:

            print(
                "请输入数字序号。"
            )

            continue

        if not (
            1
            <= index
            <= len(
                directories
            )
        ):

            print(
                "序号超出范围。"
            )

            continue

        return directories[
            index - 1
        ]


def choose_save_directory() -> Path:
    """
    按现有 datasets/side_cls 目录逐级选择：

        split
        -> shape
        -> color（如果存在）

    如果 shape 目录下面没有颜色子目录，
    就直接保存到 shape 目录。
    """

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"数据集目录不存在：{DATASET_ROOT}"
        )

    split_dir = choose_directory(
        DATASET_ROOT,
        "选择数据集：train / val / test",
    )

    shape_dir = choose_directory(
        split_dir,
        f"选择形状目录：{split_dir.name}",
    )

    color_dirs = sorted(
        [
            path
            for path in shape_dir.iterdir()
            if path.is_dir()
        ],
        key=lambda path: path.name,
    )

    if color_dirs:

        save_dir = choose_directory(
            shape_dir,
            f"选择颜色目录：{shape_dir.name}",
        )

    else:

        save_dir = shape_dir

    print()
    print(
        "=" * 64
    )

    print(
        "保存目录："
    )

    print(
        save_dir
    )

    print(
        "=" * 64
    )

    return save_dir


def next_image_index(
    save_dir: Path,
) -> int:
    """
    获取下一个图片编号。

    文件名格式：
        image_000001.jpg
        image_000002.jpg
        ...
    """

    maximum_index = 0

    for path in save_dir.glob(
        "image_*.jpg"
    ):

        stem = path.stem

        try:

            index = int(
                stem.split(
                    "_"
                )[-1]
            )

        except ValueError:

            continue

        maximum_index = max(
            maximum_index,
            index,
        )

    return maximum_index + 1


def main() -> None:

    save_dir = (
        choose_save_directory()
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_index = next_image_index(
        save_dir
    )

    camera = SideCamera()

    extractor = SideObjectExtractor(
        padding_ratio=0.15
    )

    try:

        camera.open()

        if not BACKGROUND_PATH.exists():

            raise FileNotFoundError(
                "\n固定背景图不存在："
                f"\n{BACKGROUND_PATH}"
                "\n\n请先运行："
                "\npython scripts/capture_side_fixed_background.py"
            )

        background = (
            extractor.load_background(
                BACKGROUND_PATH
            )
        )

        print(
            f"[BACKGROUND] loaded: "
            f"{BACKGROUND_PATH}"
        )

        # 相机启动后先丢弃几帧。
        current_frame = None

        for _ in range(
            20
        ):

            current_frame = (
                camera.read()
            )

        if current_frame is None:

            raise RuntimeError(
                "无法读取侧面相机画面"
            )

        if (
            current_frame.shape
            !=
            background.shape
        ):

            raise RuntimeError(
                "\n当前相机 ROI 与固定背景尺寸不一致。"
                f"\ncurrent    = {current_frame.shape}"
                f"\nbackground = {background.shape}"
                "\n请重新采集固定背景。"
            )

        print()
        print(
            "=" * 64
        )

        print(
            "SIDE SHAPE DATASET CAPTURE"
        )

        print(
            "=" * 64
        )

        print(
            f"Save dir : {save_dir}"
        )

        print(
            "S        : 保存当前物体 crop"
        )

        print(
            "Q / ESC  : 退出"
        )

        print(
            "=" * 64
        )

        while True:

            mechanical_roi = (
                camera.read()
            )

            object_result = (
                extractor.extract(
                    mechanical_roi
                )
            )

            display = (
                mechanical_roi.copy()
            )

            crop = None

            if (
                object_result.found
                and
                object_result.crop
                is not None
            ):

                crop = (
                    object_result.crop
                )

                crop_bbox = getattr(
                    object_result,
                    "crop_bbox",
                    None,
                )

                if crop_bbox is not None:

                    x1, y1, x2, y2 = (
                        crop_bbox
                    )

                    cv2.rectangle(
                        display,
                        (
                            int(
                                x1
                            ),
                            int(
                                y1
                            ),
                        ),
                        (
                            int(
                                x2
                            ),
                            int(
                                y2
                            ),
                        ),
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    display,
                    "Object: YES",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    "Side Shape Crop",
                    crop,
                )

            else:

                cv2.putText(
                    display,
                    "Object: NO",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display,
                (
                    f"Next image: "
                    f"{image_index:06d}"
                ),
                (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "S=save   Q=quit",
                (
                    16,
                    display.shape[0]
                    -
                    16,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Side Shape Dataset Capture",
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
                ord("s"),
                ord("S"),
            ):

                if crop is None:

                    print(
                        "[SAVE] 当前没有检测到物体，未保存。"
                    )

                    continue

                save_path = (
                    save_dir
                    /
                    f"image_{image_index:06d}.jpg"
                )

                success = cv2.imwrite(
                    str(
                        save_path
                    ),
                    crop,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        95,
                    ],
                )

                if not success:

                    print(
                        f"[SAVE ERROR] "
                        f"{save_path}"
                    )

                    continue

                print(
                    f"[SAVE] "
                    f"{save_path.relative_to(ROOT)}"
                )

                image_index += 1

    finally:

        camera.close()

        cv2.destroyAllWindows()

        print()
        print(
            "Shape dataset capture finished."
        )


if __name__ == "__main__":
    main()
