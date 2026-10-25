from datetime import datetime
from pathlib import Path

import cv2

from vision.cameras.side_camera import SideCamera


ROOT = Path(
    __file__
).resolve().parents[1]


SAVE_DIR = (
    ROOT
    / "outputs"
    / "debug"
    / "side_camera"
)


def main() -> None:

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera = SideCamera()

    try:

        # ======================================================
        # 打开侧面相机
        # ======================================================

        camera.open()

        # ======================================================
        # 获取固定参数
        # ======================================================

        info = (
            camera.get_stream_info()
        )

        # ======================================================
        # 打印
        # ======================================================

        print(
            "=" * 60
        )

        print(
            "SIDE CAMERA"
        )

        print(
            "=" * 60
        )

        print(
            f"device      : "
            f"{info['device']}"
        )

        print(
            f"resolution  : "
            f"{info['width']} x "
            f"{info['height']}"
        )

        print(
            f"fps         : "
            f"{info['fps']:.2f}"
        )

        print(
            f"fourcc      : "
            f"{info['fourcc']}"
        )

        print(
            f"exposure    : "
            f"{info['exposure']}"
        )

        print(
            f"gain        : "
            f"{info['gain']}"
        )

        print(
            f"white bal.  : "
            f"{info['white_balance_temperature']} K"
        )

        print(
            f"brightness  : "
            f"{info['brightness']}"
        )

        print(
            f"contrast    : "
            f"{info['contrast']}"
        )

        print(
            f"saturation  : "
            f"{info['saturation']}"
        )

        print(
            f"gamma       : "
            f"{info['gamma']}"
        )

        print(
            f"sharpness   : "
            f"{info['sharpness']}"
        )

        print(
            f"backlight   : "
            f"{info['backlight_compensation']}"
        )

        print(
            "=" * 60
        )

        print(
            "Q / ESC : 退出"
        )

        print(
            "S       : 保存原始图片"
        )

        print(
            "=" * 60
        )

        # ======================================================
        # 创建窗口
        # ======================================================

        cv2.namedWindow(
            "Side Camera",
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            "Side Camera",
            1280,
            720,
        )

        # ======================================================
        # 主循环
        # ======================================================

        while True:

            frame = camera.read()

            cv2.imshow(
                "Side Camera",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ==================================================
            # 退出
            # ==================================================

            if key in (
                ord("q"),
                27,
            ):

                break

            # ==================================================
            # 保存
            # ==================================================

            if key == ord("s"):

                timestamp = (
                    datetime
                    .now()
                    .strftime(
                        "%Y%m%d_%H%M%S_%f"
                    )
                )

                save_path = (
                    SAVE_DIR
                    / (
                        "side_camera_"
                        f"{timestamp}.jpg"
                    )
                )

                success = cv2.imwrite(
                    str(save_path),
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        95,
                    ],
                )

                if success:

                    print(
                        f"[SAVE] {save_path}"
                    )

                else:

                    print(
                        f"[ERROR] "
                        f"保存失败：{save_path}"
                    )

    finally:

        camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()