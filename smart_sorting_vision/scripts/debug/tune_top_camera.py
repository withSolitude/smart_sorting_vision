import cv2
import subprocess
from pathlib import Path


DEVICE = "/dev/video0"

WIDTH = 1920
HEIGHT = 1080
FPS = 30


ROOT = Path(__file__).resolve().parents[1]

SAVE_DIR = (
    ROOT
    / "outputs"
    / "debug"
    / "top_camera_tuning"
)


# ==========================================================
# 初始参数
# ==========================================================

exposure = 312
white_balance = 4600

brightness = 0
contrast = 35
saturation = 60

gamma = 100
sharpness = 2


# ==========================================================
# V4L2控制
# ==========================================================

def set_ctrl(name, value):

    command = [
        "v4l2-ctl",
        "-d",
        DEVICE,
        "--set-ctrl",
        f"{name}={value}",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        print(
            f"[ERROR] {name}={value}"
        )

        print(
            result.stderr.strip()
        )

        return False

    return True


# ==========================================================
# 应用当前参数
# ==========================================================

def apply_controls():

    # 自动对焦关闭
    set_ctrl(
        "focus_automatic_continuous",
        0,
    )

    # 自动曝光关闭
    # 1 = Manual Mode
    set_ctrl(
        "auto_exposure",
        1,
    )

    # 曝光
    set_ctrl(
        "exposure_time_absolute",
        exposure,
    )

    # 自动白平衡关闭
    set_ctrl(
        "white_balance_automatic",
        0,
    )

    # 白平衡
    set_ctrl(
        "white_balance_temperature",
        white_balance,
    )

    # 其他参数
    set_ctrl(
        "brightness",
        brightness,
    )

    set_ctrl(
        "contrast",
        contrast,
    )

    set_ctrl(
        "saturation",
        saturation,
    )

    set_ctrl(
        "gamma",
        gamma,
    )

    set_ctrl(
        "sharpness",
        sharpness,
    )


# ==========================================================
# 打印当前参数
# ==========================================================

def print_parameters():

    print()
    print("=" * 50)

    print(
        f"Exposure   = {exposure}"
    )

    print(
        f"WB         = {white_balance}"
    )

    print(
        f"Brightness = {brightness}"
    )

    print(
        f"Contrast   = {contrast}"
    )

    print(
        f"Saturation = {saturation}"
    )

    print(
        f"Gamma      = {gamma}"
    )

    print(
        f"Sharpness  = {sharpness}"
    )

    print("=" * 50)
    print()


# ==========================================================
# 主程序
# ==========================================================

def main():

    global exposure
    global white_balance

    global brightness
    global contrast
    global saturation

    global gamma
    global sharpness


    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ======================================================
    # 先配置视频格式
    # ======================================================

    subprocess.run(
        [
            "v4l2-ctl",
            "-d",
            DEVICE,
            "--set-fmt-video="
            "width=1920,"
            "height=1080,"
            "pixelformat=MJPG",
        ]
    )

    subprocess.run(
        [
            "v4l2-ctl",
            "-d",
            DEVICE,
            "--set-parm",
            "30",
        ]
    )


    # ======================================================
    # 打开相机
    # ======================================================

    cap = cv2.VideoCapture(
        DEVICE,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():

        raise RuntimeError(
            "无法打开顶部相机"
        )


    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FPS,
    )


    # ======================================================
    # 应用初始值
    # ======================================================

    apply_controls()


    print()
    print("=" * 60)
    print("顶部相机参数调试")
    print("=" * 60)

    print("Q：退出")
    print("S：保存图片")
    print("P：打印最终参数")

    print()
    print("曝光：")
    print("E：+20")
    print("D：-20")

    print()
    print("白平衡：")
    print("W：+100 K")
    print("X：-100 K")

    print()
    print("亮度：")
    print("B：+1")
    print("N：-1")

    print()
    print("对比度：")
    print("C：+1")
    print("V：-1")

    print()
    print("饱和度：")
    print("R：+1")
    print("F：-1")

    print()
    print("Gamma：")
    print("G：+5")
    print("H：-5")

    print()
    print("锐度：")
    print("K：+1")
    print("J：-1")

    print("=" * 60)


    image_index = 0


    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "读取相机失败"
                )

                continue


            display = frame.copy()


            # ==================================================
            # 显示参数
            # ==================================================

            cv2.putText(
                display,
                f"EXP : {exposure}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display,
                f"WB  : {white_balance} K",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display,
                f"B/C/S : "
                f"{brightness} / "
                f"{contrast} / "
                f"{saturation}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display,
                f"Gamma : {gamma}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display,
                f"Sharp : {sharpness}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )


            cv2.imshow(
                "Top Camera Tuning",
                display,
            )


            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            # ==================================================
            # 退出
            # ==================================================

            if key == ord("q"):

                break


            # ==================================================
            # 保存
            # ==================================================

            elif key == ord("s"):

                filename = (
                    SAVE_DIR
                    / f"tune_{image_index:04d}.jpg"
                )

                cv2.imwrite(
                    str(filename),
                    frame,
                )

                print(
                    f"保存：{filename}"
                )

                image_index += 1


            # ==================================================
            # 打印最终参数
            # ==================================================

            elif key == ord("p"):

                print_parameters()


            # ==================================================
            # Exposure
            # ==================================================

            elif key == ord("e"):

                exposure = min(
                    10000,
                    exposure + 20,
                )

                set_ctrl(
                    "exposure_time_absolute",
                    exposure,
                )


            elif key == ord("d"):

                exposure = max(
                    78,
                    exposure - 20,
                )

                set_ctrl(
                    "exposure_time_absolute",
                    exposure,
                )


            # ==================================================
            # White Balance
            # ==================================================

            elif key == ord("w"):

                white_balance = min(
                    12000,
                    white_balance + 100,
                )

                set_ctrl(
                    "white_balance_temperature",
                    white_balance,
                )


            elif key == ord("x"):

                white_balance = max(
                    10,
                    white_balance - 100,
                )

                set_ctrl(
                    "white_balance_temperature",
                    white_balance,
                )


            # ==================================================
            # Brightness
            # ==================================================

            elif key == ord("b"):

                brightness = min(
                    64,
                    brightness + 1,
                )

                set_ctrl(
                    "brightness",
                    brightness,
                )


            elif key == ord("n"):

                brightness = max(
                    -64,
                    brightness - 1,
                )

                set_ctrl(
                    "brightness",
                    brightness,
                )


            # ==================================================
            # Contrast
            # ==================================================

            elif key == ord("c"):

                contrast = min(
                    95,
                    contrast + 1,
                )

                set_ctrl(
                    "contrast",
                    contrast,
                )


            elif key == ord("v"):

                contrast = max(
                    0,
                    contrast - 1,
                )

                set_ctrl(
                    "contrast",
                    contrast,
                )


            # ==================================================
            # Saturation
            # ==================================================

            elif key == ord("r"):

                saturation = min(
                    100,
                    saturation + 1,
                )

                set_ctrl(
                    "saturation",
                    saturation,
                )


            elif key == ord("f"):

                saturation = max(
                    0,
                    saturation - 1,
                )

                set_ctrl(
                    "saturation",
                    saturation,
                )


            # ==================================================
            # Gamma
            # ==================================================

            elif key == ord("g"):

                gamma = min(
                    300,
                    gamma + 5,
                )

                set_ctrl(
                    "gamma",
                    gamma,
                )


            elif key == ord("h"):

                gamma = max(
                    100,
                    gamma - 5,
                )

                set_ctrl(
                    "gamma",
                    gamma,
                )


            # ==================================================
            # Sharpness
            # ==================================================

            elif key == ord("k"):

                sharpness = min(
                    7,
                    sharpness + 1,
                )

                set_ctrl(
                    "sharpness",
                    sharpness,
                )


            elif key == ord("j"):

                sharpness = max(
                    1,
                    sharpness - 1,
                )

                set_ctrl(
                    "sharpness",
                    sharpness,
                )


    finally:

        print_parameters()

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()