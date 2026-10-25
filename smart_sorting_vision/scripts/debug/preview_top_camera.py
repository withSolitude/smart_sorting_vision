import cv2


DEVICE = "/dev/video0"


def main():

    cap = cv2.VideoCapture(
        DEVICE,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开相机：{DEVICE}"
        )

    # 这里只设置视频流格式
    # 不设置曝光、白平衡等图像参数

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1920,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        1080,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30,
    )

    print("============================")
    print("纯画面预览")
    print("不会修改曝光/白平衡等参数")
    print("Q：退出")
    print("============================")

    try:

        while True:

            ret, frame = cap.read()

            if not ret or frame is None:
                print("读取图像失败")
                continue

            cv2.imshow(
                "Top Camera Preview",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()