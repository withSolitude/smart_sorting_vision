from __future__ import annotations

import subprocess
import time
from typing import Optional

import cv2
import numpy as np


class SideCamera:
    """
    智能分拣系统侧面相机统一驱动。

    ============================================================
    当前硬件
    ============================================================

    Camera:
        icspring camera

    Video Capture Device:
        /dev/video0

    Format:
        MJPG

    Resolution:
        1280 x 720

    FPS:
        30

    ============================================================
    使用原则
    ============================================================

    所有侧面视觉程序统一使用：

        camera = SideCamera()
        camera.open()
        frame = camera.read()

    其中：

        read_full()
            返回完整 1280×720 图像

        read()
            返回裁剪后的左侧长条托盘图像

    正式识别程序统一使用：

        read()

    只有：
        调整相机
        调整 ROI
        调试完整画面

    时才使用：

        read_full()

    ============================================================
    相机参数
    ============================================================

    自动曝光：
        OFF

    曝光：
        100

    Gain：
        0

    自动白平衡：
        OFF

    白平衡：
        4800 K

    其它 ISP：
        全部固定

    ============================================================
    ROI
    ============================================================

    原始图像：
        1280 × 720

    只保留左侧长条托盘。

    当前 ROI：

        x = 220 ~ 745
        y = 105 ~ 720

    得到：

        width  = 525
        height = 615

    后续如果需要微调托盘范围，
    只修改下面四个 ROI 参数。
    """

    # ==========================================================
    # 1. 相机设备
    # ==========================================================

    DEVICE = (
        "/dev/v4l/by-id/"
        "usb-icSpring_icspring_camera-video-index0"
    )

    # ==========================================================
    # 2. 原始视频参数
    # ==========================================================

    WIDTH = 1280
    HEIGHT = 720
    FPS = 30

    FOURCC = "MJPG"

    BUFFER_SIZE = 1

    # ==========================================================
    # 3. 固定 ROI
    # ==========================================================

    # ROI_X1 = 220
    # ROI_Y1 = 105

    # ROI_X2 = 785   #745
    # ROI_Y2 = 720

    ROI_X1 = 30    #30
    ROI_Y1 = 0    #10

    ROI_X2 = 805  #745
    ROI_Y2 = 720
    
    ROI_WIDTH = ROI_X2 - ROI_X1
    ROI_HEIGHT = ROI_Y2 - ROI_Y1

    # ==========================================================
    # 4. 曝光
    # ==========================================================
    # 1 = Manual Mode
    # 3 = Aperture Priority Mode

    AUTO_EXPOSURE = 1

    EXPOSURE_TIME_ABSOLUTE = 300           #100

    # ==========================================================
    # 5. Gain
    # ==========================================================

    # 有效范围：
    # 0 ~ 63

    GAIN = 0

    # ==========================================================
    # 6. 白平衡
    # ==========================================================

    WHITE_BALANCE_AUTOMATIC = 0

    # 根据实际画面对比：
    #
    # 4700：略冷
    # 4800：最中性
    # 4900：略暖
    #
    # 因此固定 4800 K

    WHITE_BALANCE_TEMPERATURE = 4850

    # ==========================================================
    # 7. 基础 ISP
    # ==========================================================

    BRIGHTNESS = 0

    CONTRAST = 38

    SATURATION = 65

    HUE = 0

    GAMMA = 100

    # ==========================================================
    # 8. 防频闪
    # ==========================================================

    # 0 = Disabled
    # 1 = 50 Hz
    # 2 = 60 Hz

    POWER_LINE_FREQUENCY = 1

    # ==========================================================
    # 9. 锐度
    # ==========================================================

    SHARPNESS = 0

    # ==========================================================
    # 10. 背光补偿
    # ==========================================================

    BACKLIGHT_COMPENSATION = 0

    # ==========================================================
    # 11. 启动预热
    # ==========================================================

    WARMUP_FRAMES = 10

    # 运行中 USB 相机掉线时的重连节流周期，避免每帧重复打开设备。
    RECONNECT_INTERVAL_S = 0.5

    # ==========================================================
    # 12. 运行时配置入口
    # ==========================================================

    @classmethod
    def configure(
        cls,
        config: dict,
    ) -> None:
        """Apply the production side-camera profile from runtime_config.yaml."""
        cls.DEVICE = str(config["device"])
        cls.WIDTH = int(config.get("width", cls.WIDTH))
        cls.HEIGHT = int(config.get("height", cls.HEIGHT))
        cls.FPS = int(config.get("fps", cls.FPS))
        cls.FOURCC = str(config.get("fourcc", cls.FOURCC))
        cls.BUFFER_SIZE = int(config.get("buffer_size", cls.BUFFER_SIZE))

        roi = config.get("roi")
        if roi is not None:
            if not isinstance(roi, (list, tuple)) or len(roi) != 4:
                raise ValueError("camera.side.roi 必须是 [x1, y1, x2, y2]")
            cls.ROI_X1, cls.ROI_Y1, cls.ROI_X2, cls.ROI_Y2 = map(int, roi)
            cls.ROI_WIDTH = cls.ROI_X2 - cls.ROI_X1
            cls.ROI_HEIGHT = cls.ROI_Y2 - cls.ROI_Y1

        if not (0 <= cls.ROI_X1 < cls.ROI_X2 <= cls.WIDTH):
            raise ValueError("camera.side.roi 的 X 范围超出图像")
        if not (0 <= cls.ROI_Y1 < cls.ROI_Y2 <= cls.HEIGHT):
            raise ValueError("camera.side.roi 的 Y 范围超出图像")

        controls = config.get("controls", {})
        mapping = {
            "auto_exposure": "AUTO_EXPOSURE",
            "exposure_time_absolute": "EXPOSURE_TIME_ABSOLUTE",
            "gain": "GAIN",
            "white_balance_automatic": "WHITE_BALANCE_AUTOMATIC",
            "white_balance_temperature": "WHITE_BALANCE_TEMPERATURE",
            "brightness": "BRIGHTNESS",
            "contrast": "CONTRAST",
            "saturation": "SATURATION",
            "hue": "HUE",
            "gamma": "GAMMA",
            "power_line_frequency": "POWER_LINE_FREQUENCY",
            "sharpness": "SHARPNESS",
            "backlight_compensation": "BACKLIGHT_COMPENSATION",
        }
        for key, attr in mapping.items():
            if key in controls:
                setattr(cls, attr, int(controls[key]))

    # ==========================================================
    # 12. 初始化
    # ==========================================================

    def __init__(self) -> None:

        self.device = self.DEVICE

        self.cap: Optional[
            cv2.VideoCapture
        ] = None
        self._online = False
        self._last_error: str | None = None
        self._next_reconnect_at = 0.0

    # ==========================================================
    # 13. 设置单个 V4L2 参数
    # ==========================================================

    def _set_control(
        self,
        name: str,
        value: int,
    ) -> None:

        command = [
            "v4l2-ctl",
            "-d",
            self.device,
            f"--set-ctrl={name}={value}",
        ]

        try:

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

        except FileNotFoundError as exc:

            raise RuntimeError(
                "系统中找不到 v4l2-ctl。\n"
                "请执行：\n"
                "sudo apt install v4l-utils"
            ) from exc

        if result.returncode != 0:

            raise RuntimeError(
                "\n侧面相机参数设置失败\n"
                f"device : {self.device}\n"
                f"name   : {name}\n"
                f"value  : {value}\n"
                f"stderr : {result.stderr.strip()}"
            )

    # ==========================================================
    # 14. 读取单个 V4L2 参数
    # ==========================================================

    def _get_control(
        self,
        name: str,
    ) -> Optional[int]:

        command = [
            "v4l2-ctl",
            "-d",
            self.device,
            f"--get-ctrl={name}",
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:

            return None

        text = result.stdout.strip()

        try:

            # 可能得到：
            #
            # brightness: 0
            #
            # 或：
            #
            # auto_exposure: 1 (Manual Mode)
            #
            # 所以取冒号后第一个字段。

            value_text = (
                text
                .split(
                    ":",
                    1,
                )[1]
                .strip()
            )

            first_value = (
                value_text
                .split()[0]
            )

            return int(
                first_value
            )

        except (
            IndexError,
            ValueError,
        ):

            return None

    # ==========================================================
    # 15. 设置所有固定参数
    # ==========================================================

    def _apply_fixed_controls(
        self,
    ) -> None:

        # ======================================================
        # 手动曝光
        # ======================================================

        self._set_control(
            "auto_exposure",
            self.AUTO_EXPOSURE,
        )

        self._set_control(
            "exposure_time_absolute",
            self.EXPOSURE_TIME_ABSOLUTE,
        )

        # ======================================================
        # Gain
        # ======================================================

        self._set_control(
            "gain",
            self.GAIN,
        )

        # ======================================================
        # 手动白平衡
        # ======================================================

        self._set_control(
            "white_balance_automatic",
            self.WHITE_BALANCE_AUTOMATIC,
        )

        self._set_control(
            "white_balance_temperature",
            self.WHITE_BALANCE_TEMPERATURE,
        )

        # ======================================================
        # 图像参数
        # ======================================================

        self._set_control(
            "brightness",
            self.BRIGHTNESS,
        )

        self._set_control(
            "contrast",
            self.CONTRAST,
        )

        self._set_control(
            "saturation",
            self.SATURATION,
        )

        self._set_control(
            "hue",
            self.HUE,
        )

        self._set_control(
            "gamma",
            self.GAMMA,
        )

        # ======================================================
        # 防频闪
        # ======================================================

        self._set_control(
            "power_line_frequency",
            self.POWER_LINE_FREQUENCY,
        )

        # ======================================================
        # 锐度
        # ======================================================

        self._set_control(
            "sharpness",
            self.SHARPNESS,
        )

        # ======================================================
        # 背光补偿
        # ======================================================

        self._set_control(
            "backlight_compensation",
            self.BACKLIGHT_COMPENSATION,
        )

    # ==========================================================
    # 16. FOURCC 转字符串
    # ==========================================================

    @staticmethod
    def _decode_fourcc(
        value: int,
    ) -> str:

        return "".join(
            chr(
                (
                    value
                    >> (8 * i)
                )
                & 0xFF
            )
            for i in range(4)
        )

    # ==========================================================
    # 17. 检查视频流
    # ==========================================================

    def _verify_stream(
        self,
    ) -> None:

        if (
            self.cap is None
            or not self.cap.isOpened()
        ):

            raise RuntimeError(
                "侧面相机尚未打开"
            )

        width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = float(
            self.cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        fourcc_value = int(
            self.cap.get(
                cv2.CAP_PROP_FOURCC
            )
        )

        fourcc = (
            self._decode_fourcc(
                fourcc_value
            )
            .strip("\x00")
        )

        errors = []

        if width != self.WIDTH:

            errors.append(
                f"width={width}, "
                f"expected={self.WIDTH}"
            )

        if height != self.HEIGHT:

            errors.append(
                f"height={height}, "
                f"expected={self.HEIGHT}"
            )

        if abs(
            fps - self.FPS
        ) > 0.5:

            errors.append(
                f"fps={fps:.2f}, "
                f"expected={self.FPS}"
            )

        if fourcc != self.FOURCC:

            errors.append(
                f"fourcc={fourcc}, "
                f"expected={self.FOURCC}"
            )

        if errors:

            raise RuntimeError(
                "侧面相机视频流参数异常：\n"
                + "\n".join(
                    errors
                )
            )

    # ==========================================================
    # 18. 检查固定 ISP 参数
    # ==========================================================

    def _verify_controls(
        self,
    ) -> None:

        expected_controls = {

            "auto_exposure":
                self.AUTO_EXPOSURE,

            "exposure_time_absolute":
                self.EXPOSURE_TIME_ABSOLUTE,

            "gain":
                self.GAIN,

            "white_balance_automatic":
                self.WHITE_BALANCE_AUTOMATIC,

            "white_balance_temperature":
                self.WHITE_BALANCE_TEMPERATURE,

            "brightness":
                self.BRIGHTNESS,

            "contrast":
                self.CONTRAST,

            "saturation":
                self.SATURATION,

            "hue":
                self.HUE,

            "gamma":
                self.GAMMA,

            "power_line_frequency":
                self.POWER_LINE_FREQUENCY,

            "sharpness":
                self.SHARPNESS,

            "backlight_compensation":
                self.BACKLIGHT_COMPENSATION,
        }

        errors = []

        for (
            name,
            expected_value,
        ) in expected_controls.items():

            actual_value = (
                self._get_control(
                    name
                )
            )

            if actual_value is None:

                errors.append(
                    f"{name}: 无法读取"
                )

                continue

            if actual_value != expected_value:

                errors.append(
                    f"{name}="
                    f"{actual_value}, "
                    f"expected="
                    f"{expected_value}"
                )

        if errors:

            raise RuntimeError(
                "侧面相机 ISP 参数"
                "回读检查失败：\n"
                + "\n".join(
                    errors
                )
            )

    # ==========================================================
    # 19. 打开相机
    # ==========================================================

    def open(
        self,
    ) -> None:

        if (
            self.cap is not None
            and self.cap.isOpened()
        ):

            return

        cap = cv2.VideoCapture(
            self.device,
            cv2.CAP_V4L2,
        )

        if not cap.isOpened():

            cap.release()

            raise RuntimeError(
                f"无法打开侧面相机："
                f"{self.device}"
            )

        self.cap = cap

        try:

            # ==================================================
            # MJPG
            # ==================================================

            fourcc = (
                cv2.VideoWriter_fourcc(
                    *self.FOURCC
                )
            )

            self.cap.set(
                cv2.CAP_PROP_FOURCC,
                fourcc,
            )

            # ==================================================
            # 原始分辨率
            # ==================================================

            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                self.WIDTH,
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.HEIGHT,
            )

            # ==================================================
            # FPS
            # ==================================================

            self.cap.set(
                cv2.CAP_PROP_FPS,
                self.FPS,
            )

            # ==================================================
            # Buffer
            # ==================================================

            self.cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                self.BUFFER_SIZE,
            )

            # ==================================================
            # 视频模式完成之后固定 ISP
            # ==================================================

            self._apply_fixed_controls()

            # ==================================================
            # 检查
            # ==================================================

            self._verify_stream()

            self._verify_controls()

            # ==================================================
            # 丢弃启动阶段前几帧
            # ==================================================

            for _ in range(
                self.WARMUP_FRAMES
            ):

                success = (
                    self.cap.grab()
                )

                if not success:

                    raise RuntimeError(
                        "侧面相机启动预热"
                        "阶段读取失败"
                    )

            self._online = True
            self._last_error = None

        except Exception:

            self.release()

            raise

    @property
    def is_online(self) -> bool:
        """返回最近一次读帧是否成功，供 Runtime 判断是否暂停分拣。"""
        return bool(self._online and self.cap is not None and self.cap.isOpened())

    @property
    def last_error(self) -> str | None:
        """返回最近一次相机错误，便于记录掉线原因。"""
        return self._last_error

    def _try_reconnect(self) -> bool:
        """以固定间隔重开相机，成功后由下一次读帧恢复业务。"""
        now = time.monotonic()
        if now < self._next_reconnect_at:
            return False
        self._next_reconnect_at = now + self.RECONNECT_INTERVAL_S
        try:
            self.release()
            self.open()
            print(f"[CAMERA][SIDE] 已重新连接: {self.device}")
            return True
        except Exception as exc:
            self._online = False
            self._last_error = str(exc)
            print(f"[CAMERA][SIDE] 掉线，等待重连: {exc}")
            return False

    def reconnect_once(self) -> bool:
        """供 Runtime 在暂停状态调用的一次重连尝试，不读取业务图像。"""
        if self.is_online:
            return True
        return self._try_reconnect()

    # ==========================================================
    # 20. 读取完整画面
    # ==========================================================

    def read_full(
        self,
    ) -> np.ndarray:

        """
        返回完整：

            1280 × 720

        只建议用于：
            调试
            ROI调整
            检查相机位置
        """

        if (
            self.cap is None
            or not self.cap.isOpened()
        ):

            raise RuntimeError(
                "侧面相机尚未打开，"
                "请先调用 camera.open()"
            )

        success, frame = (
            self.cap.read()
        )

        if (
            not success
            or frame is None
        ):

            raise RuntimeError(
                "侧面相机读取失败"
            )

        expected_shape = (
            self.HEIGHT,
            self.WIDTH,
            3,
        )

        if frame.shape != expected_shape:

            raise RuntimeError(
                "侧面相机原始图像尺寸异常：\n"
                f"actual   = {frame.shape}\n"
                f"expected = {expected_shape}"
            )

        return frame

    # ==========================================================
    # 21. 读取托盘 ROI
    # ==========================================================

    def read(
        self,
    ) -> np.ndarray | None:

        """
        正式视觉程序统一调用这个函数。

        返回的不是完整 1280×720，
        而是左侧长条托盘 ROI。
        """

        # 正式运行期间把读帧失败转换成离线状态，避免异常直接终止主线程。
        try:
            frame = self.read_full()
        except Exception as exc:
            self._online = False
            self._last_error = str(exc)
            self.release()
            if not self._try_reconnect():
                return None
            try:
                frame = self.read_full()
            except Exception as retry_exc:
                self._online = False
                self._last_error = str(retry_exc)
                self.release()
                return None

        roi = frame[
            self.ROI_Y1:self.ROI_Y2,
            self.ROI_X1:self.ROI_X2,
        ]

        # 返回独立连续内存。
        roi = roi.copy()

        expected_shape = (
            self.ROI_HEIGHT,
            self.ROI_WIDTH,
            3,
        )

        if roi.shape != expected_shape:

            raise RuntimeError(
                "侧面相机 ROI 尺寸异常：\n"
                f"actual   = {roi.shape}\n"
                f"expected = {expected_shape}"
            )

        self._online = True
        self._last_error = None
        return roi

    # ==========================================================
    # 22. ROI坐标 -> 原图坐标
    # ==========================================================

    def roi_to_full_point(
        self,
        x: int,
        y: int,
    ) -> tuple[int, int]:

        return (
            x + self.ROI_X1,
            y + self.ROI_Y1,
        )

    # ==========================================================
    # 23. 原图坐标 -> ROI坐标
    # ==========================================================

    def full_to_roi_point(
        self,
        x: int,
        y: int,
    ) -> tuple[int, int]:

        return (
            x - self.ROI_X1,
            y - self.ROI_Y1,
        )

    # ==========================================================
    # 24. 获取状态
    # ==========================================================

    def get_stream_info(
        self,
    ) -> dict:

        if (
            self.cap is None
            or not self.cap.isOpened()
        ):

            raise RuntimeError(
                "侧面相机尚未打开"
            )

        fourcc_value = int(
            self.cap.get(
                cv2.CAP_PROP_FOURCC
            )
        )

        return {

            "device":
                self.device,

            "width":
                int(
                    self.cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                ),

            "height":
                int(
                    self.cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                ),

            "fps":
                float(
                    self.cap.get(
                        cv2.CAP_PROP_FPS
                    )
                ),

            "fourcc":
                self._decode_fourcc(
                    fourcc_value
                ).strip("\x00"),

            "roi_x1":
                self.ROI_X1,

            "roi_y1":
                self.ROI_Y1,

            "roi_x2":
                self.ROI_X2,

            "roi_y2":
                self.ROI_Y2,

            "roi_width":
                self.ROI_WIDTH,

            "roi_height":
                self.ROI_HEIGHT,

            "auto_exposure":
                self.AUTO_EXPOSURE,

            "exposure":
                self.EXPOSURE_TIME_ABSOLUTE,

            "gain":
                self.GAIN,

            "white_balance_auto":
                self.WHITE_BALANCE_AUTOMATIC,

            "white_balance_temperature":
                self.WHITE_BALANCE_TEMPERATURE,

            "brightness":
                self.BRIGHTNESS,

            "contrast":
                self.CONTRAST,

            "saturation":
                self.SATURATION,

            "hue":
                self.HUE,

            "gamma":
                self.GAMMA,

            "sharpness":
                self.SHARPNESS,

            "backlight_compensation":
                self.BACKLIGHT_COMPENSATION,
        }

    # ==========================================================
    # 25. 是否打开
    # ==========================================================

    def is_opened(
        self,
    ) -> bool:

        return (
            self.cap is not None
            and self.cap.isOpened()
        )

    # ==========================================================
    # 26. 关闭相机
    # ==========================================================

    def release(
        self,
    ) -> None:

        if self.cap is not None:

            self.cap.release()

            self.cap = None

        self._online = False

    close = release

    # ==========================================================
    # 27. with 支持
    # ==========================================================

    def __enter__(
        self,
    ) -> "SideCamera":

        self.open()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.release()

    def __del__(
        self,
    ) -> None:

        self.release()
