from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class TopCamera:
    """
    顶部相机统一驱动。

    设计原则
    ------------------------------------------------------------
    1. 所有相机硬件参数只允许在本文件中定义。
    2. 其它模块禁止重新设置曝光、白平衡、分辨率等参数。
    3. read_full():
           返回原始 1920x1080 BGR 图像。

    4. read_undistorted():
           返回去畸变后的 1920x1080 BGR 图像。

    5. read():
           返回：
               去畸变
               +
               SAFE ROI

           最终输出尺寸：
               870 x 840

    6. YOLO、数据集采集、抓取点计算等正常业务，
       统一调用：

           frame = camera.read()

    7. 机械坐标标定、几何调试时可调用：

           frame = camera.read_undistorted()

    8. 相机重新标定、底层调试时才调用：

           frame = camera.read_full()
    """

    # ============================================================
    # 项目路径
    # ============================================================

    ROOT = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    # ============================================================
    # 相机设备
    #
    # 当前顶部相机：
    # /dev/video0
    #
    # 如果以后两个USB相机同时连接后 video 编号发生变化，
    # 再改成 /dev/v4l/by-id/... 的固定设备路径。
    # ============================================================

    DEVICE = (
        "/dev/v4l/by-id/"
        "usb-icSpring_icspring_camera_20240307110322-video-index0"
    )

    # ============================================================
    # 视频流固定参数
    # ============================================================

    WIDTH = 1920
    HEIGHT = 1080

    FPS = 30

    FOURCC = "MJPG"

    BUFFER_SIZE = 1

    # ============================================================
    # 曝光
    #
    # V4L2:
    #
    # auto_exposure:
    #   1 = Manual Mode
    #   3 = Aperture Priority Mode
    # ============================================================

    AUTO_EXPOSURE = 1

    EXPOSURE_TIME_ABSOLUTE = 100

    # ============================================================
    # 增益
    # ============================================================

    GAIN = 0

    # ============================================================
    # 白平衡
    # ============================================================

    WHITE_BALANCE_AUTOMATIC = 0

    WHITE_BALANCE_TEMPERATURE = 5400

    # ============================================================
    # ISP固定参数
    # ============================================================

    BRIGHTNESS = 0

    CONTRAST = 30

    SATURATION = 55

    HUE = 0

    GAMMA = 100

    # ============================================================
    # 电源频率
    #
    # 0 = Disabled
    # 1 = 50 Hz
    # 2 = 60 Hz
    #
    # 当前固定：
    # 1 = 50 Hz
    # ============================================================

    POWER_LINE_FREQUENCY = 1

    # ============================================================
    # 锐度
    # ============================================================

    SHARPNESS = 8

    # ============================================================
    # 背光补偿
    # ============================================================

    BACKLIGHT_COMPENSATION = 0

    # ============================================================
    # 标定参数文件
    # ============================================================

    CALIBRATION_FILE = (
        ROOT
        / "data"
        / "calibration"
        / "top_camera"
        / "top_camera_intrinsics.npz"
    )

    # ============================================================
    # 最终 SAFE ROI
    #
    # 坐标基于：
    #
    # 去畸变后的完整 1920x1080 图像
    #
    # 注意：
    # 这不是托盘物理边界，
    # 而是为了防止靠近托盘边缘的高物体被裁断而保留的
    # “安全检测区域”。
    # ============================================================

    SAFE_ROI_X1 = 580
    SAFE_ROI_Y1 = 120

    SAFE_ROI_X2 = 1330
    SAFE_ROI_Y2 = 927

    SAFE_ROI_WIDTH = (
        SAFE_ROI_X2
        -
        SAFE_ROI_X1
    )

    SAFE_ROI_HEIGHT = (
        SAFE_ROI_Y2
        -
        SAFE_ROI_Y1
    )

    # ============================================================
    # 相机启动后丢弃若干帧
    #
    # UVC相机刚打开时前几帧可能尚未稳定。
    # ============================================================

    WARMUP_FRAMES = 10

    # ============================================================
    # FPS验证容差
    # ============================================================

    FPS_TOLERANCE = 1.0

    # ============================================================
    # 运行时配置入口
    # ============================================================

    @classmethod
    def configure(
        cls,
        config: dict,
        *,
        calibration_file: str | Path | None = None,
    ) -> None:
        """Apply the production camera profile from runtime_config.yaml.

        TopCamera historically stored its profile as class constants and several
        helper methods are classmethods.  This method keeps that implementation
        compatible while making runtime_config.yaml the single production source.
        """
        cls.DEVICE = str(config["device"])
        cls.WIDTH = int(config.get("width", cls.WIDTH))
        cls.HEIGHT = int(config.get("height", cls.HEIGHT))
        cls.FPS = int(config.get("fps", cls.FPS))
        cls.FOURCC = str(config.get("fourcc", cls.FOURCC))
        cls.BUFFER_SIZE = int(config.get("buffer_size", cls.BUFFER_SIZE))

        roi = config.get("safe_roi")
        if roi is not None:
            if not isinstance(roi, (list, tuple)) or len(roi) != 4:
                raise ValueError("camera.top.safe_roi 必须是 [x1, y1, x2, y2]")
            cls.SAFE_ROI_X1, cls.SAFE_ROI_Y1, cls.SAFE_ROI_X2, cls.SAFE_ROI_Y2 = map(int, roi)
            cls.SAFE_ROI_WIDTH = cls.SAFE_ROI_X2 - cls.SAFE_ROI_X1
            cls.SAFE_ROI_HEIGHT = cls.SAFE_ROI_Y2 - cls.SAFE_ROI_Y1

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

        if calibration_file is not None:
            cls.CALIBRATION_FILE = Path(calibration_file)

        cls._verify_roi_definition()

    # ============================================================
    # 初始化
    # ============================================================

    def __init__(self):

        self._cap: cv2.VideoCapture | None = None

        self._opened = False

        # --------------------------------------------------------
        # 标定参数
        # --------------------------------------------------------

        self._camera_matrix: np.ndarray | None = None

        self._dist_coeffs: np.ndarray | None = None

        self._new_camera_matrix: np.ndarray | None = None

        self._calibration_image_size: tuple[int, int] | None = None

        self._valid_roi: tuple[int, int, int, int] | None = None

        self._calibration_rms: float | None = None

        # --------------------------------------------------------
        # 去畸变映射表
        # --------------------------------------------------------

        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None

    # ============================================================
    # V4L2工具检查
    # ============================================================

    @staticmethod
    def _check_v4l2_ctl_available() -> None:

        if shutil.which(
            "v4l2-ctl"
        ) is None:

            raise RuntimeError(
                "系统中没有找到 v4l2-ctl。\n"
                "\n"
                "请安装：\n"
                "sudo apt install v4l-utils"
            )

    # ============================================================
    # 设置单个V4L2控制参数
    # ============================================================

    @classmethod
    def _set_control(
        cls,
        name: str,
        value: int,
    ) -> None:

        command = [
            "v4l2-ctl",
            "-d",
            cls.DEVICE,
            "-c",
            f"{name}={value}",
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "设置相机参数失败：\n"
                f"control = {name}\n"
                f"value   = {value}\n"
                f"device  = {cls.DEVICE}\n"
                "\n"
                f"stderr:\n"
                f"{result.stderr.strip()}"
            )

    # ============================================================
    # 读取单个V4L2控制参数
    # ============================================================

    @classmethod
    def _get_control(
        cls,
        name: str,
    ) -> int:

        command = [
            "v4l2-ctl",
            "-d",
            cls.DEVICE,
            "-C",
            name,
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "读取相机参数失败：\n"
                f"control = {name}\n"
                f"device  = {cls.DEVICE}\n"
                "\n"
                f"stderr:\n"
                f"{result.stderr.strip()}"
            )

        output = (
            result.stdout
            .strip()
        )

        # --------------------------------------------------------
        # 可能输出：
        #
        # brightness: 0
        #
        # 或：
        #
        # auto_exposure: 1 (Manual Mode)
        #
        # 所以只取冒号后面的第一个token。
        # --------------------------------------------------------

        if ":" not in output:

            raise RuntimeError(
                "无法解析 v4l2-ctl 输出：\n"
                f"{output}"
            )

        value_text = (
            output
            .split(
                ":",
                1,
            )[1]
            .strip()
            .split()[0]
        )

        try:

            return int(
                value_text
            )

        except ValueError as exc:

            raise RuntimeError(
                "无法解析控制参数值：\n"
                f"{output}"
            ) from exc

    # ============================================================
    # 设置全部固定控制参数
    # ============================================================

    @classmethod
    def _apply_fixed_controls(
        cls,
    ) -> None:

        # --------------------------------------------------------
        # 先关闭自动控制。
        #
        # 顺序非常重要：
        #
        # 自动曝光关闭
        #     ↓
        # 设置曝光
        #
        # 自动白平衡关闭
        #     ↓
        # 设置色温
        # --------------------------------------------------------

        cls._set_control(
            "auto_exposure",
            cls.AUTO_EXPOSURE,
        )

        cls._set_control(
            "exposure_time_absolute",
            cls.EXPOSURE_TIME_ABSOLUTE,
        )

        cls._set_control(
            "white_balance_automatic",
            cls.WHITE_BALANCE_AUTOMATIC,
        )

        cls._set_control(
            "white_balance_temperature",
            cls.WHITE_BALANCE_TEMPERATURE,
        )

        # --------------------------------------------------------
        # 其它ISP参数
        # --------------------------------------------------------

        controls = {

            "brightness":
                cls.BRIGHTNESS,

            "contrast":
                cls.CONTRAST,

            "saturation":
                cls.SATURATION,

            "hue":
                cls.HUE,

            "gamma":
                cls.GAMMA,

            "gain":
                cls.GAIN,

            "power_line_frequency":
                cls.POWER_LINE_FREQUENCY,

            "sharpness":
                cls.SHARPNESS,

            "backlight_compensation":
                cls.BACKLIGHT_COMPENSATION,
        }

        for (
            name,
            value,
        ) in controls.items():

            cls._set_control(
                name,
                value,
            )

    # ============================================================
    # 验证全部固定控制参数
    # ============================================================

    @classmethod
    def _verify_controls(
        cls,
    ) -> None:

        expected_controls = {

            "auto_exposure":
                cls.AUTO_EXPOSURE,

            "exposure_time_absolute":
                cls.EXPOSURE_TIME_ABSOLUTE,

            "white_balance_automatic":
                cls.WHITE_BALANCE_AUTOMATIC,

            "white_balance_temperature":
                cls.WHITE_BALANCE_TEMPERATURE,

            "brightness":
                cls.BRIGHTNESS,

            "contrast":
                cls.CONTRAST,

            "saturation":
                cls.SATURATION,

            "hue":
                cls.HUE,

            "gamma":
                cls.GAMMA,

            "gain":
                cls.GAIN,

            "power_line_frequency":
                cls.POWER_LINE_FREQUENCY,

            "sharpness":
                cls.SHARPNESS,

            "backlight_compensation":
                cls.BACKLIGHT_COMPENSATION,
        }

        errors = []

        for (
            name,
            expected,
        ) in expected_controls.items():

            actual = cls._get_control(
                name
            )

            if actual != expected:

                errors.append(
                    (
                        name,
                        expected,
                        actual,
                    )
                )

        if errors:

            message = [
                "顶部相机固定参数验证失败："
            ]

            for (
                name,
                expected,
                actual,
            ) in errors:

                message.append(
                    f"{name}: "
                    f"expected={expected}, "
                    f"actual={actual}"
                )

            raise RuntimeError(
                "\n".join(
                    message
                )
            )

    # ============================================================
    # 加载相机内参
    # ============================================================

    def _load_calibration(
        self,
    ) -> None:

        calibration_file = (
            self.CALIBRATION_FILE
        )

        if not calibration_file.exists():

            raise FileNotFoundError(
                "找不到顶部相机标定参数文件：\n"
                f"{calibration_file}\n"
                "\n"
                "当前正式 TopCamera 要求存在有效的内参标定文件。"
            )

        data = np.load(
            calibration_file,
            allow_pickle=False,
        )

        required_keys = {

            "camera_matrix",
            "dist_coeffs",
            "new_camera_matrix",
            "image_size",
            "valid_roi",
            "rms",
        }

        missing_keys = (
            required_keys
            -
            set(
                data.files
            )
        )

        if missing_keys:

            raise RuntimeError(
                "顶部相机标定文件缺少参数：\n"
                f"{sorted(missing_keys)}"
            )

        self._camera_matrix = (
            data["camera_matrix"]
            .astype(
                np.float64
            )
        )

        self._dist_coeffs = (
            data["dist_coeffs"]
            .astype(
                np.float64
            )
        )

        self._new_camera_matrix = (
            data["new_camera_matrix"]
            .astype(
                np.float64
            )
        )

        image_size_array = (
            data["image_size"]
            .astype(
                np.int32
            )
        )

        self._calibration_image_size = (
            int(
                image_size_array[0]
            ),
            int(
                image_size_array[1]
            ),
        )

        valid_roi_array = (
            data["valid_roi"]
            .astype(
                np.int32
            )
        )

        self._valid_roi = tuple(
            int(value)
            for value
            in valid_roi_array
        )

        self._calibration_rms = float(
            data["rms"]
        )

        # --------------------------------------------------------
        # 内参标定分辨率必须和正式视频流一致。
        # --------------------------------------------------------

        expected_size = (
            self.WIDTH,
            self.HEIGHT,
        )

        if (
            self._calibration_image_size
            != expected_size
        ):

            raise RuntimeError(
                "标定参数分辨率与 TopCamera 固定分辨率不一致：\n"
                f"calibration = "
                f"{self._calibration_image_size}\n"
                f"camera      = "
                f"{expected_size}\n"
                "\n"
                "不能直接使用该标定文件。"
            )

    # ============================================================
    # 生成实时去畸变映射表
    # ============================================================

    def _create_undistort_maps(
        self,
    ) -> None:

        if (
            self._camera_matrix is None
            or
            self._dist_coeffs is None
            or
            self._new_camera_matrix is None
        ):

            raise RuntimeError(
                "相机标定参数尚未加载。"
            )

        image_size = (
            self.WIDTH,
            self.HEIGHT,
        )

        (
            self._map1,
            self._map2,
        ) = cv2.initUndistortRectifyMap(

            self._camera_matrix,

            self._dist_coeffs,

            None,

            self._new_camera_matrix,

            image_size,

            cv2.CV_16SC2,
        )

    # ============================================================
    # 检查SAFE ROI合法性
    # ============================================================

    @classmethod
    def _verify_roi_definition(
        cls,
    ) -> None:

        if not (
            0
            <= cls.SAFE_ROI_X1
            < cls.SAFE_ROI_X2
            <= cls.WIDTH
        ):

            raise RuntimeError(
                "顶部相机 SAFE ROI X 坐标非法。"
            )

        if not (
            0
            <= cls.SAFE_ROI_Y1
            < cls.SAFE_ROI_Y2
            <= cls.HEIGHT
        ):

            raise RuntimeError(
                "顶部相机 SAFE ROI Y 坐标非法。"
            )

        expected_width = (
            cls.SAFE_ROI_X2
            -
            cls.SAFE_ROI_X1
        )

        expected_height = (
            cls.SAFE_ROI_Y2
            -
            cls.SAFE_ROI_Y1
        )

        if (
            cls.SAFE_ROI_WIDTH
            != expected_width
        ):

            raise RuntimeError(
                "SAFE_ROI_WIDTH 定义错误。"
            )

        if (
            cls.SAFE_ROI_HEIGHT
            != expected_height
        ):

            raise RuntimeError(
                "SAFE_ROI_HEIGHT 定义错误。"
            )

    # ============================================================
    # 验证视频流
    # ============================================================

    def _verify_stream(
        self,
    ) -> None:

        if self._cap is None:

            raise RuntimeError(
                "相机尚未初始化。"
            )

        actual_width = int(
            round(
                self._cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )
        )

        actual_height = int(
            round(
                self._cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )
        )

        actual_fps = float(
            self._cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        fourcc_value = int(
            self._cap.get(
                cv2.CAP_PROP_FOURCC
            )
        )

        actual_fourcc = "".join(
            [
                chr(
                    (
                        fourcc_value
                        >>
                        (8 * i)
                    )
                    &
                    0xFF
                )
                for i in range(4)
            ]
        )

        errors = []

        if (
            actual_width
            != self.WIDTH
        ):

            errors.append(
                f"width: "
                f"expected={self.WIDTH}, "
                f"actual={actual_width}"
            )

        if (
            actual_height
            != self.HEIGHT
        ):

            errors.append(
                f"height: "
                f"expected={self.HEIGHT}, "
                f"actual={actual_height}"
            )

        if (
            abs(
                actual_fps
                -
                self.FPS
            )
            >
            self.FPS_TOLERANCE
        ):

            errors.append(
                f"fps: "
                f"expected={self.FPS}, "
                f"actual={actual_fps:.3f}"
            )

        if (
            actual_fourcc
            != self.FOURCC
        ):

            errors.append(
                f"fourcc: "
                f"expected={self.FOURCC}, "
                f"actual={actual_fourcc}"
            )

        if errors:

            raise RuntimeError(
                "顶部相机视频流参数验证失败：\n"
                +
                "\n".join(
                    errors
                )
            )

    # ============================================================
    # 打开相机
    # ============================================================

    def open(
        self,
    ) -> None:

        if self._opened:

            return

        # --------------------------------------------------------
        # 基础环境检查
        # --------------------------------------------------------

        self._check_v4l2_ctl_available()

        self._verify_roi_definition()

        # --------------------------------------------------------
        # 先加载标定参数。
        # --------------------------------------------------------

        self._load_calibration()

        self._create_undistort_maps()

        # --------------------------------------------------------
        # 打开 V4L2 相机
        # --------------------------------------------------------

        cap = cv2.VideoCapture(
            self.DEVICE,
            cv2.CAP_V4L2,
        )

        if not cap.isOpened():

            cap.release()

            raise RuntimeError(
                "无法打开顶部相机：\n"
                f"{self.DEVICE}"
            )

        self._cap = cap

        try:

            # ----------------------------------------------------
            # 设置MJPG
            # ----------------------------------------------------

            fourcc = cv2.VideoWriter_fourcc(
                *self.FOURCC
            )

            self._cap.set(
                cv2.CAP_PROP_FOURCC,
                fourcc,
            )

            # ----------------------------------------------------
            # 固定分辨率
            # ----------------------------------------------------

            self._cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                self.WIDTH,
            )

            self._cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.HEIGHT,
            )

            # ----------------------------------------------------
            # 固定FPS
            # ----------------------------------------------------

            self._cap.set(
                cv2.CAP_PROP_FPS,
                self.FPS,
            )

            # ----------------------------------------------------
            # 尽量减小OpenCV缓存
            #
            # 某些V4L2驱动不一定严格支持这个属性，
            # 所以这里设置，但不作为硬性验证条件。
            # ----------------------------------------------------

            self._cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                self.BUFFER_SIZE,
            )

            # ----------------------------------------------------
            # 应用所有固定ISP控制
            # ----------------------------------------------------

            self._apply_fixed_controls()

            # ----------------------------------------------------
            # 验证视频流
            # ----------------------------------------------------

            self._verify_stream()

            # ----------------------------------------------------
            # 验证V4L2控制参数
            # ----------------------------------------------------

            self._verify_controls()

            # ----------------------------------------------------
            # 丢弃启动阶段若干帧
            # ----------------------------------------------------

            for _ in range(
                self.WARMUP_FRAMES
            ):

                ok, _ = (
                    self._cap.read()
                )

                if not ok:

                    raise RuntimeError(
                        "顶部相机预热阶段读取失败。"
                    )

            self._opened = True

        except Exception:

            self.release()

            raise

    # ============================================================
    # 原始完整图
    #
    # 返回：
    #   1920 x 1080 x 3
    #
    # 坐标系：
    #   原始相机图像坐标系
    #
    # 仅用于：
    #   相机重新标定
    #   底层调试
    # ============================================================

    def read_full(
        self,
    ) -> np.ndarray:

        if (
            not self._opened
            or
            self._cap is None
        ):

            raise RuntimeError(
                "TopCamera 尚未打开。\n"
                "请先调用 camera.open()。"
            )

        ok, frame = (
            self._cap.read()
        )

        if (
            not ok
            or
            frame is None
        ):

            raise RuntimeError(
                "顶部相机读取图像失败。"
            )

        expected_shape = (
            self.HEIGHT,
            self.WIDTH,
            3,
        )

        if (
            frame.shape
            != expected_shape
        ):

            raise RuntimeError(
                "顶部相机原始帧尺寸异常：\n"
                f"expected = "
                f"{expected_shape}\n"
                f"actual   = "
                f"{frame.shape}"
            )

        return frame

    # ============================================================
    # 去畸变完整图
    #
    # 返回：
    #   1920 x 1080 x 3
    #
    # 坐标系：
    #   去畸变完整图坐标系
    #
    # 后续机械XY标定应基于这个坐标系。
    # ============================================================

    def read_undistorted(
        self,
    ) -> np.ndarray:

        frame = (
            self.read_full()
        )

        if (
            self._map1 is None
            or
            self._map2 is None
        ):

            raise RuntimeError(
                "顶部相机去畸变映射尚未初始化。"
            )

        undistorted = cv2.remap(
            frame,
            self._map1,
            self._map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        expected_shape = (
            self.HEIGHT,
            self.WIDTH,
            3,
        )

        if (
            undistorted.shape
            != expected_shape
        ):

            raise RuntimeError(
                "顶部相机去畸变帧尺寸异常：\n"
                f"expected = "
                f"{expected_shape}\n"
                f"actual   = "
                f"{undistorted.shape}"
            )

        return undistorted

    # ============================================================
    # 正式业务接口
    #
    # read():
    #
    # 原始图
    #   ↓
    # 去畸变
    #   ↓
    # SAFE ROI
    #   ↓
    # 返回 870 x 840 图像
    #
    # YOLO / 数据集 / Mask处理统一使用该接口。
    # ============================================================

    def read(
        self,
    ) -> np.ndarray:

        frame = (
            self.read_undistorted()
        )

        roi = frame[
            self.SAFE_ROI_Y1:
            self.SAFE_ROI_Y2,

            self.SAFE_ROI_X1:
            self.SAFE_ROI_X2,
        ]

        # --------------------------------------------------------
        # copy非常重要。
        #
        # 防止后续代码修改ROI时直接修改原frame的内存视图。
        # --------------------------------------------------------

        roi = (
            roi.copy()
        )

        expected_shape = (
            self.SAFE_ROI_HEIGHT,
            self.SAFE_ROI_WIDTH,
            3,
        )

        if (
            roi.shape
            != expected_shape
        ):

            raise RuntimeError(
                "顶部相机 SAFE ROI 尺寸异常：\n"
                f"expected = "
                f"{expected_shape}\n"
                f"actual   = "
                f"{roi.shape}"
            )

        return roi

    # ============================================================
    # ROI坐标 → 完整去畸变图坐标
    #
    # 输入：
    #   (u_roi, v_roi)
    #
    # 输出：
    #   (u_full, v_full)
    # ============================================================

    @classmethod
    def roi_to_full_point(
        cls,
        u_roi: float,
        v_roi: float,
    ) -> tuple[float, float]:

        u_full = (
            float(u_roi)
            +
            cls.SAFE_ROI_X1
        )

        v_full = (
            float(v_roi)
            +
            cls.SAFE_ROI_Y1
        )

        return (
            u_full,
            v_full,
        )

    # ============================================================
    # 完整去畸变图坐标 → ROI坐标
    # ============================================================

    @classmethod
    def full_to_roi_point(
        cls,
        u_full: float,
        v_full: float,
    ) -> tuple[float, float]:

        u_roi = (
            float(u_full)
            -
            cls.SAFE_ROI_X1
        )

        v_roi = (
            float(v_full)
            -
            cls.SAFE_ROI_Y1
        )

        return (
            u_roi,
            v_roi,
        )

    # ============================================================
    # 批量 ROI坐标 → 完整图坐标
    #
    # points:
    #   shape = (N, 2)
    # ============================================================

    @classmethod
    def roi_to_full_points(
        cls,
        points: np.ndarray,
    ) -> np.ndarray:

        points = np.asarray(
            points,
            dtype=np.float32,
        )

        if (
            points.ndim != 2
            or
            points.shape[1] != 2
        ):

            raise ValueError(
                "points 必须为 Nx2 数组。"
            )

        result = (
            points.copy()
        )

        result[:, 0] += (
            cls.SAFE_ROI_X1
        )

        result[:, 1] += (
            cls.SAFE_ROI_Y1
        )

        return result

    # ============================================================
    # 批量 完整图坐标 → ROI坐标
    # ============================================================

    @classmethod
    def full_to_roi_points(
        cls,
        points: np.ndarray,
    ) -> np.ndarray:

        points = np.asarray(
            points,
            dtype=np.float32,
        )

        if (
            points.ndim != 2
            or
            points.shape[1] != 2
        ):

            raise ValueError(
                "points 必须为 Nx2 数组。"
            )

        result = (
            points.copy()
        )

        result[:, 0] -= (
            cls.SAFE_ROI_X1
        )

        result[:, 1] -= (
            cls.SAFE_ROI_Y1
        )

        return result

    # ============================================================
    # 判断完整图中的点是否位于SAFE ROI内
    # ============================================================

    @classmethod
    def is_full_point_inside_roi(
        cls,
        u_full: float,
        v_full: float,
    ) -> bool:

        return (
            cls.SAFE_ROI_X1
            <= u_full
            < cls.SAFE_ROI_X2
            and
            cls.SAFE_ROI_Y1
            <= v_full
            < cls.SAFE_ROI_Y2
        )

    # ============================================================
    # 获取当前实际视频流信息
    # ============================================================

    def get_stream_info(
        self,
    ) -> dict[str, Any]:

        if (
            self._cap is None
            or
            not self._cap.isOpened()
        ):

            return {
                "opened": False,
                "device": self.DEVICE,
            }

        width = int(
            round(
                self._cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )
        )

        height = int(
            round(
                self._cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )
        )

        fps = float(
            self._cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        fourcc_value = int(
            self._cap.get(
                cv2.CAP_PROP_FOURCC
            )
        )

        fourcc = "".join(
            [
                chr(
                    (
                        fourcc_value
                        >>
                        (8 * i)
                    )
                    &
                    0xFF
                )
                for i in range(4)
            ]
        )

        return {

            "opened":
                True,

            "device":
                self.DEVICE,

            "width":
                width,

            "height":
                height,

            "fps":
                fps,

            "fourcc":
                fourcc,

            "safe_roi":
                (
                    self.SAFE_ROI_X1,
                    self.SAFE_ROI_Y1,
                    self.SAFE_ROI_X2,
                    self.SAFE_ROI_Y2,
                ),

            "safe_roi_width":
                self.SAFE_ROI_WIDTH,

            "safe_roi_height":
                self.SAFE_ROI_HEIGHT,

            "calibration_file":
                str(
                    self.CALIBRATION_FILE
                ),

            "calibration_rms":
                self._calibration_rms,

            "calibration_valid_roi":
                self._valid_roi,
        }

    # ============================================================
    # 获取固定相机参数
    # ============================================================

    @classmethod
    def get_fixed_controls(
        cls,
    ) -> dict[str, int]:

        return {

            "auto_exposure":
                cls.AUTO_EXPOSURE,

            "exposure_time_absolute":
                cls.EXPOSURE_TIME_ABSOLUTE,

            "gain":
                cls.GAIN,

            "white_balance_automatic":
                cls.WHITE_BALANCE_AUTOMATIC,

            "white_balance_temperature":
                cls.WHITE_BALANCE_TEMPERATURE,

            "brightness":
                cls.BRIGHTNESS,

            "contrast":
                cls.CONTRAST,

            "saturation":
                cls.SATURATION,

            "hue":
                cls.HUE,

            "gamma":
                cls.GAMMA,

            "power_line_frequency":
                cls.POWER_LINE_FREQUENCY,

            "sharpness":
                cls.SHARPNESS,

            "backlight_compensation":
                cls.BACKLIGHT_COMPENSATION,
        }

    # ============================================================
    # 是否已经打开
    # ============================================================

    @property
    def is_opened(
        self,
    ) -> bool:

        return (
            self._opened
            and
            self._cap is not None
            and
            self._cap.isOpened()
        )

    # ============================================================
    # 释放
    # ============================================================

    def release(
        self,
    ) -> None:

        if self._cap is not None:

            self._cap.release()

        self._cap = None

        self._opened = False

    # 与 SideCamera / main.py 统一生命周期接口。
    close = release

    # ============================================================
    # Context Manager
    #
    # 可写：
    #
    # with TopCamera() as camera:
    #     frame = camera.read()
    # ============================================================

    def __enter__(
        self,
    ) -> "TopCamera":

        self.open()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.release()
