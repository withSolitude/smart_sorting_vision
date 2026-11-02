from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class PixelToRobotTransform:
    """
    顶部相机像素坐标 (u, v) -> 机械坐标 (X_mm, Y_mm)。

    重要：
        新标定把 H 保存于完整去畸变图坐标系，ROI 只决定输入像素的原点。
        相机几何和去畸变不变时，调整 ROI 不需要重标；旧 ROI 标定可显式迁移。
    """

    def __init__(
        self,
        homography: np.ndarray,
        image_width: int,
        image_height: int,
        *,
        source: str = "TopCamera.read()",
        tcp_offset_x_mm: float = 0.0,
        tcp_offset_y_mm: float = 0.0,
        metadata: dict | None = None,
        coordinate_space: str = "roi",
        camera_config: dict | None = None,
    ) -> None:
        H = np.asarray(homography, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"homography 必须为 3x3，实际为 {H.shape}")
        if not np.all(np.isfinite(H)):
            raise ValueError("homography 含 NaN/Inf")

        self.H = H
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.source = str(source)
        self.tcp_offset_x_mm = float(tcp_offset_x_mm)
        self.tcp_offset_y_mm = float(tcp_offset_y_mm)
        self.metadata = dict(metadata or {})
        if coordinate_space not in {"roi", "undistorted_full"}:
            raise ValueError(f"不支持的像素标定坐标系：{coordinate_space}")
        self.coordinate_space = coordinate_space
        self._input_origin = np.zeros(2, dtype=np.float64)
        self._input_size = (self.image_width, self.image_height)
        if camera_config is not None:
            self.configure_camera(camera_config)

    def configure_camera(self, camera_config: dict) -> None:
        """绑定当前 YAML 的裁剪框，不修改保存的完整图 H。"""
        if self.coordinate_space == "roi":
            # 老文件不知道旧裁剪原点，保留原行为；迁移后才能解除尺寸绑定。
            return
        full_size = (int(camera_config["width"]), int(camera_config["height"]))
        if full_size != (self.image_width, self.image_height):
            raise ValueError(
                f"完整图分辨率{full_size}与标定{(self.image_width, self.image_height)}不同；"
                "请核对原始采集和去畸变配置。ROI宽高不需要与标定宽高相同。"
            )
        x1, y1, x2, y2 = map(int, camera_config["safe_roi"])
        if not (0 <= x1 < x2 <= full_size[0] and 0 <= y1 < y2 <= full_size[1]):
            raise ValueError("safe_roi 必须位于完整去畸变图内")
        self._input_origin = np.array([x1, y1], dtype=np.float64)
        self._input_size = (x2 - x1, y2 - y1)

    @classmethod
    def load(cls, path: str | Path, *, camera_config: dict | None = None) -> "PixelToRobotTransform":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            homography=np.asarray(data["homography"], dtype=np.float64),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            source=data.get("source", "TopCamera.read()"),
            tcp_offset_x_mm=float(data.get("tcp_offset_x_mm", 0.0)),
            tcp_offset_y_mm=float(data.get("tcp_offset_y_mm", 0.0)),
            metadata=data.get("metadata", {}),
            coordinate_space=data.get("coordinate_space", "roi"),
            camera_config=camera_config,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 2 if self.coordinate_space == "undistorted_full" else 1,
            "coordinate_space": self.coordinate_space,
            "source": self.source,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "homography": self.H.tolist(),
            "tcp_offset_x_mm": self.tcp_offset_x_mm,
            "tcp_offset_y_mm": self.tcp_offset_y_mm,
            "metadata": self.metadata,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def check_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.ndim < 2:
            raise ValueError("frame 无效")

        h, w = frame.shape[:2]
        if (w, h) != self._input_size:
            raise RuntimeError(
                "当前 TopCamera.read() 图像尺寸与标定时不一致：\n"
                f"current     = {w} x {h}\n"
                f"expected    = {self._input_size[0]} x {self._input_size[1]}\n"
                + ("旧文件使用ROI局部坐标；可用 migrate_pixel_to_robot_roi 换算到完整图坐标。"
                   if self.coordinate_space == "roi" else
                   "当前图像与YAML裁剪框不符；请核对脚本是否读取同一份相机配置。")
            )

    def pixel_to_robot(
        self,
        u: float,
        v: float,
        *,
        apply_tcp_offset: bool = True,
    ) -> tuple[float, float]:
        if not (math.isfinite(u) and math.isfinite(v)):
            raise ValueError("像素坐标含 NaN/Inf")

        point = np.array([[[float(u), float(v)]]], dtype=np.float64)
        point += self._input_origin
        result = cv2.perspectiveTransform(point, self.H)[0, 0]

        x_mm = float(result[0])
        y_mm = float(result[1])

        if apply_tcp_offset:
            x_mm += self.tcp_offset_x_mm
            y_mm += self.tcp_offset_y_mm

        if not (math.isfinite(x_mm) and math.isfinite(y_mm)):
            raise RuntimeError("像素到机械坐标变换结果为 NaN/Inf")

        return x_mm, y_mm

    def pixels_to_robot(
        self,
        points: Iterable[tuple[float, float]],
        *,
        apply_tcp_offset: bool = True,
    ) -> np.ndarray:
        pts = np.asarray(list(points), dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float64)

        out = cv2.perspectiveTransform(pts + self._input_origin, self.H).reshape(-1, 2)
        if apply_tcp_offset:
            out[:, 0] += self.tcp_offset_x_mm
            out[:, 1] += self.tcp_offset_y_mm
        return out

    def robot_to_pixels(
        self,
        points: Iterable[tuple[float, float]],
        *,
        apply_tcp_offset: bool = True,
    ) -> np.ndarray:
        """机械坐标反投影到当前ROI，供抓取搜索、夹爪包络和禁区绘制共用。"""
        pts = np.asarray(list(points), dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float64)
        if apply_tcp_offset:
            pts -= [self.tcp_offset_x_mm, self.tcp_offset_y_mm]
        full_pixels = cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)
        return full_pixels - self._input_origin

    def robot_to_pixel(
        self,
        x_mm: float,
        y_mm: float,
        *,
        apply_tcp_offset: bool = True,
    ) -> tuple[float, float]:
        """单点机械坐标反投影，使用与正向转换相同的ROI原点。"""
        out = self.robot_to_pixels([(x_mm, y_mm)], apply_tcp_offset=apply_tcp_offset)[0]
        return float(out[0]), float(out[1])

    def to_full_image(self, old_roi: Iterable[int], full_size: tuple[int, int]) -> "PixelToRobotTransform":
        """只做已知裁剪原点的精确换算；不重新拟合，也不更改TCP偏置。"""
        if self.coordinate_space != "roi":
            raise ValueError("该文件已经使用完整图坐标，无需重复迁移")
        x1, y1, x2, y2 = map(int, old_roi)
        width, height = map(int, full_size)
        if (x2 - x1, y2 - y1) != (self.image_width, self.image_height):
            raise ValueError("旧ROI宽高与旧标定记录不匹配，不能据此换算")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("旧ROI必须位于完整图内")
        translation = np.array([[1.0, 0.0, -x1], [0.0, 1.0, -y1], [0.0, 0.0, 1.0]])
        metadata = dict(self.metadata)
        metadata["calibration_roi"] = [x1, y1, x2, y2]
        if "points_pixel" in metadata:
            metadata["points_pixel_roi"] = metadata["points_pixel"]
            metadata["points_pixel"] = (np.asarray(metadata["points_pixel"], dtype=float) + [x1, y1]).tolist()
        return type(self)(
            self.H @ translation, width, height, source="TopCamera.read_undistorted()",
            tcp_offset_x_mm=self.tcp_offset_x_mm, tcp_offset_y_mm=self.tcp_offset_y_mm,
            metadata=metadata, coordinate_space="undistorted_full",
        )

    def robot_angle_from_image_angle(
        self,
        u: float,
        v: float,
        image_angle_deg: float,
        *,
        sample_length_px: float = 50.0,
    ) -> float:
        theta = math.radians(float(image_angle_deg))
        u2 = float(u) + sample_length_px * math.cos(theta)
        v2 = float(v) + sample_length_px * math.sin(theta)

        x1, y1 = self.pixel_to_robot(u, v, apply_tcp_offset=False)
        x2, y2 = self.pixel_to_robot(u2, v2, apply_tcp_offset=False)

        return math.degrees(math.atan2(y2 - y1, x2 - x1))
