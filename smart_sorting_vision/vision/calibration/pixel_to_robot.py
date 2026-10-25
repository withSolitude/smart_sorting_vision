from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class PixelToRobotTransform:
    """
    顶部相机最终视觉图像坐标 (u, v) -> 机械坐标 (X_mm, Y_mm)。

    重要：
        H 只对“生成它时所使用的 TopCamera.read() 输出坐标系”有效。
        只要后续改变了相机位置、分辨率、去畸变参数、裁剪 ROI，必须重新标定。
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

    @classmethod
    def load(cls, path: str | Path) -> "PixelToRobotTransform":
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
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
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
        if w != self.image_width or h != self.image_height:
            raise RuntimeError(
                "当前 TopCamera.read() 图像尺寸与标定时不一致：\n"
                f"current     = {w} x {h}\n"
                f"calibration = {self.image_width} x {self.image_height}\n"
                "请停止运行并重新确认相机/去畸变/ROI配置。"
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

        out = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        if apply_tcp_offset:
            out[:, 0] += self.tcp_offset_x_mm
            out[:, 1] += self.tcp_offset_y_mm
        return out

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
