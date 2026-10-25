from __future__ import annotations

from importlib import import_module
from typing import Any

import cv2
import numpy as np


class QRDetector:
    """
    侧面二维码检测/解码器 V5：ZXing-C++ 快速主路径 + OpenCV 曲面回退。

    正式流程：
        1) ZXing-C++ QR-only 直接解码（平面/轻微弯曲，最快）
        2) ZXing 失败 -> OpenCV detectAndDecodeCurved()（圆柱/圆台/圆锥曲面）
        3) OpenCV 曲面解码仍失败 -> 使用 OpenCV 找到的候选四角做轻量展开，
           再交给 ZXing-C++ 解码（最终兜底）

    重要：
        - OpenCV 只有真正 decode 出非空文本时才会 qr_present=True；
        - 仅 detect 到四角不会判定为二维码；
        - 因此仍然避免普通黑色/蓝色物体的假阳性。
    """

    def __init__(
        self,
        *,
        try_rotate: bool = True,
        try_downscale: bool = False,
        try_invert: bool = False,
        max_input_side: int = 640,
        opencv_curved_fallback: bool = True,
        zxing_rectify_fallback: bool = True,
        curved_warp_size: int = 360,
        curved_quiet_zone: int = 28,
        curved_gammas: tuple[float, ...] = (0.72, 0.82, 0.90, 1.10, 1.22),
        **legacy_options: Any,
    ) -> None:
        self.try_rotate = bool(try_rotate)
        self.try_downscale = bool(try_downscale)
        self.try_invert = bool(try_invert)
        self.max_input_side = max(0, int(max_input_side))

        self.opencv_curved_fallback = bool(opencv_curved_fallback)
        self.zxing_rectify_fallback = bool(zxing_rectify_fallback)
        self.curved_warp_size = max(240, int(curved_warp_size))
        self.curved_quiet_zone = max(8, int(curved_quiet_zone))
        self.curved_gammas = tuple(float(v) for v in curved_gammas)
        self.legacy_options = dict(legacy_options)

        try:
            self.zxingcpp = import_module("zxingcpp")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "缺少 ZXing-C++ Python 绑定。请执行：\n"
                "    python3 -m pip install zxing-cpp==3.1.1\n"
                "安装完成后重新运行侧面识别。"
            ) from exc

        self.qr_format = self.zxingcpp.BarcodeFormat.QRCode
        self._opencv_detector = cv2.QRCodeDetector()

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None:
            raise ValueError("QRDetector 输入图像为空")
        if not isinstance(image, np.ndarray):
            raise TypeError("QRDetector 输入必须为 numpy.ndarray")
        if image.size == 0:
            raise ValueError("QRDetector 输入图像尺寸为 0")
        if image.ndim == 2:
            return
        if image.ndim == 3 and image.shape[2] == 3:
            return
        raise ValueError(f"QRDetector 只支持灰度/BGR uint8 图像：{image.shape}")

    @staticmethod
    def empty_result() -> dict[str, Any]:
        return {
            "found": False,
            "method": "zxingcpp",
            "backend": "zxing-cpp",
            "confidence": 0.0,
            "points": None,
            "finder_centers": [],
            "verified_finders": 0,
            "center": None,
            "normalized_center": None,
            "text": "",
            "data_hex": "",
            "format": "",
            "content_type": "",
            "orientation_deg": None,
            "ec_level": "",
            "symbology_identifier": "",
            "input_scale": 1.0,
            "curved_recovery_used": False,
            "curved_axis": "",
            "curved_gamma": None,
            "curved_candidate_found": False,
            "opencv_curved_attempted": False,
            "opencv_curved_decoded": False,
        }

    @staticmethod
    def _normalized_center(center: tuple[float, float] | None, image_shape):
        if center is None:
            return None
        h, w = image_shape[:2]
        if w <= 0 or h <= 0:
            return None
        return (float(center[0] / w), float(center[1] / h))

    @staticmethod
    def _point_xy(point: Any) -> tuple[float, float]:
        return float(point.x), float(point.y)

    @staticmethod
    def _safe_str(value: Any) -> str:
        try:
            return str(value)
        except Exception:
            return ""

    def _prepare_input(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        maximum = max(h, w)
        if self.max_input_side <= 0 or maximum <= self.max_input_side:
            return image, 1.0
        scale = float(self.max_input_side) / float(maximum)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        work = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return work, scale

    def _read_zxing(self, image: np.ndarray):
        return self.zxingcpp.read_barcode(
            image,
            formats=self.qr_format,
            try_rotate=self.try_rotate,
            try_downscale=self.try_downscale,
            try_invert=self.try_invert,
            return_errors=False,
        )

    def _barcode_points(
        self,
        barcode: Any,
        *,
        inverse_scale: float,
        image_shape,
    ) -> np.ndarray | None:
        try:
            pos = barcode.position
            ordered = (
                pos.top_left,
                pos.top_right,
                pos.bottom_right,
                pos.bottom_left,
            )
            pts = np.asarray([self._point_xy(p) for p in ordered], dtype=np.float32)
            pts *= float(inverse_scale)
        except Exception:
            return None

        if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
            return None
        h, w = image_shape[:2]
        pts[:, 0] = np.clip(pts[:, 0], 0.0, max(0.0, float(w - 1)))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, max(0.0, float(h - 1)))
        return pts

    def _barcode_result(
        self,
        barcode: Any,
        *,
        image_shape,
        points: np.ndarray | None,
        method: str,
        input_scale: float,
        curved_axis: str = "",
        curved_gamma: float | None = None,
        candidate_found: bool = False,
    ) -> dict[str, Any]:
        try:
            valid = bool(barcode.valid)
        except Exception:
            valid = True
        if not valid:
            return self.empty_result()

        center = None
        if points is not None:
            c = np.mean(points, axis=0)
            center = (float(c[0]), float(c[1]))

        try:
            raw_bytes = bytes(barcode.bytes)
            data_hex = raw_bytes.hex()
        except Exception:
            data_hex = ""

        try:
            orientation = int(barcode.orientation)
        except Exception:
            orientation = None

        return {
            "found": True,
            "method": method,
            "backend": "zxing-cpp",
            "confidence": 1.0,
            "points": points,
            "finder_centers": [],
            "verified_finders": 0,
            "center": center,
            "normalized_center": self._normalized_center(center, image_shape),
            "text": self._safe_str(getattr(barcode, "text", "")),
            "data_hex": data_hex,
            "format": self._safe_str(getattr(barcode, "format", "")),
            "content_type": self._safe_str(getattr(barcode, "content_type", "")),
            "orientation_deg": orientation,
            "ec_level": self._safe_str(getattr(barcode, "ec_level", "")),
            "symbology_identifier": self._safe_str(
                getattr(barcode, "symbology_identifier", "")
            ),
            "input_scale": float(input_scale),
            "curved_recovery_used": method == "zxingcpp_curved_rectify",
            "curved_axis": curved_axis,
            "curved_gamma": curved_gamma,
            "curved_candidate_found": bool(candidate_found),
            "opencv_curved_attempted": method == "opencv_curved_decode",
            "opencv_curved_decoded": False,
        }

    def _opencv_result(
        self,
        *,
        text: str,
        points: np.ndarray | None,
        image_shape,
    ) -> dict[str, Any]:
        center = None
        if points is not None:
            c = np.mean(points, axis=0)
            center = (float(c[0]), float(c[1]))

        return {
            "found": True,
            "method": "opencv_curved_decode",
            "backend": "opencv-curved",
            "confidence": 1.0,
            "points": points,
            "finder_centers": [],
            "verified_finders": 0,
            "center": center,
            "normalized_center": self._normalized_center(center, image_shape),
            "text": str(text),
            "data_hex": str(text).encode("utf-8", errors="replace").hex(),
            "format": "QRCode",
            "content_type": "Text",
            "orientation_deg": None,
            "ec_level": "",
            "symbology_identifier": "",
            "input_scale": 1.0,
            "curved_recovery_used": True,
            "curved_axis": "opencv",
            "curved_gamma": None,
            "curved_candidate_found": points is not None,
            "opencv_curved_attempted": True,
            "opencv_curved_decoded": True,
        }

    @staticmethod
    def _sanitize_quad(points: Any, image_shape) -> np.ndarray | None:
        if points is None:
            return None
        try:
            quad = np.asarray(points, dtype=np.float32).reshape(4, 2).copy()
        except Exception:
            return None
        if not np.all(np.isfinite(quad)):
            return None
        h, w = image_shape[:2]
        quad[:, 0] = np.clip(quad[:, 0], 0.0, max(0.0, float(w - 1)))
        quad[:, 1] = np.clip(quad[:, 1], 0.0, max(0.0, float(h - 1)))
        area = abs(float(cv2.contourArea(quad)))
        if area < max(400.0, 0.002 * float(h * w)):
            return None
        return quad

    def _try_opencv_curved(
        self, image: np.ndarray
    ) -> tuple[dict[str, Any] | None, np.ndarray | None, bool]:
        """
        调用 OpenCV 原生曲面 QR 解码。

        返回：(成功结果或None, 候选四角或None, 是否实际尝试过)
        只有 decoded text 非空才算成功。
        """
        if not self.opencv_curved_fallback:
            return None, None, False

        text = ""
        points = None
        try:
            out = self._opencv_detector.detectAndDecodeCurved(image)
            if isinstance(out, tuple):
                if len(out) >= 1:
                    text = str(out[0] or "")
                if len(out) >= 2:
                    points = self._sanitize_quad(out[1], image.shape)
            else:
                text = str(out or "")
        except Exception:
            # 某些 OpenCV 构建可能没有 curved API；后续仍可走候选四角+ZXing 展开。
            text = ""
            points = None

        if text:
            return (
                self._opencv_result(text=text, points=points, image_shape=image.shape),
                points,
                True,
            )

        # detectAndDecodeCurved 失败时经常仍会返回 points；若没有，再单独 detect 一次。
        if points is None:
            try:
                ok, raw_points = self._opencv_detector.detect(image)
                if ok:
                    points = self._sanitize_quad(raw_points, image.shape)
            except Exception:
                points = None

        return None, points, True

    def _warp_candidate(self, image: np.ndarray, quad: np.ndarray) -> np.ndarray:
        size = self.curved_warp_size
        qz = min(self.curved_quiet_zone, size // 5)
        dst = np.asarray(
            [
                [qz, qz],
                [size - qz - 1, qz],
                [size - qz - 1, size - qz - 1],
                [qz, size - qz - 1],
            ],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(
            np.asarray(quad, dtype=np.float32).reshape(4, 2), dst
        )
        return cv2.warpPerspective(
            image,
            M,
            (size, size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    @staticmethod
    def _nonlinear_axis_warp(image: np.ndarray, gamma: float, axis: str) -> np.ndarray:
        h, w = image.shape[:2]
        yy, xx = np.indices((h, w), dtype=np.float32)

        if axis == "x":
            u = (xx / max(1.0, float(w - 1))) * 2.0 - 1.0
            src_u = np.sign(u) * np.power(np.abs(u), float(gamma))
            map_x = (src_u + 1.0) * 0.5 * float(w - 1)
            map_y = yy
        elif axis == "y":
            u = (yy / max(1.0, float(h - 1))) * 2.0 - 1.0
            src_u = np.sign(u) * np.power(np.abs(u), float(gamma))
            map_y = (src_u + 1.0) * 0.5 * float(h - 1)
            map_x = xx
        else:
            return image

        return cv2.remap(
            image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    def _try_decode_variant(self, image: np.ndarray):
        try:
            barcode = self._read_zxing(image)
        except Exception:
            barcode = None
        if barcode is not None:
            return barcode

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        try:
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6)).apply(gray)
            return self._read_zxing(clahe)
        except Exception:
            return None

    def _recover_with_zxing_rectify(
        self, image: np.ndarray, quad: np.ndarray | None
    ) -> dict[str, Any] | None:
        if not self.zxing_rectify_fallback or quad is None:
            return None

        base = self._warp_candidate(image, quad)

        barcode = self._try_decode_variant(base)
        if barcode is not None:
            return self._barcode_result(
                barcode,
                image_shape=image.shape,
                points=quad,
                method="zxingcpp_curved_rectify",
                input_scale=1.0,
                curved_axis="perspective",
                curved_gamma=1.0,
                candidate_found=True,
            )

        for axis in ("x", "y"):
            for gamma in self.curved_gammas:
                candidate = self._nonlinear_axis_warp(base, gamma, axis)
                barcode = self._try_decode_variant(candidate)
                if barcode is None:
                    continue
                return self._barcode_result(
                    barcode,
                    image_shape=image.shape,
                    points=quad,
                    method="zxingcpp_curved_rectify",
                    input_scale=1.0,
                    curved_axis=axis,
                    curved_gamma=float(gamma),
                    candidate_found=True,
                )
        return None

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        self._validate_image(image)

        # ---------- 1. ZXing-C++ 极速主路径 ----------
        work, scale = self._prepare_input(image)
        try:
            barcode = self._read_zxing(work)
        except Exception as exc:
            result = self.empty_result()
            result["method"] = "zxingcpp_error"
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result

        if barcode is not None:
            inverse_scale = 1.0 / max(scale, 1e-12)
            points = self._barcode_points(
                barcode,
                inverse_scale=inverse_scale,
                image_shape=image.shape,
            )
            return self._barcode_result(
                barcode,
                image_shape=image.shape,
                points=points,
                method="zxingcpp_decode",
                input_scale=float(scale),
            )

        # ---------- 2. OpenCV 原生曲面 QR 回退 ----------
        opencv_result, quad, attempted = self._try_opencv_curved(image)
        if opencv_result is not None:
            return opencv_result

        # ---------- 3. OpenCV 定位 + ZXing 展开兜底 ----------
        recovered = self._recover_with_zxing_rectify(image, quad)
        if recovered is not None:
            recovered["opencv_curved_attempted"] = bool(attempted)
            return recovered

        result = self.empty_result()
        result["opencv_curved_attempted"] = bool(attempted)
        result["curved_candidate_found"] = quad is not None
        return result

    @staticmethod
    def prepare_debug_binary(image: np.ndarray) -> np.ndarray:
        QRDetector._validate_image(image)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )

    @staticmethod
    def draw_result(
        image: np.ndarray,
        raw_result: dict[str, Any],
        stable_result: dict[str, Any] | None = None,
    ) -> np.ndarray:
        display = image.copy()
        final_found = (
            bool(raw_result.get("found", False))
            if stable_result is None
            else bool(stable_result.get("found", False))
        )

        points = raw_result.get("points")
        if raw_result.get("found", False) and points is not None:
            try:
                pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(display, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass

        cv2.putText(
            display,
            "QR: YES" if final_found else "QR: NO",
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0) if final_found else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"Backend: {raw_result.get('method', 'zxingcpp')}",
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if raw_result.get("found", False):
            text = str(raw_result.get("text", ""))
            if len(text) > 36:
                text = text[:33] + "..."
            cv2.putText(
                display,
                f"Text: {text}",
                (15, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return display
