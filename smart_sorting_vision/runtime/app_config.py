from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from vision.calibration.yaw_servo import YawServoCalibration


class ConfigError(ValueError):
    """Raised when runtime_config.yaml is incomplete or inconsistent."""


def _as_float_pair(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{name} 必须是长度为 2 的数组")
    try:
        a = float(value[0])
        b = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} 必须由数值组成") from exc
    if a > b:
        raise ConfigError(f"{name} 必须按 [min, max] 排列，当前={value}")
    return a, b


def _resolve_path(project_root: Path, value: str | Path, *, name: str) -> str:
    if value is None:
        raise ConfigError(f"缺少路径配置: {name}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def load_runtime_config(project_root: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Load, resolve and minimally validate the production runtime config.

    Relative resource paths are always resolved from the project root, so the
    application behaves the same regardless of the current working directory.
    """

    root = Path(project_root).resolve()
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"配置不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = deepcopy(raw)

    for section in ("stm32", "camera", "model", "calibration", "mechanics", "vision", "runtime"):
        if section not in cfg:
            raise ConfigError(f"缺少配置段: {section}")

    for camera_name in ("top", "side"):
        camera_cfg = cfg["camera"].get(camera_name)
        if not isinstance(camera_cfg, dict):
            raise ConfigError(f"camera.{camera_name} 必须是对象")
        if not camera_cfg.get("device"):
            raise ConfigError(f"缺少 camera.{camera_name}.device")
        for key in ("width", "height", "fps", "fourcc", "buffer_size", "controls"):
            if key not in camera_cfg:
                raise ConfigError(f"缺少 camera.{camera_name}.{key}")
    if "safe_roi" not in cfg["camera"]["top"] or "roi" not in cfg["camera"]["side"]:
        raise ConfigError("camera.top.safe_roi 和 camera.side.roi 必须统一写在 YAML")
    top_calibration = cfg["calibration"].get("top_camera")
    if not isinstance(top_calibration, dict):
        raise ConfigError("缺少 calibration.top_camera")
    for key in (
        "image_dir",
        "pattern_type",
        "pattern_size",
        "circle_spacing_mm",
        "fix_k3",
        "new_camera_alpha",
        "min_valid_images",
        "warning_error_px",
        "max_view_error_px",
    ):
        if key not in top_calibration:
            raise ConfigError(f"缺少 calibration.top_camera.{key}")
    if str(top_calibration["pattern_type"]).lower() not in {
        "symmetric_circles",
        "circles",
        "circle_grid",
    }:
        raise ConfigError(
            "calibration.top_camera.pattern_type 当前只支持 symmetric_circles"
        )
    pattern_size = top_calibration["pattern_size"]
    if (
        not isinstance(pattern_size, (list, tuple))
        or len(pattern_size) != 2
        or any(int(value) < 3 for value in pattern_size)
    ):
        raise ConfigError(
            "calibration.top_camera.pattern_size 必须是至少为3x3的 [列, 行]"
        )
    if float(top_calibration["circle_spacing_mm"]) <= 0:
        raise ConfigError(
            "calibration.top_camera.circle_spacing_mm 必须大于0"
        )
    if not isinstance(top_calibration["fix_k3"], bool):
        raise ConfigError("calibration.top_camera.fix_k3 必须是 true/false")
    try:
        alpha = float(top_calibration["new_camera_alpha"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "calibration.top_camera.new_camera_alpha 必须是数值"
        ) from exc
    if not 0.0 <= alpha <= 1.0:
        raise ConfigError(
            "calibration.top_camera.new_camera_alpha 必须在 [0, 1] 范围内"
        )
    try:
        max_view_error = float(top_calibration["max_view_error_px"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "calibration.top_camera.max_view_error_px 必须是数值"
        ) from exc
    if max_view_error <= 0:
        raise ConfigError(
            "calibration.top_camera.max_view_error_px 必须大于0"
        )
    background_capture = cfg["calibration"].get("side_background_capture")
    if not isinstance(background_capture, dict):
        raise ConfigError("缺少 calibration.side_background_capture")
    for key in ("settle_frames", "frame_count"):
        if key not in background_capture:
            raise ConfigError(f"缺少 calibration.side_background_capture.{key}")
    if "scheduler" not in cfg["runtime"] or "warmup" not in cfg["runtime"]:
        raise ConfigError("runtime.scheduler 和 runtime.warmup 必须统一写在 YAML")
    debug_cfg = cfg.get("debug")
    if not isinstance(debug_cfg, dict) or "frames_dir" not in debug_cfg or "dataset_dir" not in debug_cfg:
        raise ConfigError("debug.frames_dir 和 debug.dataset_dir 必须统一写在 YAML")

    # Resolve every production resource path once, at the unique config entry.
    cfg["model"]["top"]["path"] = _resolve_path(
        root, cfg["model"]["top"]["path"], name="model.top.path"
    )
    cfg["model"]["side"]["path"] = _resolve_path(
        root, cfg["model"]["side"]["path"], name="model.side.path"
    )
    cfg["calibration"]["top_intrinsics"] = _resolve_path(
        root, cfg["calibration"]["top_intrinsics"], name="calibration.top_intrinsics"
    )
    cfg["calibration"]["pixel_to_robot"] = _resolve_path(
        root, cfg["calibration"]["pixel_to_robot"], name="calibration.pixel_to_robot"
    )
    cfg["calibration"]["side_background"] = _resolve_path(
        root, cfg["calibration"]["side_background"], name="calibration.side_background"
    )

    workspace = cfg["mechanics"].get("workspace_mm", {})
    x_limits = _as_float_pair(workspace.get("x"), name="mechanics.workspace_mm.x")
    y_limits = _as_float_pair(workspace.get("y"), name="mechanics.workspace_mm.y")
    cfg["mechanics"]["workspace_mm"]["x"] = list(x_limits)
    cfg["mechanics"]["workspace_mm"]["y"] = list(y_limits)

    # 抓取方向和TCP共用此标定，不再在其他块另存一组Yaw范围。
    try:
        YawServoCalibration.from_config(cfg["calibration"]["yaw_servo"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"calibration.yaw_servo 无效: {exc}") from exc

    core_cfg = cfg["vision"]["top"].get("shape_core")
    if core_cfg is not None:
        try:
            if not isinstance(core_cfg["enabled"], bool):
                raise ValueError("enabled 必须为 true/false")
            for name in ("quadrilateral", "polygon", "triangle", "round"):
                if not 0.0 < float(core_cfg["ratios"][name]) < 1.0:
                    raise ValueError(f"ratios.{name} 必须在0与1之间")
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"vision.top.shape_core 无效: {exc}") from exc

    top_tcp_comp = cfg["vision"]["top"].get("yaw_tcp_compensation", {})
    if bool(top_tcp_comp.get("enabled", False)):
        try:
            reference_axis = float(top_tcp_comp["reference_axis_deg"])
            offset_axis0 = top_tcp_comp["offset_axis0_mm"]
            if not isinstance(offset_axis0, (list, tuple)) or len(offset_axis0) != 2:
                raise ValueError("offset_axis0_mm 必须为 [x, y]")
            float(offset_axis0[0])
            float(offset_axis0[1])
            visual_to_command = top_tcp_comp.get(
                "visual_to_command_matrix",
                [[1.0, 0.0], [0.0, 1.0]],
            )
            if (
                not isinstance(visual_to_command, (list, tuple))
                or len(visual_to_command) != 2
                or any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in visual_to_command)
            ):
                raise ValueError("visual_to_command_matrix 必须为 2x2 数值矩阵")
            a, b = float(visual_to_command[0][0]), float(visual_to_command[0][1])
            c, d = float(visual_to_command[1][0]), float(visual_to_command[1][1])
            if abs(a * d - b * c) < 1e-9:
                raise ValueError("visual_to_command_matrix 不可逆/退化")
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"vision.top.yaw_tcp_compensation 无效: {exc}") from exc
        _ = reference_axis

    descent_drift = cfg["vision"]["top"].get("descent_drift_compensation", {})
    if bool(descent_drift.get("enabled", False)):
        try:
            signed_drift_mm = float(descent_drift.get("signed_drift_mm", 0.0))
            relative_angle_deg = float(descent_drift.get("relative_angle_deg", 90.0))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"vision.top.descent_drift_compensation 无效: {exc}") from exc
        if not (-100.0 <= signed_drift_mm <= 100.0):
            raise ConfigError(
                "vision.top.descent_drift_compensation.signed_drift_mm 数值异常，"
                "应填写实际毫米级横向漂移"
            )
        if not (-360.0 <= relative_angle_deg <= 360.0):
            raise ConfigError(
                "vision.top.descent_drift_compensation.relative_angle_deg 必须在 [-360, 360]"
            )

    # 动态预开口只使用 YAML 中的线性公式参数；PC 和 STM32 必须保持一致。
    preopen = (
        cfg["vision"]["top"].get("grasp_orientation", {}).get("dynamic_preopen", {}) or {}
    )
    if bool(preopen.get("enabled", False)):
        try:
            if preopen.get("configured") is not True:
                raise ValueError("configured 必须为 true")
            margin = float(preopen["margin_mm"])
            minimum = float(preopen["min_command_mm"])
            maximum = float(preopen["max_command_mm"])
            closed = float(preopen["closed_servo_deg"])
            standby = float(preopen["standby_servo_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"vision.top.grasp_orientation.dynamic_preopen 无效: {exc}") from exc
        if margin <= 0.0 or minimum <= 0.0 or maximum <= minimum or maximum > 120.0:
            raise ConfigError("dynamic_preopen 的 margin/min/max 数值无效")
        if maximum <= margin:
            raise ConfigError("dynamic_preopen.margin_mm 过大，没有可用物体宽度")
        if not (-135.0 <= closed <= 135.0 and -135.0 <= standby <= 135.0) or closed == standby:
            raise ConfigError("dynamic_preopen 的 closed/standby 舵机角无效")

    top_consensus = cfg["vision"]["top"].get("consensus", {})
    side_consensus = cfg["vision"]["side"].get("consensus", {})
    for name, item in (("vision.top.consensus", top_consensus), ("vision.side.consensus", side_consensus)):
        window = int(item.get("window", 0))
        required = int(item.get("required", 0))
        if window <= 0 or required <= 0 or required > window:
            raise ConfigError(f"{name} 无效: required={required}, window={window}")

    # 二维码只有两层开关：是否运行识别，以及是否把解码内容纳入多帧一致性。
    # 内容开关关闭时，正式流程只使用 qr_present，不会因为不同文本阻塞 READY。
    side_recognition = cfg["vision"]["side"].get("recognition")
    if not isinstance(side_recognition, dict):
        raise ConfigError("缺少 vision.side.recognition")
    qr_cfg = side_recognition.get("qr")
    if not isinstance(qr_cfg, dict):
        raise ConfigError("缺少 vision.side.recognition.qr")
    for key, default in (
        ("enabled", True),
        ("check_content_consistency", False),
    ):
        if key not in qr_cfg:
            qr_cfg[key] = default
        if not isinstance(qr_cfg[key], bool):
            raise ConfigError(f"vision.side.recognition.qr.{key} 必须是 true/false")

    shape_cfg = side_recognition.get("shape")
    if not isinstance(shape_cfg, dict):
        raise ConfigError("缺少 vision.side.recognition.shape")
    if "enabled" in shape_cfg and not isinstance(shape_cfg["enabled"], bool):
        raise ConfigError("vision.side.recognition.shape.enabled 必须是 true/false")
    labels = shape_cfg.get("no_object_labels", ["no_object", "empty", "background"])
    if not isinstance(labels, (list, tuple)) or not any(str(label).strip() for label in labels):
        raise ConfigError(
            "vision.side.recognition.shape.no_object_labels 必须至少包含一个类别名"
        )
    shape_cfg["no_object_labels"] = [str(label).strip() for label in labels if str(label).strip()]

    crop_cfg = cfg["vision"]["side"].get("crop_extractor")
    if not isinstance(crop_cfg, dict) or not isinstance(crop_cfg.get("extractor"), dict):
        raise ConfigError("缺少 vision.side.crop_extractor.extractor")

    return cfg
