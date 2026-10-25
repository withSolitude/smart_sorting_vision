from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


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

    top_tcp_comp = cfg["vision"]["top"].get("yaw_tcp_compensation", {})
    if bool(top_tcp_comp.get("enabled", False)):
        try:
            reference_axis = float(top_tcp_comp["reference_axis_deg"])
            offset_axis0 = top_tcp_comp["offset_axis0_mm"]
            if not isinstance(offset_axis0, (list, tuple)) or len(offset_axis0) != 2:
                raise ValueError("offset_axis0_mm 必须为 [x, y]")
            float(offset_axis0[0])
            float(offset_axis0[1])
            min_axis = float(top_tcp_comp.get("min_axis_deg", -87.0))
            max_axis = float(top_tcp_comp.get("max_axis_deg", 89.0))
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
        if min_axis >= max_axis:
            raise ConfigError("vision.top.yaw_tcp_compensation: min_axis_deg 必须小于 max_axis_deg")
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

    # Dynamic claw preopen is configured only in this YAML. The dense
    # millimetre-to-servo lookup is compiled into the STM32 header.
    preopen = (
        cfg["vision"]["top"].get("grasp_orientation", {}).get("dynamic_preopen", {}) or {}
    )
    if bool(preopen.get("enabled", False)):
        try:
            if preopen.get("configured") is not True:
                raise ValueError("configured 必须为 true")
            if preopen.get("profile_kind") not in ("cad_four_bar", "measured"):
                raise ValueError("profile_kind 必须为 cad_four_bar 或 measured")
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

    return cfg
