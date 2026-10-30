"""检查 YAML 预开口线性公式与 PC/MCU 生成结果是否一致。

此检查不连接串口，也不运动夹爪；它只验证当前 ``runtime_config.yaml`` 中
的公式参数，避免旧版查表测试误读新公式。
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.app_config import load_runtime_config
from vision.grasp.claw_preopen_profile import ClawPreopenProfile, firmware_header


def main() -> int:
    cfg = load_runtime_config(ROOT, ROOT / "config" / "runtime_config.yaml")
    data = cfg["vision"]["top"]["grasp_orientation"]["dynamic_preopen"]
    profile = ClawPreopenProfile.from_runtime_config(data)
    low = profile.angle_for_gap(profile.min_command_mm)
    high = profile.angle_for_gap(profile.max_command_mm)
    if not low > high:
        raise AssertionError("开口变大时舵机角度必须减小")
    if profile.command_for_width(1.0) < profile.min_command_mm:
        raise AssertionError("最小物体宽度没有被夹在最小命令范围内")
    if "kClosedAngleDeg" not in firmware_header(data):
        raise AssertionError("MCU 公式头文件没有生成闭合角参数")
    print("预开口公式检查通过")
    print(f"gap range: {profile.min_command_mm:.1f}..{profile.max_command_mm:.1f} mm")
    print(f"servo range: {low:.2f}..{high:.2f} deg")
    print(f"margin: {profile.margin_mm:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
