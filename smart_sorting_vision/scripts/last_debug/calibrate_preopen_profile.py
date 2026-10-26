"""根据实测点拟合夹爪预开口的线性公式。

输入格式为 ``舵机角度:内侧间隙_mm``，脚本只做离线拟合，不连接串口、不
运动电机、不自动覆盖 YAML。生成的两个文件需要人工检查后分别安装到 PC
和 MCU。
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.grasp.claw_preopen_profile import ClawPreopenProfile, firmware_header


def fit_angle(points: list[tuple[float, float]]) -> tuple[float, float]:
    """拟合 angle = intercept + slope * gap，返回截距和斜率。"""
    gaps = [gap for _, gap in points]
    angles = [angle for angle, _ in points]
    mean_gap = sum(gaps) / len(gaps)
    mean_angle = sum(angles) / len(angles)
    denominator = sum((gap - mean_gap) ** 2 for gap in gaps)
    if denominator <= 1e-12:
        raise ValueError("所有测量间隙不能相同")
    slope = sum((gap - mean_gap) * (angle - mean_angle)
                for angle, gap in points) / denominator
    intercept = mean_angle - slope * mean_gap
    if not all(math.isfinite(v) for v in (intercept, slope)):
        raise ValueError("拟合结果不是有限数值")
    return intercept, slope


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point", action="append", required=True,
        help="实测点，格式为 servo_deg:gap_mm；至少3个，建议5~7个",
    )
    parser.add_argument("--margin-mm", type=float, default=5.0,
                        help="物体宽度之外的总安全余量")
    parser.add_argument("--confirmed-safe", action="store_true",
                        help="确认每个点均为无碰撞的 PREOPEN 实测位置")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="新的输出目录；不能已存在")
    args = parser.parse_args()
    if not args.confirmed_safe:
        parser.error("先完成空载/低速实测，再加 --confirmed-safe")
    if len(args.point) < 3 or len(args.point) > 32:
        parser.error("--point 至少3个，最多32个")

    points: list[tuple[float, float]] = []
    for raw in args.point:
        try:
            servo, gap = (float(value) for value in raw.split(":", 1))
        except ValueError as exc:
            parser.error(f"无效测量点 {raw!r}，应为 servo_deg:gap_mm")
        if not math.isfinite(servo) or not math.isfinite(gap):
            parser.error(f"测量点必须是有限数值：{raw!r}")
        if not -135.0 <= servo <= 135.0 or gap <= 0.0:
            parser.error(f"超出范围的测量点：{raw!r}")
        points.append((servo, gap))

    gaps = [gap for _, gap in points]
    if len(set(round(gap, 6) for gap in gaps)) != len(gaps):
        parser.error("每个测量点的间隙必须不同")
    closed, slope = fit_angle(points)
    max_gap = max(gaps)
    min_gap = min(gaps)
    standby = closed + slope * max_gap
    if slope >= 0.0:
        parser.error("拟合斜率应为负值：间隙变大时舵机角度应减小")
    if args.margin_mm <= 0.0 or args.margin_mm >= max_gap:
        parser.error("margin-mm 必须大于0且小于最大实测间隙")

    data = {
        "schema_version": 1,
        "profile_kind": "measured_linear",
        "configured": True,
        "hardware_validated": True,
        "created_at": datetime.now().astimezone().isoformat(),
        "source": "user measured servo angle and inner gap; linear fit",
        "closed_servo_deg": closed,
        "standby_servo_deg": standby,
        "margin_mm": float(args.margin_mm),
        "min_command_mm": min_gap,
        "max_command_mm": max_gap,
        "samples": [
            {"servo_deg": servo, "gap_mm": gap}
            for servo, gap in points
        ],
    }
    try:
        profile = ClawPreopenProfile(data)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"拟合出的公式无效：{exc}")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    runtime = {
        key: data[key]
        for key in (
            "profile_kind", "configured", "hardware_validated",
            "closed_servo_deg", "standby_servo_deg", "margin_mm",
            "min_command_mm", "max_command_mm",
        )
    }
    runtime["enabled"] = True
    (out_dir / "dynamic_preopen.yaml").write_text(
        yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (out_dir / "claw_preopen_profile.h").write_text(
        firmware_header(data), encoding="utf-8"
    )
    (out_dir / "preopen_measurements.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"拟合公式：angle = {profile.closed_servo_deg:.4f} + "
          f"({profile.standby_servo_deg - profile.closed_servo_deg:.4f}) "
          f"* gap / {profile.max_command_mm:.4f}")
    print(f"允许命令开口：{profile.min_command_mm:.1f}..{profile.max_command_mm:.1f} mm")
    print(f"输出目录：{out_dir}")
    print("请人工检查后再安装 dynamic_preopen.yaml 和 claw_preopen_profile.h。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
