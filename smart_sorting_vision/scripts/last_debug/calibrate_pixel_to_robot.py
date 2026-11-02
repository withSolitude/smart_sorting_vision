"""用固定点标定顶部完整去畸变图像到 XY 坐标的单应矩阵。

流程
----
1. 在画面中依次点击 4 个标定点（左上、右上、右下、左下）；
2. 加 `--verify-point` 时再点击 1 个**验证点**（强烈建议，默认不启用）；
3. 依次输入每个点的机械 XY；
4. 用 4 点拟合单应，若启用验证点则用第 5 点做独立残差验证；
5. 打印方向自查表并检查工作区包含关系，异常时拒绝保存。

为什么要第 5 点
---------------
4 点单应是**精确解**，残差恒为 0，因此点位与机械坐标配错（例如把图像
左上配到了机械右下）时，拟合结果依然"完美"，但整张图会绕某个中心旋转
180°，表现为"物体在图像右上角，夹爪却跑到左下角"。

第 5 点不参与拟合，能真实反映映射是否正确：配对正确时残差通常是毫米级，
配对旋转时残差会达到几十上百毫米。

保存完整图坐标，正式流水线自动补上 ROI 原点，后续裁剪不必重标。
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from vision.calibration.pixel_to_robot import PixelToRobotTransform

from .common import ROOT, load_context

CALIBRATION_POINT_COUNT = 4
DEFAULT_VERIFY_TOLERANCE_MM = 5.0


# ============================================================
# 校验与报告
# ============================================================

def check_workspace(robot_points, cfg, *, tolerance_mm: float = 1e-6) -> bool:
    """所有标定点必须落在 mechanics.workspace_mm 内（配置缺失时跳过）。"""
    workspace = (cfg.get("mechanics") or {}).get("workspace_mm") or {}
    x_limits = workspace.get("x")
    y_limits = workspace.get("y")
    if not x_limits or not y_limits:
        print("[SKIP] 配置中没有 mechanics.workspace_mm，跳过工作区检查。")
        return True
    ok = True
    for index, (x_mm, y_mm) in enumerate(robot_points, start=1):
        inside = (
            x_limits[0] - tolerance_mm <= x_mm <= x_limits[1] + tolerance_mm
            and y_limits[0] - tolerance_mm <= y_mm <= y_limits[1] + tolerance_mm
        )
        if not inside:
            ok = False
            print(
                f"[WARN] P{index} 机械坐标 ({x_mm:.2f}, {y_mm:.2f}) "
                f"不在工作区 X={x_limits} Y={y_limits} 内，请检查是否输错或符号写反。"
            )
    return ok


def report_direction(robot_points) -> None:
    """打印“图像里的方向”在机械坐标里对应什么，供人工与实机对照。"""
    tl, tr, _br, bl = robot_points[:4]
    to_right = np.asarray(tr, dtype=np.float64) - np.asarray(tl, dtype=np.float64)
    to_down = np.asarray(bl, dtype=np.float64) - np.asarray(tl, dtype=np.float64)
    print("-" * 66)
    print("方向自查（请与实机对照）：")
    print(f"  图像向右 (P1->P2): ΔX={to_right[0]:+8.2f} mm  ΔY={to_right[1]:+8.2f} mm")
    print(f"  图像向下 (P1->P4): ΔX={to_down[0]:+8.2f} mm  ΔY={to_down[1]:+8.2f} mm")
    print(
        "  → 若这两行与“在实机上把夹爪往右/往下移动时 X/Y 的实际变化”相反，\n"
        "    说明点序或坐标配色反了（会让整张图旋转 180°），请重新标定。"
    )
    print("-" * 66)


def print_verify_result(
    verify_error_mm: float | None,
    tolerance_mm: float,
) -> bool:
    if verify_error_mm is None:
        print("[SKIP] 未启用第 5 点验证：本次标定没有独立残差检查。")
        return True

    print(f"验证点残差 = {verify_error_mm:.2f} mm (阈值 {tolerance_mm:.1f} mm)")
    if verify_error_mm <= tolerance_mm:
        print("[OK] 第 5 点验证通过，映射可信。")
        return True

    print(
        "[FAIL] 第 5 点残差过大，映射不可信。\n"
        "       最常见原因：4 个点的点击顺序与输入的机械坐标顺序不一致，\n"
        "       或某个坐标的符号/数字写错。请重新标定。"
    )
    return False


# ============================================================
# main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runtime_config.yaml")
    parser.add_argument(
        "--verify-point",
        action="store_true",
        help="额外点 1 个验证点做独立残差检查（强烈建议，但不改变默认 4 点流程）",
    )
    parser.add_argument(
        "--tolerance-mm",
        type=float,
        default=DEFAULT_VERIFY_TOLERANCE_MM,
        help="第 5 点允许的最大残差（mm）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使校验失败也保存（会写入明显警告）",
    )
    args = parser.parse_args()

    cfg, TopCamera, _ = load_context(ROOT / args.config)
    output = ROOT / cfg["calibration"]["pixel_to_robot"]

    point_count = CALIBRATION_POINT_COUNT + (1 if args.verify_point else 0)

    camera = TopCamera()
    try:
        camera.open()
        # 内参已经生效，但此处不裁ROI、不使用多边形掩膜。
        frame = camera.read_undistorted()
        points: list[tuple[float, float]] = []

        def on_click(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < point_count:
                points.append((float(x), float(y)))
                label = f"P{len(points)}" if len(points) <= 4 else "VERIFY"
                print(f"[PIXEL] {label} = ({float(x):.1f}, {float(y):.1f})")

        window = "pixel to robot: click 4 points + 1 verify"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_click)
        while True:
            view = frame.copy()
            for i, (u, v) in enumerate(points, start=1):
                color = (0, 0, 255) if i <= 4 else (255, 0, 255)
                label = f"P{i}" if i <= 4 else "V"
                cv2.circle(view, (round(u), round(v)), 6, color, 2)
                cv2.putText(view, label, (round(u) + 8, round(v) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            if len(points) < point_count:
                instruction = (
                    f"click {len(points) + 1}/{point_count}"
                    + (" (4 points then 1 verify)" if point_count > 4 else "")
                    + "; R clear; Backspace undo; Q quit"
                )
            else:
                instruction = f"{point_count}/{point_count} complete; Enter continue"
            cv2.putText(view, instruction, (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
            cv2.imshow(window, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return
            if key in (ord("r"), ord("R")):
                points.clear()
                print("[PIXEL] 已清空，重新点击")
                continue
            if key in (8, 127):
                if points:
                    removed = points.pop()
                    print(f"[PIXEL] 撤销：{removed}")
                continue
            if len(points) == point_count and key in (10, 13):
                break

        robot = []
        for i, (u, v) in enumerate(points, start=1):
            label = f"P{i}" if i <= 4 else "验证点"
            x, y = map(float, input(f"{label} pixel=({u:.1f},{v:.1f}) 输入 X_mm Y_mm: ").split())
            robot.append((x, y))

        calibration_pixel = points[:4]
        calibration_robot = robot[:4]
        verify_pixel = points[4] if len(points) > 4 else None
        verify_robot = robot[4] if len(robot) > 4 else None

        H = cv2.getPerspectiveTransform(
            np.float32(calibration_pixel),
            np.float32(calibration_robot),
        )

        verify_error_mm = None
        if verify_pixel is not None:
            predicted = cv2.perspectiveTransform(
                np.asarray([[verify_pixel]], dtype=np.float64), H
            )[0, 0]
            verify_error_mm = float(
                np.linalg.norm(predicted - np.asarray(verify_robot, dtype=np.float64))
            )
            print(
                f"[VERIFY] pixel=({verify_pixel[0]:.1f},{verify_pixel[1]:.1f}) "
                f"预测=({predicted[0]:.2f},{predicted[1]:.2f}) "
                f"实测=({verify_robot[0]:.2f},{verify_robot[1]:.2f})"
            )

        print()
        report_direction(calibration_robot)
        workspace_ok = check_workspace(calibration_robot, cfg)
        verify_ok = print_verify_result(verify_error_mm, args.tolerance_mm)

        trustworthy = workspace_ok and verify_ok
        if not trustworthy and not args.force:
            print()
            print("[ABORT] 标定未保存。确认要继续保存请加 --force。")
            return

        det = float(np.linalg.det(np.asarray(H, dtype=np.float64)[:2, :2]))
        print(f"单应线性部分 det = {det:+.6f}")
        if det < 0:
            print(
                "[WARN] det < 0：该映射包含镜像（左右手系被翻转）。\n"
                "       通常意味着 4 个点的顺序绕向与机械坐标的绕向相反。"
            )

        metadata = {
            "point_count": 4,
            "points_pixel": calibration_pixel,
            "points_robot": calibration_robot,
            "verify_tolerance_mm": args.tolerance_mm,
            "workspace_ok": workspace_ok,
            "force_saved": not trustworthy,
        }
        if verify_error_mm is not None:
            metadata["verify_error_mm"] = verify_error_mm
            metadata["verify_pixel"] = list(verify_pixel)
            metadata["verify_robot"] = list(verify_robot)

        transform = PixelToRobotTransform(
            H, frame.shape[1], frame.shape[0], source="TopCamera.read_undistorted()",
            metadata=metadata,
            coordinate_space="undistorted_full",
        )
        transform.save(output)

        points_record = {
            "pixel": calibration_pixel,
            "robot": calibration_robot,
        }
        if verify_pixel is not None:
            points_record["verify_pixel"] = list(verify_pixel)
            points_record["verify_robot"] = list(verify_robot)
            points_record["verify_error_mm"] = verify_error_mm
        output.with_suffix(".points.json").write_text(
            json.dumps(points_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[SAVE] {output}")
        if not trustworthy:
            print("[WARN] 该文件是在校验未通过的情况下强制保存的，请勿直接用于比赛。")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
