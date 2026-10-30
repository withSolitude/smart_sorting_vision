"""顶部相机单物体抓取规划预览。

这个脚本只运行顶部视觉和抓取规划，不打开 STM32，不发送任何运动命令。
画面中会显示当前选中的一个物体、规划抓取点、闭合轴方向、夹爪纵向方向、
机械坐标、协议 Yaw、实际物理 Yaw 和动态预开口，适合先检查视觉规划是否合理。

运行：
    python -m scripts.last_debug.test_single_object_grasp
    python -m scripts.last_debug.test_single_object_grasp --object-index 2

按键：
    [ / ]  切换可显示的候选物体
    R      清空多帧稳定计数
    P      暂停/继续画面
    S      保存当前标注图
    I      在终端打印当前抓取规划
    Q/ESC 退出
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.app_config import load_runtime_config
from vision.calibration.yaw_servo import YawServoCalibration
from vision.cameras.top_camera import TopCamera
from vision.pipeline.top_pick_pipeline import TopPickPipeline


def _build_pipeline(cfg: dict) -> TopPickPipeline:
    """从正式 YAML 构造与比赛主程序相同的顶部抓取规划器。"""
    # 复用正式调试预览脚本的配置构造，避免维护第二套参数。
    from scripts.last_use.test_top_gripper_rect_overlay import build_top_pipeline

    return build_top_pipeline(cfg)


def _draw_polygon(image: np.ndarray, polygon, color, thickness: int = 2) -> None:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) >= 2:
        cv2.polylines(
            image,
            [np.round(points).astype(np.int32)],
            True,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_cross(image: np.ndarray, center, color, size: int = 10) -> None:
    x, y = int(round(center[0])), int(round(center[1]))
    cv2.drawMarker(
        image,
        (x, y),
        color,
        cv2.MARKER_CROSS,
        2 * size,
        2,
        cv2.LINE_AA,
    )


def _draw_arrow(image: np.ndarray, p0, p1, color, label: str) -> None:
    start = (int(round(p0[0])), int(round(p0[1])))
    end = (int(round(p1[0])), int(round(p1[1])))
    cv2.arrowedLine(image, start, end, color, 2, cv2.LINE_AA, tipLength=0.18)
    cv2.putText(
        image,
        label,
        (end[0] + 6, end[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_robot_direction(
    image: np.ndarray,
    pipeline: TopPickPipeline,
    candidate,
    physical_yaw_deg: float,
    *,
    length_mm: float = 20.0,
) -> None:
    """把机器人 XY 中的闭合轴和纵向轴反投影到当前 ROI 图像。"""
    try:
        center_px = tuple(float(v) for v in candidate.grasp_pixel)
        center_robot = np.asarray(
            pipeline.transform.pixel_to_robot(
                center_px[0], center_px[1], apply_tcp_offset=False
            ),
            dtype=np.float64,
        )
        theta = math.radians(float(physical_yaw_deg))
        closing = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
        longitudinal = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)

        # 方向线从抓取点向两边延伸；同一条轴的反向等价于180°周期。
        close_end = pipeline.transform.robot_to_pixel(
            *(center_robot + length_mm * closing), apply_tcp_offset=False
        )
        long_end = pipeline.transform.robot_to_pixel(
            *(center_robot + length_mm * longitudinal), apply_tcp_offset=False
        )
        _draw_arrow(image, center_px, close_end, (255, 0, 255), "closing axis")
        _draw_arrow(image, center_px, long_end, (0, 220, 255), "longitudinal")
    except (TypeError, ValueError, RuntimeError, IndexError):
        # 方向显示失败不影响候选规划和其它信息显示。
        return


def _put_panel(image: np.ndarray, lines: list[str]) -> None:
    if not lines:
        return
    scale = 0.52
    line_h = 22
    widths = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for line in lines]
    width = min(image.shape[1] - 20, max(widths, default=0) + 18)
    height = line_h * len(lines) + 12
    cv2.rectangle(image, (8, 8), (8 + width, 8 + height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (15, 30 + index * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _select_candidate(result, selected_index: int | None):
    candidates = list(getattr(result, "candidates", []) or [])
    if not candidates:
        return None
    if selected_index is not None:
        matched = [c for c in candidates if int(c.object_index) == int(selected_index)]
        if matched:
            return max(matched, key=lambda c: float(c.score))
    # READY 后优先显示共识结果对应的候选，保证画面中的点就是将要使用的点。
    stable_pixel = getattr(result, "grasp_pixel", None)
    if stable_pixel is not None:
        sx, sy = float(stable_pixel[0]), float(stable_pixel[1])
        return min(
            candidates,
            key=lambda c: (float(c.grasp_pixel[0]) - sx) ** 2
            + (float(c.grasp_pixel[1]) - sy) ** 2,
        )
    return max(candidates, key=lambda c: float(c.score))


def _print_candidate(candidate, yaw: YawServoCalibration) -> None:
    physical = yaw.physical_axis_deg(float(candidate.yaw_deg))
    servo = yaw.command_deg(float(candidate.yaw_deg))
    print("[GRASP PLAN]")
    print(f"  object_index = {candidate.object_index}")
    print(f"  pixel        = ({candidate.grasp_pixel[0]}, {candidate.grasp_pixel[1]})")
    print(f"  robot_xy_mm  = ({candidate.x_mm:.3f}, {candidate.y_mm:.3f})")
    print(f"  yaw_axis_deg = {candidate.yaw_deg:+.3f}")
    print(f"  physical_deg = {physical:+.3f}")
    print(f"  servo_deg    = {servo:+.3f}")
    print(f"  preopen_mm   = {candidate.preopen_mm}")
    print(f"  width_mm     = {candidate.grasp_width_mm}")
    print(f"  clearance_mm = {candidate.gripper_clearance_mm:.3f}")
    print(f"  score        = {candidate.score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单物体抓取点和方向规划预览（不连接下位机）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "runtime_config.yaml",
        help="运行配置 YAML，默认使用正式配置",
    )
    parser.add_argument(
        "--object-index",
        type=int,
        default=None,
        help="固定显示模型返回的 object_index；不填则显示得分最高的一个",
    )
    parser.add_argument("--save-dir", type=Path, default=ROOT / "outputs" / "debug" / "single_grasp")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_runtime_config(ROOT, config_path.resolve())
    yaw_calibration = YawServoCalibration.from_config(cfg["calibration"]["yaw_servo"])

    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )
    camera = TopCamera()
    pipeline = _build_pipeline(cfg)
    window = "SINGLE OBJECT GRASP PLAN"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1500, 900)

    selected_index = args.object_index
    paused = False
    last_frame = None
    last_result = None
    save_count = 0
    fps_t0 = time.perf_counter()
    fps_count = 0
    fps = 0.0

    print("[INFO] 只做顶部视觉规划，不连接 STM32，不会驱动电机。")
    print("[INFO] 绿色=目标轮廓/抓取点，紫色=夹爪闭合轴，青色=夹爪纵向轴。")
    print("[INFO] [ / ] 切换物体，R 重置稳定计数，P 暂停，S 保存，I 打印，Q/ESC 退出。")

    try:
        camera.open()
        while True:
            if not paused or last_frame is None:
                frame = camera.read()
                if frame is None:
                    continue
                last_frame = frame.copy()
                last_result = pipeline.process(frame)
                fps_count += 1
                elapsed = time.perf_counter() - fps_t0
                if elapsed >= 1.0:
                    fps = fps_count / elapsed
                    fps_count = 0
                    fps_t0 = time.perf_counter()

            view = last_frame.copy()
            result = last_result
            candidate = _select_candidate(result, selected_index)

            # 其它候选只画细轮廓，当前选中的目标突出显示。
            for other in getattr(result, "candidates", []) or []:
                if candidate is not None and other is candidate:
                    continue
                _draw_polygon(view, other.polygon, (0, 180, 180), 1)

            if candidate is not None:
                _draw_polygon(view, candidate.polygon, (0, 255, 0), 2)
                for footprint in candidate.gripper_footprint_polygons_px:
                    overlay = view.copy()
                    _draw_polygon(overlay, footprint, (0, 255, 0), 2)
                    cv2.addWeighted(overlay, 0.30, view, 0.70, 0.0, dst=view)
                    _draw_polygon(view, footprint, (0, 255, 0), 2)

                _draw_cross(view, candidate.grasp_pixel, (0, 255, 0), 12)
                physical_yaw = yaw_calibration.physical_axis_deg(float(candidate.yaw_deg))
                _draw_robot_direction(view, pipeline, candidate, physical_yaw)
                gx, gy = candidate.grasp_pixel
                cv2.putText(
                    view,
                    f"object={candidate.object_index} score={candidate.score:.3f}",
                    (int(gx) + 14, max(24, int(gy) - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            ready = bool(getattr(result, "ready", False)) if result is not None else False
            panel = [
                f"FPS={fps:.1f}  paused={paused}",
                f"objects={0 if result is None else result.object_count}  candidates={0 if result is None else result.candidate_count}",
                f"READY={ready}  consensus={0 if result is None else result.votes}/{0 if result is None else result.required_hits}",
                f"selected={selected_index if selected_index is not None else 'best-score'}",
            ]
            if candidate is not None:
                physical_yaw = yaw_calibration.physical_axis_deg(float(candidate.yaw_deg))
                servo_yaw = yaw_calibration.command_deg(float(candidate.yaw_deg))
                panel.extend(
                    [
                        f"pixel=({candidate.grasp_pixel[0]},{candidate.grasp_pixel[1]})",
                        f"XY=({candidate.x_mm:.1f},{candidate.y_mm:.1f}) mm",
                        f"yaw axis={candidate.yaw_deg:+.1f}  physical={physical_yaw:+.1f}  servo={servo_yaw:+.1f}",
                        f"width={candidate.grasp_width_mm} mm  preopen={candidate.preopen_mm} mm",
                        f"clearance={candidate.gripper_clearance_mm:.2f} mm  mode={candidate.orientation_mode}",
                    ]
                )
            elif result is not None:
                panel.append(f"reason={result.reason}")
            _put_panel(view, panel)

            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("p"), ord("P")):
                paused = not paused
            elif key in (ord("r"), ord("R")):
                pipeline.reset()
                print("[INFO] consensus reset")
            elif key in (ord("["), ord("]")):
                available = sorted(
                    {int(c.object_index) for c in getattr(result, "candidates", []) or []}
                )
                if available:
                    if selected_index not in available:
                        selected_index = available[0]
                    else:
                        step = -1 if key == ord("[") else 1
                        selected_index = available[(available.index(selected_index) + step) % len(available)]
                    print(f"[INFO] selected object_index={selected_index}")
            elif key in (ord("s"), ord("S")):
                args.save_dir.mkdir(parents=True, exist_ok=True)
                save_count += 1
                path = args.save_dir / f"single_grasp_{save_count:03d}.png"
                cv2.imwrite(str(path), view)
                print(f"[SAVE] {path}")
            elif key in (ord("i"), ord("I")):
                if candidate is None:
                    print("[GRASP PLAN] no safe candidate")
                else:
                    _print_candidate(candidate, yaw_calibration)
    finally:
        try:
            camera.close()
        finally:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
