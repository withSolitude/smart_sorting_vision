"""顶部相机单物体抓取规划预览。

默认只运行顶部视觉和抓取规划，不打开 STM32。按 Space 后才会连接下位机，
执行一次动态预开口 + PICK_TARGET 实抓，并在第一阶段完成后发送 STOP。
画面中会显示当前选中的一个物体、规划抓取点、闭合轴方向、夹爪纵向方向、
机械坐标、协议 Yaw、实际物理 Yaw 和动态预开口。

运行：
    python -m scripts.last_debug.test_single_object_grasp
    python -m scripts.last_debug.test_single_object_grasp --object-index 2

按键：
    [ / ]  切换可显示的候选物体
    Space    在下位机已收到物理 START 且画面 READY 时实抓一次
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
    parser = argparse.ArgumentParser(
        description="单物体抓取点/方向预览；按 Space 后执行一次真实抓取"
    )
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
    link = None
    link_state = {"started": False, "last_error": ""}
    pick_helpers = None
    drift_compensator = None
    runtime_cfg = cfg.get("runtime", {}) or {}
    ack_timeout_s = float(runtime_cfg.get("pick_ack_timeout_s", 1.5))
    ready_timeout_s = float(runtime_cfg.get("recognition_ready_timeout_s", 18.0))
    retry_interval_s = float(runtime_cfg.get("retry_interval_s", 0.25))
    last_frame = None
    last_result = None
    save_count = 0
    fps_t0 = time.perf_counter()
    fps_count = 0
    fps = 0.0

    print("[INFO] 启动时只打开顶部相机；按 Space 才会打开 STM32 并执行一次实抓。")
    print("[INFO] 绿色=目标轮廓/抓取点，紫色=夹爪闭合轴，青色=夹爪纵向轴。")
    print("[INFO] 实抓前必须先按下机械物理 START，并等待 READY。")
    print("[INFO] [ / ] 切换物体，Space 实抓，R 重置，P 暂停，S 保存，I 打印，Q/ESC 退出。")

    def load_pick_helpers():
        """延迟加载已验证的单次抓取事务，避免预览启动时打开串口。"""
        from scripts.last_use.test_top_pick_only_debug import (
            PhysicalStop,
            UserQuit,
            build_descent_drift_compensator,
            execute_pick,
            make_frozen_target,
            send_stop_and_wait,
        )

        return {
            "PhysicalStop": PhysicalStop,
            "UserQuit": UserQuit,
            "build_descent_drift_compensator": build_descent_drift_compensator,
            "execute_pick": execute_pick,
            "make_frozen_target": make_frozen_target,
            "send_stop_and_wait": send_stop_and_wait,
        }

    def open_pick_link():
        """第一次实抓时打开串口并先让下位机回到安全 STOP。"""
        nonlocal link, pick_helpers, drift_compensator
        if link is not None and link.is_open:
            return True

        from communication.stm32_link import STM32Link

        pick_helpers = pick_helpers or load_pick_helpers()
        drift_compensator = drift_compensator or pick_helpers[
            "build_descent_drift_compensator"
        ](cfg["vision"]["top"])
        link = STM32Link(
            device=cfg["stm32"]["device"],
            baudrate=cfg["stm32"]["baud"],
        )
        link.open()
        pick_helpers["send_stop_and_wait"](link, link_state, timeout_s=2.0)
        print("[STM32] 已同步到 STOP；请按机械物理 START 后再按 Space。")
        return True

    def poll_link_state() -> None:
        """只读取启动/停止/错误状态，保持预览循环不阻塞。"""
        if link is None or not link.is_open:
            return
        try:
            from communication.sorting_protocol import Mode

            for message in link.poll():
                print(f"[RX] {link.describe(message)}")
                if message.mode == Mode.MCU_START_WORK:
                    link_state["started"] = True
                    link_state["last_error"] = ""
                    print("[READY] 已收到机械 START，可以按 Space 实抓。")
                elif message.mode in (Mode.MCU_STOP_WORK, Mode.MCU_STOP_ACK):
                    link_state["started"] = False
                elif message.mode == Mode.MCU_ERROR:
                    link_state["started"] = False
                    link_state["last_error"] = f"MCU_ERROR code={int(message.box_id)}"
        except Exception as exc:
            link_state["started"] = False
            link_state["last_error"] = str(exc)
            print(f"[STM32][ERROR] {exc}")

    try:
        camera.open()
        while True:
            poll_link_state()
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
                f"MCU={'STARTED' if link_state['started'] else 'STOPPED/OFFLINE'}",
            ]
            if link_state["last_error"]:
                panel.append(f"MCU error={link_state['last_error']}")
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
            elif key == 32:
                # 只有多帧 READY 且下位机收到机械 START 才允许实抓。
                if result is None or not result.ready:
                    print("[BLOCK] 当前抓取规划还没有 READY，继续等待稳定结果。")
                    continue
                try:
                    open_pick_link()
                    if not link_state["started"]:
                        print("[BLOCK] 尚未收到 MCU_START_WORK，请先按机械物理 START。")
                        continue
                    pick_helpers = pick_helpers or load_pick_helpers()
                    frozen = pick_helpers["make_frozen_target"](
                        result,
                        yaw_calibration,
                        drift_compensator,
                    )
                    print(
                        "[REAL PICK] "
                        f"XY=({frozen['command_x_mm']:.2f},{frozen['command_y_mm']:.2f}) "
                        f"Yaw={frozen['yaw_deg']:+.1f} "
                        f"physical={frozen['physical_yaw_deg']:+.1f} "
                        f"preopen={frozen['preopen_mm']:.1f}"
                    )
                    pick_helpers["execute_pick"](
                        link,
                        camera,
                        frozen,
                        link_state,
                        ack_timeout_s=ack_timeout_s,
                        ready_timeout_s=ready_timeout_s,
                        retry_interval_s=retry_interval_s,
                    )
                    # 单次测试完成后主动 STOP，避免继续进入比赛分拣流程。
                    pick_helpers["send_stop_and_wait"](link, link_state, timeout_s=2.0)
                    print("[DONE] 单次抓取第一阶段完成，已发送 STOP。")
                    pipeline.reset()
                    last_result = None
                except Exception as exc:
                    if pick_helpers is not None and isinstance(
                        exc, pick_helpers["UserQuit"]
                    ):
                        break
                    if pick_helpers is not None and isinstance(
                        exc, pick_helpers["PhysicalStop"]
                    ):
                        print(f"[STOP] {exc}")
                        if link is not None and link.is_open:
                            pick_helpers["send_stop_and_wait"](link, link_state, timeout_s=2.0)
                        continue
                    print(f"[REAL PICK][FAIL] {exc}")
                    if link is not None and link.is_open and pick_helpers is not None:
                        pick_helpers["send_stop_and_wait"](link, link_state, timeout_s=2.0)
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
        if link is not None:
            try:
                if link.is_open:
                    pick_helpers = pick_helpers or load_pick_helpers()
                    pick_helpers["send_stop_and_wait"](link, link_state, timeout_s=1.5)
            except Exception as exc:
                print(f"[WARN] cleanup STOP failed: {exc}")
            try:
                link.close()
            except Exception:
                pass
        try:
            camera.release()
        finally:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
