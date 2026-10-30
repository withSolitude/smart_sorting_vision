"""独立测试 Intel RealSense D455 的 RGB、深度和 RGB-深度对齐。

本脚本不读取 runtime_config.yaml，也不打开工程中已有的普通 USB 相机。
运行前请确认只有一个程序占用 D455。按 S 保存一组 RGB、16 位深度、深度
伪彩、对齐叠加图和 JSON 信息；按 Q 或 ESC 退出。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280, help="RGB/深度宽度")
    parser.add_argument("--height", type=int, default=720, help="RGB/深度高度")
    parser.add_argument("--fps", type=int, default=30, help="RGB/深度帧率")
    parser.add_argument(
        "--max-depth-mm", type=float, default=3000.0,
        help="深度伪彩显示上限，单位毫米；不影响保存的原始深度",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/debug/realsense_d455"),
        help="S 键保存目录",
    )
    parser.add_argument(
        "--frames", type=int, default=0,
        help="自动采集帧数；0 表示一直运行直到按 Q/ESC 或 Ctrl-C",
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="无显示器时使用；配合 --save-every 自动保存，不能使用 S 键",
    )
    parser.add_argument(
        "--save-every", type=int, default=0,
        help="每 N 帧自动保存一组；0 表示只按 S 保存",
    )
    return parser.parse_args()


def device_info(device, rs) -> dict:
    """读取设备信息；不同 SDK 版本缺少某个字段时继续运行。"""
    result = {}
    for name, key_name in (
        ("name", "name"),
        ("serial", "serial_number"),
        ("firmware", "firmware_version"),
    ):
        try:
            result[name] = device.get_info(getattr(rs.camera_info, key_name))
        except Exception:
            result[name] = "unknown"
    try:
        result["usb_type"] = device.get_info(rs.camera_info.usb_type_descriptor)
    except Exception:
        result["usb_type"] = "unknown"
    return result


def intrinsics_to_dict(stream_profile) -> dict:
    """把 RealSense 内参对象转换成可写入 JSON 的普通字典。"""
    try:
        intr = stream_profile.as_video_stream_profile().get_intrinsics()
    except Exception:
        return {}
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(value) for value in intr.coeffs],
    }


def colorize_depth(depth_u16: np.ndarray, depth_scale: float, max_depth_mm: float) -> np.ndarray:
    """把深度原始值转换为可视化伪彩图，零值保持黑色。"""
    depth_mm = depth_u16.astype(np.float32) * float(depth_scale) * 1000.0
    ratio = np.clip(depth_mm / max(float(max_depth_mm), 1.0), 0.0, 1.0)
    gray = (ratio * 255.0).astype(np.uint8)
    visible = depth_u16 > 0
    gray[~visible] = 0
    result = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    result[~visible] = 0
    return result


def depth_summary(depth_u16: np.ndarray, depth_scale: float) -> dict:
    """统计当前深度图有效像素和中心点深度。"""
    height, width = depth_u16.shape[:2]
    valid = depth_u16[depth_u16 > 0]
    cx, cy = width // 2, height // 2
    center_raw = int(depth_u16[cy, cx])
    summary = {
        "width": int(width),
        "height": int(height),
        "valid_ratio": float(valid.size / depth_u16.size),
        "center_depth_mm": float(center_raw * depth_scale * 1000.0) if center_raw else None,
    }
    if valid.size:
        values_mm = valid.astype(np.float32) * float(depth_scale) * 1000.0
        summary.update({
            "min_depth_mm": float(np.min(values_mm)),
            "median_depth_mm": float(np.median(values_mm)),
            "max_depth_mm": float(np.max(values_mm)),
        })
    else:
        summary.update({"min_depth_mm": None, "median_depth_mm": None, "max_depth_mm": None})
    return summary


def save_capture(
    output_dir: Path,
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    depth_color: np.ndarray,
    overlay: np.ndarray,
    metadata: dict,
) -> None:
    """保存一组可复查的 RGB、深度和元数据文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    paths = {
        "rgb": output_dir / f"{stamp}_rgb.jpg",
        "depth_raw": output_dir / f"{stamp}_depth.png",
        "depth_color": output_dir / f"{stamp}_depth_color.jpg",
        "overlay": output_dir / f"{stamp}_overlay.jpg",
        "metadata": output_dir / f"{stamp}.json",
    }
    if not cv2.imwrite(str(paths["rgb"]), color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"保存 RGB 失败：{paths['rgb']}")
    if not cv2.imwrite(str(paths["depth_raw"]), depth_u16):
        raise RuntimeError(f"保存原始深度失败：{paths['depth_raw']}")
    if not cv2.imwrite(str(paths["depth_color"]), depth_color, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"保存深度伪彩失败：{paths['depth_color']}")
    if not cv2.imwrite(str(paths["overlay"]), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"保存叠加图失败：{paths['overlay']}")
    metadata = dict(metadata)
    metadata["files"] = {key: path.name for key, path in paths.items() if key != "metadata"}
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[SAVE] {paths['rgb']}")
    print(f"       depth raw: {paths['depth_raw']}")
    print(f"       metadata : {paths['metadata']}")


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("width、height 和 fps 必须大于0")
    if args.frames < 0 or args.save_every < 0:
        raise ValueError("frames 和 save-every 不能小于0")

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[ERROR] 当前 Python 没有安装 pyrealsense2。")
        print("请先在 Jetson 安装 librealsense2 和匹配版本的 pyrealsense2，再运行本脚本。")
        return 2

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)

    frame_count = 0
    pipeline_started = False
    try:
        profile = pipeline.start(config)
        pipeline_started = True
        device = profile.get_device()
        info = device_info(device, rs)
        depth_sensor = device.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        color_profile = profile.get_stream(rs.stream.color)
        depth_profile = profile.get_stream(rs.stream.depth)
        print("[DEVICE]", json.dumps(info, ensure_ascii=False))
        print(f"[STREAM] color={args.width}x{args.height}@{args.fps}")
        print(f"[STREAM] depth={args.width}x{args.height}@{args.fps}")
        print(f"[DEPTH] scale={depth_scale} m/raw")
        print("[KEY] S=保存一组数据，Q/ESC=退出，SPACE=暂停/继续")
        if args.no_gui:
            print("[MODE] no-gui；达到 --frames 后自动退出，按 Ctrl-C 退出")

        paused = False
        last_tick = time.monotonic()
        while args.frames == 0 or frame_count < args.frames:
            if not paused:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                if color.ndim != 3 or depth.ndim != 2:
                    print("[WARN] 收到的帧格式异常，跳过")
                    continue
                depth_color = colorize_depth(depth, depth_scale, args.max_depth_mm)
                overlay = cv2.addWeighted(color, 0.60, depth_color, 0.40, 0.0)
                summary = depth_summary(depth, depth_scale)
                now = time.monotonic()
                measured_fps = 1.0 / max(now - last_tick, 1e-6)
                last_tick = now
                cv2.putText(
                    overlay,
                    f"center={summary['center_depth_mm']} mm  valid={summary['valid_ratio']:.1%}  fps={measured_fps:.1f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
                )
                frame_count += 1

                save_requested = bool(
                    args.save_every > 0 and frame_count % args.save_every == 0
                )
                if args.no_gui and args.save_every == 0 and frame_count == 1:
                    save_requested = True
                if not args.no_gui:
                    cv2.imshow("RealSense RGB", color)
                    cv2.imshow("RealSense Depth Colorized", depth_color)
                    cv2.imshow("RealSense Aligned Overlay", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    if key == 32:
                        paused = True
                        print("[PAUSE] 按 SPACE 继续")
                    elif key in (ord("s"), ord("S")):
                        save_requested = True
                if save_requested:
                    save_capture(
                        args.output_dir,
                        color,
                        depth,
                        depth_color,
                        overlay,
                        {
                            "timestamp": datetime.now().astimezone().isoformat(),
                            "device": info,
                            "depth_scale_m_per_raw": depth_scale,
                            "aligned_depth_to": "color",
                            "depth_summary": summary,
                            "color_intrinsics": intrinsics_to_dict(color_profile),
                            "depth_intrinsics": intrinsics_to_dict(depth_profile),
                        },
                    )
                continue

            if not args.no_gui:
                key = cv2.waitKey(30) & 0xFF
                if key == 32:
                    paused = False
                    print("[RESUME]")
                elif key in (ord("q"), ord("Q"), 27):
                    break
            else:
                time.sleep(0.05)
    except RuntimeError as exc:
        print(f"[ERROR] RealSense 启动或取帧失败：{exc}")
        print("先检查 USB 3.x 连接、librealsense/pyrealsense2 版本、设备权限和是否已有程序占用相机。")
        return 3
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl-C")
    finally:
        if pipeline_started:
            pipeline.stop()
        if not args.no_gui:
            cv2.destroyAllWindows()
    print(f"[DONE] frames={frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
