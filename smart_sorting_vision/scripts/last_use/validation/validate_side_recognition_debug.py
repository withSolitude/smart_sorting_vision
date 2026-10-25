from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "vision").is_dir() and (parent / "config").is_dir():
            return parent
    raise RuntimeError("无法自动找到 smart_sorting_vision 项目根目录")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.cameras.side_camera import SideCamera
from vision.processors.side_object_extractor import SideObjectExtractor
from vision.color.color_classifier import ColorClassifier
from vision.code_reader.qr_detector import QRDetector
from vision.shape.shape_classifier import ShapeClassifier
from vision.pipeline.side_recognizer import SideRecognizer
from vision.pipeline.side_recognition_consensus import SideRecognitionConsensus


DEFAULT_CONFIG = ROOT / "config" / "runtime_config.yaml"
SAVE_DIR = ROOT / "data" / "validation" / "side_recognition_debug"


# 调试窗口中文字体。优先复用项目在 Ubuntu/Jetson 上常见的 Noto CJK。
CHINESE_FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


@lru_cache(maxsize=16)
def get_ui_font(font_size: int):
    for path in CHINESE_FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), int(font_size))
    return ImageFont.load_default()


def draw_unicode_items(
    image: np.ndarray,
    items: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]],
    *,
    panel_height: int | None = None,
) -> None:
    """只转换需要绘字的区域，避免整帧 Pillow 转换拖慢调试。"""
    h = image.shape[0] if panel_height is None else min(int(panel_height), image.shape[0])
    if h <= 0:
        return
    rgb = cv2.cvtColor(image[:h], cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    drawer = ImageDraw.Draw(pil)
    for text, pos, size, color_bgr in items:
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        drawer.text(pos, str(text), font=get_ui_font(size), fill=color_rgb, stroke_width=1, stroke_fill=(0, 0, 0))
    image[:h] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"配置文件不是字典结构：{path}")
    return data


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def label_panel(image: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], min(120, out.shape[0])), (0, 0, 0), -1)
    items = [(title, (12, 6), 18, (255, 255, 255))]
    y = 34
    for line in lines[:4]:
        items.append((str(line), (12, y), 14, (220, 220, 220)))
        y += 20
    draw_unicode_items(out, items, panel_height=120)
    return out


def mask_to_bgr(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def difference_to_heatmap(score: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if score is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = np.asarray(score, dtype=np.float32)
    if arr.shape[:2] != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    vmax = float(np.percentile(arr, 99.5)) if arr.size else 1.0
    vmax = max(vmax, 1.0)
    norm = np.clip(arr * (255.0 / vmax), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def crop_with_mask(crop: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    if mask is None:
        return crop.copy()
    m = mask
    if m.shape[:2] != crop.shape[:2]:
        m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(crop, crop, mask=m.astype(np.uint8))


def draw_qr(crop: np.ndarray, qr_result: dict[str, Any] | None) -> np.ndarray:
    out = crop.copy()
    if not qr_result:
        return out
    points = qr_result.get("points")
    if points is not None:
        pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) >= 3:
            cv2.polylines(out, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
    for center in qr_result.get("finder_centers", []) or []:
        try:
            p = (int(round(center[0])), int(round(center[1])))
            cv2.circle(out, p, 5, (255, 0, 255), -1, cv2.LINE_AA)
        except Exception:
            pass
    return out


def topk_text(shape_result: dict[str, Any] | None) -> str:
    if not shape_result:
        return "-"
    items = shape_result.get("topk", []) or []
    parts: list[str] = []
    for item in items[:3]:
        parts.append(f"{item.get('shape', '?')}:{float(item.get('confidence', 0.0)):.2f}")
    return "  ".join(parts) if parts else "-"


def color_feature_text(color_result: dict[str, Any] | None) -> str:
    if not color_result:
        return "-"
    feat = color_result.get("features", {}) or {}
    keys = [
        ("h_mean", "H"),
        ("h_median", "H"),
        ("s_median", "S"),
        ("v_median", "V"),
        ("lab_l_median", "L"),
    ]
    parts: list[str] = []
    used: set[str] = set()
    for key, label in keys:
        if label in used or key not in feat:
            continue
        try:
            parts.append(f"{label}={float(feat[key]):.1f}")
            used.add(label)
        except Exception:
            pass
    if not parts:
        try:
            parts.append(f"S={float(color_result.get('global_s_median', 0.0)):.1f}")
            parts.append(f"C={float(color_result.get('global_lab_chroma_median', 0.0)):.1f}")
        except Exception:
            pass
    return " ".join(parts) if parts else "-"


def save_debug_sample(
    *,
    roi: np.ndarray,
    object_result: Any,
    raw_result: dict[str, Any] | None,
    stable_state: dict[str, Any] | None,
    timing: dict[str, float],
) -> Path:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    prefix = SAVE_DIR / f"side_debug_{stamp}"
    cv2.imwrite(str(prefix) + "_roi.jpg", roi)
    if object_result is not None:
        if getattr(object_result, "crop", None) is not None:
            cv2.imwrite(str(prefix) + "_crop.jpg", object_result.crop)
        if getattr(object_result, "crop_mask", None) is not None:
            cv2.imwrite(str(prefix) + "_crop_mask.png", object_result.crop_mask)
        if getattr(object_result, "object_mask", None) is not None:
            cv2.imwrite(str(prefix) + "_object_mask.png", object_result.object_mask)
        if getattr(object_result, "core_mask", None) is not None:
            cv2.imwrite(str(prefix) + "_core_mask.png", object_result.core_mask)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "object": {
            "found": bool(getattr(object_result, "found", False)) if object_result is not None else False,
            "reason": str(getattr(object_result, "reason", "")) if object_result is not None else "",
            "bbox": getattr(object_result, "bbox", None) if object_result is not None else None,
            "crop_bbox": getattr(object_result, "crop_bbox", None) if object_result is not None else None,
            "area": int(getattr(object_result, "area", 0)) if object_result is not None else 0,
            "area_ratio": float(getattr(object_result, "area_ratio", 0.0)) if object_result is not None else 0.0,
        },
        "raw_result": json_safe(raw_result),
        "stable_state": json_safe(stable_state),
        "timing_ms": timing,
    }
    with (Path(str(prefix) + "_report.json")).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {prefix}_*")
    return prefix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="侧面相机综合识别调试：背景差分 + Color + Shape + QR + 多帧投票")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--no-shape", action="store_true", help="临时跳过YOLO形状模型，只调前景/颜色/QR")
    p.add_argument("--background-frames", type=int, default=40, help="按B重采空背景时平均的帧数")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = read_yaml(config_path)

    side_cfg = cfg.get("camera", {}).get("side")
    if not isinstance(side_cfg, dict):
        raise RuntimeError("runtime_config.yaml 缺少 camera.side")
    SideCamera.configure(side_cfg)

    calibration_cfg = cfg.get("calibration", {}) or {}
    background_path = resolve_project_path(calibration_cfg.get(
        "side_background",
        "data/calibration/side_camera/side_mechanical_roi_background.png",
    ))

    model_cfg = cfg.get("model", {}).get("side", {}) or {}
    model_path = resolve_project_path(model_cfg.get(
        "path",
        "outputs/training/side_cls/v2_yolo26n_cls/weights/best.pt",
    ))

    vision_side_cfg = cfg.get("vision", {}).get("side", {}) or {}
    consensus_cfg = vision_side_cfg.get("consensus", {}) or {}
    consensus_window = int(consensus_cfg.get("window", 5))
    consensus_required = int(consensus_cfg.get("required", 4))

    camera = SideCamera()
    extractor = SideObjectExtractor(
        padding_ratio=float(vision_side_cfg.get("padding_ratio", 0.15))
    )
    color_classifier = ColorClassifier()
    qr_detector = QRDetector()

    shape_classifier = None
    if not args.no_shape:
        if not model_path.exists():
            raise FileNotFoundError(f"侧面形状模型不存在：{model_path}")
        shape_classifier = ShapeClassifier(
            model_path=model_path,
            imgsz=int(model_cfg.get("imgsz", 224)),
            device=model_cfg.get("device", 0),
            confidence_threshold=float(model_cfg.get("confidence_threshold", 0.50)),
            topk=int(model_cfg.get("topk", 3)),
        )

    recognizer = None
    if shape_classifier is not None:
        recognizer = SideRecognizer(
            color_classifier=color_classifier,
            qr_detector=qr_detector,
            shape_classifier=shape_classifier,
        )

    consensus = SideRecognitionConsensus(
        window_size=consensus_window,
        required_hits=consensus_required,
    )

    if background_path.exists():
        extractor.load_background(background_path)
        print(f"[BACKGROUND] loaded: {background_path}")
    else:
        print(f"[BACKGROUND] 尚不存在：{background_path}")
        print("             清空托盘并让机构处于 Recognition 姿态后按 B 采集。")

    print("=" * 76)
    print("SIDE RECOGNITION DEBUG")
    print(f"config     : {config_path}")
    print(f"background : {background_path}")
    print(f"shape model: {'OFF' if args.no_shape else model_path}")
    print(f"consensus  : {consensus_required}/{consensus_window}")
    print("B=重采空背景  R=清投票  S=保存样本  P=打印结果  Q/ESC=退出")
    print("=" * 76)

    camera.open()
    try:
        # 预热相机，避免首帧把旧曝光状态带进统计。
        for _ in range(max(10, int(getattr(SideCamera, "WARMUP_FRAMES", 10)))):
            camera.read()

        smooth_fps = 0.0
        prev_loop = time.perf_counter()
        raw_history: deque[tuple[str, bool, str]] = deque(maxlen=60)
        frame_counter = 0

        # 调试时用户会直接把 A 物体换成 B 物体。Consensus READY 后会锁存，
        # 如果不按 R，旧版窗口会一直显示上一件。这里要求连续 3 帧完整结果
        # 与锁存候选不同才自动开启新物体会话；单帧 QR 抖动不会误触发。
        latched_mismatch_frames = 0
        last_object_result = None
        last_raw_result = None
        last_stable_state = None
        last_timing = {"extract": 0.0, "color": 0.0, "qr": 0.0, "shape": 0.0, "total": 0.0}

        while True:
            frame_counter += 1
            mechanical_roi = camera.read()
            h, w = mechanical_roi.shape[:2]

            if extractor.background_image is None:
                object_result = None
                raw_result = None
                stable_state = {
                    "ready": False,
                    "votes": 0,
                    "window": 0,
                    "required_hits": consensus_required,
                    "candidate": None,
                    "result": None,
                }
                timing = {"extract": 0.0, "color": 0.0, "qr": 0.0, "shape": 0.0, "total": 0.0}
            else:
                t0 = time.perf_counter()
                te = time.perf_counter()
                object_result = extractor.extract(mechanical_roi)
                extract_ms = (time.perf_counter() - te) * 1000.0

                raw_result = None
                if (
                    object_result.found
                    and object_result.crop is not None
                    and object_result.crop_mask is not None
                ):
                    tc = time.perf_counter()
                    color_result = color_classifier.classify(object_result.crop, object_result.crop_mask)
                    color_ms = (time.perf_counter() - tc) * 1000.0

                    tq = time.perf_counter()
                    qr_result = qr_detector.detect(object_result.crop)
                    qr_ms = (time.perf_counter() - tq) * 1000.0

                    shape_selection = {
                        "qr_suppression_applied": False,
                        "selected_mode": "raw",
                    }

                    if shape_classifier is not None:
                        ts = time.perf_counter()
                        shape_raw = shape_classifier.classify(object_result.crop)
                        shape_result = shape_raw

                        if bool(qr_result.get("found", False)):
                            suppressed = SideRecognizer._qr_suppressed_shape_image(
                                object_result.crop,
                                object_result.crop_mask,
                                qr_result,
                            )
                            if suppressed is not None:
                                shape_selection["qr_suppression_applied"] = True
                                shape_clean = shape_classifier.classify(suppressed)
                                raw_conf = float(shape_raw.get("confidence", 0.0))
                                clean_conf = float(shape_clean.get("confidence", 0.0))
                                shape_selection.update({
                                    "raw_shape": shape_raw.get("raw_shape", shape_raw.get("shape", "unknown")),
                                    "raw_confidence": raw_conf,
                                    "suppressed_shape": shape_clean.get("raw_shape", shape_clean.get("shape", "unknown")),
                                    "suppressed_confidence": clean_conf,
                                })
                                if clean_conf >= raw_conf + 0.03:
                                    shape_result = shape_clean
                                    shape_selection["selected_mode"] = "qr_suppressed"

                        shape_ms = (time.perf_counter() - ts) * 1000.0
                    else:
                        shape_result = {
                            "found": True,
                            "shape": "shape_off",
                            "raw_shape": "shape_off",
                            "confidence": 1.0,
                            "topk": [],
                            "reason": "disabled",
                        }
                        shape_ms = 0.0

                    raw_result = {
                        "object_found": True,
                        "color": color_result.get("color", "unknown"),
                        "color_confidence": float(color_result.get("confidence", 0.0)),
                        "qr_present": bool(qr_result.get("found", False)),
                        "qr_text": str(qr_result.get("text", "")),
                        "shape": shape_result.get("shape", "unknown"),
                        "shape_confidence": float(shape_result.get("confidence", 0.0)),
                        "details": {
                            "color": color_result,
                            "qr": qr_result,
                            "shape": shape_result,
                            "shape_selection": shape_selection,
                        },
                    }
                    stable_state = consensus.update(raw_result)

                    raw_key = (
                        str(raw_result["color"]),
                        bool(raw_result["qr_present"]),
                        str(raw_result["shape"]),
                    )
                    raw_history.append(raw_key)

                    candidate = stable_state.get("candidate") if stable_state else None
                    if (
                        stable_state
                        and stable_state.get("latched", False)
                        and candidate is not None
                        and tuple(candidate) != raw_key
                    ):
                        latched_mismatch_frames += 1
                    else:
                        latched_mismatch_frames = 0

                    if latched_mismatch_frames >= 3:
                        print(
                            "[AUTO RESET] 连续3帧与锁存结果不同："
                            f"old={candidate} new={raw_key}"
                        )
                        consensus.reset()
                        raw_history.clear()
                        stable_state = consensus.update(raw_result)
                        raw_history.append(raw_key)
                        latched_mismatch_frames = 0
                else:
                    color_ms = qr_ms = shape_ms = 0.0
                    consensus.reset()
                    stable_state = {
                        "ready": False,
                        "votes": 0,
                        "window": 0,
                        "required_hits": consensus_required,
                        "candidate": None,
                        "result": None,
                    }

                total_ms = (time.perf_counter() - t0) * 1000.0
                timing = {
                    "extract": extract_ms,
                    "color": color_ms,
                    "qr": qr_ms,
                    "shape": shape_ms,
                    "total": total_ms,
                }

            last_object_result = object_result
            last_raw_result = raw_result
            last_stable_state = stable_state
            last_timing = timing

            # ---------------- 主 ROI 画面 ----------------
            roi_display = mechanical_roi.copy()
            reason = "background_missing"
            area_text = ""
            if object_result is not None:
                reason = str(object_result.reason)
                area_text = f"area={object_result.area} ({object_result.area_ratio * 100.0:.2f}%)"
                if object_result.crop_bbox is not None:
                    x1, y1, x2, y2 = map(int, object_result.crop_bbox)
                    cv2.rectangle(roi_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if object_result.bbox is not None:
                    x1, y1, x2, y2 = map(int, object_result.bbox)
                    cv2.rectangle(roi_display, (x1, y1), (x2, y2), (255, 255, 0), 1)

            raw_line = "RAW: -"
            detail_line = ""
            shape_topk = "-"
            color_feat = "-"
            if raw_result is not None:
                color_detail = raw_result["details"]["color"]
                qr_detail = raw_result["details"]["qr"]
                shape_detail = raw_result["details"]["shape"]
                raw_line = (
                    f"RAW color={raw_result['color']}({raw_result['color_confidence']:.2f}) "
                    f"QR={'Y' if raw_result['qr_present'] else 'N'} "
                    f"shape={raw_result['shape']}({raw_result['shape_confidence']:.2f})"
                )
                shape_selection = raw_result["details"].get("shape_selection", {}) or {}
                qr_text = str(qr_detail.get("text", ""))
                if len(qr_text) > 24:
                    qr_text = qr_text[:21] + "..."
                curved_axis = str(qr_detail.get("curved_axis", "") or "")
                curved_gamma = qr_detail.get("curved_gamma", None)
                method = str(qr_detail.get("method", "") or "")
                curved_tag = ""
                if method == "opencv_curved_decode":
                    curved_tag = " curve=opencv_curved"
                elif method == "zxingcpp_curved_rectify":
                    curved_tag = f" curve={curved_axis}:{curved_gamma}"
                elif bool(qr_detail.get("curved_candidate_found", False)):
                    curved_tag = " curve=candidate_only"
                detail_line = (
                    f"QRmethod={qr_detail.get('method', '-') or '-'} "
                    f"qconf={float(qr_detail.get('confidence', 0.0)):.2f} "
                    f"ori={qr_detail.get('orientation_deg', '-')} "
                    f"text={qr_text or '-'} "
                    f"shapeMode={shape_selection.get('selected_mode', 'raw')}"
                    f"{curved_tag}"
                )
                shape_topk = topk_text(shape_detail)
                color_feat = color_feature_text(color_detail)

            ready = bool(stable_state and stable_state.get("ready", False))
            stable_line = f"CONS {stable_state.get('votes', 0)}/{stable_state.get('window', 0)} need={consensus_required}"
            if ready and stable_state.get("result") is not None:
                final = stable_state["result"]
                stable_line += (
                    f" READY {final.get('color')} | QR={'Y' if final.get('qr_present') else 'N'} | {final.get('shape')}"
                )

            cv2.rectangle(roi_display, (0, 0), (w, 154), (0, 0, 0), -1)
            lines = [
                raw_line,
                stable_line,
                f"extract={reason} {area_text}",
                f"ColorFeat {color_feat}",
                f"ShapeTop3 {shape_topk}",
                f"ms ext={timing['extract']:.1f} col={timing['color']:.1f} qr={timing['qr']:.1f} shape={timing['shape']:.1f} total={timing['total']:.1f}",
                detail_line,
            ]
            text_items = []
            y = 4
            for i, text in enumerate(lines):
                color = (0, 255, 0) if (i == 1 and ready) else (255, 255, 255)
                text_items.append((str(text), (12, y), 14, color))
                y += 20
            draw_unicode_items(roi_display, text_items, panel_height=154)

            now = time.perf_counter()
            dt = now - prev_loop
            prev_loop = now
            if dt > 0:
                fps = 1.0 / dt
                smooth_fps = fps if smooth_fps <= 0 else 0.90 * smooth_fps + 0.10 * fps
            cv2.putText(roi_display, f"FPS={smooth_fps:.1f}  B=background R=reset S=save P=print Q=quit",
                        (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)

            # ---------------- Mask 2x2 ----------------
            background = extractor.background_image
            bg_view = background if background is not None else np.zeros_like(mechanical_roi)
            diff_view = difference_to_heatmap(
                getattr(object_result, "difference_score", None) if object_result is not None else None,
                (h, w),
            )
            core_view = mask_to_bgr(
                getattr(object_result, "core_mask", None) if object_result is not None else None,
                (h, w),
            )
            object_view = mask_to_bgr(
                getattr(object_result, "object_mask", None) if object_result is not None else None,
                (h, w),
            )
            cell_w, cell_h = 420, 300
            masks = np.vstack([
                np.hstack([
                    label_panel(fit_image(bg_view, cell_w, cell_h), "BACKGROUND", [str(background_path.name)]),
                    label_panel(fit_image(diff_view, cell_w, cell_h), "DIFFERENCE HEATMAP", [reason]),
                ]),
                np.hstack([
                    label_panel(fit_image(core_view, cell_w, cell_h), "CORE MASK", []),
                    label_panel(fit_image(object_view, cell_w, cell_h), "OBJECT MASK", [area_text]),
                ]),
            ])

            # ---------------- Crop 2x2 ----------------
            if object_result is not None and object_result.crop is not None:
                crop = object_result.crop
                crop_mask = object_result.crop_mask
                masked = crop_with_mask(crop, crop_mask)
                qr_view = crop.copy()
                color_mask_view = np.zeros(crop.shape, dtype=np.uint8)
                if raw_result is not None:
                    qr_view = draw_qr(crop, raw_result["details"]["qr"])
                    stable_mask = raw_result["details"]["color"].get("stable_mask")
                    color_mask_view = mask_to_bgr(stable_mask, crop.shape[:2])
            else:
                crop = masked = qr_view = color_mask_view = np.zeros((240, 320, 3), dtype=np.uint8)

            crop_grid = np.vstack([
                np.hstack([
                    label_panel(fit_image(crop, cell_w, cell_h), "RAW CROP", [raw_line]),
                    label_panel(fit_image(masked, cell_w, cell_h), "MASKED CROP", [color_feat]),
                ]),
                np.hstack([
                    label_panel(fit_image(color_mask_view, cell_w, cell_h), "COLOR STABLE MASK", [f"color={raw_result.get('color') if raw_result else '-'}"]),
                    label_panel(
                        fit_image(qr_view, cell_w, cell_h),
                        "QR DEBUG",
                        [
                            f"found={raw_result.get('qr_present') if raw_result else False}",
                            f"text={(raw_result or {}).get('qr_text', '') or '-'}",
                            shape_topk,
                        ],
                    ),
                ]),
            ])

            cv2.imshow("SIDE DEBUG - ROI", roi_display)
            cv2.imshow("SIDE DEBUG - MASKS", masks)
            cv2.imshow("SIDE DEBUG - CROP", crop_grid)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                consensus.reset()
                raw_history.clear()
                latched_mismatch_frames = 0
                print("[RESET] consensus cleared")
            elif key in (ord("b"), ord("B")):
                print("[BACKGROUND] 请确认：托盘为空、机械已在正式 Recognition 姿态、灯光保持正式状态。")
                print(f"[BACKGROUND] collecting {args.background_frames} frames ...")
                background = extractor.capture_background(
                    camera,
                    frame_count=max(1, int(args.background_frames)),
                    settle_frames=15,
                )
                extractor.save_background(background_path)
                consensus.reset()
                raw_history.clear()
                latched_mismatch_frames = 0
                print(f"[BACKGROUND] saved: {background_path} shape={background.shape}")
            elif key in (ord("s"), ord("S")):
                save_debug_sample(
                    roi=mechanical_roi,
                    object_result=object_result,
                    raw_result=raw_result,
                    stable_state=stable_state,
                    timing=timing,
                )
            elif key in (ord("p"), ord("P")):
                print("\n" + "=" * 76)
                print(f"frame={frame_counter} FPS={smooth_fps:.2f}")
                print(f"object={getattr(object_result, 'found', False)} reason={reason} {area_text}")
                print(json.dumps(json_safe(raw_result), ensure_ascii=False, indent=2))
                print("CONSENSUS")
                print(json.dumps(json_safe(stable_state), ensure_ascii=False, indent=2))
                if raw_history:
                    print("LAST60 COUNTS")
                    for item, count in Counter(raw_history).most_common():
                        print(f"  {count:2d}x {item}")
                print("=" * 76 + "\n")

    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
