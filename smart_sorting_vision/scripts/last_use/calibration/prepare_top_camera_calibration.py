"""第2步：检查顶部相机、采集/计算内参、准备XY示教参考姿态。

放到 smart_sorting_vision/scripts/last_use/calibration/ 下。
python -m scripts.last_use.calibration.prepare_top_camera_calibration --help

check/capture/fit/install 不打开 STM32 串口。只有 reference 会发送运动相关命令。
本工具不训练模型、不计算像素到机器人H，不自动修改runtime_config.yaml。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "runtime/app_config.py").is_file() and (parent / "vision/cameras/top_camera.py").is_file():
            return parent
    return Path(__file__).resolve().parents[3]


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_context(args):
    root = Path(args.project_root).expanduser().resolve()
    if not (root / "runtime/app_config.py").is_file():
        raise RuntimeError("请把scripts目录复制进工程根目录，或通过 --project-root 指定已有工程。")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from runtime.app_config import load_runtime_config
    from vision.cameras.top_camera import TopCamera
    config_path = resolve(root, args.config)
    cfg = load_runtime_config(root, config_path)
    TopCamera.configure(cfg["camera"]["top"], calibration_file=cfg["calibration"]["top_intrinsics"])
    return root, cfg, TopCamera


def inspect_intrinsics(path: Path, size: tuple[int, int]) -> dict:
    required = {"camera_matrix", "dist_coeffs", "new_camera_matrix", "image_size", "valid_roi", "rms"}
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"内参缺字段：{sorted(missing)}")
        arrays = {name: data[name].copy() for name in required}
    if arrays["image_size"].size != 2:
        raise ValueError("image_size必须是[width,height]")
    actual_size = tuple(int(v) for v in arrays["image_size"].reshape(-1))
    if actual_size != size:
        raise ValueError(f"内参分辨率{actual_size}与配置{size}不同；不能直接使用。")
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"内参{name}含NaN/Inf")
    for name in ("camera_matrix", "new_camera_matrix"):
        mat = arrays[name]
        if mat.shape != (3, 3) or mat[0, 0] <= 0 or mat[1, 1] <= 0:
            raise ValueError(f"{name}不是有效的3x3相机矩阵")
        if not np.allclose(mat[2], [0, 0, 1]):
            raise ValueError(f"{name}最后一行应为[0,0,1]")
    if arrays["dist_coeffs"].size not in (4, 5, 8, 12, 14):
        raise ValueError("dist_coeffs长度不符合OpenCV针孔模型")
    if arrays["valid_roi"].size != 4 or arrays["rms"].size != 1:
        raise ValueError("valid_roi/rms形状不正确")
    rms = float(arrays["rms"].reshape(-1)[0])
    if rms < 0:
        raise ValueError("rms不能为负数")
    return {"path": str(path), "sha256": digest(path), "image_size": list(actual_size),
            "rms_px": rms, "valid_roi": arrays["valid_roi"].astype(int).reshape(-1).tolist()}


def raw_camera_class(TopCamera):
    class RawCalibrationCamera(TopCamera):
        # 正式TopCamera保持不变；这个专用子类只采原始图，不需要旧内参。
        def _load_calibration(self):
            pass

        def _create_undistort_maps(self):
            pass

        def read_undistorted(self):
            raise RuntimeError("原始标定相机禁止调用去畸变接口")

        def read(self):
            raise RuntimeError("原始标定相机只允许read_full()")

    return RawCalibrationCamera


def find_board(frame, board, *, accurate=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if accurate and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, board, flags=flags)
        if found and corners is not None:
            return True, corners.astype(np.float32)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board, flags=flags)
    if not found or corners is None:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners.astype(np.float32)


def object_points(board, square_mm):
    points = np.zeros((board[0] * board[1], 3), np.float32)
    points[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * float(square_mm)
    return points


def panel(frame, label, width=640, height=400):
    canvas = np.zeros((height, width, 3), np.uint8)
    scale = min(width / frame.shape[1], (height - 32) / frame.shape[0])
    small = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    y, x = 32, (width - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    cv2.putText(canvas, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_png(path, frame):
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"无法写入图像：{path}")


def check(args, root, cfg, TopCamera):
    camera_cfg = cfg["camera"]["top"]
    size = (int(camera_cfg["width"]), int(camera_cfg["height"]))
    intrinsic_path = resolve(root, args.intrinsics) if args.intrinsics else Path(cfg["calibration"]["top_intrinsics"])
    info = inspect_intrinsics(intrinsic_path, size)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("[INFO] 拟合RMS是历史拟合误差，不代表当前安装状态或机器人XY精度。")
    TopCamera.configure(camera_cfg, calibration_file=intrinsic_path)
    camera = TopCamera()
    x1, y1, x2, y2 = map(int, camera_cfg["safe_roi"])
    roi_size = [x2 - x1, y2 - y1]
    mapping_path = Path(cfg["calibration"]["pixel_to_robot"])
    mapping_info = {"path": str(mapping_path), "exists": mapping_path.is_file()}
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        old_size = [mapping.get("image_width"), mapping.get("image_height")]
        mapping_info.update(sha256=digest(mapping_path), image_size=old_size,
                            dimensions_match=(old_size == roi_size))
        print(f"[H] image_size={old_size}; 当前ROI={roi_size}")
        if old_size != roi_size:
            print("[WARN] H的图像尺寸不一致，必须重做H。")
    print("[KEY] G=开关辅助网格；S=保存同一帧的原图/去畸变全图/ROI与参数；Q/ESC=退出")
    print("[INFO] 本命令不打开串口、不移动机构，不修改配置。")
    grid = False
    try:
        camera.open()
        cv2.namedWindow("Step2 Top Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Step2 Top Camera", 1500, 420)
        while True:
            raw = camera.read_full()
            # 与TopCamera.read_undistorted()/read()使用完全相同的映射表和插值。
            und = cv2.remap(raw, camera._map1, camera._map2, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
            roi = und[y1:y2, x1:x2].copy()
            if roi.shape[:2] != (roi_size[1], roi_size[0]):
                raise RuntimeError("ROI尺寸异常")
            und_display, roi_display = und.copy(), roi.copy()
            cv2.rectangle(und_display, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 3)
            if grid:
                for target in (und_display, roi_display):
                    h, w = target.shape[:2]
                    for i in range(1, 4):
                        cv2.line(target, (w * i // 4, 0), (w * i // 4, h - 1), (120, 120, 120), 1)
                        cv2.line(target, (0, h * i // 4), (w - 1, h * i // 4), (120, 120, 120), 1)
            view = np.hstack([panel(raw, f"RAW {size[0]}x{size[1]}"),
                              panel(und_display, "UNDISTORTED + SAFE ROI"),
                              panel(roi_display, f"PRODUCTION ROI {roi_size[0]}x{roi_size[1]}")])
            cv2.imshow("Step2 Top Camera", view)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("g"), ord("G")):
                grid = not grid
            if key in (ord("s"), ord("S")):
                out = root / "data/calibration/top_camera/setup_checks" / stamp()
                out.mkdir(parents=True)
                save_png(out / "raw.png", raw)
                save_png(out / "undistorted.png", und)
                save_png(out / "roi.png", roi)
                save_png(out / "comparison.png", view)
                save_json(out / "setup_report.json", {
                    "created_at": datetime.now().isoformat(), "intrinsics": info,
                    "top_config": camera_cfg, "actual_stream": camera.get_stream_info(),
                    "mapping": mapping_info, "robot_geometry_verified": False,
                    "note": "同尺寸不能证明相机姿态、ROI原点、H或TCP仍有效；本检查不验收机器人坐标。"})
                print(f"[SAVE] {out}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def capture(args, root, cfg, TopCamera):
    board = (args.cols, args.rows)
    if min(board) < 3 or not math.isfinite(args.square_mm) or args.square_mm <= 0:
        raise ValueError("内角点每边至少3个，square-mm必须是正数")
    out = resolve(root, args.session)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"采集目录非空，请换一个新目录以保留旧记录：{out}")
    images = out / "images"
    images.mkdir(parents=True)
    camera = raw_camera_class(TopCamera)()
    meta = {"created_at": datetime.now().isoformat(), "source": "raw_full_frame",
            "board_inner_corners": list(board), "square_size_mm": args.square_mm,
            "top_config": cfg["camera"]["top"], "image_size": [TopCamera.WIDTH, TopCamera.HEIGHT],
            "captures": []}
    save_json(out / "capture_session.json", meta)
    print(f"[SESSION] {out}\n[BOARD] {board[0]}x{board[1]} 内角点，方格边长{args.square_mm}mm")
    print("[KEY] SPACE=保存完整检测到棋盘格的原始图；Q/ESC=退出")
    print("[INFO] 保持相机不动，移动并倾斜棋盘。建议20~30张，覆盖全画面并改变距离。")
    last_corners = None
    try:
        camera.open()
        meta["actual_stream"] = camera.get_stream_info()
        cv2.namedWindow("Capture RAW Intrinsics", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Capture RAW Intrinsics", 1280, 720)
        while True:
            frame = camera.read_full()
            found, corners = find_board(frame, board)
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, board, corners, True)
            cv2.putText(display, f"RAW | found={found} | saved={len(meta['captures'])} | SPACE save / Q quit",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow("Capture RAW Intrinsics", display)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key != 32:
                continue
            if not found:
                print("[SKIP] 未检测到完整棋盘内角点")
                continue
            if last_corners is not None:
                shift = float(np.mean(np.linalg.norm(corners.reshape(-1, 2) - last_corners.reshape(-1, 2), axis=1)))
                if shift < 5.0:
                    print("[SKIP] 与上一张几乎相同；移动/倾斜棋盘后再保存")
                    continue
            path = images / f"raw_{len(meta['captures']):03d}.png"
            save_png(path, frame)  # 保存未画角点、未去畸变、未裁剪的图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            center = corners.reshape(-1, 2).mean(axis=0)
            meta["captures"].append({"file": str(path.relative_to(out)), "sha256": digest(path),
                "corner_center_px": center.tolist(), "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var())})
            save_json(out / "capture_session.json", meta)
            last_corners = corners.copy()
            print(f"[SAVE {len(meta['captures'])}] {path.name}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        save_json(out / "capture_session.json", meta)
    print(f"[DONE] {len(meta['captures'])}张。下一步：fit --session {out}")


def fit_arrays(objects, observations, size, alpha):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objects, observations, size, None, None)
    errors = []
    for obj, obs, rv, tv in zip(objects, observations, rvecs, tvecs):
        predicted, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((obs.reshape(-1, 2) - predicted.reshape(-1, 2)) ** 2, axis=1)))))
    newK, valid_roi = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
    if not all(np.isfinite(a).all() for a in (K, dist, newK, np.asarray(errors))) or not math.isfinite(rms):
        raise RuntimeError("拟合产生NaN/Inf，请检查棋盘/图片")
    return float(rms), K, dist, newK, valid_roi, errors


def fit(args, root, cfg, TopCamera):
    out = resolve(root, args.session)
    meta = json.loads((out / "capture_session.json").read_text(encoding="utf-8"))
    if meta.get("source") != "raw_full_frame":
        raise ValueError("只接受本工具记录的原始完整帧采集目录")
    size = tuple(meta["image_size"])
    if size != (TopCamera.WIDTH, TopCamera.HEIGHT):
        raise ValueError("采集尺寸与当前正式配置不同，请统一后重新采集")
    board = tuple(int(v) for v in meta["board_inner_corners"])
    square_mm = float(meta["square_size_mm"])
    if min(board) < 3 or not math.isfinite(square_mm) or square_mm <= 0:
        raise ValueError("采集记录的棋盘参数无效")
    if not 0 <= args.alpha <= 1 or args.min_images < 12:
        raise ValueError("alpha必须在[0,1]，min-images至少12")
    objects, observations, used, rejected, selected_images = [], [], [], [], []
    template = object_points(board, square_mm)
    for item in meta["captures"]:
        path = (out / item["file"]).resolve()
        if not path.is_relative_to(out):
            raise ValueError("采集图片路径不能越出session目录")
        if not path.exists():
            rejected.append({"file": item["file"], "reason": "missing"})
            continue
        if digest(path) != item["sha256"]:
            raise ValueError(f"图片被修改过，不能混用裁剪/缩放图：{path.name}")
        frame = cv2.imread(str(path))
        if frame is None or (frame.shape[1], frame.shape[0]) != size:
            raise ValueError(f"图片尺寸或内容异常：{path}")
        found, corners = find_board(frame, board, accurate=True)
        if not found:
            rejected.append({"file": item["file"], "reason": "corners_not_found"})
            continue
        objects.append(template.copy())
        observations.append(corners)
        used.append(item["file"])
        selected_images.append(path)
        print(f"[CORNERS] {path.name}", flush=True)
    if len(used) < args.min_images:
        raise RuntimeError(f"有效图片{len(used)}张，至少需要{args.min_images}张；补充不同倾角/位置的采集。")
    rms, K, dist, newK, valid_roi, errors = fit_arrays(objects, observations, size, args.alpha)
    candidate = out / "top_camera_intrinsics_candidate.npz"
    np.savez(candidate, camera_matrix=K, dist_coeffs=dist, new_camera_matrix=newK,
             image_size=np.asarray(size, np.int32), image_width=size[0], image_height=size[1],
             valid_roi=np.asarray(valid_roi, np.int32), roi=np.asarray(valid_roi, np.int32),
             rms=np.float64(rms), reprojection_error=np.float64(rms), alpha=np.float64(args.alpha),
             square_size_mm=np.float64(square_mm), chessboard_size=np.asarray(board, np.int32))
    inspect_intrinsics(candidate, size)
    report = {"created_at": datetime.now().isoformat(), "candidate": str(candidate),
              "candidate_sha256": digest(candidate), "image_size": list(size), "alpha": args.alpha,
              "board_inner_corners": list(board), "square_size_mm": square_mm,
              "valid_images": len(used), "rms_px": rms, "max_view_rmse_px": max(errors),
              "rejected": rejected, "per_view": [{"file": f, "rmse_px": e} for f, e in zip(used, errors)],
              "robot_xy_verified": False,
              "note": "拟合误差仅用于诊断；需检查姿态覆盖和实时直线去畸变效果，不能凭低RMS验收XY。"}
    save_json(out / "intrinsics_fit_report.json", report)
    sample = cv2.imread(str(selected_images[0]))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, size, cv2.CV_16SC2)
    und = cv2.remap(sample, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    save_png(out / "fit_preview.png", np.hstack([panel(sample, "RAW"), panel(und, "CANDIDATE UNDISTORTED")]))
    for f, e in sorted(zip(used, errors), key=lambda x: x[1], reverse=True):
        print(f"{e:.4f}px {f}" + ("  CHECK IMAGE" if e >= .8 else ""))
    print(f"[FIT] valid={len(used)} RMS={rms:.4f}px max_view={max(errors):.4f}px")
    print(f"[CANDIDATE] {candidate}\n[INFO] 尚未替换正式内参。先用check --intrinsics检查这个候选。")


def install(args, root, cfg, TopCamera):
    candidate = resolve(root, args.candidate)
    target = Path(cfg["calibration"]["top_intrinsics"])
    if candidate == target:
        raise ValueError("candidate不能就是正式文件本身")
    info = inspect_intrinsics(candidate, (TopCamera.WIDTH, TopCamera.HEIGHT))
    report_path = candidate.parent / "intrinsics_fit_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("candidate_sha256") != info["sha256"] or int(report.get("valid_images", 0)) < 12:
        raise ValueError("候选与拟合报告不一致，或有效图片不足12张")
    backup_dir = target.parent / "intrinsics_backups" / stamp()
    backup_dir.mkdir(parents=True)
    for source, name in [(target, "top_camera_intrinsics_previous.npz"),
                         (Path(cfg["calibration"]["pixel_to_robot"]), "pixel_to_robot_previous.json"),
                         (resolve(root, args.config), "runtime_config_previous.yaml")]:
        if source.is_file():
            shutil.copy2(source, backup_dir / name)
    temp = target.with_name(target.name + ".tmp")
    shutil.copy2(candidate, temp)
    temp.replace(target)
    save_json(backup_dir / "installation.json", {"created_at": datetime.now().isoformat(),
              "installed": info, "target": str(target), "dependent_geometry_requires_recalibration": True})
    print(f"[INSTALLED] {target}\n[BACKUP] {backup_dir}")
    print("[NEXT] 重启相机程序；重新建立/验收H，并复查Yaw/TCP/边界。旧H未自动删除，也没有新增运行拦截。")


def wait_events(link, expected, timeout_s, *, x=None, y=None, clock=time.monotonic, sleep=time.sleep):
    from communication.sorting_protocol import Mode
    received = set()
    deadline = clock() + timeout_s
    while clock() < deadline:
        for frame in link.poll():
            print("[RX]", link.describe(frame))
            if frame.mode in (Mode.MCU_ERROR, Mode.MCU_BUSY, Mode.MCU_START_WORK, Mode.MCU_STOP_WORK):
                raise RuntimeError(f"等待过程中收到{frame.mode.name}，停止后续操作，code={frame.box_id}")
            if frame.mode not in expected:
                continue
            if x is not None and abs(frame.x_mm - x) > .51:
                continue
            if y is not None and abs(frame.y_mm - y) > .51:
                continue
            received.add(frame.mode)
        if received >= set(expected):
            return
        sleep(.003)
    missing = [m.name for m in expected if m not in received]
    raise TimeoutError(f"等待{missing}超时；不要忽略超时继续标定。")


def reference(args, root, cfg, TopCamera):
    from communication.sorting_protocol import Mode, SortingFrame
    from communication.stm32_link import STM32Link
    if not all(math.isfinite(v) for v in (args.x, args.y, args.yaw_servo_deg, args.timeout)):
        raise ValueError("参考姿态参数不能含NaN/Inf")
    xlim, ylim = cfg["mechanics"]["workspace_mm"]["x"], cfg["mechanics"]["workspace_mm"]["y"]
    if not (max(-77, xlim[0]) <= args.x <= min(0, xlim[1]) and
            max(0, ylim[0]) <= args.y <= min(55, ylim[1])):
        raise ValueError("XY超出当前固件与配置共有的合法范围")
    if not -60 <= args.yaw_servo_deg <= 135 or args.timeout <= 0:
        raise ValueError("当前固件Yaw舵机命令范围[-60,135]，timeout需为正数")
    # 协议0.1单位量化，记录实际下发值。
    x, y, yaw = [round(v * 10) / 10 for v in (args.x, args.y, args.yaw_servo_deg)]
    print("[MOTION] 本命令会发STOP并等待回位，随后移动XY和Yaw。先清空机构、确认机械原点和抬升状态。")
    print("[POSE] 当前上传固件默认Lift=-70deg、Claw=-35deg；本协议不能单独设置或回读它们。")
    print("[NOTE] 已处于Disable时STOP不会强制重发所有舵机姿态；必须现场核对Lift/Claw。")
    print(f"[TARGET] XY=({x},{y})mm; Yaw SERVO={yaw}deg（不是机械闭合轴方向）")
    link = STM32Link(device=cfg["stm32"]["device"], baudrate=cfg["stm32"]["baud"])
    completed = False
    try:
        link.open()
        link.send_frame(SortingFrame(mode=Mode.STOP))
        wait_events(link, {Mode.MCU_STOP_ACK}, args.timeout)
        link.send_frame(SortingFrame(mode=Mode.CALIB_MOVE, x_mm=x, y_mm=y))
        wait_events(link, {Mode.MCU_CALIB_MOVE_ACK, Mode.MCU_CALIB_REACHED}, args.timeout, x=x, y=y)
        link.send_frame(SortingFrame(mode=Mode.CALIB_YAW, x_mm=yaw))
        wait_events(link, {Mode.MCU_CALIB_YAW_ACK, Mode.MCU_CALIB_YAW_REACHED}, 5., x=yaw, y=0)
        out = root / "data/calibration/top_camera/reference_sessions" / stamp()
        save_json(out / "reference_pose_record.json", {
            "created_at": datetime.now().isoformat(), "source_firmware": "Control_fixed(1).zip",
            "commanded_xy_mm": [x, y], "commanded_yaw_servo_deg": yaw,
            "expected_firmware_lift_deg": -70., "expected_firmware_claw_deg": -35.,
            "lift_claw_actually_measured": False, "physical_closing_axis_deg": None,
            "note": "仅记录已确认握手的命令。XY为脉冲位置，Yaw为时间到位；Lift/Claw仍需人工确认。"})
        completed = True
        print(f"[COMMANDS ACKNOWLEDGED] {out / 'reference_pose_record.json'}")
        print("[NEXT] 确认夹爪实际姿态后保持Yaw/Lift/开口不变，再做XY示教。")
    finally:
        if link.is_open and not completed:
            try:
                link.send_frame(SortingFrame(mode=Mode.STOP))
            except Exception:
                pass
        link.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-root", default=str(project_root()))
    common.add_argument("--config", default="config/runtime_config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", parents=[common], help="只预览相机，不动机械；检查现有或候选内参")
    p.add_argument("--intrinsics", default=None, help="可选：候选npz；不安装，只预览")
    p = sub.add_parser("capture", parents=[common], help="采集RAW全图，无需旧内参")
    p.add_argument("--session", required=True, help="一个新的空目录，相对工程根目录")
    p.add_argument("--cols", type=int, default=6, help="棋盘内角点列数，不是方格数")
    p.add_argument("--rows", type=int, default=9, help="棋盘内角点行数，不是方格数")
    p.add_argument("--square-mm", type=float, required=True, help="实测单个方格边长mm")
    p = sub.add_parser("fit", parents=[common], help="计算候选内参和逐图误差，不替换正式文件")
    p.add_argument("--session", required=True)
    p.add_argument("--min-images", type=int, default=12)
    p.add_argument("--alpha", type=float, default=0., help="与旧项目一致默认0；更改会影响H")
    p = sub.add_parser("install", parents=[common], help="备份并安装已经检查过的候选内参")
    p.add_argument("--candidate", required=True)
    p = sub.add_parser("reference", parents=[common], help="会运动：STOP等待回位，再设置XY/Yaw")
    p.add_argument("--x", type=float, default=0.)
    p.add_argument("--y", type=float, default=0.)
    p.add_argument("--yaw-servo-deg", type=float, default=90.)
    p.add_argument("--timeout", type=float, default=30.)
    return parser.parse_args()


def main():
    args = parse_args()
    root, cfg, TopCamera = load_context(args)
    {"check": check, "capture": capture, "fit": fit, "install": install, "reference": reference}[args.command](args, root, cfg, TopCamera)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STOP] 用户退出")
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
