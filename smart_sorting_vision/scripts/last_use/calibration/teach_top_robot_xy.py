from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# File location: scripts/last_use/calibration/teach_top_robot_xy.py
# parents[3] is the smart_sorting_vision project root.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from communication.sorting_protocol import Mode, SortingFrame
from communication.stm32_link import STM32Link
from runtime.app_config import load_runtime_config
from vision.calibration.pixel_to_robot import PixelToRobotTransform
from vision.cameras.top_camera import TopCamera


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fixed-marker XY teaching calibration: capture a clear tray image, click marker centers, "
            "then teach the robot XY coordinate of each marker through CALIB_MOVE."
        )
    )
    p.add_argument("--config", default=str(ROOT / "config" / "runtime_config.yaml"))
    p.add_argument("--ack-timeout", type=float, default=1.0)
    p.add_argument("--reach-timeout", type=float, default=15.0)
    p.add_argument("--ransac-mm", type=float, default=2.0)
    p.add_argument(
        "--resume",
        default=None,
        metavar="SESSION_DIR|latest",
        help=(
            "Resume an interrupted teach session. Pass the session directory, "
            "its timestamp directory name, or 'latest'."
        ),
    )
    return p.parse_args()


def wait_for(link: STM32Link, mode: Mode, timeout: float) -> SortingFrame:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in link.poll():
            print("[RX]", link.describe(frame))

            if frame.mode == Mode.MCU_ERROR:
                raise RuntimeError(f"STM32 MCU_ERROR code={frame.box_id}")

            if frame.mode == Mode.MCU_BUSY:
                raise RuntimeError(
                    f"STM32 MCU_BUSY while waiting for {mode.name}; "
                    "MCU is not in a state that accepts this calibration command"
                )

            if frame.mode == Mode.MCU_START_WORK:
                raise RuntimeError(
                    f"MCU entered START_WORK while waiting for {mode.name}; "
                    "run STOP synchronization before calibration"
                )

            if frame.mode == mode:
                return frame

        time.sleep(0.002)

    raise TimeoutError(f"timeout waiting for {mode.name}")


def synchronize_mcu_stop(
    link: STM32Link,
    *,
    timeout_s: float = 20.0,
    retry_interval_s: float = 0.40,
    poll_sleep_s: float = 0.010,
) -> None:
    """Force the STM32 into the safe STOPPED state before XY calibration.

    This intentionally mirrors the production startup synchronization used by
    main.py: old START/ACK frames are discarded, STOP is retransmitted, and
    calibration is not allowed to continue until the MCU reports the safe
    MCU_STOP_ACK.
    """
    timeout_s = max(1.0, float(timeout_s))
    retry_interval_s = max(0.10, float(retry_interval_s))
    poll_sleep_s = max(0.001, float(poll_sleep_s))

    print(
        "[SYNC] MCU calibration synchronization: "
        "force STOP -> wait safe MCU_STOP_ACK"
    )

    stale_count = 0
    for _ in range(5):
        stale_frames = link.poll()
        if not stale_frames:
            break
        stale_count += len(stale_frames)
        for stale in stale_frames:
            print(f"[SYNC][DRAIN] {stale.mode.name}")
        time.sleep(0.005)

    if stale_count:
        print(f"[SYNC] drained {stale_count} stale frame(s)")

    deadline = time.monotonic() + timeout_s
    next_send_at = 0.0
    tx_count = 0

    while True:
        now = time.monotonic()
        if now >= deadline:
            raise RuntimeError(
                "MCU calibration sync timeout: "
                f"no safe MCU_STOP_ACK within {timeout_s:.1f}s"
            )

        if now >= next_send_at:
            link.send_frame(SortingFrame(mode=Mode.STOP))
            tx_count += 1
            if tx_count == 1:
                print("[SYNC][TX] STOP(0x7F)")
            else:
                print(f"[SYNC][RETRY] STOP(0x7F) #{tx_count}")
            next_send_at = now + retry_interval_s

        for frame in link.poll():
            mode = frame.mode

            if mode == Mode.MCU_STOP_ACK:
                print(
                    "[SYNC] READY: MCU safe STOP confirmed; "
                    "XY calibration enabled"
                )
                return

            if mode == Mode.MCU_ERROR:
                print(
                    "[SYNC][RX] MCU_ERROR "
                    f"code={frame.box_id}; keep forcing STOP"
                )
                continue

            if mode == Mode.MCU_START_WORK:
                print(
                    "[SYNC][DROP] stale/early MCU_START_WORK ignored; "
                    "keep forcing STOP"
                )
                continue

            if mode == Mode.MCU_STOP_WORK:
                print(
                    "[SYNC][RX] MCU_STOP_WORK; "
                    "waiting for safe MCU_STOP_ACK"
                )
                continue

            print(f"[SYNC][DROP] {mode.name}")

        time.sleep(poll_sleep_s)


def capture_reference(camera: TopCamera) -> np.ndarray:
    window = "XY teach - C capture clear tray | Q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        frame = camera.read()
        if frame is None:
            continue
        display = frame.copy()
        cv2.putText(
            display,
            "Move mechanism away. C=capture clear tray, Q=quit",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt
        if key in (ord("c"), ord("C")):
            cv2.destroyWindow(window)
            return frame.copy()


def click_markers(image: np.ndarray) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    window = "XY teach - click fixed markers"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        display = image.copy()
        for i, (u, v) in enumerate(points, start=1):
            pt = (int(round(u)), int(round(v)))
            cv2.drawMarker(display, pt, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(display, f"P{i:02d}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "L-click add | R-click undo | ENTER finish (>=4) | Q quit",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt
        if key in (10, 13) and len(points) >= 4:
            cv2.destroyWindow(window)
            return points


def marked_reference(image: np.ndarray, points, active_index: int | None = None) -> np.ndarray:
    out = image.copy()
    for i, (u, v) in enumerate(points):
        pt = (int(round(u)), int(round(v)))
        color = (0, 0, 255) if i == active_index else (0, 255, 255)
        cv2.drawMarker(out, pt, color, cv2.MARKER_CROSS, 28, 2)
        cv2.circle(out, pt, 8, color, 2)
        cv2.putText(out, f"P{i+1:02d}", (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def teach_sessions_dir() -> Path:
    return (
        ROOT
        / "data"
        / "calibration"
        / "top_camera"
        / "robot_mapping"
        / "teach_sessions"
    )


def resolve_resume_session(value: str) -> Path:
    """Resolve --resume from an absolute path, relative path, timestamp, or latest."""
    base = teach_sessions_dir()

    if value.strip().lower() == "latest":
        candidates = [
            path
            for path in base.iterdir()
            if path.is_dir()
            and (path / "reference.jpg").is_file()
            and (path / "image_points.csv").is_file()
            and (path / "taught_points.csv").is_file()
        ] if base.is_dir() else []

        if not candidates:
            raise FileNotFoundError(
                f"No resumable teach session found under: {base}"
            )
        return max(candidates, key=lambda path: path.stat().st_mtime).resolve()

    requested = Path(value).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.extend((Path.cwd() / requested, base / requested))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(f"Resume session directory not found: {value}")


def load_resume_session(
    session: Path,
) -> tuple[np.ndarray, list[tuple[float, float]], list[dict]]:
    """Load and validate the immutable image points plus saved taught points."""
    reference_path = session / "reference.jpg"
    image_points_path = session / "image_points.csv"
    taught_points_path = session / "taught_points.csv"

    missing = [
        str(path.name)
        for path in (reference_path, image_points_path, taught_points_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Resume session is incomplete; missing: {', '.join(missing)}"
        )

    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference is None:
        raise RuntimeError(f"Cannot read reference image: {reference_path}")

    points: list[tuple[float, float]] = []
    point_by_id: dict[str, tuple[float, float]] = {}
    with image_points_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"point_id", "u", "v"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(
                f"Invalid image_points.csv fields: {reader.fieldnames}"
            )
        for row in reader:
            point_id = row["point_id"].strip()
            if point_id in point_by_id:
                raise RuntimeError(f"Duplicate image point ID: {point_id}")
            uv = (float(row["u"]), float(row["v"]))
            point_by_id[point_id] = uv
            points.append(uv)

    if len(points) < 4:
        raise RuntimeError(
            f"Resume session has only {len(points)} image points; need at least 4"
        )

    taught: list[dict] = []
    taught_ids: set[str] = set()
    with taught_points_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"point_id", "u", "v", "x_mm", "y_mm"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(
                f"Invalid taught_points.csv fields: {reader.fieldnames}"
            )
        for row in reader:
            point_id = row["point_id"].strip()
            if point_id in taught_ids:
                raise RuntimeError(f"Duplicate taught point ID: {point_id}")
            if point_id not in point_by_id:
                raise RuntimeError(
                    f"Taught point {point_id} is absent from image_points.csv"
                )

            u = float(row["u"])
            v = float(row["v"])
            expected_u, expected_v = point_by_id[point_id]
            if abs(u - expected_u) > 1e-6 or abs(v - expected_v) > 1e-6:
                raise RuntimeError(
                    f"Pixel mismatch for {point_id}: taught=({u}, {v}), "
                    f"image_points=({expected_u}, {expected_v})"
                )

            taught.append(
                {
                    "point_id": point_id,
                    "u": u,
                    "v": v,
                    "x_mm": float(row["x_mm"]),
                    "y_mm": float(row["y_mm"]),
                }
            )
            taught_ids.add(point_id)

    return reference, points, taught


def save_taught_points(session: Path, taught: list[dict]) -> None:
    """Atomically save progress so Ctrl+C or a power loss cannot truncate the CSV."""
    target = session / "taught_points.csv"
    temporary = session / "taught_points.csv.tmp"
    fieldnames = ["point_id", "u", "v", "x_mm", "y_mm"]

    with temporary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(taught)
        f.flush()
        os.fsync(f.fileno())

    temporary.replace(target)


def send_move(link, x_mm, y_mm, x_limits, y_limits, ack_timeout, reach_timeout):
    if not (x_limits[0] <= x_mm <= x_limits[1] and y_limits[0] <= y_mm <= y_limits[1]):
        raise ValueError(f"outside workspace: ({x_mm}, {y_mm})")

    # Normally CALIB_REACHED returns the MCU to STOPPED.  If a physical START
    # event arrived between teaching commands, do not silently discard it: the
    # MCU is then in a running state and CALIB_MOVE would correctly return BUSY.
    need_resync = False
    for stale in link.poll():
        print("[DROP OLD RX]", link.describe(stale))
        if stale.mode in (
            Mode.MCU_START_WORK,
            Mode.MCU_STOP_WORK,
            Mode.MCU_BUSY,
            Mode.MCU_ERROR,
        ):
            need_resync = True

    if need_resync:
        print("[SYNC] runtime-state frame detected before CALIB_MOVE; re-synchronizing STOP")
        synchronize_mcu_stop(link)

    tx = SortingFrame(mode=Mode.CALIB_MOVE, x_mm=x_mm, y_mm=y_mm)
    raw = link.send_frame(tx)
    print("[TX]", raw.hex(" ").upper())

    # Receive ACK and REACHED in one loop. Some serial reads contain both
    # frames; two independent wait_for() calls would return on ACK and silently
    # discard the REACHED frame from the same poll batch.
    sent_at = time.monotonic()
    ack_deadline = sent_at + max(0.1, float(ack_timeout))
    reach_deadline = sent_at + max(0.1, float(ack_timeout)) + max(
        0.1, float(reach_timeout)
    )
    ack: SortingFrame | None = None
    reached: SortingFrame | None = None

    while time.monotonic() < reach_deadline:
        for frame in link.poll():
            print("[RX]", link.describe(frame))

            if frame.mode == Mode.MCU_ERROR:
                raise RuntimeError(f"STM32 MCU_ERROR code={frame.box_id}")
            if frame.mode == Mode.MCU_BUSY:
                raise RuntimeError(
                    "STM32 MCU_BUSY while waiting for calibration movement"
                )
            if frame.mode == Mode.MCU_START_WORK:
                raise RuntimeError(
                    "MCU entered START_WORK during calibration movement"
                )

            if frame.mode == Mode.MCU_CALIB_MOVE_ACK:
                if abs(frame.x_mm - x_mm) > 0.2 or abs(frame.y_mm - y_mm) > 0.2:
                    raise RuntimeError(
                        f"ACK mismatch: target=({x_mm},{y_mm}), "
                        f"ack=({frame.x_mm},{frame.y_mm})"
                    )
                ack = frame

            elif frame.mode == Mode.MCU_CALIB_REACHED:
                if abs(frame.x_mm - x_mm) > 0.5 or abs(frame.y_mm - y_mm) > 0.5:
                    raise RuntimeError(
                        f"REACHED mismatch: target=({x_mm},{y_mm}), "
                        f"reached=({frame.x_mm},{frame.y_mm})"
                    )
                reached = frame

        if ack is not None and reached is not None:
            return float(reached.x_mm), float(reached.y_mm)

        now = time.monotonic()
        if ack is None and now >= ack_deadline:
            raise TimeoutError("timeout waiting for MCU_CALIB_MOVE_ACK")

        time.sleep(0.002)

    if ack is None:
        raise TimeoutError("timeout waiting for MCU_CALIB_MOVE_ACK")
    raise TimeoutError("timeout waiting for MCU_CALIB_REACHED")


def fit_and_save(session: Path, image_shape, taught: list[dict], ransac_mm: float):
    if len(taught) < 4:
        raise RuntimeError("Need at least 4 taught points")
    image_pts = np.asarray([[r["u"], r["v"]] for r in taught], dtype=np.float64)
    robot_pts = np.asarray([[r["x_mm"], r["y_mm"]] for r in taught], dtype=np.float64)
    H, mask = cv2.findHomography(image_pts, robot_pts, cv2.RANSAC, float(ransac_mm))
    if H is None:
        raise RuntimeError("cv2.findHomography failed")
    predicted = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(predicted - robot_pts, axis=1)
    inliers = mask.reshape(-1).astype(bool) if mask is not None else np.ones(len(taught), dtype=bool)

    h, w = image_shape[:2]
    candidate = session / "pixel_to_robot_candidate.json"
    transform = PixelToRobotTransform(
        homography=H,
        image_width=w,
        image_height=h,
        source="TopCamera.read() fixed-marker teaching calibration",
        metadata={
            "calibration_session": str(session),
            "point_count": len(taught),
            "inlier_count": int(np.count_nonzero(inliers)),
            "ransac_threshold_mm": float(ransac_mm),
            "fit_mean_error_mm": float(np.mean(errors)),
            "fit_rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
            "fit_max_error_mm": float(np.max(errors)),
        },
    )
    transform.save(candidate)
    report = {
        "candidate": str(candidate),
        "point_count": len(taught),
        "inlier_count": int(np.count_nonzero(inliers)),
        "mean_error_mm": float(np.mean(errors)),
        "rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
        "max_error_mm": float(np.max(errors)),
        "errors_mm": [float(v) for v in errors],
    }
    (session / "fit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    cfg = load_runtime_config(ROOT, args.config)
    x_limits = tuple(map(float, cfg["mechanics"]["workspace_mm"]["x"]))
    y_limits = tuple(map(float, cfg["mechanics"]["workspace_mm"]["y"]))

    TopCamera.configure(cfg["camera"]["top"], calibration_file=cfg["calibration"]["top_intrinsics"])
    camera = TopCamera()
    link = STM32Link(device=cfg["stm32"]["device"], baudrate=cfg["stm32"]["baud"])

    try:
        if args.resume:
            session = resolve_resume_session(args.resume)
            reference, points, taught = load_resume_session(session)
            marked = marked_reference(reference, points)

            confirmed_ids = {row["point_id"] for row in taught}
            remaining_ids = [
                f"P{i:02d}"
                for i in range(1, len(points) + 1)
                if f"P{i:02d}" not in confirmed_ids
            ]

            print("=" * 72)
            print("RESUME FIXED-MARKER XY TEACHING")
            print("=" * 72)
            print(f"session          : {session}")
            print(f"image points     : {len(points)}")
            print(f"confirmed points : {len(taught)}")
            if taught:
                print(f"last saved point : {taught[-1]['point_id']}")
            if remaining_ids:
                print(f"next point       : {remaining_ids[0]}")
            else:
                print("next point       : none; refitting existing points")
            print("The old reference image and pixel clicks are being reused.")
            print("=" * 72)
        else:
            session = teach_sessions_dir() / datetime.now().strftime("%Y%m%d_%H%M%S")
            session.mkdir(parents=True, exist_ok=False)

            camera.open()
            print("Stage A: move the mechanism out of the top-camera view, then press C in the preview.")
            reference = capture_reference(camera)
            cv2.imwrite(str(session / "reference.jpg"), reference)
            points = click_markers(reference)
            marked = marked_reference(reference, points)
            cv2.imwrite(str(session / "reference_marked.jpg"), marked)
            camera.close()

            with (session / "image_points.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["point_id", "u", "v"])
                writer.writeheader()
                for i, (u, v) in enumerate(points, start=1):
                    writer.writerow({"point_id": f"P{i:02d}", "u": u, "v": v})

            taught = []
            save_taught_points(session, taught)

        print("\nStage B: teach the mechanical XY of each fixed marker.")
        print("Commands:")
        print("  m X Y   -> absolute CALIB_MOVE to X,Y")
        print("  j dX dY -> fine relative jog from the last reached XY")
        print("  c       -> confirm the last reached XY for this marker")
        print("  c X Y   -> record a known manually positioned XY without commanding motion")
        print("  s       -> skip current marker; q -> quit")
        print("Keep the pointer/Yaw/Lift/Claw geometry unchanged during all teaching points.")
        print(f"Workspace X={x_limits}, Y={y_limits}")
        link.open()

        # CALIB_MOVE is accepted by the MCU only from STOPPED.  Do not rely on
        # the mechanism merely looking stationary: WAIT_PICK_TARGET and other
        # runtime states can also look stationary but must reject calibration.
        synchronize_mcu_stop(link)

        confirmed_ids = {row["point_id"] for row in taught}

        window = "XY teach - current fixed marker"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        for i, (u, v) in enumerate(points):
            point_id = f"P{i+1:02d}"

            if point_id in confirmed_ids:
                saved = next(row for row in taught if row["point_id"] == point_id)
                print(
                    f"[RESUME] skip {point_id}: already confirmed -> "
                    f"robot=({saved['x_mm']:.2f},{saved['y_mm']:.2f})"
                )
                continue

            active = marked_reference(reference, points, active_index=i)
            cv2.imshow(window, active)
            cv2.waitKey(1)
            last_reached: tuple[float, float] | None = None

            while True:
                command = input(f"\n{point_id} pixel=({u:.1f},{v:.1f}) > ").strip()
                if not command:
                    continue
                parts = command.split()
                op = parts[0].lower()
                if op == "q":
                    raise KeyboardInterrupt
                if op == "s":
                    print(f"[SKIP] {point_id}")
                    break
                if op == "m" and len(parts) == 3:
                    x_mm, y_mm = float(parts[1]), float(parts[2])
                    last_reached = send_move(
                        link, x_mm, y_mm, x_limits, y_limits, args.ack_timeout, args.reach_timeout
                    )
                    print(f"[REACHED] {last_reached}")
                    continue
                if op == "j" and len(parts) == 3:
                    if last_reached is None:
                        print("No current XY for relative jog. Use 'm X Y' once first.")
                        continue
                    dx_mm, dy_mm = float(parts[1]), float(parts[2])
                    x_mm = last_reached[0] + dx_mm
                    y_mm = last_reached[1] + dy_mm
                    last_reached = send_move(
                        link, x_mm, y_mm, x_limits, y_limits, args.ack_timeout, args.reach_timeout
                    )
                    print(f"[JOG REACHED] {last_reached}")
                    continue
                if op == "c" and len(parts) == 1:
                    if last_reached is None:
                        print("No reached XY yet. Use 'm X Y' first, or 'c X Y' if you positioned it manually.")
                        continue
                    x_mm, y_mm = last_reached
                elif op == "c" and len(parts) == 3:
                    x_mm, y_mm = float(parts[1]), float(parts[2])
                    if not (x_limits[0] <= x_mm <= x_limits[1] and y_limits[0] <= y_mm <= y_limits[1]):
                        print("Outside workspace; not recorded.")
                        continue
                else:
                    print("Invalid command. Use: m X Y | j dX dY | c | c X Y | s | q")
                    continue

                row = {"point_id": point_id, "u": u, "v": v, "x_mm": x_mm, "y_mm": y_mm}
                taught.append(row)
                confirmed_ids.add(point_id)
                save_taught_points(session, taught)
                print(f"[CONFIRMED] {point_id}: pixel=({u:.1f},{v:.1f}) -> robot=({x_mm:.2f},{y_mm:.2f})")
                break

        cv2.destroyAllWindows()
        if len(taught) < 4:
            raise RuntimeError("Fewer than 4 points confirmed")
        report = fit_and_save(session, reference.shape, taught, args.ransac_mm)
        print("\n" + "=" * 72)
        print("FIXED-MARKER PIXEL -> ROBOT XY FIT")
        print("=" * 72)
        print(f"points         : {report['point_count']}")
        print(f"RANSAC inliers : {report['inlier_count']}/{report['point_count']}")
        print(f"mean error     : {report['mean_error_mm']:.3f} mm")
        print(f"RMSE           : {report['rmse_mm']:.3f} mm")
        print(f"max error      : {report['max_error_mm']:.3f} mm")
        print(f"candidate      : {report['candidate']}")
        print("Do independent validation before replacing production pixel_to_robot.json.")
        return 0
    except KeyboardInterrupt:
        print("\n[STOP] teaching aborted by user")
        return 130
    finally:
        cv2.destroyAllWindows()
        try:
            if link.is_open:
                try:
                    link.send_frame(SortingFrame(mode=Mode.STOP))
                except Exception:
                    pass
        finally:
            link.close()
            camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
