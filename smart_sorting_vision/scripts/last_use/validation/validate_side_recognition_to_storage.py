from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from communication.sorting_protocol import (
    Mode,
    SortingProtocol,
    make_storage_test,
    make_stop,
)
from communication.stm32_link import STM32Link
from vision.cameras.side_camera import SideCamera
from vision.pipeline.side_sort_pipeline import SideSortPipeline
from vision.storage.object_info import ObjectInfo
from vision.storage.storage_manager import StorageManager


# 独立储物测试的通信容错参数。
# STORAGE_TEST 的 ACK 应该几乎立即返回；机械完成才可能需要较长时间。
ACK_RETRY_PERIOD_S = 0.35
ACK_TIMEOUT_S = 3.0
DONE_TIMEOUT_S = 30.0
STARTUP_STOP_TIMEOUT_S = 2.0


def load_cfg() -> dict:
    path = ROOT / "config" / "runtime_config.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def abs_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def safe_close_camera(camera) -> None:
    for name in ("release", "close"):
        fn = getattr(camera, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            return


def frame_hex(frame) -> str:
    try:
        return SortingProtocol.to_hex(SortingProtocol.pack(frame))
    except Exception:
        return "<pack failed>"


def main() -> None:
    cfg = load_cfg()
    SideCamera.configure(cfg["camera"]["side"])

    side_cfg = cfg.get("vision", {}).get("side", {})
    consensus_cfg = side_cfg.get("consensus", {})
    model_cfg = cfg["model"]["side"]

    camera = SideCamera()
    pipeline = SideSortPipeline(
        background_path=abs_path(cfg["calibration"]["side_background"]),
        shape_model_path=abs_path(model_cfg["path"]),
        shape_device=model_cfg.get("device", 0),
        shape_imgsz=int(model_cfg.get("imgsz", 224)),
        shape_confidence_threshold=float(model_cfg.get("confidence_threshold", 0.50)),
        shape_topk=int(model_cfg.get("topk", 3)),
        padding_ratio=float(side_cfg.get("padding_ratio", 0.15)),
        consensus_window=int(consensus_cfg.get("window", 5)),
        consensus_required=int(consensus_cfg.get("required", 4)),
    )

    storage = StorageManager(box_count=int(cfg.get("storage", {}).get("box_count", 6)))
    link = STM32Link(
        device=str(cfg["stm32"]["device"]),
        baudrate=int(cfg["stm32"].get("baud", 115200)),
    )

    output_dir = ROOT / "data" / "validation" / "side_recognition_to_storage"
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen = None
    pending = None
    logs = []
    lower_stopped = False
    startup_stop_sent_t = 0.0

    def cancel_pending(reason: str) -> None:
        nonlocal pending
        if pending is None:
            return
        rid = pending.get("reservation_id")
        if rid is not None:
            try:
                storage.cancel(rid)
            except Exception:
                pass
        print(f"[FAIL] {reason}; pending transaction cancelled")
        logs.append({**pending, "failed": reason, "finished_t": time.monotonic()})
        pending = None

    def transmit_storage_pending(*, retry: bool = False) -> None:
        nonlocal pending
        if pending is None:
            return
        frame = make_storage_test(pending["box_id"])
        link.send_frame(frame)
        pending["last_send_t"] = time.monotonic()
        pending["send_count"] += 1
        tag = "RETRY" if retry else "TX"
        print(
            f"[{tag}] STORAGE_TEST box={pending['box_id']} "
            f"source={pending['source']} try={pending['send_count']} "
            f"raw={frame_hex(frame)}"
        )

    def begin_storage(box_id: int, *, reservation=None, source="manual") -> bool:
        nonlocal pending
        if pending is not None:
            print(
                f"[TEST] storage command already pending: box={pending['box_id']} "
                f"ack={int(pending['ack_t'] is not None)}"
            )
            return False
        if not lower_stopped:
            print("[TEST] lower is not confirmed STOPPED; press S or wait for STOP ACK")
            return False

        now = time.monotonic()
        pending = {
            "box_id": int(box_id),
            "source": source,
            "reservation_id": None if reservation is None else reservation.reservation_id,
            "sent_t": now,
            "last_send_t": 0.0,
            "send_count": 0,
            "ack_t": None,
        }
        transmit_storage_pending(retry=False)
        return True

    try:
        link.open()
        camera.open()
        print("=" * 78)
        print("SIDE RECOGNITION -> STORAGE STANDALONE TEST (robust handshake)")
        print("This script actively forces lower to STOPPED before storage test.")
        print("P=store recognized object | 1..6=manual box test | R=recognize again | S=STOP | Q=quit")
        print("=" * 78)

        # 旧脚本只提示“下位机必须STOPPED”，却没有真正建立STOP握手。
        # 这里启动后立即发送STOP，收到 0x85/0x88 后才允许 STORAGE_TEST。
        stop_frame = make_stop()
        link.send_frame(stop_frame)
        startup_stop_sent_t = time.monotonic()
        print(f"[TX] STOP raw={frame_hex(stop_frame)}")
        print("[WAIT] waiting MCU_STOP_ACK/MCU_STOP_WORK before enabling storage test")

        while True:
            frame = camera.read()
            result = pipeline.process(frame)
            now = time.monotonic()

            for rx in link.poll():
                print("[RX]", STM32Link.describe(rx))

                if rx.mode in (Mode.MCU_STOP_ACK, Mode.MCU_STOP_WORK):
                    lower_stopped = True
                    if rx.mode == Mode.MCU_STOP_ACK:
                        print("[STATE] lower confirmed STOPPED by STOP_ACK")
                    continue

                if rx.mode == Mode.MCU_START_WORK:
                    lower_stopped = False
                    print("[WARN] lower entered START/RUNNING during standalone storage test")
                    if pending is not None:
                        cancel_pending("lower left STOPPED (MCU_START_WORK)")
                    # 独立测试模式不需要正式START，自动拉回STOPPED。
                    link.send_frame(make_stop())
                    startup_stop_sent_t = now
                    print("[TX] STOP again to restore standalone-test state")
                    continue

                if pending is None:
                    continue

                if rx.mode == Mode.MCU_STORAGE_TEST_ACK:
                    if rx.box_id != pending["box_id"]:
                        print(
                            f"[WARN] STORAGE_TEST_ACK box mismatch: rx={rx.box_id} "
                            f"pending={pending['box_id']}"
                        )
                        continue
                    if pending["ack_t"] is None:
                        pending["ack_t"] = now
                        print(
                            f"[ACK] box={rx.box_id} "
                            f"elapsed={now - pending['sent_t']:.3f}s; mechanical action accepted"
                        )
                    continue

                if rx.mode == Mode.MCU_STORAGE_TEST_DONE:
                    if rx.box_id != pending["box_id"]:
                        print(
                            f"[WARN] STORAGE_TEST_DONE box mismatch: rx={rx.box_id} "
                            f"pending={pending['box_id']}"
                        )
                        continue

                    done_t = now
                    entry = dict(pending)
                    entry["ack_elapsed_s"] = (
                        None if pending["ack_t"] is None else pending["ack_t"] - pending["sent_t"]
                    )
                    entry["done_elapsed_s"] = done_t - pending["sent_t"]
                    rid = pending.get("reservation_id")
                    if rid is not None:
                        commit = storage.commit(rid)
                        entry["commit"] = commit.__dict__
                        print(f"[DONE] box={rx.box_id} commit={commit.success} count={commit.count}")
                    else:
                        print(f"[DONE] manual box={rx.box_id}")
                    logs.append(entry)
                    pending = None
                    frozen = None
                    pipeline.reset()
                    continue

                if rx.mode == Mode.MCU_BUSY:
                    cancel_pending("MCU_BUSY: lower was not STOPPED or another action was active")
                    continue

                if rx.mode == Mode.MCU_ERROR:
                    cancel_pending(f"MCU_ERROR code={rx.box_id}")
                    continue

            # 启动STOP没有任何回包时给明确提示，而不是静默等待。
            if not lower_stopped and pending is None and now - startup_stop_sent_t > STARTUP_STOP_TIMEOUT_S:
                print(
                    "[WARN] no STOP_ACK for >2s. UART RX works only if other MCU frames are seen; "
                    "check lower firmware protocol/state."
                )
                startup_stop_sent_t = now
                link.send_frame(make_stop())

            # STORAGE_TEST 未ACK：重发同一个命令；不会重复启动机械，
            # 正确下位机会对重复命令只重发ACK。
            if pending is not None:
                if pending["ack_t"] is None:
                    if now - pending["sent_t"] >= ACK_TIMEOUT_S:
                        cancel_pending(
                            "no MCU_STORAGE_TEST_ACK within 3s; lower firmware likely lacks STORAGE_TEST(0x05) "
                            "or command is not reaching the STORAGE_TEST handler"
                        )
                    elif now - pending["last_send_t"] >= ACK_RETRY_PERIOD_S:
                        transmit_storage_pending(retry=True)
                elif now - pending["ack_t"] >= DONE_TIMEOUT_S:
                    cancel_pending(
                        "ACK received but no MCU_STORAGE_TEST_DONE within 30s; protocol works, "
                        "check storage mechanism/DM4310 state machine"
                    )

            if frozen is None and result.ready:
                frozen = ObjectInfo(
                    color=result.color,
                    shape=result.shape,
                    qr_present=result.qr_present,
                )
                print(f"[READY] color={result.color} shape={result.shape} QR={int(result.qr_present)}")
                print("        press P to reserve a box and execute storage")

            view = frame.copy()
            y = 28
            lines = [
                f"SIDE -> STORAGE | lower_stopped={int(lower_stopped)}",
                f"vision: found={int(result.found)} ready={int(result.ready)} votes={result.votes}/{result.required_hits}",
            ]
            if frozen is not None:
                lines.append(f"FROZEN color={frozen.color} QR={int(frozen.qr_present)}  (shape in console)")
            if pending is not None:
                lines.append(
                    f"STORAGE box={pending['box_id']} ack={int(pending['ack_t'] is not None)} "
                    f"try={pending['send_count']}"
                )
            lines.append("P store | 1..6 manual box | R reset | S stop | Q quit")
            for text in lines:
                cv2.putText(view, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
                y += 26
            cv2.imshow("Side Recognition -> Storage", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                if pending is not None:
                    print("[TEST] cannot reset vision while storage transaction is pending")
                else:
                    frozen = None
                    pipeline.reset()
                    print("[VISION] reset")
            elif key == ord("s"):
                link.send_frame(make_stop())
                startup_stop_sent_t = now
                lower_stopped = False
                if pending is not None:
                    cancel_pending("operator STOP")
                print("[TX] STOP")
            elif ord("1") <= key <= ord("6"):
                begin_storage(key - ord("0"), source="manual")
            elif key == ord("p"):
                # 先检查pending，再reserve；修复旧脚本“pending时仍reserve导致box1被占用，
                # 第二次P变成box2”的事务泄漏。
                if pending is not None:
                    print("[TEST] storage command already pending; do not reserve another box")
                    continue
                if not lower_stopped:
                    print("[TEST] lower not STOPPED; storage command not sent")
                    continue
                if frozen is None:
                    print("[TEST] no stable side result yet")
                    continue

                reservation = storage.reserve(frozen)
                if not reservation.success or reservation.reservation_id is None:
                    print(f"[STORAGE] reserve failed: {reservation.reason}")
                    continue

                print(
                    f"[STORAGE] {frozen.key()} -> box {reservation.box_id} "
                    f"({'new' if reservation.is_new_box else 'existing'})"
                )
                if not begin_storage(
                    reservation.box_id,
                    reservation=reservation,
                    source="vision",
                ):
                    storage.cancel(reservation.reservation_id)

    finally:
        try:
            if pending is not None and pending.get("reservation_id") is not None:
                storage.cancel(pending["reservation_id"])
        except Exception:
            pass
        safe_close_camera(camera)
        link.close()
        cv2.destroyAllWindows()

        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "events": logs,
            "storage_status": storage.get_status(),
        }
        path = output_dir / f"side_to_storage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[REPORT] {path}")


if __name__ == "__main__":
    main()
