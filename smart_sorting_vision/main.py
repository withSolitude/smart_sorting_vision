"""Smart Sorting System production entry point."""

from pathlib import Path
from datetime import datetime
import json
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent

from communication.stm32_link import LinkOfflineError, STM32Link
from communication.sorting_protocol import SortingFrame, Mode
from runtime.app_config import load_runtime_config
from runtime.sorting_runtime import SortingRuntime
from vision.cameras.side_camera import SideCamera
from vision.cameras.top_camera import TopCamera
from vision.models.top_segmenter import TopSegmenter
from vision.pipeline.side_sort_pipeline import SideSortPipeline
from vision.pipeline.top_pick_pipeline import TopPickPipeline
from vision.calibration.yaw_tcp_compensation import YawTcpCompensator
from vision.calibration.descent_drift_compensation import DescentDriftCompensator
from vision.storage.storage_manager import StorageManager


class SmartSortingApp:
    def __init__(self):
        self.root = ROOT
        self.config_path = self.root / "config" / "runtime_config.yaml"
        self.config = load_runtime_config(self.root, self.config_path)

        self.stm32 = None
        self.top_camera = None
        self.side_camera = None
        self.top_pipeline = None
        self.side_pipeline = None
        self.storage = None
        self.runtime = None
        self.display = None
        self.runtime_thread = None
        self._log_lock = threading.Lock()
        self._session_log_path = None

    def check_files(self):
        required = {
            "top model": self.config["model"]["top"]["path"],
            "side model": self.config["model"]["side"]["path"],
            "top intrinsics": self.config["calibration"]["top_intrinsics"],
            "pixel_to_robot": self.config["calibration"]["pixel_to_robot"],
            "side background": self.config["calibration"]["side_background"],
        }
        surface_model = self.config["model"].get(
            "top_surface",
            {},
        )
        if bool(surface_model.get("enabled", False)):
            required["top surface model"] = surface_model["path"]
        for name, file in required.items():
            if not Path(file).exists():
                raise FileNotFoundError(f"缺少{name}: {file}")
        print("[OK] Files")

    def _print_runtime_mapping(self):
        x_limits = self.config["mechanics"]["workspace_mm"]["x"]
        y_limits = self.config["mechanics"]["workspace_mm"]["y"]
        top_consensus = self.config["vision"]["top"]["consensus"]
        side_consensus = self.config["vision"]["side"]["consensus"]

        print("[CONFIG] file:", self.config_path)
        print("[CONFIG] top camera :", self.config["camera"]["top"]["device"])
        print("[CONFIG] side camera:", self.config["camera"]["side"]["device"])
        print(f"[CONFIG] workspace X={x_limits} mm, Y={y_limits} mm")
        print(
            "[CONFIG] consensus "
            f"top={top_consensus['required']}/{top_consensus['window']} "
            f"side={side_consensus['required']}/{side_consensus['window']}"
        )
        surface_model = self.config["model"].get(
            "top_surface",
            {},
        )
        print(
            "[CONFIG] top surface model "
            f"enabled={bool(surface_model.get('enabled', False))}"
        )
        qr_cfg = self.config["vision"]["side"]["recognition"]["qr"]
        print(
            "[CONFIG] qr "
            f"enabled={bool(qr_cfg.get('enabled', True))} "
            f"content_check={bool(qr_cfg.get('check_content_consistency', False))}"
        )
        shape_cfg = self.config["vision"]["side"]["recognition"]["shape"]
        print(
            "[CONFIG] side shape "
            f"enabled={bool(shape_cfg.get('enabled', True))} "
            f"no_object_labels={shape_cfg.get('no_object_labels', [])}"
        )
        runtime_cfg = self.config["runtime"]
        print(
            "[CONFIG] competition "
            f"min_items={runtime_cfg['minimum_items_before_empty_finish']} "
            f"empty_hold={runtime_cfg['empty_finish_hold_s']}s "
            f"retry={runtime_cfg['retry_interval_s']}s"
        )

    def _warmup_inference_backends(self):
        """Eagerly initialize inference backends before WAIT START.

        This intentionally happens before SortingRuntime.spin() starts accepting
        competition work.  Any CUDA/TensorRT/shape mismatch therefore fails at
        startup instead of failing after MCU_START_WORK.
        """
        runtime_cfg = self.config["runtime"]
        warmup_cfg = runtime_cfg["warmup"]

        if not bool(warmup_cfg.get("enabled", True)):
            print("[WARMUP] disabled")
            return

        top_runs = max(1, int(warmup_cfg.get("top_runs", 3)))
        side_runs = max(1, int(warmup_cfg.get("side_runs", 2)))

        print(
            "[WARMUP] inference backends "
            f"top={top_runs} side={side_runs}"
        )

        top_frame = self.top_camera.read()
        if top_frame is None:
            raise RuntimeError("顶部相机预热取帧失败")

        top_stats = self.top_pipeline.segmenter.warmup(
            top_frame,
            runs=top_runs,
        )
        if self.top_pipeline.surface_segmenter is not None:
            surface_stats = self.top_pipeline.surface_segmenter.warmup(
                top_frame,
                runs=top_runs,
            )
        else:
            surface_stats = {
                "first_ms": 0.0,
                "steady_avg_ms": 0.0,
            }
        self.top_pipeline.reset()

        if self.side_pipeline.shape_classifier is not None:
            side_stats = self.side_pipeline.shape_classifier.warmup(
                runs=side_runs,
            )
        else:
            side_stats = {
                "first_ms": 0.0,
                "steady_avg_ms": 0.0,
            }
        self.side_pipeline.reset()

        print(
            "[WARMUP] READY "
            f"top_first={top_stats['first_ms']:.1f}ms "
            f"top_steady={top_stats['steady_avg_ms']:.1f}ms "
            f"surface_first={surface_stats['first_ms']:.1f}ms "
            f"surface_steady={surface_stats['steady_avg_ms']:.1f}ms "
            f"side_first={side_stats['first_ms']:.1f}ms "
            f"side_steady={side_stats['steady_avg_ms']:.1f}ms"
        )

    def initialize(self):
        print("==============================")
        print(" SMART SORTING SYSTEM ")
        print("==============================")

        self.check_files()
        self._print_runtime_mapping()

        self.stm32 = STM32Link(
            device=self.config["stm32"]["device"],
            baudrate=self.config["stm32"]["baud"],
        )
        # USB-UART 设备可能比 Jetson 进程晚几秒枚举。启动阶段持续等待，
        # 不把一次设备枚举失败误判成程序故障。
        while True:
            try:
                self.stm32.open()
                break
            except LinkOfflineError as exc:
                print(f"[STM32][WAIT] {exc}; retry in 1s")
                time.sleep(1.0)
        print("[OK] STM32")

        # 上位机进程一启动就先发一次STOP，尽快终止MCU可能残留的上一轮动作。
        # 这里只是“预STOP”，不把早到的ACK当成最终同步完成；
        # TensorRT Warmup完成后还会执行一次完整的安全同步握手。
        try:
            self.stm32.send_frame(SortingFrame(mode=Mode.STOP))
            print("[SYNC][PRESTOP] STOP(0x7F) sent")
        except Exception as exc:
            print(f"[SYNC][WARN] early STOP send failed: {exc}")

        # C01: runtime_config.yaml is now the only production camera profile.
        TopCamera.configure(
            self.config["camera"]["top"],
            calibration_file=self.config["calibration"]["top_intrinsics"],
        )
        SideCamera.configure(self.config["camera"]["side"])

        self.top_camera = TopCamera()
        self.top_camera.open()
        print("[OK] Top Camera")

        self.side_camera = SideCamera()
        self.side_camera.open()
        print("[OK] Side Camera")

        top_model_cfg = self.config["model"]["top"]
        top_surface_model_cfg = self.config["model"].get(
            "top_surface",
            {},
        )
        top_vision_cfg = self.config["vision"]["top"]
        top_consensus = top_vision_cfg["consensus"]
        top_orientation = top_vision_cfg["grasp_orientation"]
        top_preopen = top_orientation["dynamic_preopen"]
        top_gripper_collision = top_vision_cfg["gripper_collision"]
        top_grasp_search = top_gripper_collision["grasp_search"]
        top_surface_matching = top_vision_cfg.get(
            "surface_matching",
            {},
        )

        top_surface_segmenter = None
        if bool(top_surface_model_cfg.get("enabled", False)):
            top_surface_segmenter = TopSegmenter(
                model_path=top_surface_model_cfg["path"],
                conf=top_surface_model_cfg["conf"],
                imgsz=top_surface_model_cfg["imgsz"],
                device=top_surface_model_cfg["device"],
            )
        top_tcp_comp = top_vision_cfg["yaw_tcp_compensation"]
        branch1_axis_correction_deg = float(top_tcp_comp["branch1_axis_correction_deg"])
        branch1_closing_bias_mm = float(top_tcp_comp["branch1_closing_bias_mm"])
        branch1_longitudinal_bias_mm = float(top_tcp_comp["branch1_longitudinal_bias_mm"])

        x_limits = self.config["mechanics"]["workspace_mm"]["x"]
        y_limits = self.config["mechanics"]["workspace_mm"]["y"]
        mechanics_cfg = self.config["mechanics"]
        forbidden_zones_mm = mechanics_cfg["forbidden_zones_mm"]
        allowed_polygon_mm = top_gripper_collision["allowed_polygon_mm"]

        yaw_tcp_compensator = None
        if bool(top_tcp_comp["enabled"]):
            offset_axis0 = top_tcp_comp["offset_axis0_mm"]
            visual_to_command = top_tcp_comp["visual_to_command_matrix"]
            yaw_tcp_compensator = YawTcpCompensator(
                reference_axis_deg=float(top_tcp_comp["reference_axis_deg"]),
                offset_axis0_x_mm=float(offset_axis0[0]),
                offset_axis0_y_mm=float(offset_axis0[1]),
                min_axis_deg=float(top_tcp_comp["min_axis_deg"]),
                max_axis_deg=float(top_tcp_comp["max_axis_deg"]),
                visual_to_command_matrix=visual_to_command,
                closing_bias_mm=float(top_tcp_comp["closing_bias_mm"]),
                longitudinal_bias_mm=float(top_tcp_comp["longitudinal_bias_mm"]),
            )
            print(
                "[CONFIG] Yaw-TCP compensation ON: "
                f"ref={yaw_tcp_compensator.reference_axis_deg:.3f} deg, "
                f"d0=({yaw_tcp_compensator.offset_axis0_x_mm:.3f},"
                f" {yaw_tcp_compensator.offset_axis0_y_mm:.3f}) mm, "
                f"Jinv={yaw_tcp_compensator.visual_to_command_matrix}, "
                f"closing_bias={yaw_tcp_compensator.closing_bias_mm:+.3f} mm, "
                f"long_bias={yaw_tcp_compensator.longitudinal_bias_mm:+.3f} mm, "
                f"branch1_axis_corr={branch1_axis_correction_deg:+.3f} deg, "
                f"branch1_closing_bias={branch1_closing_bias_mm:+.3f} mm, "
                f"branch1_long_bias={branch1_longitudinal_bias_mm:+.3f} mm"
            )
        else:
            print("[CONFIG] Yaw-TCP compensation OFF (verification required)")

        descent_drift_cfg = top_vision_cfg["descent_drift_compensation"]
        descent_drift_compensator = None
        if bool(descent_drift_cfg["enabled"]):
            drift_matrix = (
                top_tcp_comp["visual_to_command_matrix"]
                if bool(descent_drift_cfg["use_visual_to_command_matrix"])
                else [[1.0, 0.0], [0.0, 1.0]]
            )
            descent_drift_compensator = DescentDriftCompensator(
                signed_drift_mm=float(descent_drift_cfg["signed_drift_mm"]),
                relative_angle_deg=float(descent_drift_cfg["relative_angle_deg"]),
                visual_to_command_matrix=drift_matrix,
            )
            print(
                "[CONFIG] Z-descent lateral compensation ON: "
                f"drift={descent_drift_compensator.signed_drift_mm:+.3f} mm, "
                f"relative={descent_drift_compensator.relative_angle_deg:+.1f} deg"
            )
        else:
            print("[CONFIG] Z-descent lateral compensation OFF")

        if bool(top_gripper_collision["enabled"]):
            print(
                "[CONFIG] Gripper footprint collision ON: "
                f"finger={float(top_gripper_collision['finger_length_mm']):.1f}x"
                f"{float(top_gripper_collision['finger_thickness_mm']):.1f}mm, "
                f"object_margin={float(top_gripper_collision['object_margin_mm']):.1f}mm, "
                f"wall_margin={float(top_gripper_collision['wall_margin_mm']):.1f}mm, "
                f"fixed_zones={len(forbidden_zones_mm)}"
            )
            if bool(top_grasp_search["enabled"]):
                print(
                    "[CONFIG] Multi-pose grasp search ON: "
                    f"yaw={top_grasp_search['yaw_offsets_deg']}, "
                    f"long={top_grasp_search['longitudinal_offsets_mm']}mm, "
                    f"hard_gap={float(top_grasp_search['hard_min_clearance_mm']):.2f}mm"
                )
        else:
            print("[CONFIG] Gripper footprint collision OFF")

        self.top_pipeline = TopPickPipeline(
            model_path=top_model_cfg["path"],
            calibration_path=self.config["calibration"]["pixel_to_robot"],
            conf=top_model_cfg["conf"],
            imgsz=top_model_cfg["imgsz"],
            device=top_model_cfg["device"],
            surface_segmenter=top_surface_segmenter,
            surface_match_class_name=top_surface_matching["match_class_name"],
            surface_match_center_max_px=top_surface_matching["match_center_max_px"],
            surface_min_area_ratio=top_surface_matching["min_area_ratio"],
            surface_required=top_surface_matching["required"],
            obstacle_margin=top_vision_cfg["obstacle_margin_px"],
            border_margin=top_vision_cfg["border_margin_px"],
            min_safe_area=top_vision_cfg["min_safe_area_px"],
            min_clearance=top_vision_cfg["min_clearance_px"],
            consensus_window=top_consensus["window"],
            consensus_required=top_consensus["required"],
            max_track_center_shift_px=top_consensus["max_track_center_shift_px"],
            max_grasp_spread_mm=top_consensus["max_grasp_spread_mm"],
            reset_after_misses=top_consensus["reset_after_misses"],
            orientation_enabled=top_orientation["enabled"],
            grip_long_sides=top_orientation["grip_long_sides"],
            min_orientation_aspect_ratio=top_orientation["min_aspect_ratio"],
            default_grasp_axis_robot_deg=top_orientation["default_grasp_axis_robot_deg"],
            yaw_alignment_offset_deg=top_orientation["yaw_alignment_offset_deg"],
            max_yaw_spread_deg=top_orientation["max_yaw_spread_deg"],
            grasp_width_slack_ratio=top_orientation["width_slack_ratio"],
            grasp_support_tolerance_px=top_orientation["support_tolerance_px"],
            grasp_min_contact_span_px=top_orientation["min_contact_span_px"],
            grasp_circle_circularity_threshold=top_orientation["circle_circularity_threshold"],
            grasp_circle_min_vertices=top_orientation["circle_min_vertices"],
            grasp_min_opening_mm=top_orientation["min_opening_mm"],
            grasp_max_opening_mm=top_orientation["max_opening_mm"],
            dynamic_preopen_enabled=top_preopen["enabled"],
            preopen_margin_mm=top_preopen["margin_mm"],
            preopen_command_min_mm=top_preopen["min_command_mm"],
            preopen_command_max_mm=top_preopen["max_command_mm"],

            # 动态预开口公式参数传入 TopPickPipeline；PC 只发送净开口(mm)。
            # 保留旧参数名以兼容调试脚本；生产配置不再使用查表类型。
            preopen_profile_kind="linear",
            preopen_configured=bool(top_preopen["configured"]),
            preopen_closed_servo_deg=float(top_preopen["closed_servo_deg"]),
            preopen_standby_servo_deg=float(top_preopen["standby_servo_deg"]),

            gripper_collision_enabled=top_gripper_collision["enabled"],
            gripper_finger_length_mm=top_gripper_collision["finger_length_mm"],
            gripper_finger_thickness_mm=top_gripper_collision["finger_thickness_mm"],
            gripper_finger_longitudinal_offset_mm=top_gripper_collision["finger_longitudinal_offset_mm"],
            gripper_opening_uncertainty_mm=top_gripper_collision["opening_uncertainty_mm"],
            gripper_object_margin_mm=top_gripper_collision["object_margin_mm"],
            gripper_wall_margin_mm=top_gripper_collision["wall_margin_mm"],
            gripper_fallback_opening_mm=top_gripper_collision["fallback_opening_mm"],
            gripper_forbidden_zones_mm=forbidden_zones_mm,
            gripper_allowed_region_mm=top_gripper_collision["allowed_region_mm"],
            gripper_allowed_polygon_mm=allowed_polygon_mm,
            gripper_object_overlap_tolerance_px=top_gripper_collision["object_overlap_tolerance_px"],
            gripper_enforce_image_border=top_gripper_collision["enforce_image_border"],
            gripper_image_border_margin_px=top_gripper_collision["image_border_margin_px"],
            gripper_body_enabled=top_gripper_collision["body"]["enabled"],
            gripper_body_width_mm=top_gripper_collision["body"]["width_mm"],
            gripper_body_length_mm=top_gripper_collision["body"]["length_mm"],
            gripper_body_longitudinal_offset_mm=top_gripper_collision["body"]["longitudinal_offset_mm"],
            gripper_debug_log_rejections=top_gripper_collision["debug_log_rejections"],
            grasp_search_enabled=top_grasp_search["enabled"],
            grasp_search_yaw_offsets_deg=top_grasp_search["yaw_offsets_deg"],
            grasp_search_longitudinal_offsets_mm=top_grasp_search["longitudinal_offsets_mm"],
            grasp_search_closing_offsets_mm=top_grasp_search["closing_offsets_mm"],
            grasp_search_hard_min_clearance_mm=top_grasp_search["hard_min_clearance_mm"],
            grasp_search_preferred_clearance_mm=top_grasp_search["preferred_clearance_mm"],
            grasp_search_min_target_overlap_mm=top_grasp_search["min_target_overlap_mm"],
            grasp_search_min_side_gap_mm=top_grasp_search["min_side_gap_mm"],
            grasp_search_yaw_penalty_per_deg=top_grasp_search["yaw_penalty_per_deg"],
            grasp_search_center_penalty_per_mm=top_grasp_search["center_penalty_per_mm"],
            grasp_search_max_trials_per_object=top_grasp_search["max_trials_per_object"],
            x_limits_mm=x_limits,
            y_limits_mm=y_limits,
            yaw_tcp_compensator=yaw_tcp_compensator,
            branch1_axis_correction_deg=branch1_axis_correction_deg,
            branch1_closing_bias_mm=branch1_closing_bias_mm,
            branch1_longitudinal_bias_mm=branch1_longitudinal_bias_mm,
        )

        side_model_cfg = self.config["model"]["side"]
        side_vision_cfg = self.config["vision"]["side"]
        side_consensus = side_vision_cfg["consensus"]

        side_recognition_cfg = side_vision_cfg["recognition"]
        side_crop_cfg = side_vision_cfg["crop_extractor"]
        side_extractor_cfg = side_crop_cfg["extractor"]
        side_color_cfg = dict(side_recognition_cfg["color"])
        side_qr_cfg = dict(side_recognition_cfg["qr"])
        enable_color = bool(side_color_cfg.pop("enabled"))
        enable_qr = bool(side_qr_cfg.pop("enabled"))
        # QR 的内容校验不是解码器参数，而是多帧共识策略参数。
        # 从 QRDetector 配置中取出，避免把策略配置误传给底层解码器。
        qr_check_content = bool(side_qr_cfg.pop("check_content_consistency", False))
        side_shape_cfg = dict(side_recognition_cfg["shape"])
        enable_shape = bool(side_shape_cfg.pop("enabled"))
        no_object_labels = tuple(
            str(label)
            for label in side_shape_cfg.pop(
                "no_object_labels",
                ("no_object", "empty", "background"),
            )
        )

        self.side_pipeline = SideSortPipeline(
            background_path=self.config["calibration"]["side_background"],
            shape_model_path=side_model_cfg["path"],
            shape_device=side_model_cfg["device"],
            shape_imgsz=side_model_cfg["imgsz"],
            shape_confidence_threshold=side_model_cfg["confidence_threshold"],
            shape_topk=side_model_cfg["topk"],
            padding_ratio=side_vision_cfg["padding_ratio"],
            consensus_window=side_consensus["window"],
            consensus_required=side_consensus["required"],
            extractor_config=side_extractor_cfg,
            color_config=side_color_cfg,
            qr_config=side_qr_cfg,
            qr_check_content=qr_check_content,
            no_object_labels=no_object_labels,
            enable_color=enable_color,
            enable_qr=enable_qr,
            enable_shape=enable_shape,
        )

        # 必须在正式 WAIT START 前完成。这样 TensorRT execution context、
        # CUDA kernel、PyTorch/NumPy 后处理都不会拖到第一件货物。
        self._warmup_inference_backends()
        print("[OK] Pipeline + Warmup")

        self.storage = StorageManager(box_count=int(self.config["storage"]["box_count"]))
        print("[OK] Storage")

        runtime_cfg = self.config["runtime"]
        scheduler_cfg = runtime_cfg["scheduler"]
        camera_watchdog_cfg = runtime_cfg["camera_watchdog"]
        self.runtime = SortingRuntime(
            top_camera=self.top_camera,
            side_camera=self.side_camera,
            top_pipeline=self.top_pipeline,
            side_pipeline=self.side_pipeline,
            storage_manager=self.storage,
            stm32=self.stm32,
            pick_ack_timeout_s=runtime_cfg["pick_ack_timeout_s"],
            recognition_ready_timeout_s=runtime_cfg["recognition_ready_timeout_s"],
            side_recognition_timeout_s=runtime_cfg["side_recognition_timeout_s"],
            retry_pick_ready_timeout_s=runtime_cfg["retry_pick_ready_timeout_s"],
            sort_ack_timeout_s=runtime_cfg["sort_ack_timeout_s"],
            sort_done_timeout_s=runtime_cfg["sort_done_timeout_s"],
            top_no_frame_timeout_s=camera_watchdog_cfg["top_no_frame_timeout_s"],
            side_no_frame_timeout_s=camera_watchdog_cfg["side_no_frame_timeout_s"],
            pick_x_limits_mm=x_limits,
            pick_y_limits_mm=y_limits,
            descent_drift_compensator=descent_drift_compensator,
            branch1_axis_correction_deg=branch1_axis_correction_deg,
            retry_interval_s=runtime_cfg["retry_interval_s"],
            link_reconnect_interval_s=runtime_cfg["link_reconnect_interval_s"],
            heartbeat_interval_s=runtime_cfg["heartbeat_interval_s"],
            link_dead_timeout_s=runtime_cfg["link_dead_timeout_s"],
            minimum_items_before_empty_finish=runtime_cfg["minimum_items_before_empty_finish"],
            empty_finish_hold_s=runtime_cfg["empty_finish_hold_s"],
            max_items=runtime_cfg["max_items"],
            reject_unknown=runtime_cfg["reject_unknown"],
            on_finished=self._on_finished,
            on_session_start=self._on_session_start,
            loop_sleep_idle_s=scheduler_cfg["idle_sleep_s"],
            loop_sleep_wait_s=scheduler_cfg["wait_sleep_s"],
            loop_sleep_vision_s=scheduler_cfg["vision_sleep_s"],
        )

        # --------------------------------------------------------
        # Jetson / STM32 上电同步
        #
        # Warmup已经完成、正式WAIT START尚未开始时，Jetson主动发送STOP。
        # 下位机只有在夹爪、XY、储物机构全部安全回到可启动状态后，
        # 才回MCU_STOP_ACK。这样每次开机都从同一个确定状态开始，
        # 第一次实体按键短按必定对应START，而不是受上一轮/残留状态影响。
        # --------------------------------------------------------
        sync_cfg = runtime_cfg["startup_sync"]
        if bool(sync_cfg["enabled"]):
            while True:
                try:
                    self.runtime.synchronize_startup(
                        timeout_s=sync_cfg["timeout_s"],
                        retry_interval_s=sync_cfg["retry_interval_s"],
                        poll_sleep_s=sync_cfg["poll_sleep_s"],
                    )
                    break
                except (LinkOfflineError, OSError) as exc:
                    # 上电同步也采用可恢复策略；重新打开串口后继续发送 STOP，
                    # 直到 MCU 回报真正安全的 MCU_STOP_ACK。
                    print(f"[SYNC][WAIT] {exc}; reconnect and retry")
                    self.stm32.reconnect_once()
                    time.sleep(1.0)
                except RuntimeError as exc:
                    # 只有同步超时/串口异常可重试；配置或协议编程错误必须
                    # 立即暴露，避免用无限重试掩盖真正的软件缺陷。
                    text = str(exc).lower()
                    recoverable = "timeout" in text or "serial" in text or "safe mcu_stop_ack" in text
                    if not recoverable:
                        raise
                    print(f"[SYNC][WAIT] {exc}; reconnect and retry")
                    self.stm32.reconnect_once()
                    time.sleep(1.0)
            print("[OK] MCU synchronized")
        else:
            print("[WARN] MCU startup synchronization disabled")

        print("[OK] Runtime")

        log_dir = self.root / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_log_path = log_dir / f"competition_{stamp}.jsonl"
        print("[LOG]", self._session_log_path)

    def _on_session_start(self, runtime):
        print(
            f"[COMPETITION] START cycle={runtime.cycle_id} "
            f"already_completed={runtime.completed_count}"
        )

    def _on_finished(self, record):
        safe = {k: v for k, v in record.items() if k != "image_bgr"}
        safe["time"] = datetime.now().isoformat(timespec="seconds")

        if self._session_log_path is not None:
            with self._log_lock:
                with self._session_log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            safe,
                            ensure_ascii=False
                        ) + "\n"
                    )

        if self.display is not None:

            # ==========================================
            # 颜色英文 -> 中文
            # ==========================================

            color_map = {
                "red": "红色",
                "orange": "橙色",
                "yellow": "黄色",
                "green": "绿色",
                "cyan": "青色",
                "blue": "蓝色",
                "purple": "紫色",
                "black": "黑色",
                "white": "白色",
                "gray": "灰色",
                "grey": "灰色",
            }


            # 获取识别结果
            raw_color = str(
                record.get(
                    "color",
                    "未知颜色"
                )
            )

            shape = str(
                record.get(
                    "shape",
                    "未知形状"
                )
            )


            # 转中文颜色
            color_cn = color_map.get(
                raw_color.lower(),
                raw_color
            )


            # ==========================================
            # GUI显示格式
            #
            # 原:
            # blue / 五棱柱 / 二维码 → 1号箱
            #
            # 修改:
            # 五棱柱_蓝色
            #
            # ==========================================

            product_name = (
                f"{shape}_{color_cn}"
            )


            self.display.add_finished_item(
                sequence=int(record["sequence"]),
                product_name=product_name,
                image_bgr=record.get("image_bgr"),
                success_count=int(record["success_count"]),
            )

    def run(self):
        ui_cfg = self.config.get("ui", {})
        if not bool(ui_cfg.get("enabled", True)):
            self.runtime.spin()
            return

        try:
            # Jetson + OpenCV + system PyQt5 compatibility:
            # cv2 imported earlier may overwrite QT_QPA_PLATFORM_PLUGIN_PATH
            # with .venv/.../cv2/qt/plugins. Restore the system Qt5 plugins
            # immediately before importing/creating QApplication.
            import os

            system_qt_plugin_root = "/usr/lib/aarch64-linux-gnu/qt5/plugins"
            system_qt_platform_dir = (
                "/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms"
            )

            if os.path.isdir(system_qt_platform_dir):
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = (
                    system_qt_platform_dir
                )
                os.environ["QT_PLUGIN_PATH"] = system_qt_plugin_root

            # OpenCV wheels may also set their bundled Qt font directory.
            os.environ.pop("QT_QPA_FONTDIR", None)

            print(
                "[GUI] Qt platform plugins:",
                os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"),
            )

            from PyQt5.QtWidgets import QApplication
            from ui.sorting_display import SortingDisplayWindow
        except Exception as exc:
            print(f"[WARN] GUI unavailable, fallback to console: {exc}")
            self.runtime.spin()
            return

        qt_app = QApplication.instance() or QApplication(sys.argv)
        self.display = SortingDisplayWindow(
            fullscreen=bool(ui_cfg.get("fullscreen", True)),
            max_visible_rows=int(ui_cfg.get("max_visible_rows", 6)),
        )

        self.runtime_thread = threading.Thread(
            target=self.runtime.spin,
            name="competition-runtime",
            daemon=True,
        )
        self.runtime_thread.start()

        qt_app.exec_()

        self.runtime.shutdown(send_stop=True)
        if self.runtime_thread.is_alive():
            self.runtime_thread.join(timeout=2.0)

    def close(self):
        print("[SYSTEM] shutdown")

        if self.runtime is not None:
            try:
                self.runtime.shutdown(send_stop=True)
                if self.runtime_thread is not None and self.runtime_thread.is_alive():
                    self.runtime_thread.join(timeout=2.0)
            except Exception as exc:
                print(f"[WARN] runtime shutdown failed: {exc}")

        for name, resource in (
            ("top_camera", self.top_camera),
            ("side_camera", self.side_camera),
            ("stm32", self.stm32),
        ):
            if resource is None:
                continue
            close = getattr(resource, "close", None)
            if not callable(close):
                close = getattr(resource, "release", None)
            if not callable(close):
                print(f"[WARN] {name} has no close/release method")
                continue
            try:
                close()
            except Exception as exc:
                print(f"[WARN] {name} close failed: {exc}")

        self.runtime = None
        self.top_camera = None
        self.side_camera = None
        self.stm32 = None


app = None


def handler(sig, frame):
    global app
    if app:
        app.close()
    sys.exit(0)


def main():
    global app
    signal.signal(signal.SIGINT, handler)
    app = SmartSortingApp()
    try:
        app.initialize()
        app.run()
    finally:
        app.close()


if __name__ == "__main__":
    main()
