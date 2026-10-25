from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time

import cv2
import numpy as np

from vision.models.top_segmenter import TopSegmenter
from vision.grasp.collision_aware_grasp_planner import (
    CollisionAwareGraspPlanner,
)
from vision.calibration.pixel_to_robot import (
    PixelToRobotTransform,
)
from vision.pipeline.top_pick_consensus import (
    TopPickConsensus,
)
from vision.grasp.orientation_planner import (
    axial_difference_deg,
    plan_grasp_axis,
    plan_parallel_edge_grasp_options,
)
from vision.grasp.gripper_footprint import GripperFootprintModel
from vision.grasp.claw_preopen_profile import ClawPreopenProfile
from vision.grasp.grasp_pose_search import (
    axis_vectors,
    build_search_offsets,
    evaluate_target_engagement,
    normalize_axial_deg,
)
from vision.calibration.yaw_tcp_compensation import YawTcpCompensator


@dataclass
class TopPickCandidate:
    object_index: int
    class_name: str
    confidence: float

    polygon: np.ndarray

    center_pixel: tuple[float, float]
    grasp_pixel: tuple[int, int]

    x_mm: float
    y_mm: float

    # 机械XY坐标系中的夹爪闭合轴角度。
    # 不是舵机原始角度；180°周期，正式输出位于 [-90, 90)。
    yaw_deg: float
    object_long_axis_deg: float
    aspect_ratio: float
    orientation_reliable: bool
    orientation_mode: str
    rectangularity: float
    circularity: float
    hull_vertex_count: int
    closing_axis_image_deg: float
    grasp_width_px: float
    grasp_width_mm: float | None
    contact_type: str
    contact_quality: float
    support_overlap_px: float
    support_min_point_px: tuple[float, float] | None
    support_max_point_px: tuple[float, float] | None

    clearance_px: float
    self_clearance_px: float
    obstacle_clearance_px: float
    # 近似“夹爪从物体两侧落下后，距离最近其他物体还剩多少空间”。
    # = 抓取中心到最近其他物体像素距离 - 当前抓取半宽；用于优先抓周围更空的物体。
    approach_gap_px: float
    safe_area_px: int

    # 真实夹爪二维包络到最近“其他物体/固定挡板”的最小距离。
    # 该值已经考虑当前 yaw 与当前动态预开口，不再只是抓取中心点距离。
    gripper_clearance_px: float
    gripper_clearance_mm: float

    # 两个指爪（以及可选主体）的名义二维包络，仅用于调试/预览绘制。
    gripper_footprint_polygons_px: list[np.ndarray]

    # 视觉估计的动态预开口。None 时下位机自动回退旧固定预收缩角。
    preopen_mm: float | None

    # 同一物体多姿态搜索相对基础 antipodal 姿态的修正量。
    search_yaw_offset_deg: float
    search_longitudinal_offset_mm: float
    search_closing_offset_mm: float
    search_trial_count: int
    search_safe_count: int

    score: float

    # 双边抓取物理Yaw分支：0=当前轴向侧，1=180°等效另一侧。
    # 放在末尾并提供默认值，避免影响已有调试代码的构造参数。
    yaw_branch: int = 0


@dataclass
class TopPickPipelineResult:
    ready: bool

    votes: int
    required_hits: int
    window_size: int

    object_count: int
    candidate_count: int

    grasp_pixel: tuple[int, int] | None

    x_mm: float | None
    y_mm: float | None
    yaw_deg: float | None
    grasp_width_mm: float | None
    preopen_mm: float | None

    class_name: str | None
    confidence: float
    clearance_px: float

    reason: str

    inference_ms: float

    candidates: list[TopPickCandidate]

    # 与yaw_deg配套的物理Yaw分支；READY时必须为0或1。
    # 放在末尾提供默认值，兼容原有测试/调试构造方式。
    yaw_branch: int | None = None


class TopPickPipeline:
    """
    顶部视觉正式抓取流水线。

    输入：
        TopCamera.read() 返回的一帧正式图像

    内部：
        YOLO segmentation
            ↓
        每个实例的 polygon
            ↓
        CollisionAwareGraspPlanner
            ↓
        grasp pixel
            ↓
        PixelToRobotTransform
            ↓
        robot X/Y
            ↓
        TopPickConsensus
            ↓
        READY

    输出：
        TopPickPipelineResult

    重要：
        1. 本类不打开相机。
        2. 本类不发送 STM32。
        3. P1~P4 不是抓取范围限制。
        4. 整个 TopCamera.read() 图像均允许进行 Homography 转换。
        5. 机械极限限制应在后续 robot_config / 状态机层设置。
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        calibration_path: str | Path,
        conf: float = 0.35,
        imgsz: int = 832,
        device=0,
        obstacle_margin: int = 12,
        border_margin: int = 12,
        min_safe_area: int = 50,
        min_clearance: float = 3.0,
        consensus_window: int = 2,
        consensus_required: int = 2,
        max_track_center_shift_px: float = 55.0,
        max_grasp_spread_mm: float = 3.0,
        reset_after_misses: int = 3,
        orientation_enabled: bool = True,
        grip_long_sides: bool = True,
        min_orientation_aspect_ratio: float = 1.20,
        default_grasp_axis_robot_deg: float = 0.0,
        yaw_alignment_offset_deg: float = 0.0,
        max_yaw_spread_deg: float = 12.0,
        grasp_width_slack_ratio: float = 0.15,
        grasp_support_tolerance_px: float = 3.0,
        grasp_min_contact_span_px: float = 3.0,
        grasp_circle_circularity_threshold: float = 0.985,
        grasp_circle_min_vertices: int = 8,
        grasp_min_opening_mm: float | None = None,
        grasp_max_opening_mm: float | None = None,
        dynamic_preopen_enabled: bool = True,
        preopen_margin_mm: float = 6.0,
        preopen_command_min_mm: float | None = None,
        preopen_command_max_mm: float | None = None,
        preopen_profile_kind: str = "cad_four_bar",
        preopen_configured: bool = False,
        preopen_closed_servo_deg: float = -15.0,
        preopen_standby_servo_deg: float = -95.0,
        gripper_collision_enabled: bool = True,
        gripper_finger_length_mm: float = 20.0,
        gripper_finger_thickness_mm: float = 9.0,
        gripper_finger_longitudinal_offset_mm: float = 0.0,
        gripper_opening_uncertainty_mm: float = 1.5,
        gripper_object_margin_mm: float = 1.0,
        gripper_wall_margin_mm: float = 2.0,
        gripper_fallback_opening_mm: float = 38.3,
        gripper_forbidden_zones_mm: list | None = None,
        gripper_allowed_region_mm: tuple[float, float, float, float] | None = None,
        gripper_allowed_polygon_mm: list | None = None,
        gripper_object_overlap_tolerance_px: int = 4,
        gripper_enforce_image_border: bool = True,
        gripper_image_border_margin_px: int = 2,
        gripper_body_enabled: bool = False,
        gripper_body_width_mm: float = 50.0,
        gripper_body_length_mm: float = 20.0,
        gripper_body_longitudinal_offset_mm: float = -25.0,
        gripper_debug_log_rejections: bool = False,
        grasp_search_enabled: bool = True,
        grasp_search_yaw_offsets_deg=(0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0),
        grasp_search_longitudinal_offsets_mm=(0.0, -2.0, 2.0, -4.0, 4.0),
        grasp_search_closing_offsets_mm=(0.0,),
        grasp_search_hard_min_clearance_mm: float = 1.0,
        grasp_search_preferred_clearance_mm: float = 4.0,
        grasp_search_min_target_overlap_mm: float = 1.5,
        grasp_search_min_side_gap_mm: float = 0.25,
        grasp_search_yaw_penalty_per_deg: float = 0.12,
        grasp_search_center_penalty_per_mm: float = 0.35,
        grasp_search_max_trials_per_object: int = 40,
        x_limits_mm: (
            tuple[float, float] | None
        ) = None,
        y_limits_mm: (
            tuple[float, float] | None
        ) = None,
        yaw_tcp_compensator: YawTcpCompensator | None = None,
        branch1_axis_correction_deg: float = 0.0,
        branch1_closing_bias_mm: float = 0.0,
        branch1_longitudinal_bias_mm: float = 0.0,
    ) -> None:
        self.model_path = Path(
            model_path
        )

        self.calibration_path = Path(
            calibration_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "顶部模型不存在：\n"
                f"{self.model_path}"
            )

        if not self.calibration_path.exists():
            raise FileNotFoundError(
                "像素->机械标定文件不存在：\n"
                f"{self.calibration_path}"
            )

        self.segmenter = TopSegmenter(
            model_path=self.model_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
        )

        self.planner = (
            CollisionAwareGraspPlanner(
                obstacle_margin=
                    obstacle_margin,
                border_margin=
                    border_margin,
                min_safe_area=
                    min_safe_area,
                min_clearance=
                    min_clearance,
            )
        )

        self.transform = (
            PixelToRobotTransform.load(
                self.calibration_path
            )
        )

        self.consensus = TopPickConsensus(
            window_size=
                consensus_window,
            required_hits=
                consensus_required,
            max_track_center_shift_px=
                max_track_center_shift_px,
            max_grasp_spread_mm=
                max_grasp_spread_mm,
            reset_after_misses=
                reset_after_misses,
            max_yaw_spread_deg=
                max_yaw_spread_deg,
        )

        self.orientation_enabled = bool(orientation_enabled)
        self.grip_long_sides = bool(grip_long_sides)
        self.min_orientation_aspect_ratio = float(min_orientation_aspect_ratio)
        self.default_grasp_axis_robot_deg = float(default_grasp_axis_robot_deg)
        self.yaw_alignment_offset_deg = float(yaw_alignment_offset_deg)
        self.grasp_width_slack_ratio = float(grasp_width_slack_ratio)
        self.grasp_support_tolerance_px = float(grasp_support_tolerance_px)
        self.grasp_min_contact_span_px = float(grasp_min_contact_span_px)
        self.grasp_circle_circularity_threshold = float(grasp_circle_circularity_threshold)
        self.grasp_circle_min_vertices = int(grasp_circle_min_vertices)
        self.grasp_min_opening_mm = (None if grasp_min_opening_mm is None else float(grasp_min_opening_mm))
        self.grasp_max_opening_mm = (None if grasp_max_opening_mm is None else float(grasp_max_opening_mm))
        self.dynamic_preopen_enabled = bool(dynamic_preopen_enabled)
        self.preopen_margin_mm = float(preopen_margin_mm)
        self.preopen_command_min_mm = (
            None if preopen_command_min_mm is None else float(preopen_command_min_mm)
        )
        self.preopen_command_max_mm = (
            None if preopen_command_max_mm is None else float(preopen_command_max_mm)
        )
        self.claw_preopen_profile = ClawPreopenProfile.from_runtime_config({
            "profile_kind": preopen_profile_kind,
            "configured": preopen_configured,
            "margin_mm": preopen_margin_mm,
            "min_command_mm": preopen_command_min_mm,
            "max_command_mm": preopen_command_max_mm,
            "closed_servo_deg": preopen_closed_servo_deg,
            "standby_servo_deg": preopen_standby_servo_deg,
        })
        if not self.dynamic_preopen_enabled:
            raise ValueError("New claw requires configured dynamic preopen; fixed legacy fallback is disabled")
        # Net opening, margin and limits must come from the same profile.
        # YAML is the only PC-side production configuration. The dense lookup
        # remains compiled into the MCU header, because STM32 does not read YAML.
        self.preopen_margin_mm = self.claw_preopen_profile.margin_mm
        self.preopen_command_min_mm = self.claw_preopen_profile.min_command_mm
        self.preopen_command_max_mm = self.claw_preopen_profile.max_command_mm
        print(f"[CLAW] {self.claw_preopen_profile.kind} preopen profile: {self.preopen_command_min_mm:.1f}.."
              f"{self.preopen_command_max_mm:.1f} mm, margin={self.preopen_margin_mm:.1f} mm")
        self.gripper_debug_log_rejections = bool(gripper_debug_log_rejections)
        self.gripper_footprint = GripperFootprintModel(
            transform=self.transform,
            enabled=bool(gripper_collision_enabled),
            finger_length_mm=float(gripper_finger_length_mm),
            finger_thickness_mm=float(gripper_finger_thickness_mm),
            finger_longitudinal_offset_mm=float(gripper_finger_longitudinal_offset_mm),
            opening_uncertainty_mm=float(gripper_opening_uncertainty_mm),
            object_margin_mm=float(gripper_object_margin_mm),
            wall_margin_mm=float(gripper_wall_margin_mm),
            fallback_opening_mm=float(gripper_fallback_opening_mm),
            forbidden_zones_mm=gripper_forbidden_zones_mm or [],
            allowed_region_mm=gripper_allowed_region_mm,
            allowed_polygon_mm=gripper_allowed_polygon_mm,
            object_overlap_tolerance_px=int(gripper_object_overlap_tolerance_px),
            enforce_image_border=bool(gripper_enforce_image_border),
            image_border_margin_px=int(gripper_image_border_margin_px),
            body_enabled=bool(gripper_body_enabled),
            body_width_mm=float(gripper_body_width_mm),
            body_length_mm=float(gripper_body_length_mm),
            body_longitudinal_offset_mm=float(gripper_body_longitudinal_offset_mm),
        )
        self.last_gripper_rejections: list[dict] = []
        self.last_gripper_search_alternatives: list[dict] = []

        self.grasp_search_enabled = bool(grasp_search_enabled)
        self.grasp_search_hard_min_clearance_mm = max(0.0, float(grasp_search_hard_min_clearance_mm))
        self.grasp_search_preferred_clearance_mm = max(
            self.grasp_search_hard_min_clearance_mm,
            float(grasp_search_preferred_clearance_mm),
        )
        self.grasp_search_min_target_overlap_mm = max(0.0, float(grasp_search_min_target_overlap_mm))
        self.grasp_search_min_side_gap_mm = max(0.0, float(grasp_search_min_side_gap_mm))
        self.grasp_search_yaw_penalty_per_deg = max(0.0, float(grasp_search_yaw_penalty_per_deg))
        self.grasp_search_center_penalty_per_mm = max(0.0, float(grasp_search_center_penalty_per_mm))
        self.grasp_search_max_trials_per_object = max(1, int(grasp_search_max_trials_per_object))
        if self.grasp_search_enabled:
            self.grasp_search_offsets = build_search_offsets(
                grasp_search_yaw_offsets_deg,
                grasp_search_longitudinal_offsets_mm,
                grasp_search_closing_offsets_mm,
                max_trials=self.grasp_search_max_trials_per_object,
            )
        else:
            self.grasp_search_offsets = [(0.0, 0.0, 0.0)]

        self._H_inv = np.linalg.inv(np.asarray(self.transform.H, dtype=np.float64))
        if not np.all(np.isfinite(self._H_inv)):
            raise ValueError("pixel_to_robot homography inverse invalid")

        self.yaw_tcp_compensator = yaw_tcp_compensator

        # branch=1 专用微调：
        # - axis_correction_deg：另一物理侧的实际夹爪方向角修正；
        # - closing_bias_mm：另一物理侧沿夹爪闭合轴的抓取中心左右修正；
        # - longitudinal_bias_mm：另一物理侧沿夹爪前后方向
        #   （闭合轴 +90°）的抓取中心修正。
        #
        # branch=0 完全不受这三个参数影响。
        self.branch1_axis_correction_deg = float(branch1_axis_correction_deg)
        self.branch1_closing_bias_mm = float(branch1_closing_bias_mm)
        self.branch1_longitudinal_bias_mm = float(branch1_longitudinal_bias_mm)

        if not math.isfinite(self.branch1_axis_correction_deg):
            raise ValueError("branch1_axis_correction_deg must be finite")
        if not math.isfinite(self.branch1_closing_bias_mm):
            raise ValueError("branch1_closing_bias_mm must be finite")
        if not math.isfinite(self.branch1_longitudinal_bias_mm):
            raise ValueError("branch1_longitudinal_bias_mm must be finite")
        if abs(self.branch1_axis_correction_deg) > 15.0:
            raise ValueError(
                "branch1_axis_correction_deg absolute value must be <= 15 deg; "
                "larger error should be recalibrated instead of patched"
            )
        if abs(self.branch1_closing_bias_mm) > 10.0:
            raise ValueError(
                "branch1_closing_bias_mm absolute value must be <= 10 mm"
            )
        if abs(self.branch1_longitudinal_bias_mm) > 10.0:
            raise ValueError(
                "branch1_longitudinal_bias_mm absolute value must be <= 10 mm"
            )

        if self.min_orientation_aspect_ratio < 1.0:
            raise ValueError("min_orientation_aspect_ratio must be >= 1.0")
        if self.grasp_width_slack_ratio < 0.0:
            raise ValueError("grasp_width_slack_ratio must be >= 0")
        if self.grasp_support_tolerance_px <= 0.0:
            raise ValueError("grasp_support_tolerance_px must be > 0")
        if self.grasp_min_contact_span_px < 0.0:
            raise ValueError("grasp_min_contact_span_px must be >= 0")
        if self.preopen_margin_mm < 0.0:
            raise ValueError("preopen_margin_mm must be >= 0")
        if (
            self.preopen_command_min_mm is not None
            and self.preopen_command_max_mm is not None
            and self.preopen_command_min_mm > self.preopen_command_max_mm
        ):
            raise ValueError("preopen_command_min_mm cannot exceed preopen_command_max_mm")
        if (
            self.grasp_min_opening_mm is not None
            and self.grasp_max_opening_mm is not None
            and self.grasp_min_opening_mm > self.grasp_max_opening_mm
        ):
            raise ValueError("grasp_min_opening_mm cannot exceed grasp_max_opening_mm")

        self.required_hits = int(
            consensus_required
        )

        self.window_size = int(
            consensus_window
        )

        self.x_limits_mm = (
            tuple(
                map(
                    float,
                    x_limits_mm,
                )
            )
            if x_limits_mm is not None
            else None
        )

        self.y_limits_mm = (
            tuple(
                map(
                    float,
                    y_limits_mm,
                )
            )
            if y_limits_mm is not None
            else None
        )

    def reset(
        self,
    ) -> None:
        self.consensus.reset()

    @staticmethod
    def _polygon_center(
        polygon: np.ndarray,
    ) -> tuple[float, float]:
        polygon = np.asarray(
            polygon,
            dtype=np.float32,
        )

        moments = cv2.moments(
            polygon
        )

        if abs(
            moments["m00"]
        ) > 1e-9:
            return (
                float(
                    moments["m10"]
                    /
                    moments["m00"]
                ),
                float(
                    moments["m01"]
                    /
                    moments["m00"]
                ),
            )

        mean = np.mean(
            polygon,
            axis=0,
        )

        return (
            float(mean[0]),
            float(mean[1]),
        )

    @staticmethod
    def _normalize_polygon(
        polygon,
    ) -> np.ndarray | None:
        if polygon is None:
            return None

        polygon = np.asarray(
            polygon,
            dtype=np.float32,
        )

        if (
            polygon.ndim != 2
            or
            polygon.shape[1] != 2
            or
            len(polygon) < 3
        ):
            return None

        if not np.all(
            np.isfinite(
                polygon
            )
        ):
            return None

        return polygon

    def _inside_machine_limits(
        self,
        x_mm: float,
        y_mm: float,
    ) -> bool:
        if self.x_limits_mm is not None:
            x_min, x_max = (
                self.x_limits_mm
            )

            if not (
                x_min
                <=
                x_mm
                <=
                x_max
            ):
                return False

        if self.y_limits_mm is not None:
            y_min, y_max = (
                self.y_limits_mm
            )

            if not (
                y_min
                <=
                y_mm
                <=
                y_max
            ):
                return False

        return True

    def _robot_to_pixel(
        self,
        point_robot_mm: tuple[float, float] | np.ndarray,
    ) -> tuple[float, float] | None:
        point = np.asarray(point_robot_mm, dtype=np.float64).reshape(1, 1, 2)
        try:
            out = cv2.perspectiveTransform(point, self._H_inv)[0, 0]
        except Exception:
            return None
        if not np.all(np.isfinite(out)):
            return None
        return float(out[0]), float(out[1])

    def _preopen_for_width(self, width_mm: float | None) -> float | None:
        if not self.dynamic_preopen_enabled or width_mm is None:
            raise ValueError("new claw requires a measured width; no legacy fixed preopen")
        return self.claw_preopen_profile.command_for_width(width_mm)

    def _physical_yaw_for_branch(
        self,
        axial_yaw_deg: float,
        branch: int,
    ) -> float:
        """把[-90,90)轴向抓取角展开成唯一的有向物理Yaw。

        branch=0:
            保持原轴向角，不施加 branch=1 微调。

        branch=1:
            先切换到相差180°的另一物理侧，再叠加
            branch1_axis_correction_deg。

        注意：
        这个返回值就是 Jetson 做 Yaw-TCP 补偿时使用的真实物理Yaw，
        STM32 必须使用完全相同的 branch=1 角度修正。
        """
        axial = normalize_axial_deg(float(axial_yaw_deg))

        if int(branch) == 0:
            return float(axial)

        if int(branch) == 1:
            physical = (
                axial + 180.0
                if axial < 0.0
                else axial - 180.0
            )
            physical += self.branch1_axis_correction_deg

            # 统一到 [-180, 180)
            physical = ((physical + 180.0) % 360.0) - 180.0
            return float(physical)

        raise ValueError(f"yaw branch must be 0 or 1, got {branch}")

    def _workspace_margin_mm(self, x_mm: float, y_mm: float) -> float:
        """返回候选点到当前XY工作区最近边界的余量；余量越大越稳。"""
        margins: list[float] = []
        if self.x_limits_mm is not None:
            x_min, x_max = self.x_limits_mm
            margins.extend((float(x_mm) - x_min, x_max - float(x_mm)))
        if self.y_limits_mm is not None:
            y_min, y_max = self.y_limits_mm
            margins.extend((float(y_mm) - y_min, y_max - float(y_mm)))
        return float(min(margins)) if margins else 0.0

    def _select_yaw_branch(
        self,
        base_x_mm: float,
        base_y_mm: float,
        axial_yaw_deg: float,
    ) -> tuple[float, float, int, float] | None:
        """评估两个180°等效物理Yaw分支，并选择实际可执行的一侧。

        选择规则只新增双边能力，不改变原有抓取姿态/碰撞评分：
        1) 当前Yaw-TCP补偿器必须接受该物理Yaw；
        2) 补偿后的XY必须位于现有机械工作区；
        3) 两边都可用时，优先工作区边界余量更大的分支；
        4) 完全同分时优先branch 0，保持原行为。
        """
        # 没有Yaw-TCP补偿时无法安全区分两种偏心物理姿态，保持原branch 0。
        if self.yaw_tcp_compensator is None:
            if self._inside_machine_limits(base_x_mm, base_y_mm):
                return float(base_x_mm), float(base_y_mm), 0, float(axial_yaw_deg)
            return None

        feasible: list[tuple[float, int, float, float, float]] = []
        for branch in (0, 1):
            physical_yaw_deg = self._physical_yaw_for_branch(axial_yaw_deg, branch)
            try:
                compensated = self.yaw_tcp_compensator.apply(
                    base_x_mm,
                    base_y_mm,
                    physical_yaw_deg,
                )
            except (ValueError, RuntimeError):
                continue

            x_cmd = float(compensated.x_mm)
            y_cmd = float(compensated.y_mm)

            # --------------------------------------------------------
            # branch=1 专用二维抓取中心修正
            #
            # closing:
            #   沿真实夹爪闭合轴 [cos(theta), sin(theta)]。
            #
            # longitudinal:
            #   沿闭合轴逆时针 +90°：
            #   [-sin(theta), cos(theta)]。
            #
            # 两个修正先定义在视觉/H物理平面(mm)，再复用现有 J^-1
            # (visual_to_command_matrix) 转成真正的 XY 命令毫米。
            # branch=0 完全不受影响。
            # --------------------------------------------------------
            if int(branch) == 1:
                closing_bias = float(
                    self.branch1_closing_bias_mm
                )
                longitudinal_bias = float(
                    self.branch1_longitudinal_bias_mm
                )

                if (
                    abs(closing_bias) > 1.0e-12
                    or abs(longitudinal_bias) > 1.0e-12
                ):
                    theta = math.radians(
                        float(physical_yaw_deg)
                    )

                    close_x = math.cos(theta)
                    close_y = math.sin(theta)

                    long_x = -math.sin(theta)
                    long_y = math.cos(theta)

                    bias_vx = (
                        closing_bias * close_x
                        + longitudinal_bias * long_x
                    )
                    bias_vy = (
                        closing_bias * close_y
                        + longitudinal_bias * long_y
                    )

                    matrix = (
                        self.yaw_tcp_compensator
                        .visual_to_command_matrix
                    )
                    bias_dx = (
                        matrix[0][0] * bias_vx
                        + matrix[0][1] * bias_vy
                    )
                    bias_dy = (
                        matrix[1][0] * bias_vx
                        + matrix[1][1] * bias_vy
                    )

                    x_cmd += float(bias_dx)
                    y_cmd += float(bias_dy)

            if not self._inside_machine_limits(x_cmd, y_cmd):
                continue

            workspace_margin = self._workspace_margin_mm(x_cmd, y_cmd)
            feasible.append(
                (workspace_margin, int(branch), x_cmd, y_cmd, float(physical_yaw_deg))
            )

        if not feasible:
            return None

        # margin最大优先；完全同分时branch 0优先。
        feasible.sort(key=lambda item: (-item[0], item[1]))
        _, branch, x_cmd, y_cmd, physical_yaw_deg = feasible[0]
        return x_cmd, y_cmd, branch, physical_yaw_deg

    def _build_candidates(
        self,
        frame: np.ndarray,
        objects: list[dict],
        *,
        fast_only: bool = False,
    ) -> list[TopPickCandidate]:
        self.last_gripper_rejections = []
        self.last_gripper_search_alternatives = []

        valid_objects: list[tuple[int, dict, np.ndarray]] = []
        for list_index, obj in enumerate(objects):
            polygon = self._normalize_polygon(obj.get("polygon"))
            if polygon is None:
                continue
            valid_objects.append((list_index, obj, polygon))

        candidates: list[TopPickCandidate] = []

        for list_index, obj, target_polygon in valid_objects:
            other_polygons = [
                polygon
                for other_index, _, polygon in valid_objects
                if other_index != list_index
            ]

            grasp_result = self.planner.plan(
                target_polygon=target_polygon,
                other_polygons=other_polygons,
                image_shape=frame.shape,
            )
            if not grasp_result.valid or grasp_result.grasp_point is None:
                continue

            base_grasp_x, base_grasp_y = grasp_result.grasp_point
            center_pixel = self._polygon_center(target_polygon)

            # ---------- 先求一个几何上合理的基础姿态 ----------
            orientation = None
            parallel_edge_options = []
            if self.orientation_enabled:
                try:
                    orientation = plan_grasp_axis(
                        polygon=target_polygon,
                        grasp_pixel=(base_grasp_x, base_grasp_y),
                        transform=self.transform,
                        grip_long_sides=self.grip_long_sides,
                        min_aspect_ratio=self.min_orientation_aspect_ratio,
                        default_grasp_axis_robot_deg=self.default_grasp_axis_robot_deg,
                        yaw_alignment_offset_deg=self.yaw_alignment_offset_deg,
                        safe_mask=grasp_result.safe_mask,
                        width_slack_ratio=self.grasp_width_slack_ratio,
                        support_tolerance_px=self.grasp_support_tolerance_px,
                        min_contact_span_px=self.grasp_min_contact_span_px,
                        circle_circularity_threshold=self.grasp_circle_circularity_threshold,
                        circle_min_vertices=self.grasp_circle_min_vertices,
                    )
                    if orientation.grasp_pixel_override is not None:
                        base_grasp_x, base_grasp_y = orientation.grasp_pixel_override
                    base_yaw_deg = float(orientation.grasp_axis_robot_deg)
                    long_axis_deg = float(orientation.long_axis_robot_deg)
                    aspect_ratio = float(orientation.aspect_ratio)
                    orientation_reliable = bool(orientation.reliable)
                    orientation_mode = str(orientation.mode)
                    rectangularity = float(orientation.rectangularity)
                    circularity = float(orientation.circularity)
                    hull_vertex_count = int(orientation.hull_vertex_count)
                    closing_axis_image_deg = float(orientation.closing_axis_image_deg)
                    base_grasp_width_px = float(orientation.grasp_width_px)
                    base_grasp_width_mm = orientation.grasp_width_mm
                    contact_type = str(orientation.contact_type)
                    contact_quality = float(orientation.contact_quality)
                    support_overlap_px = float(orientation.support_overlap_px)
                    support_min_point_px = orientation.support_min_point_px
                    support_max_point_px = orientation.support_max_point_px

                    # 除主姿态外，再保留所有真实“相对平行边 edge-edge”抓取族。
                    # 例如长方形会同时得到两组互相约90°的闭合轴：夹长边、夹短边。
                    # 这两组在几何上都是真正的平行夹爪抓取，不再因为开口更大而被提前丢弃。
                    parallel_edge_options = plan_parallel_edge_grasp_options(
                        polygon=target_polygon,
                        grasp_pixel=(base_grasp_x, base_grasp_y),
                        transform=self.transform,
                        default_grasp_axis_robot_deg=self.default_grasp_axis_robot_deg,
                        yaw_alignment_offset_deg=self.yaw_alignment_offset_deg,
                        safe_mask=grasp_result.safe_mask,
                        width_slack_ratio=self.grasp_width_slack_ratio,
                        support_tolerance_px=self.grasp_support_tolerance_px,
                        min_contact_span_px=self.grasp_min_contact_span_px,
                        circle_circularity_threshold=self.grasp_circle_circularity_threshold,
                        circle_min_vertices=self.grasp_circle_min_vertices,
                    )
                except Exception:
                    base_yaw_deg = float(self.default_grasp_axis_robot_deg)
                    long_axis_deg = float(self.default_grasp_axis_robot_deg)
                    aspect_ratio = 1.0
                    orientation_reliable = False
                    orientation_mode = "geometry_fallback"
                    rectangularity = 0.0
                    circularity = 0.0
                    hull_vertex_count = 0
                    closing_axis_image_deg = 0.0
                    base_grasp_width_px = 0.0
                    base_grasp_width_mm = None
                    contact_type = "none"
                    contact_quality = 0.0
                    support_overlap_px = 0.0
                    support_min_point_px = None
                    support_max_point_px = None
                    parallel_edge_options = []
            else:
                base_yaw_deg = float(self.default_grasp_axis_robot_deg)
                long_axis_deg = float(self.default_grasp_axis_robot_deg)
                aspect_ratio = 1.0
                orientation_reliable = False
                orientation_mode = "disabled_fallback"
                rectangularity = 0.0
                circularity = 0.0
                hull_vertex_count = 0
                closing_axis_image_deg = 0.0
                base_grasp_width_px = 0.0
                base_grasp_width_mm = None
                contact_type = "none"
                contact_quality = 0.0
                support_overlap_px = 0.0
                support_min_point_px = None
                support_max_point_px = None
                parallel_edge_options = []

            # 建立“等价基础抓取族”。主姿态始终保留；如果物体存在其它成对平行边，
            # 将它们也作为同等级基础方向交给后续真实夹爪碰撞搜索。
            orientation_family = [orientation] if orientation is not None else []
            for option in parallel_edge_options:
                if not any(
                    abs(axial_difference_deg(option.grasp_axis_robot_deg, old.grasp_axis_robot_deg)) < 1.0
                    for old in orientation_family
                ):
                    orientation_family.append(option)
            if not orientation_family:
                orientation_family_yaws = [float(base_yaw_deg)]
                orientation_family_contact = [float(contact_quality)]
            else:
                orientation_family_yaws = [float(o.grasp_axis_robot_deg) for o in orientation_family]
                orientation_family_contact = [float(o.contact_quality) for o in orientation_family]

            try:
                target_robot = self.transform.pixels_to_robot(
                    target_polygon,
                    apply_tcp_offset=False,
                )
                base_center_robot = np.asarray(
                    self.transform.pixel_to_robot(
                        float(base_grasp_x),
                        float(base_grasp_y),
                        apply_tcp_offset=False,
                    ),
                    dtype=np.float64,
                )
            except Exception:
                continue

            if (
                target_robot.ndim != 2
                or target_robot.shape[1] != 2
                or len(target_robot) < 3
                or not np.all(np.isfinite(target_robot))
                or not np.all(np.isfinite(base_center_robot))
            ):
                continue

            # 每个目标只生成一次其它物体/固定挡板的整图 mask 和距离场。
            # 否则 20~40 个姿态逐一做 distanceTransform 会非常慢。
            collision_context = self.gripper_footprint.prepare_context(
                other_polygons=other_polygons,
                image_shape=frame.shape,
            )

            confidence = float(obj.get("confidence", 0.0))
            clearance = float(grasp_result.clearance)
            center_gap_base_px = max(
                0.0,
                float(grasp_result.obstacle_clearance)
                - max(0.0, float(base_grasp_width_px) * 0.5),
            )

            if self.grasp_search_enabled:
                trial_specs = []
                seen_trial_keys = set()

                if fast_only:
                    # ====================================================
                    # 快速路径：
                    #
                    # 只检查每个真实基础抓取族的“原始姿态”：
                    #   yaw 偏移 = 0
                    #   longitudinal 偏移 = 0
                    #   closing 偏移 = 0
                    #
                    # 对一个普通物体通常只需检查 1~2 个姿态。
                    # 只要整帧中存在任意安全候选，process() 就直接使用
                    # 这些快速候选，不再执行几十/上百个完整搜索姿态。
                    #
                    # 若整帧一个安全候选都没有，再自动退回原来的完整搜索。
                    # ====================================================
                    for family_yaw_deg in orientation_family_yaws:
                        family_delta = axial_difference_deg(
                            family_yaw_deg,
                            base_yaw_deg,
                        )
                        final_yaw = normalize_axial_deg(
                            base_yaw_deg + family_delta
                        )
                        key = (
                            round(float(final_yaw), 4),
                            0.0,
                            0.0,
                        )
                        if key in seen_trial_keys:
                            continue
                        seen_trial_keys.add(key)
                        trial_specs.append(
                            (
                                float(family_delta),
                                0.0,
                                0.0,
                            )
                        )
                else:
                    # ====================================================
                    # 完整搜索：
                    #
                    # 保持原来的搜索行为完全不变。
                    # 只有快速路径在整帧中找不到任何安全候选时才进入这里。
                    # ====================================================
                    # 把每个真实平行边抓取族都作为一个“零代价基准方向”，并在每个基准
                    # 周围应用原有的小角度/小位移搜索。按 local-spec 外层、family 内层交错，
                    # 可避免 max_trials 时第一组方向把第二组方向全部挤掉。
                    for (
                        local_yaw_offset_deg,
                        longitudinal_offset_mm,
                        closing_offset_mm,
                    ) in self.grasp_search_offsets:
                        for family_yaw_deg in orientation_family_yaws:
                            family_delta = axial_difference_deg(
                                family_yaw_deg,
                                base_yaw_deg,
                            )
                            combined_yaw_offset = float(
                                family_delta + local_yaw_offset_deg
                            )
                            final_yaw = normalize_axial_deg(
                                base_yaw_deg + combined_yaw_offset
                            )
                            key = (
                                round(float(final_yaw), 4),
                                round(float(longitudinal_offset_mm), 4),
                                round(float(closing_offset_mm), 4),
                            )
                            if key in seen_trial_keys:
                                continue
                            seen_trial_keys.add(key)
                            trial_specs.append(
                                (
                                    combined_yaw_offset,
                                    float(longitudinal_offset_mm),
                                    float(closing_offset_mm),
                                )
                            )
                            if (
                                len(trial_specs)
                                >= self.grasp_search_max_trials_per_object
                            ):
                                break
                        if (
                            len(trial_specs)
                            >= self.grasp_search_max_trials_per_object
                        ):
                            break
            else:
                trial_specs = [(0.0, 0.0, 0.0)]
            best_candidate: TopPickCandidate | None = None
            best_trial_record: dict | None = None
            safe_trial_records: list[dict] = []
            reason_counts: dict[str, int] = {}
            trial_count = 0
            safe_count = 0
            stored_rejects = 0

            def reject(reason: str, *, px=None, yaw=None, opening=None, polygons=None, meta=None) -> None:
                nonlocal stored_rejects
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                # 预览只保存有限数量的红框，避免35个候选全部铺满画面。
                if stored_rejects >= 10:
                    return
                record = {
                    "object_index": int(obj.get("index", list_index)),
                    "grasp_pixel": (
                        (int(round(px[0])), int(round(px[1]))) if px is not None else
                        (int(base_grasp_x), int(base_grasp_y))
                    ),
                    "yaw_deg": float(base_yaw_deg if yaw is None else yaw),
                    "opening_mm": None if opening is None else float(opening),
                    "reason": str(reason),
                    "footprint_polygons_px": [
                        np.asarray(poly, dtype=np.float32).copy()
                        for poly in (polygons or [])
                    ],
                }
                if meta:
                    record.update(meta)
                self.last_gripper_rejections.append(record)
                stored_rejects += 1

            for yaw_offset_deg, longitudinal_offset_mm, closing_offset_mm in trial_specs:
                trial_count += 1
                yaw_deg = normalize_axial_deg(base_yaw_deg + yaw_offset_deg)
                closing_axis, longitudinal_axis = axis_vectors(yaw_deg)
                candidate_center_robot = (
                    base_center_robot
                    + float(longitudinal_offset_mm) * longitudinal_axis
                    + float(closing_offset_mm) * closing_axis
                )

                candidate_px_f = self._robot_to_pixel(candidate_center_robot)
                if candidate_px_f is None:
                    reject("candidate_robot_to_pixel_failed", yaw=yaw_deg)
                    continue
                candidate_px = (float(candidate_px_f[0]), float(candidate_px_f[1]))

                # 中心仍必须落在目标本体内，不能为了避障把夹爪中心移到物体外面。
                inside = cv2.pointPolygonTest(
                    np.asarray(target_polygon, dtype=np.float32).reshape(-1, 1, 2),
                    candidate_px,
                    False,
                )
                if inside < 0:
                    reject(
                        "grasp_center_outside_target",
                        px=candidate_px,
                        yaw=yaw_deg,
                        meta={
                            "yaw_offset_deg": float(yaw_offset_deg),
                            "longitudinal_offset_mm": float(longitudinal_offset_mm),
                            "closing_offset_mm": float(closing_offset_mm),
                        },
                    )
                    continue

                # 先按该 yaw 重新计算真实物体宽度；旋转夹爪后开口必须跟着变化。
                # 初次用零开口只读取 width，然后再算动态 preopen 并做完整接触检查。
                width_probe = evaluate_target_engagement(
                    target_robot,
                    candidate_center_robot,
                    yaw_deg,
                    1.0e6,
                    finger_length_mm=self.gripper_footprint.finger_length_mm,
                    finger_longitudinal_offset_mm=self.gripper_footprint.finger_longitudinal_offset_mm,
                    min_side_gap_mm=0.0,
                    min_longitudinal_overlap_mm=0.0,
                )
                trial_width_mm = float(width_probe.width_mm)
                if not np.isfinite(trial_width_mm) or trial_width_mm <= 0.0:
                    reject("invalid_trial_width", px=candidate_px, yaw=yaw_deg)
                    continue

                if self.grasp_min_opening_mm is not None and trial_width_mm < self.grasp_min_opening_mm:
                    reject("below_min_grasp_width", px=candidate_px, yaw=yaw_deg)
                    continue
                if self.grasp_max_opening_mm is not None and trial_width_mm > self.grasp_max_opening_mm:
                    reject("above_max_grasp_width", px=candidate_px, yaw=yaw_deg)
                    continue

                try:
                    preopen_mm = self._preopen_for_width(trial_width_mm)
                except ValueError as exc:
                    reject("preopen_out_of_calibrated_range", px=candidate_px, yaw=yaw_deg,
                           meta={"preopen_error": str(exc)})
                    continue
                engagement_opening = (
                    self.gripper_footprint.fallback_opening_mm
                    if preopen_mm is None
                    else float(preopen_mm)
                )
                engagement = evaluate_target_engagement(
                    target_robot,
                    candidate_center_robot,
                    yaw_deg,
                    engagement_opening,
                    finger_length_mm=self.gripper_footprint.finger_length_mm,
                    finger_longitudinal_offset_mm=self.gripper_footprint.finger_longitudinal_offset_mm,
                    min_side_gap_mm=self.grasp_search_min_side_gap_mm,
                    min_longitudinal_overlap_mm=self.grasp_search_min_target_overlap_mm,
                )
                if not engagement.fits_between_fingers:
                    reject(
                        "target_not_between_fingers",
                        px=candidate_px,
                        yaw=yaw_deg,
                        opening=preopen_mm,
                        meta={
                            "min_side_gap_mm": float(engagement.min_side_gap_mm),
                            "yaw_offset_deg": float(yaw_offset_deg),
                            "longitudinal_offset_mm": float(longitudinal_offset_mm),
                            "closing_offset_mm": float(closing_offset_mm),
                        },
                    )
                    continue
                if not engagement.has_longitudinal_overlap:
                    reject(
                        "target_contact_overlap_too_small",
                        px=candidate_px,
                        yaw=yaw_deg,
                        opening=preopen_mm,
                        meta={
                            "target_overlap_mm": float(engagement.longitudinal_overlap_mm),
                            "yaw_offset_deg": float(yaw_offset_deg),
                            "longitudinal_offset_mm": float(longitudinal_offset_mm),
                            "closing_offset_mm": float(closing_offset_mm),
                        },
                    )
                    continue

                footprint_check = self.gripper_footprint.check(
                    grasp_pixel=candidate_px,
                    yaw_deg=float(yaw_deg),
                    opening_mm=preopen_mm,
                    other_polygons=other_polygons,
                    image_shape=frame.shape,
                    context=collision_context,
                )
                if not footprint_check.safe:
                    reject(
                        str(footprint_check.reason),
                        px=candidate_px,
                        yaw=yaw_deg,
                        opening=preopen_mm,
                        polygons=footprint_check.footprint_polygons_px,
                        meta={
                            "yaw_offset_deg": float(yaw_offset_deg),
                            "longitudinal_offset_mm": float(longitudinal_offset_mm),
                            "closing_offset_mm": float(closing_offset_mm),
                        },
                    )
                    continue

                if float(footprint_check.clearance_mm) < self.grasp_search_hard_min_clearance_mm:
                    reject(
                        "clearance_below_hard_min",
                        px=candidate_px,
                        yaw=yaw_deg,
                        opening=preopen_mm,
                        polygons=footprint_check.footprint_polygons_px,
                        meta={"clearance_mm": float(footprint_check.clearance_mm)},
                    )
                    continue

                try:
                    x_mm, y_mm = self.transform.pixel_to_robot(
                        candidate_px[0],
                        candidate_px[1],
                        apply_tcp_offset=True,
                    )
                except Exception:
                    reject("candidate_pixel_transform_failed", px=candidate_px, yaw=yaw_deg)
                    continue

                if not (np.isfinite(x_mm) and np.isfinite(y_mm)):
                    reject("candidate_xy_not_finite", px=candidate_px, yaw=yaw_deg)
                    continue

                # 双边Yaw：同一个轴向抓取角同时检查两个180°等效物理姿态。
                # TCP补偿必须使用最终真实执行的physical_yaw，而PICK_TARGET仍发送
                # 原来的[-90,90)轴向yaw；branch通过独立命令发送给STM32。
                selected = self._select_yaw_branch(
                    float(x_mm),
                    float(y_mm),
                    float(yaw_deg),
                )
                if selected is None:
                    reject("dual_yaw_branch_unavailable", px=candidate_px, yaw=yaw_deg)
                    continue

                x_mm, y_mm, yaw_branch, physical_yaw_deg = selected

                safe_count += 1

                center_offset_mm = math.hypot(
                    float(longitudinal_offset_mm),
                    float(closing_offset_mm),
                )
                gripper_gap_score_mm = min(
                    max(float(footprint_check.clearance_mm), 0.0),
                    self.grasp_search_preferred_clearance_mm,
                )
                # 找到当前姿态最近的真实基础抓取族。对于“另一对相对平行边”，
                # 它本身就是合法基准方向，不应该因为与主方向相差约90°而被罚分。
                family_distances = [
                    abs(axial_difference_deg(yaw_deg, family_yaw))
                    for family_yaw in orientation_family_yaws
                ]
                nearest_family_index = int(np.argmin(family_distances))
                yaw_deviation_from_family = float(family_distances[nearest_family_index])
                family_contact_quality = float(orientation_family_contact[nearest_family_index])
                contact_score = max(0.0, min(family_contact_quality, 1.0))
                confidence_score = max(0.0, min(confidence, 1.0))

                # 安全间隙是主评分项；只有“偏离最近真实抓取族”的小角度才罚分。
                # 因此矩形夹长边和夹短边在这里地位相同，最终由真实碰撞/间隙决定。
                score = (
                    10.0 * gripper_gap_score_mm
                    + 0.25 * min(center_gap_base_px, 60.0)
                    + 0.20 * min(clearance, 40.0)
                    + 10.0 * contact_score
                    + 5.0 * confidence_score
                    - self.grasp_search_yaw_penalty_per_deg * yaw_deviation_from_family
                    - self.grasp_search_center_penalty_per_mm * center_offset_mm
                )

                # 用当前物理宽度修正用于显示/兼容的像素宽度估计。
                if orientation_family and nearest_family_index < len(orientation_family):
                    trial_family = orientation_family[nearest_family_index]
                    ref_width_mm = trial_family.grasp_width_mm
                    ref_width_px = float(trial_family.grasp_width_px)
                else:
                    trial_family = None
                    ref_width_mm = base_grasp_width_mm
                    ref_width_px = float(base_grasp_width_px)

                if ref_width_mm is not None and float(ref_width_mm) > 1e-6:
                    trial_width_px = max(
                        0.0,
                        ref_width_px * trial_width_mm / float(ref_width_mm),
                    )
                else:
                    trial_width_px = max(0.0, ref_width_px)

                candidate = TopPickCandidate(
                    object_index=int(obj.get("index", list_index)),
                    class_name=str(obj.get("class_name", "")),
                    confidence=confidence,
                    polygon=target_polygon,
                    center_pixel=center_pixel,
                    grasp_pixel=(int(round(candidate_px[0])), int(round(candidate_px[1]))),
                    x_mm=float(x_mm),
                    y_mm=float(y_mm),
                    yaw_deg=float(yaw_deg),
                    yaw_branch=int(yaw_branch),
                    object_long_axis_deg=float(long_axis_deg),
                    aspect_ratio=float(aspect_ratio),
                    orientation_reliable=bool(
                        trial_family.reliable if trial_family is not None else orientation_reliable
                    ),
                    orientation_mode=str(
                        trial_family.mode if trial_family is not None else orientation_mode
                    ),
                    rectangularity=float(rectangularity),
                    circularity=float(circularity),
                    hull_vertex_count=int(hull_vertex_count),
                    closing_axis_image_deg=float(
                        trial_family.closing_axis_image_deg
                        if trial_family is not None else closing_axis_image_deg
                    ),
                    grasp_width_px=float(trial_width_px),
                    grasp_width_mm=float(trial_width_mm),
                    contact_type=str(
                        trial_family.contact_type if trial_family is not None else contact_type
                    ),
                    contact_quality=float(family_contact_quality),
                    support_overlap_px=float(
                        trial_family.support_overlap_px
                        if trial_family is not None else support_overlap_px
                    ),
                    support_min_point_px=(
                        trial_family.support_min_point_px
                        if trial_family is not None else support_min_point_px
                    ),
                    support_max_point_px=(
                        trial_family.support_max_point_px
                        if trial_family is not None else support_max_point_px
                    ),
                    clearance_px=clearance,
                    self_clearance_px=float(grasp_result.self_clearance),
                    obstacle_clearance_px=float(grasp_result.obstacle_clearance),
                    approach_gap_px=float(center_gap_base_px),
                    safe_area_px=int(grasp_result.safe_area),
                    gripper_clearance_px=float(footprint_check.clearance_px),
                    gripper_clearance_mm=float(footprint_check.clearance_mm),
                    gripper_footprint_polygons_px=[
                        np.asarray(poly, dtype=np.float32).copy()
                        for poly in footprint_check.footprint_polygons_px
                    ],
                    preopen_mm=None if preopen_mm is None else float(preopen_mm),
                    search_yaw_offset_deg=float(yaw_offset_deg),
                    search_longitudinal_offset_mm=float(longitudinal_offset_mm),
                    search_closing_offset_mm=float(closing_offset_mm),
                    search_trial_count=0,  # 结束搜索后统一写入
                    search_safe_count=0,
                    score=float(score),
                )

                trial_record = {
                    "object_index": int(obj.get("index", list_index)),
                    "grasp_pixel": candidate.grasp_pixel,
                    "yaw_deg": float(yaw_deg),
                    "yaw_branch": int(yaw_branch),
                    "physical_yaw_deg": float(physical_yaw_deg),
                    "opening_mm": None if preopen_mm is None else float(preopen_mm),
                    "clearance_mm": float(footprint_check.clearance_mm),
                    "score": float(score),
                    "yaw_offset_deg": float(yaw_offset_deg),
                    "longitudinal_offset_mm": float(longitudinal_offset_mm),
                    "closing_offset_mm": float(closing_offset_mm),
                    "footprint_polygons_px": [
                        np.asarray(poly, dtype=np.float32).copy()
                        for poly in footprint_check.footprint_polygons_px
                    ],
                }
                safe_trial_records.append(trial_record)

                if best_candidate is None or candidate.score > best_candidate.score:
                    best_candidate = candidate
                    best_trial_record = trial_record

            if best_candidate is None:
                if self.gripper_debug_log_rejections:
                    reason_text = ", ".join(
                        f"{key}:{value}"
                        for key, value in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
                    )
                    log_tag = (
                        "[GRASP FAST]"
                        if fast_only
                        else "[GRASP SEARCH]"
                    )
                    print(
                        f"{log_tag} "
                        f"obj={int(obj.get('index', list_index))} "
                        f"trials={trial_count} safe=0 reasons={reason_text or 'none'}"
                    )
                continue

            best_candidate.search_trial_count = int(trial_count)
            best_candidate.search_safe_count = int(safe_count)
            candidates.append(best_candidate)

            # 预览只保留得分最高的少量“安全但未选中”姿态，画成黄色。
            alternatives = sorted(
                safe_trial_records,
                key=lambda item: float(item.get("score", 0.0)),
                reverse=True,
            )
            for record in alternatives:
                if record is best_trial_record:
                    continue
                self.last_gripper_search_alternatives.append(record)
                if sum(
                    1
                    for item in self.last_gripper_search_alternatives
                    if int(item.get("object_index", -9999)) == int(obj.get("index", list_index))
                ) >= 6:
                    break

            if self.gripper_debug_log_rejections:
                log_tag = (
                    "[GRASP FAST]"
                    if fast_only
                    else "[GRASP SEARCH]"
                )
                print(
                    f"{log_tag} "
                    f"obj={best_candidate.object_index} "
                    f"trials={trial_count} safe={safe_count} "
                    f"best_yaw={best_candidate.yaw_deg:+.1f}deg "
                    f"branch={best_candidate.yaw_branch} "
                    f"physical={self._physical_yaw_for_branch(best_candidate.yaw_deg, best_candidate.yaw_branch):+.1f}deg "
                    f"dyaw={best_candidate.search_yaw_offset_deg:+.1f}deg "
                    f"dlong={best_candidate.search_longitudinal_offset_mm:+.1f}mm "
                    f"gap={best_candidate.gripper_clearance_mm:.2f}mm"
                )

        return candidates

    @staticmethod
    def _candidate_to_consensus_dict(
        candidate: TopPickCandidate,
    ) -> dict:
        return {
            "center_pixel":
                candidate.center_pixel,
            "grasp_x":
                candidate.grasp_pixel[0],
            "grasp_y":
                candidate.grasp_pixel[1],
            "x_mm":
                candidate.x_mm,
            "y_mm":
                candidate.y_mm,
            "yaw_deg":
                candidate.yaw_deg,
            # TopPickConsensus会保留额外字段到history；不改原共识算法，
            # 这里只让Pipeline在READY时确认两帧选择的是同一物理分支。
            "yaw_branch":
                candidate.yaw_branch,
            "orientation_reliable":
                candidate.orientation_reliable,
            "orientation_mode":
                candidate.orientation_mode,
            "rectangularity":
                candidate.rectangularity,
            "aspect_ratio":
                candidate.aspect_ratio,
            "confidence":
                candidate.confidence,
            "clearance":
                candidate.clearance_px,
            "grasp_width_mm":
                candidate.grasp_width_mm,
            "preopen_mm":
                candidate.preopen_mm,
            "approach_gap_px":
                candidate.approach_gap_px,
            "score":
                candidate.score,
            "class_name":
                candidate.class_name,
        }

    def process(
        self,
        frame: np.ndarray,
    ) -> TopPickPipelineResult:
        if frame is None:
            raise ValueError(
                "frame 不能为 None"
            )

        # 标定时与运行时必须使用
        # 完全相同的 TopCamera.read() 输出尺寸。
        self.transform.check_frame(
            frame
        )

        start = time.perf_counter()

        yolo_result = (
            self.segmenter.predict(
                frame
            )
        )

        objects = (
            self.segmenter.get_objects(
                yolo_result
            )
        )

        inference_ms = (
            (
                time.perf_counter()
                -
                start
            )
            *
            1000.0
        )

        # ============================================================
        # 两阶段抓取搜索：
        #
        # 1) 快速路径：
        #    每个物体只检查 1~2 个自然基础抓取姿态。
        #
        # 2) 只有整帧一个安全候选都找不到时，
        #    才回退到原来的完整多姿态搜索。
        #
        # 这样物体很多但存在明显可抓目标时，不再为所有物体
        # 跑几十/上百个候选姿态。
        # ============================================================
        if self.grasp_search_enabled:
            candidates = self._build_candidates(
                frame,
                objects,
                fast_only=True,
            )

            if not candidates:
                if self.gripper_debug_log_rejections:
                    print(
                        "[GRASP FAST] no safe candidate in quick pass; "
                        "fallback to full search"
                    )

                candidates = self._build_candidates(
                    frame,
                    objects,
                    fast_only=False,
                )
        else:
            candidates = self._build_candidates(
                frame,
                objects,
                fast_only=False,
            )

        stable = (
            self.consensus.update(
                [
                    self
                    ._candidate_to_consensus_dict(
                        candidate
                    )
                    for candidate
                    in candidates
                ]
            )
        )

        # 不改TopPickConsensus原有逻辑：候选dict中的yaw_branch会原样保存在history。
        # READY时要求有效投票使用同一个物理分支，避免两帧在两侧之间跳变。
        stable_yaw_branch: int | None = None
        branch_consistent = True
        if bool(stable.ready):
            valid_history = [
                item
                for item in self.consensus.history
                if item is not None
            ]
            branch_values = [
                int(item.get("yaw_branch", 0))
                for item in valid_history
            ]
            if branch_values and all(value in (0, 1) for value in branch_values):
                if len(set(branch_values)) == 1:
                    stable_yaw_branch = int(branch_values[0])
                else:
                    branch_consistent = False
            else:
                branch_consistent = False

        output_ready = bool(stable.ready) and branch_consistent
        output_reason = (
            str(stable.reason)
            if not bool(stable.ready)
            else ("ready" if branch_consistent else "yaw_branch_mixed")
        )

        return TopPickPipelineResult(
            ready=
                output_ready,
            votes=
                int(stable.votes),
            required_hits=
                self.required_hits,
            window_size=
                self.window_size,
            object_count=
                len(objects),
            candidate_count=
                len(candidates),
            grasp_pixel=
                stable.grasp_pixel,
            x_mm=
                stable.x_mm,
            y_mm=
                stable.y_mm,
            yaw_deg=
                stable.yaw_deg if output_ready else None,
            yaw_branch=
                stable_yaw_branch if output_ready else None,
            grasp_width_mm=
                stable.grasp_width_mm,
            preopen_mm=
                stable.preopen_mm,
            class_name=
                stable.class_name,
            confidence=
                float(
                    stable.confidence
                ),
            clearance_px=
                float(
                    stable.clearance_px
                ),
            reason=
                output_reason,
            inference_ms=
                float(inference_ms),
            candidates=
                candidates,
        )
