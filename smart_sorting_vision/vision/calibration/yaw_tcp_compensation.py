from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class YawTcpCompensationResult:
    x_mm: float
    y_mm: float
    delta_x_mm: float
    delta_y_mm: float
    visual_delta_x_mm: float
    visual_delta_y_mm: float


class YawTcpCompensator:
    """Yaw 旋转引起的 TCP 平面偏心补偿。

    ``offset_axis0_*`` 是由顶部视觉/H 测得的 TCP 旋转偏心，因此
    ``R(reference)d - R(theta)d`` 首先得到的是“视觉/H 平面中的 TCP 位移”。

    由于同一机械 XY 命令在该视觉平面中的实际位移并非严格 1:1，
    需要再通过 ``visual_to_command_matrix`` 将视觉位移换算成 XY 机构命令：

        delta_visual(theta) = R(reference) d - R(theta) d
        delta_command       = J^{-1} delta_visual
        command_xy          = base_xy + delta_command

    其中 J 来自固定 Yaw 下的 CENTER / X+ / X- / Y+ / Y- 五点标定。

    注意：yaw 必须使用唯一物理分支。虽然闭合轴在纯方向意义上 180° 等价，
    但存在 TCP 偏心时 theta 与 theta+180° 的真实 TCP 位置并不等价。
    """

    def __init__(
        self,
        *,
        reference_axis_deg: float,
        offset_axis0_x_mm: float,
        offset_axis0_y_mm: float,
        min_axis_deg: float = -87.0,
        max_axis_deg: float = 89.0,
        visual_to_command_matrix: tuple[tuple[float, float], tuple[float, float]] | list[list[float]] | None = None,
        closing_bias_mm: float = 0.0,
        longitudinal_bias_mm: float = 0.0,
    ) -> None:
        values = (
            reference_axis_deg,
            offset_axis0_x_mm,
            offset_axis0_y_mm,
            min_axis_deg,
            max_axis_deg,
            closing_bias_mm,
            longitudinal_bias_mm,
        )
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Yaw TCP compensation 参数含 NaN/Inf")
        if float(min_axis_deg) >= float(max_axis_deg):
            raise ValueError("min_axis_deg 必须小于 max_axis_deg")

        self.reference_axis_deg = float(reference_axis_deg)
        self.offset_axis0_x_mm = float(offset_axis0_x_mm)
        self.offset_axis0_y_mm = float(offset_axis0_y_mm)
        self.min_axis_deg = float(min_axis_deg)
        self.max_axis_deg = float(max_axis_deg)
        # 固定抓取中心修正，均定义在“夹爪局部坐标系”：
        #
        # closing_bias_mm:
        #   正值沿当前 closing axis（夹爪闭合轴）正方向移动。
        #
        # longitudinal_bias_mm:
        #   正值沿 closing axis + 90°（夹爪前后方向）移动。
        #
        # 这两个量都是“下降前”的固定抓取中心/TCP微调，
        # 不是 Z 下降过程中产生的机械漂移补偿。
        self.closing_bias_mm = float(closing_bias_mm)
        self.longitudinal_bias_mm = float(longitudinal_bias_mm)

        if visual_to_command_matrix is None:
            visual_to_command_matrix = ((1.0, 0.0), (0.0, 1.0))
        try:
            a = float(visual_to_command_matrix[0][0])
            b = float(visual_to_command_matrix[0][1])
            c = float(visual_to_command_matrix[1][0])
            d = float(visual_to_command_matrix[1][1])
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError("visual_to_command_matrix 必须是 2x2 数值矩阵") from exc
        if not all(math.isfinite(v) for v in (a, b, c, d)):
            raise ValueError("visual_to_command_matrix 含 NaN/Inf")
        det = a * d - b * c
        if abs(det) < 1e-9:
            raise ValueError("visual_to_command_matrix 不可逆/退化")
        self.visual_to_command_matrix = ((a, b), (c, d))

        self._reference_offset = self.offset_at(self.reference_axis_deg)

    @staticmethod
    def _rotate(x_mm: float, y_mm: float, theta_deg: float) -> tuple[float, float]:
        theta = math.radians(float(theta_deg))
        c = math.cos(theta)
        s = math.sin(theta)
        return (
            c * float(x_mm) - s * float(y_mm),
            s * float(x_mm) + c * float(y_mm),
        )

    def axis_supported(self, axis_deg: float) -> bool:
        axis = float(axis_deg)
        return math.isfinite(axis) and self.min_axis_deg <= axis <= self.max_axis_deg

    def offset_at(self, axis_deg: float) -> tuple[float, float]:
        return self._rotate(
            self.offset_axis0_x_mm,
            self.offset_axis0_y_mm,
            float(axis_deg),
        )

    def visual_correction(self, axis_deg: float) -> tuple[float, float]:
        """需要在顶部视觉/H平面抵消的 TCP 位移。"""
        axis = float(axis_deg)
        if not self.axis_supported(axis):
            raise ValueError(
                "Yaw 超出 TCP 补偿已验证/可执行范围: "
                f"{axis:.3f} deg not in "
                f"[{self.min_axis_deg:.3f}, {self.max_axis_deg:.3f}]"
            )
        ox, oy = self.offset_at(axis)
        ref_x, ref_y = self._reference_offset
        vx = ref_x - ox
        vy = ref_y - oy

        # ------------------------------------------------------------
        # 固定抓取中心微调（夹爪局部坐标系）
        #
        # closing axis:
        #     [cos(theta), sin(theta)]
        #
        # longitudinal axis = closing axis + 90 deg:
        #     [-sin(theta), cos(theta)]
        #
        # 两个偏置先在视觉/H物理平面(mm)中叠加，再统一经过 J^-1
        # 转换成真正的 XY 机构命令增量。
        # ------------------------------------------------------------
        theta = math.radians(axis)
        c = math.cos(theta)
        s = math.sin(theta)

        if abs(self.closing_bias_mm) > 1e-12:
            vx += self.closing_bias_mm * c
            vy += self.closing_bias_mm * s

        if abs(self.longitudinal_bias_mm) > 1e-12:
            vx += -self.longitudinal_bias_mm * s
            vy += self.longitudinal_bias_mm * c

        return vx, vy

    def correction(self, axis_deg: float) -> tuple[float, float]:
        """返回真正应发送给 XY 机构的命令增量。"""
        vx, vy = self.visual_correction(axis_deg)
        m = self.visual_to_command_matrix
        return (
            m[0][0] * vx + m[0][1] * vy,
            m[1][0] * vx + m[1][1] * vy,
        )

    def apply(self, x_mm: float, y_mm: float, axis_deg: float) -> YawTcpCompensationResult:
        if not (math.isfinite(float(x_mm)) and math.isfinite(float(y_mm))):
            raise ValueError("原始 XY 含 NaN/Inf")
        vdx, vdy = self.visual_correction(axis_deg)
        dx, dy = self.correction(axis_deg)
        x = float(x_mm) + dx
        y = float(y_mm) + dy
        if not (math.isfinite(x) and math.isfinite(y)):
            raise RuntimeError("Yaw TCP 补偿结果为 NaN/Inf")
        return YawTcpCompensationResult(
            x_mm=x,
            y_mm=y,
            delta_x_mm=dx,
            delta_y_mm=dy,
            visual_delta_x_mm=vdx,
            visual_delta_y_mm=vdy,
        )
