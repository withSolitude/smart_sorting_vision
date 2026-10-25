from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DescentDriftCompensationResult:
    x_mm: float
    y_mm: float
    command_delta_x_mm: float
    command_delta_y_mm: float
    predicted_drift_x_mm: float
    predicted_drift_y_mm: float
    precomp_visual_x_mm: float
    precomp_visual_y_mm: float


class DescentDriftCompensator:
    """Pre-compensate repeatable lateral TCP drift caused by Z descent.

    Angle convention:
      closing_axis_deg is the physical gripper *closing axis* in robot XY.
      The observed descent drift is assumed to be at
          closing_axis_deg + relative_angle_deg
      where relative_angle_deg defaults to +90 deg (perpendicular to the line
      connecting the two jaws / closing axis).

    signed_drift_mm is the observed TCP drift magnitude in the calibrated robot
    plane. Positive means along +relative_angle_deg; negative flips it 180 deg.

    The desired opposite physical shift is converted into actual XY stage command
    millimetres through visual_to_command_matrix (the same J^-1 calibrated for
    Yaw-TCP compensation). This keeps physical-plane mm and stage-command mm from
    being silently treated as identical.
    """

    def __init__(
        self,
        *,
        signed_drift_mm: float,
        relative_angle_deg: float = 90.0,
        visual_to_command_matrix=None,
    ) -> None:
        self.signed_drift_mm = float(signed_drift_mm)
        self.relative_angle_deg = float(relative_angle_deg)

        matrix = (
            [[1.0, 0.0], [0.0, 1.0]]
            if visual_to_command_matrix is None
            else visual_to_command_matrix
        )
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.shape != (2, 2) or not np.all(np.isfinite(arr)):
            raise ValueError("visual_to_command_matrix must be finite 2x2")
        if abs(float(np.linalg.det(arr))) < 1e-9:
            raise ValueError("visual_to_command_matrix is singular")
        self.visual_to_command_matrix = arr

    def apply(
        self,
        x_mm: float,
        y_mm: float,
        closing_axis_deg: float,
    ) -> DescentDriftCompensationResult:
        x = float(x_mm)
        y = float(y_mm)
        yaw = float(closing_axis_deg)
        if not all(math.isfinite(v) for v in (x, y, yaw, self.signed_drift_mm)):
            raise ValueError("non-finite descent-drift input")

        drift_angle = math.radians(yaw + self.relative_angle_deg)
        drift_x = self.signed_drift_mm * math.cos(drift_angle)
        drift_y = self.signed_drift_mm * math.sin(drift_angle)

        # Before Z descends, move by the *opposite* physical-plane vector.
        precomp_visual = np.array([-drift_x, -drift_y], dtype=np.float64)
        command_delta = self.visual_to_command_matrix @ precomp_visual

        dx = float(command_delta[0])
        dy = float(command_delta[1])
        return DescentDriftCompensationResult(
            x_mm=x + dx,
            y_mm=y + dy,
            command_delta_x_mm=dx,
            command_delta_y_mm=dy,
            predicted_drift_x_mm=float(drift_x),
            predicted_drift_y_mm=float(drift_y),
            precomp_visual_x_mm=float(precomp_visual[0]),
            precomp_visual_y_mm=float(precomp_visual[1]),
        )
