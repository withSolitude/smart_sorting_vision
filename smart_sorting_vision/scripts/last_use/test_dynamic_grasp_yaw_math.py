from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from communication.sorting_protocol import Mode, SortingFrame, SortingProtocol, make_pick_target
from vision.calibration.pixel_to_robot import PixelToRobotTransform
from vision.grasp.orientation_planner import (
    axial_difference_deg,
    axial_mean_deg,
    plan_grasp_axis,
)


def rect(cx, cy, long_len, short_len, angle_deg):
    t = np.deg2rad(angle_deg)
    e1 = np.array([np.cos(t), np.sin(t)]) * long_len / 2.0
    e2 = np.array([-np.sin(t), np.cos(t)]) * short_len / 2.0
    c = np.array([cx, cy], dtype=float)
    return np.vstack([c-e1-e2, c+e1-e2, c+e1+e2, c-e1+e2]).astype(np.float32)


def main():
    # Identity pixel->robot direction, enough to test axial geometry.
    tf = PixelToRobotTransform(np.eye(3), 1000, 1000)

    # Object long axis = +30 deg. Contacting its two LONG sides means closing axis = +120 deg,
    # which is the same undirected axis as -60 deg.
    o = plan_grasp_axis(
        polygon=rect(400, 300, 160, 50, 30),
        grasp_pixel=(400, 300),
        transform=tf,
        grip_long_sides=True,
        min_aspect_ratio=1.2,
    )
    assert o.reliable
    assert abs(axial_difference_deg(o.long_axis_robot_deg, 30.0)) < 0.5
    assert abs(axial_difference_deg(o.grasp_axis_robot_deg, -60.0)) < 0.5

    # Near-square: orientation is intentionally ignored.
    sq = plan_grasp_axis(
        polygon=rect(300, 200, 100, 95, 43),
        grasp_pixel=(300, 200),
        transform=tf,
        grip_long_sides=True,
        min_aspect_ratio=1.2,
        default_grasp_axis_robot_deg=12.0,
    )
    assert not sq.reliable
    assert abs(axial_difference_deg(sq.grasp_axis_robot_deg, 12.0)) < 1e-6

    # Axial mean must treat +89 and -89 as almost the same axis, not average them to 0.
    m = axial_mean_deg([89.0, -89.0])
    assert abs(abs(m) - 90.0) < 0.2

    # Protocol byte[7] signed yaw roundtrip; 1 degree/LSB.
    f = make_pick_target(-40.2, 23.1, yaw_deg=-37.4)
    raw = SortingProtocol.pack(f)
    assert raw[2] == int(Mode.PICK_TARGET)
    d = SortingProtocol.unpack(raw)
    assert abs(d.x_mm + 40.2) < 1e-6
    assert abs(d.y_mm - 23.1) < 1e-6
    assert d.yaw_deg == -37.0

    ack = SortingFrame(mode=Mode.MCU_PICK_TARGET_ACK, x_mm=-40.2, y_mm=23.1, yaw_deg=64.0)
    assert SortingProtocol.unpack(SortingProtocol.pack(ack)).yaw_deg == 64.0

    print("DYNAMIC GRASP YAW MATH/PROTOCOL TESTS PASSED")


if __name__ == "__main__":
    main()
