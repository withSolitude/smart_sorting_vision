#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.calibration.pixel_to_robot import PixelToRobotTransform
from vision.grasp.gripper_footprint import GripperFootprintModel


def main() -> None:
    # 1 px = 1 mm 的人工 Homography，只验证几何逻辑。
    t = PixelToRobotTransform(np.eye(3), 300, 300)
    model = GripperFootprintModel(
        transform=t,
        finger_length_mm=20.0,
        finger_thickness_mm=9.0,
        opening_uncertainty_mm=1.5,
        object_margin_mm=1.0,
        wall_margin_mm=2.0,
        object_overlap_tolerance_px=4,
        forbidden_zones_mm=[{
            "name": "wall",
            "polygon": [[190, 0], [210, 8], [205, 300], [185, 292]],
        }],
        allowed_polygon_mm=[[20, 20], [280, 20], [270, 280], [30, 280]],
        enforce_image_border=True,
    )

    safe_obstacle = np.asarray([[150, 90], [165, 90], [165, 110], [150, 110]], dtype=np.float32)
    r1 = model.check(
        grasp_pixel=(100, 100), yaw_deg=0.0, opening_mm=30.0,
        other_polygons=[safe_obstacle], image_shape=(300, 300, 3),
    )
    assert r1.safe, r1

    hit_obstacle = np.asarray([[117, 93], [128, 93], [128, 107], [117, 107]], dtype=np.float32)
    r2 = model.check(
        grasp_pixel=(100, 100), yaw_deg=0.0, opening_mm=30.0,
        other_polygons=[hit_obstacle], image_shape=(300, 300, 3),
    )
    assert not r2.safe and r2.reason == "other_object_collision", r2

    r3 = model.check(
        grasp_pixel=(165, 100), yaw_deg=0.0, opening_mm=30.0,
        other_polygons=[], image_shape=(300, 300, 3),
    )
    assert not r3.safe and r3.reason == "fixed_wall_collision", r3

    # 托盘四点是允许区，内部安全、靠边越界时拒绝。
    inside = model.check(
        grasp_pixel=(100, 150), yaw_deg=20.0, opening_mm=20.0,
        other_polygons=[], image_shape=(300, 300, 3),
    )
    assert inside.safe, inside

    near_edge = model.check(
        grasp_pixel=(30, 50), yaw_deg=90.0, opening_mm=20.0,
        other_polygons=[], image_shape=(300, 300, 3),
    )
    assert not near_edge.safe and near_edge.reason == "allowed_polygon_collision", near_edge

    # 兼容旧 rect 配置。
    legacy = GripperFootprintModel(
        transform=t,
        finger_length_mm=20.0,
        finger_thickness_mm=9.0,
        forbidden_zones_mm=[{"name": "legacy_wall", "rect": [190, 0, 210, 300]}],
        enforce_image_border=True,
    )
    r4 = legacy.check(
        grasp_pixel=(165, 100), yaw_deg=0.0, opening_mm=30.0,
        other_polygons=[], image_shape=(300, 300, 3),
    )
    assert not r4.safe and r4.reason == "fixed_wall_collision", r4

    # 斜四边形保持自身形状，不扩成轴对齐矩形。
    skew = GripperFootprintModel(
        transform=t,
        finger_length_mm=10.0,
        finger_thickness_mm=4.0,
        opening_uncertainty_mm=0.0,
        object_margin_mm=0.0,
        wall_margin_mm=0.0,
        forbidden_zones_mm=[{
            "name": "skew_wall",
            "polygon": [[180, 20], [195, 20], [230, 280], [215, 280]],
        }],
        enforce_image_border=False,
    )
    r5 = skew.check(
        grasp_pixel=(190, 250), yaw_deg=90.0, opening_mm=10.0,
        other_polygons=[], image_shape=(300, 300, 3),
    )
    assert r5.safe, r5

    print("gripper footprint tests: PASS")


if __name__ == "__main__":
    main()
