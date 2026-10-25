#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.grasp.grasp_pose_search import (
    build_search_offsets,
    evaluate_target_engagement,
    normalize_axial_deg,
)


def main() -> None:
    # 20 x 10 mm 矩形，中心在原点。
    rect = np.asarray(
        [[-10.0, -5.0], [10.0, -5.0], [10.0, 5.0], [-10.0, 5.0]],
        dtype=np.float64,
    )

    g0 = evaluate_target_engagement(
        rect,
        (0.0, 0.0),
        0.0,
        23.0,
        finger_length_mm=5.0,
        min_side_gap_mm=0.25,
        min_longitudinal_overlap_mm=1.5,
    )
    assert abs(g0.width_mm - 20.0) < 1e-6
    assert g0.valid
    assert abs(g0.min_side_gap_mm - 1.5) < 1e-6

    # 沿闭合轴偏 2 mm 后，一侧已经没有足够开口余量，必须拒绝。
    g_bad_close = evaluate_target_engagement(
        rect,
        (2.0, 0.0),
        0.0,
        23.0,
        finger_length_mm=5.0,
        min_side_gap_mm=0.25,
        min_longitudinal_overlap_mm=1.5,
    )
    assert not g_bad_close.fits_between_fingers

    # 旋转90°后，闭合方向宽度应变为10mm。
    g90 = evaluate_target_engagement(
        rect,
        (0.0, 0.0),
        90.0,
        13.0,
        finger_length_mm=5.0,
        min_side_gap_mm=0.25,
        min_longitudinal_overlap_mm=1.5,
    )
    assert abs(g90.width_mm - 10.0) < 1e-6
    assert g90.valid

    # 180°轴周期归一化。
    assert abs(normalize_axial_deg(100.0) - (-80.0)) < 1e-9
    assert abs(normalize_axial_deg(-100.0) - 80.0) < 1e-9

    offsets = build_search_offsets(
        [0, -5, 5],
        [0, -2, 2],
        [0],
        max_trials=20,
    )
    assert len(offsets) == 9
    assert offsets[0] == (0.0, 0.0, 0.0)

    print("grasp pose search tests: PASS")


if __name__ == "__main__":
    main()
