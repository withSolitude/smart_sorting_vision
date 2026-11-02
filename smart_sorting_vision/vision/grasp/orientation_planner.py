from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


def normalize_axial_deg(angle_deg: float) -> float:
    """Normalize a 180-degree-periodic axis to [-90, 90)."""
    angle = (float(angle_deg) + 90.0) % 180.0 - 90.0
    if abs(angle) < 1e-12:
        return 0.0
    return angle


def axial_difference_deg(a_deg: float, b_deg: float) -> float:
    """Signed shortest difference a-b for an undirected (180° periodic) axis."""
    return normalize_axial_deg(float(a_deg) - float(b_deg))


def axial_mean_deg(angles_deg: list[float] | np.ndarray) -> float:
    """Mean of 180-degree-periodic angles using the doubled-angle method."""
    values = np.asarray(angles_deg, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("angles_deg cannot be empty")
    doubled = np.deg2rad(values * 2.0)
    s = float(np.mean(np.sin(doubled)))
    c = float(np.mean(np.cos(doubled)))
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return normalize_axial_deg(float(values[0]))
    return normalize_axial_deg(0.5 * math.degrees(math.atan2(s, c)))


@dataclass(frozen=True)
class ObjectOrientation:
    # Legacy/debug fields kept for compatibility with the existing pipeline.
    long_axis_image_deg: float
    long_axis_robot_deg: float
    grasp_axis_robot_deg: float
    aspect_ratio: float
    reliable: bool

    # Production general-shape planner metadata.
    # antipodal_edge_edge: two parallel supporting edges -> strongest grasp
    # antipodal_edge_vertex: one supporting edge + opposite vertex (typical triangle)
    # antipodal_vertex_vertex: usable but lower-confidence support pair
    # symmetric_fallback: true circle / rotationally ambiguous object
    mode: str = "symmetric_fallback"
    rectangularity: float = 0.0
    circularity: float = 0.0
    hull_vertex_count: int = 0

    closing_axis_image_deg: float = 0.0
    grasp_pixel_override: tuple[int, int] | None = None
    grasp_width_px: float = 0.0
    grasp_width_mm: float | None = None
    contact_type: str = "none"
    contact_quality: float = 0.0
    support_overlap_px: float = 0.0
    support_min_point_px: tuple[float, float] | None = None
    support_max_point_px: tuple[float, float] | None = None
    support_min_span_px: float = 0.0
    support_max_span_px: float = 0.0


@dataclass(frozen=True)
class _SupportFeature:
    kind: str  # edge / vertex
    point: np.ndarray
    interval_t: tuple[float, float]
    span_px: float


@dataclass(frozen=True)
class _AxisCandidate:
    n: np.ndarray
    t: np.ndarray
    axis_image_deg: float
    axis_robot_deg: float
    width_px: float
    min_feature: _SupportFeature
    max_feature: _SupportFeature
    overlap_px: float
    contact_quality: float
    overlap_quality: float
    ideal_point: np.ndarray
    score_base: float


def _validate_polygon(polygon: np.ndarray) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("polygon must be Nx2 with at least 3 points")
    if not np.all(np.isfinite(pts)):
        raise ValueError("polygon contains NaN/Inf")
    return pts


def _polygon_centroid(pts: np.ndarray) -> np.ndarray:
    contour = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    m = cv2.moments(contour)
    if abs(m["m00"]) > 1e-9:
        return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float64)
    return np.mean(np.asarray(pts, dtype=np.float64), axis=0)


def _convex_hull_simplified(pts: np.ndarray) -> np.ndarray:
    contour = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(contour)
    perimeter = float(cv2.arcLength(hull, True))
    if perimeter <= 1e-6:
        raise ValueError("degenerate polygon hull")

    # YOLO masks often contain many collinear points along one real edge.
    # Simplify only the convex hull, conservatively, so triangles/pentagons/etc.
    # keep their real faces while pixel stair-steps do not create fake axes.
    epsilon = max(0.75, 0.008 * perimeter)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    simp = approx.reshape(-1, 2).astype(np.float64)
    if len(simp) < 3:
        simp = hull.reshape(-1, 2).astype(np.float64)
    return simp


def polygon_long_axis_image_deg(polygon: np.ndarray) -> tuple[float, float]:
    """Return (minimum-area-rectangle long-axis image angle, long/short ratio)."""
    pts = _validate_polygon(polygon).astype(np.float32)
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    box = cv2.boxPoints(rect).astype(np.float64)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    long_i = int(np.argmax(lengths))
    long_len = float(lengths[long_i])
    short_len = float(np.min(lengths))
    if long_len <= 1e-6 or short_len <= 1e-6:
        raise ValueError("degenerate minimum-area rectangle")
    vx, vy = edges[long_i]
    angle = math.degrees(math.atan2(float(vy), float(vx)))
    return normalize_axial_deg(angle), long_len / short_len


def polygon_rectangularity(polygon: np.ndarray) -> tuple[float, int]:
    """Return (area / minAreaRect area, approximate corner count)."""
    pts = _validate_polygon(polygon).astype(np.float32)
    contour = pts.reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(contour)))
    rect = cv2.minAreaRect(contour)
    w, h = map(float, rect[1])
    rect_area = w * h
    if rect_area <= 1e-6:
        return 0.0, 0
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 1e-6:
        return area / rect_area, 0
    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    return float(area / rect_area), int(len(approx))


def _shape_metrics(hull: np.ndarray) -> tuple[float, int]:
    contour = hull.astype(np.float32).reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = 0.0 if perimeter <= 1e-9 else float(4.0 * math.pi * area / (perimeter * perimeter))
    return circularity, int(len(hull))


def _support_feature(
    hull: np.ndarray,
    *,
    n: np.ndarray,
    t: np.ndarray,
    extreme: float,
    side: str,
    tolerance_px: float,
    min_contact_span_px: float,
) -> _SupportFeature:
    proj_n = hull @ n
    best_edge = None
    best_span = -1.0

    for i in range(len(hull)):
        j = (i + 1) % len(hull)
        if (
            abs(float(proj_n[i]) - extreme) <= tolerance_px
            and abs(float(proj_n[j]) - extreme) <= tolerance_px
        ):
            a = hull[i]
            b = hull[j]
            ta = float(a @ t)
            tb = float(b @ t)
            span = abs(tb - ta)
            if span > best_span:
                best_span = span
                best_edge = (a, b, min(ta, tb), max(ta, tb))

    if best_edge is not None and best_span >= float(min_contact_span_px):
        a, b, lo, hi = best_edge
        return _SupportFeature(
            kind="edge",
            point=(a + b) * 0.5,
            interval_t=(float(lo), float(hi)),
            span_px=float(best_span),
        )

    idx = int(np.argmin(proj_n) if side == "min" else np.argmax(proj_n))
    point = hull[idx].copy()
    tv = float(point @ t)
    return _SupportFeature(
        kind="vertex",
        point=point,
        interval_t=(tv, tv),
        span_px=0.0,
    )


def _contact_metrics(a: _SupportFeature, b: _SupportFeature, width_px: float) -> tuple[float, float, float]:
    lo = max(a.interval_t[0], b.interval_t[0])
    hi = min(a.interval_t[1], b.interval_t[1])
    overlap = max(0.0, hi - lo)

    edge_count = int(a.kind == "edge") + int(b.kind == "edge")
    if edge_count == 2:
        contact_quality = 1.0
        denom = max(1.0, 0.5 * min(a.span_px, b.span_px))
        overlap_quality = min(1.0, overlap / denom)
    elif edge_count == 1:
        contact_quality = 0.78
        edge = a if a.kind == "edge" else b
        vertex = b if a.kind == "edge" else a
        tv = vertex.interval_t[0]
        if edge.interval_t[0] <= tv <= edge.interval_t[1]:
            overlap_quality = 0.85
        else:
            dist = min(abs(tv - edge.interval_t[0]), abs(tv - edge.interval_t[1]))
            overlap_quality = max(0.0, 0.60 * (1.0 - dist / max(width_px, 1.0)))
    else:
        contact_quality = 0.45
        overlap_quality = 0.20

    return float(contact_quality), float(overlap_quality), float(overlap)


def _closest_safe_point(
    ideal: np.ndarray,
    preferred: np.ndarray,
    *,
    safe_mask: np.ndarray | None,
    n: np.ndarray,
    width_px: float,
) -> tuple[int, int]:
    if safe_mask is None:
        return int(round(float(ideal[0]))), int(round(float(ideal[1])))

    mask = np.asarray(safe_mask)
    h, w = mask.shape[:2]
    ix = int(round(float(ideal[0])))
    iy = int(round(float(ideal[1])))
    if 0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0:
        return ix, iy

    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return int(round(float(preferred[0]))), int(round(float(preferred[1])))

    q = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    delta = q - ideal.reshape(1, 2)
    dn = np.abs(delta @ n)
    # Tangential movement is much less harmful than moving away from the
    # mid-plane between both support lines, because normal error makes one jaw
    # contact much earlier than the other.
    dt2 = np.maximum(0.0, np.sum(delta * delta, axis=1) - dn * dn)
    normal_limit = max(2.0, 0.18 * float(width_px))
    ok = dn <= normal_limit
    if np.any(ok):
        q2 = q[ok]
        dn2 = dn[ok]
        dt22 = dt2[ok]
        pref_d2 = np.sum((q2 - preferred.reshape(1, 2)) ** 2, axis=1)
        cost = 9.0 * dn2 * dn2 + dt22 + 0.03 * pref_d2
        best = q2[int(np.argmin(cost))]
        return int(round(float(best[0]))), int(round(float(best[1])))

    px = int(round(float(preferred[0])))
    py = int(round(float(preferred[1])))
    if 0 <= px < w and 0 <= py < h and mask[py, px] > 0:
        return px, py

    d2 = np.sum((q - ideal.reshape(1, 2)) ** 2, axis=1)
    best = q[int(np.argmin(d2))]
    return int(round(float(best[0]))), int(round(float(best[1])))


def _estimate_width_mm(transform, feature_a: _SupportFeature, feature_b: _SupportFeature, axis_robot_deg: float) -> float | None:
    try:
        ax, ay = transform.pixel_to_robot(
            float(feature_a.point[0]), float(feature_a.point[1]), apply_tcp_offset=True
        )
        bx, by = transform.pixel_to_robot(
            float(feature_b.point[0]), float(feature_b.point[1]), apply_tcp_offset=True
        )
        theta = math.radians(float(axis_robot_deg))
        nr = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        d = np.array([float(bx) - float(ax), float(by) - float(ay)], dtype=np.float64)
        return float(abs(d @ nr))
    except Exception:
        return None


def _candidate_axes(
    hull: np.ndarray,
    *,
    centroid: np.ndarray,
    preferred_grasp: np.ndarray,
    transform,
    default_axis_robot_deg: float,
    yaw_alignment_offset_deg: float,
    width_slack_ratio: float,
    support_tolerance_px: float,
    min_contact_span_px: float,
    keep_all_widths: bool = False,
) -> list[_AxisCandidate]:
    raw: list[_AxisCandidate] = []
    seen_axes: list[float] = []

    u_ref, v_ref = map(float, preferred_grasp)
    preferred_robot = normalize_axial_deg(float(default_axis_robot_deg) + float(yaw_alignment_offset_deg))

    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        edge_len = float(np.linalg.norm(edge))
        if edge_len <= 1e-6:
            continue
        t = edge / edge_len
        n = np.array([-t[1], t[0]], dtype=np.float64)
        axis_image = normalize_axial_deg(math.degrees(math.atan2(float(n[1]), float(n[0]))))

        # Remove duplicate normals produced by multiple collinear hull segments.
        if any(abs(axial_difference_deg(axis_image, old)) < 0.75 for old in seen_axes):
            continue
        seen_axes.append(axis_image)

        proj_n = hull @ n
        min_n = float(np.min(proj_n))
        max_n = float(np.max(proj_n))
        width = max_n - min_n
        if width <= 1e-6:
            continue

        tol = max(float(support_tolerance_px), 0.015 * width)
        fmin = _support_feature(
            hull,
            n=n,
            t=t,
            extreme=min_n,
            side="min",
            tolerance_px=tol,
            min_contact_span_px=min_contact_span_px,
        )
        fmax = _support_feature(
            hull,
            n=n,
            t=t,
            extreme=max_n,
            side="max",
            tolerance_px=tol,
            min_contact_span_px=min_contact_span_px,
        )
        contact_quality, overlap_quality, overlap = _contact_metrics(fmin, fmax, width)

        centroid_t = float(centroid @ t)
        if fmin.kind == "edge" and fmax.kind == "edge" and overlap > 0.0:
            lo = max(fmin.interval_t[0], fmax.interval_t[0])
            hi = min(fmin.interval_t[1], fmax.interval_t[1])
            target_t = float(np.clip(centroid_t, lo, hi))
        elif fmin.kind == "edge":
            target_t = float(np.clip(centroid_t, fmin.interval_t[0], fmin.interval_t[1]))
        elif fmax.kind == "edge":
            target_t = float(np.clip(centroid_t, fmax.interval_t[0], fmax.interval_t[1]))
        else:
            target_t = centroid_t

        mid_n = 0.5 * (min_n + max_n)
        ideal = n * mid_n + t * target_t
        axis_robot = normalize_axial_deg(
            float(transform.robot_angle_from_image_angle(u_ref, v_ref, axis_image))
            + float(yaw_alignment_offset_deg)
        )
        # The lower MCU now executes one unique physical Yaw pose and the gripper TCP
        # has a large eccentricity.  Therefore theta and theta+180 deg are NOT
        # physically interchangeable for motion planning, even though they describe
        # the same undirected closing axis.  Use ordinary physical-angle distance
        # here so +80 deg is not treated as "near" -85 deg.
        rotation_cost = abs(float(axis_robot) - float(preferred_robot)) / 90.0
        center_offset = float(np.linalg.norm(ideal - centroid)) / max(width, 1.0)

        # Width penalty is added after the absolute minimum width is known.
        score_base = (
            3.5 * contact_quality
            + 1.2 * overlap_quality
            - 0.30 * min(center_offset, 2.0)
            - 0.18 * rotation_cost
        )
        raw.append(
            _AxisCandidate(
                n=n,
                t=t,
                axis_image_deg=float(axis_image),
                axis_robot_deg=float(axis_robot),
                width_px=float(width),
                min_feature=fmin,
                max_feature=fmax,
                overlap_px=float(overlap),
                contact_quality=float(contact_quality),
                overlap_quality=float(overlap_quality),
                ideal_point=ideal,
                score_base=float(score_base),
            )
        )

    if not raw:
        return []

    # Legacy single-base planning keeps the historical near-minimum-opening
    # preference. The multi-base planner can request every strong antipodal
    # family so opposite parallel short/long sides are treated equally.
    if keep_all_widths:
        return raw

    min_width = min(c.width_px for c in raw)
    slack = max(0.01, float(width_slack_ratio))
    allowed = min_width * (1.0 + slack) + 0.75
    kept: list[_AxisCandidate] = []
    for c in raw:
        if c.width_px > allowed:
            continue
        width_penalty = max(0.0, (c.width_px / max(min_width, 1e-9) - 1.0) / slack)
        kept.append(
            _AxisCandidate(
                **{**c.__dict__, "score_base": float(c.score_base - 1.15 * width_penalty)}
            )
        )
    return kept or [min(raw, key=lambda c: c.width_px)]



def plan_parallel_edge_grasp_options(
    *,
    polygon: np.ndarray,
    grasp_pixel: tuple[float, float],
    transform,
    default_grasp_axis_robot_deg: float = 0.0,
    yaw_alignment_offset_deg: float = 0.0,
    safe_mask: np.ndarray | None = None,
    width_slack_ratio: float = 0.15,
    support_tolerance_px: float = 3.0,
    min_contact_span_px: float = 6.0,
    circle_circularity_threshold: float = 0.985,
    circle_min_vertices: int = 8,
) -> list[ObjectOrientation]:
    """Return every strong edge-edge antipodal grasp family.

    This intentionally does not discard a wider valid jaw axis merely because
    another axis needs a smaller opening. For a rectangle (and other shapes
    with multiple pairs of opposite parallel edges), gripping either pair is a
    legitimate parallel-jaw grasp. Higher-level collision search chooses the
    safest family in the current scene.
    """
    pts = _validate_polygon(polygon)
    image_long_deg, aspect_ratio = polygon_long_axis_image_deg(pts)
    rectangularity, _ = polygon_rectangularity(pts)
    hull = _convex_hull_simplified(pts)
    circularity, hull_vertices = _shape_metrics(hull)
    centroid = _polygon_centroid(pts)
    preferred = np.array([float(grasp_pixel[0]), float(grasp_pixel[1])], dtype=np.float64)

    robot_long_deg = normalize_axial_deg(
        transform.robot_angle_from_image_angle(
            float(preferred[0]),
            float(preferred[1]),
            image_long_deg,
        )
    )

    if (
        circularity >= float(circle_circularity_threshold)
        and hull_vertices >= int(circle_min_vertices)
    ):
        return []

    all_axes = _candidate_axes(
        hull,
        centroid=centroid,
        preferred_grasp=preferred,
        transform=transform,
        default_axis_robot_deg=default_grasp_axis_robot_deg,
        yaw_alignment_offset_deg=yaw_alignment_offset_deg,
        width_slack_ratio=width_slack_ratio,
        support_tolerance_px=support_tolerance_px,
        min_contact_span_px=min_contact_span_px,
        keep_all_widths=True,
    )

    strong = [
        c for c in all_axes
        if (
            c.min_feature.kind == "edge"
            and c.max_feature.kind == "edge"
            and c.overlap_px > 0.0
        )
    ]
    if not strong:
        return []

    preferred_robot = normalize_axial_deg(
        float(default_grasp_axis_robot_deg) + float(yaw_alignment_offset_deg)
    )
    strong.sort(
        key=lambda c: (
            abs(float(c.axis_robot_deg) - float(preferred_robot)),
            -float(c.contact_quality),
            -float(c.overlap_quality),
            float(c.width_px),
        )
    )

    out: list[ObjectOrientation] = []
    seen_robot_axes: list[float] = []
    for cand in strong:
        if any(
            abs(axial_difference_deg(cand.axis_robot_deg, old_axis)) < 1.0
            for old_axis in seen_robot_axes
        ):
            continue
        seen_robot_axes.append(float(cand.axis_robot_deg))

        grasp_override = _closest_safe_point(
            cand.ideal_point,
            preferred,
            safe_mask=safe_mask,
            n=cand.n,
            width_px=cand.width_px,
        )
        width_mm = _estimate_width_mm(
            transform,
            cand.min_feature,
            cand.max_feature,
            cand.axis_robot_deg,
        )
        out.append(
            ObjectOrientation(
                long_axis_image_deg=float(image_long_deg),
                long_axis_robot_deg=float(robot_long_deg),
                grasp_axis_robot_deg=float(cand.axis_robot_deg),
                aspect_ratio=float(aspect_ratio),
                reliable=True,
                mode="antipodal_edge_edge",
                rectangularity=float(rectangularity),
                circularity=float(circularity),
                hull_vertex_count=int(hull_vertices),
                closing_axis_image_deg=float(cand.axis_image_deg),
                grasp_pixel_override=grasp_override,
                grasp_width_px=float(cand.width_px),
                grasp_width_mm=None if width_mm is None else float(width_mm),
                contact_type="edge_edge",
                contact_quality=float(cand.contact_quality),
                support_overlap_px=float(cand.overlap_px),
                support_min_point_px=(
                    float(cand.min_feature.point[0]),
                    float(cand.min_feature.point[1]),
                ),
                support_max_point_px=(
                    float(cand.max_feature.point[0]),
                    float(cand.max_feature.point[1]),
                ),
                support_min_span_px=float(cand.min_feature.span_px),
                support_max_span_px=float(cand.max_feature.span_px),
            )
        )
    return out

def plan_grasp_axis(
    *,
    polygon: np.ndarray,
    grasp_pixel: tuple[float, float],
    transform,
    grip_long_sides: bool = True,
    min_aspect_ratio: float = 1.20,
    default_grasp_axis_robot_deg: float = 0.0,
    yaw_alignment_offset_deg: float = 0.0,
    safe_mask: np.ndarray | None = None,
    width_slack_ratio: float = 0.15,
    support_tolerance_px: float = 3.0,
    min_contact_span_px: float = 6.0,
    circle_circularity_threshold: float = 0.985,
    circle_min_vertices: int = 8,
) -> ObjectOrientation:
    """Plan a production parallel-jaw grasp for a simple top-view shape.

    This replaces shape-specific triangle/square/pentagon rules with one
    geometric rule:

      1. Build the convex hull of the segmented polygon.
      2. Enumerate antipodal parallel-jaw closing axes from hull edge normals
         (rotating-calipers principle).
      3. Keep axes near the minimum required jaw opening.
      4. Prefer true edge-edge support; edge-vertex (normal for triangles) is
         accepted; vertex-vertex is lowest confidence.
      5. Prefer contact intervals with useful tangential overlap and a grasp
         mid-plane near the polygon centroid.
      6. Move the grasp point to the mid-plane between the two support lines,
         then project it into the collision planner's safe mask if necessary.
      7. True near-circles are the only normal fallback case because their Yaw
         is physically meaningless. Regular triangles/pentagons/hexagons etc.
         still use their real support faces.

    The convex hull makes this conservative for concave simple shapes: jaws are
    planned against the outer support boundary rather than entering a recess.
    """
    del grip_long_sides, min_aspect_ratio  # retained only for API compatibility

    pts = _validate_polygon(polygon)
    image_long_deg, aspect_ratio = polygon_long_axis_image_deg(pts)
    rectangularity, _ = polygon_rectangularity(pts)
    hull = _convex_hull_simplified(pts)
    circularity, hull_vertices = _shape_metrics(hull)
    centroid = _polygon_centroid(pts)
    preferred = np.array([float(grasp_pixel[0]), float(grasp_pixel[1])], dtype=np.float64)

    robot_long_deg = normalize_axial_deg(
        transform.robot_angle_from_image_angle(float(preferred[0]), float(preferred[1]), image_long_deg)
    )
    fallback_axis = normalize_axial_deg(
        float(default_grasp_axis_robot_deg) + float(yaw_alignment_offset_deg)
    )

    # A real circle has no stable edge direction. Do NOT apply this to ordinary
    # regular polygons: triangle/pentagon/hexagon/octagon should still expose
    # their support faces to the antipodal planner.
    circle_like = (
        circularity >= float(circle_circularity_threshold)
        and hull_vertices >= int(circle_min_vertices)
    )
    if circle_like:
        return ObjectOrientation(
            long_axis_image_deg=float(image_long_deg),
            long_axis_robot_deg=float(robot_long_deg),
            grasp_axis_robot_deg=float(fallback_axis),
            aspect_ratio=float(aspect_ratio),
            reliable=False,
            mode="symmetric_fallback",
            rectangularity=float(rectangularity),
            circularity=float(circularity),
            hull_vertex_count=int(hull_vertices),
            closing_axis_image_deg=float(image_long_deg),
            grasp_pixel_override=(int(round(preferred[0])), int(round(preferred[1]))),
            contact_type="symmetric",
        )

    candidates = _candidate_axes(
        hull,
        centroid=centroid,
        preferred_grasp=preferred,
        transform=transform,
        default_axis_robot_deg=default_grasp_axis_robot_deg,
        yaw_alignment_offset_deg=yaw_alignment_offset_deg,
        width_slack_ratio=width_slack_ratio,
        support_tolerance_px=support_tolerance_px,
        min_contact_span_px=min_contact_span_px,
    )
    if not candidates:
        return ObjectOrientation(
            long_axis_image_deg=float(image_long_deg),
            long_axis_robot_deg=float(robot_long_deg),
            grasp_axis_robot_deg=float(fallback_axis),
            aspect_ratio=float(aspect_ratio),
            reliable=False,
            mode="geometry_fallback",
            rectangularity=float(rectangularity),
            circularity=float(circularity),
            hull_vertex_count=int(hull_vertices),
            closing_axis_image_deg=float(image_long_deg),
            grasp_pixel_override=(int(round(preferred[0])), int(round(preferred[1]))),
            contact_type="none",
        )

    # Multiple axes can be physically equivalent (square, regular triangle,
    # regular pentagon, hexagon...). Pixel-level contour jitter must not make
    # the selected family jump by 60/72/90 degrees from frame to frame.
    # First keep all candidates whose geometric score is essentially tied,
    # then use minimum rotation from the configured reference axis as a
    # deterministic tie-breaker. This preserves true geometry while making
    # symmetric polygons stable.
    best_score = max(c.score_base for c in candidates)
    tied = [c for c in candidates if c.score_base >= best_score - 0.18]
    preferred_robot = normalize_axial_deg(
        float(default_grasp_axis_robot_deg) + float(yaw_alignment_offset_deg)
    )
    # IMPORTANT: use physical branch distance, not 180-degree axial distance.
    # With TCP eccentricity, e.g. +87 deg and -93 deg have the same closing
    # axis but place the TCP on opposite sides of the rotation center.  The
    # production lower MCU also no longer silently switches to theta+180.
    rotation_info = [
        (
            abs(float(c.axis_robot_deg) - float(preferred_robot)),
            float(c.axis_robot_deg) - float(preferred_robot),
            c,
        )
        for c in tied
    ]
    min_rotation = min(item[0] for item in rotation_info)
    # For regular N-gons the reference axis can lie exactly halfway between
    # two equally valid face normals. Tiny mask noise would otherwise switch
    # between them every frame. Treat rotations within 2.5° as equivalent and
    # use a fixed signed-direction preference as the final deterministic tie.
    rotation_tied = [item for item in rotation_info if item[0] <= min_rotation + 2.5]
    _, _, best = min(
        rotation_tied,
        key=lambda item: (
            0 if item[1] >= 0.0 else 1,
            item[0],
            item[2].width_px,
            -item[2].contact_quality,
            -item[2].overlap_quality,
        ),
    )
    grasp_override = _closest_safe_point(
        best.ideal_point,
        preferred,
        safe_mask=safe_mask,
        n=best.n,
        width_px=best.width_px,
    )

    if best.min_feature.kind == "edge" and best.max_feature.kind == "edge":
        contact_type = "edge_edge"
    elif best.min_feature.kind == "edge" or best.max_feature.kind == "edge":
        contact_type = "edge_vertex"
    else:
        contact_type = "vertex_vertex"

    width_mm = _estimate_width_mm(
        transform,
        best.min_feature,
        best.max_feature,
        best.axis_robot_deg,
    )

    # edge-vertex is the correct antipodal structure for a triangle and is not
    # treated as an error. vertex-vertex is allowed but marked lower confidence.
    reliable = contact_type != "vertex_vertex" or best.contact_quality >= 0.45
    mode = f"antipodal_{contact_type}"

    return ObjectOrientation(
        long_axis_image_deg=float(image_long_deg),
        long_axis_robot_deg=float(robot_long_deg),
        grasp_axis_robot_deg=float(best.axis_robot_deg),
        aspect_ratio=float(aspect_ratio),
        reliable=bool(reliable),
        mode=mode,
        rectangularity=float(rectangularity),
        circularity=float(circularity),
        hull_vertex_count=int(hull_vertices),
        closing_axis_image_deg=float(best.axis_image_deg),
        grasp_pixel_override=grasp_override,
        grasp_width_px=float(best.width_px),
        grasp_width_mm=None if width_mm is None else float(width_mm),
        contact_type=contact_type,
        contact_quality=float(best.contact_quality),
        support_overlap_px=float(best.overlap_px),
        support_min_point_px=(float(best.min_feature.point[0]), float(best.min_feature.point[1])),
        support_max_point_px=(float(best.max_feature.point[0]), float(best.max_feature.point[1])),
        support_min_span_px=float(best.min_feature.span_px),
        support_max_span_px=float(best.max_feature.span_px),
    )
