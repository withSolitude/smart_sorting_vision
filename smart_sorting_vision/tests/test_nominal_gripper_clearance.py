from __future__ import annotations

import unittest
from unittest import mock

import cv2
import numpy as np

from vision.calibration.pixel_to_robot import PixelToRobotTransform
from vision.grasp.gripper_footprint import GripperFootprintModel


def rectangle(x0, y0, x1, y1):
    return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


class NominalGripperClearanceTests(unittest.TestCase):
    def setUp(self):
        # Synthetic, documented scale: 0.5 mm / px. No camera or robot required.
        self.transform = PixelToRobotTransform(np.diag([0.5, 0.5, 1.0]), 300, 300)
        self.target = rectangle(100, 100, 140, 140)

    def model(self, **kwargs):
        config = dict(
            transform=self.transform,
            finger_length_mm=5.0,
            finger_thickness_mm=2.7,
            opening_uncertainty_mm=0.0,
            object_margin_mm=0.0,
            wall_margin_mm=0.0,
            image_border_margin_px=0,
            object_overlap_tolerance_px=4,
        )
        config.update(kwargs)
        return GripperFootprintModel(**config)

    def check(self, model=None, *, target=None, others=(), **kwargs):
        target = self.target if target is None else target
        config = dict(
            grasp_pixel=(120.0, 120.0),
            yaw_deg=0.0,
            opening_mm=28.0,  # 20 mm object + 8 mm total pre-open allowance.
            target_polygon_robot_mm=target * 0.5,
            other_polygons=list(others),
            image_shape=(300, 300, 3),
        )
        config.update(kwargs)
        return (model or self.model()).check_nominal(**config)

    def test_equal_touching_cubes_only_parallel_to_contact_axis_is_clear(self):
        neighbor = rectangle(140, 100, 180, 140)
        blocked = self.check(others=[neighbor], yaw_deg=0.0)
        clear = self.check(others=[neighbor], yaw_deg=90.0)
        self.assertEqual(blocked.reason, "other_object_collision")
        self.assertFalse(blocked.safe)
        self.assertTrue(clear.safe, clear)
        self.assertGreater(clear.clearance_mm, 0.0)

    def test_five_pixel_gap_neighbor_is_not_ignored(self):
        # Old center-bearing heuristic chose 90 degrees and hit the upper object.
        neighbors = [rectangle(140, 130, 180, 170), rectangle(100, 55, 140, 95)]
        clear = self.check(others=neighbors, yaw_deg=0.0)
        blocked = self.check(others=neighbors, yaw_deg=90.0)
        self.assertTrue(clear.safe, clear)
        self.assertFalse(blocked.safe)
        self.assertEqual(blocked.reason, "other_object_collision")

    def test_long_neighbor_uses_its_outline_not_its_center(self):
        neighbor = rectangle(140, -60, 180, 140)
        self.assertFalse(self.check(others=[neighbor], yaw_deg=0.0).safe)
        self.assertTrue(self.check(others=[neighbor], yaw_deg=90.0).safe)

    def test_narrow_real_gap_is_not_rejected_by_configured_expansion(self):
        neighbor = rectangle(155, 100, 195, 140)
        model = self.model(object_margin_mm=5.0, opening_uncertainty_mm=4.0)
        nominal = self.check(model, others=[neighbor])
        expanded = model.check(
            grasp_pixel=(120, 120), yaw_deg=0.0, opening_mm=28.0,
            other_polygons=[neighbor], image_shape=(300, 300, 3),
        )
        self.assertTrue(nominal.safe, nominal)
        self.assertGreater(nominal.clearance_mm, 0.0)
        self.assertFalse(expanded.safe)

    def test_closing_sweep_catches_obstacle_inside_static_preopen(self):
        target = rectangle(110, 110, 130, 130)
        obstacle = rectangle(145, 116, 150, 124)
        model = self.model()
        static = model.check(
            grasp_pixel=(120, 120), yaw_deg=0.0, opening_mm=40.0,
            other_polygons=[obstacle], image_shape=(300, 300, 3),
        )
        sweep = self.check(model, target=target, others=[obstacle], opening_mm=40.0)
        self.assertTrue(static.safe, static)
        self.assertFalse(sweep.safe)
        self.assertEqual(sweep.reason, "other_object_collision")
        self.assertGreater(sweep.object_collision_pixels, 4)

    def test_no_neighbors_has_finite_positive_clearance(self):
        result = self.check()
        self.assertTrue(result.safe, result)
        self.assertTrue(np.isfinite(result.clearance_mm))
        self.assertGreater(result.clearance_mm, 0.0)
        self.assertEqual(len(result.footprint_polygons_px), 2)

    def test_disabled_expanded_checker_does_not_disable_nominal_checker(self):
        model = self.model(enabled=False)
        neighbor = rectangle(140, 100, 180, 140)
        self.assertEqual(model.check(
            grasp_pixel=(120, 120), yaw_deg=0.0, opening_mm=28.0,
            other_polygons=[neighbor], image_shape=(300, 300, 3),
        ).reason, "disabled")
        self.assertEqual(self.check(model, others=[neighbor]).reason, "other_object_collision")

    def test_fixed_wall_blocks_closing_path(self):
        model = self.model(forbidden_zones_mm=[{"rect": [72.5, 58.0, 75.0, 62.0]}])
        result = self.check(model, target=rectangle(110, 110, 130, 130), opening_mm=40.0)
        self.assertEqual(result.reason, "fixed_wall_collision")
        self.assertGreater(result.wall_collision_pixels, 0)

    def test_allowed_region_uses_nominal_dimensions_and_zero_margin(self):
        model = self.model(allowed_region_mm=[43, 57, 77, 63], wall_margin_mm=10.0)
        result = self.check(model)
        self.assertTrue(result.safe, result)
        self.assertAlmostEqual(result.clearance_mm, 0.3, places=5)
        blocked = self.check(self.model(allowed_region_mm=[44, 57, 77, 63]))
        self.assertEqual(blocked.reason, "allowed_region_collision")

    def test_allowed_polygon_blocks_nominal_geometry_outside(self):
        allowed = rectangle(43, 57, 77, 63)
        self.assertTrue(self.check(self.model(allowed_polygon_mm=allowed, wall_margin_mm=10)).safe)
        blocked = self.check(self.model(allowed_polygon_mm=rectangle(44, 57, 77, 63)))
        self.assertEqual(blocked.reason, "allowed_polygon_collision")

    def test_concave_allowed_region_cannot_be_crossed_between_inside_corners(self):
        # A narrow notch intersects the left swept rectangle, but none of its corners.
        allowed = np.asarray([
            [40, 50], [46, 50], [46, 60], [47, 60],
            [47, 50], [80, 50], [80, 70], [40, 70],
        ], dtype=np.float64)
        result = self.check(self.model(allowed_polygon_mm=allowed))
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "allowed_polygon_collision")

    def test_image_border_does_not_apply_configured_border_margin(self):
        target = rectangle(14.4, 100, 54.4, 140)
        model = self.model(image_border_margin_px=20, wall_margin_mm=10.0)
        inside = self.check(model, target=target, grasp_pixel=(34.4, 120.0))
        outside = self.check(
            model, target=rectangle(12.4, 100, 52.4, 140), grasp_pixel=(32.4, 120.0),
        )
        self.assertTrue(inside.safe, inside)
        self.assertEqual(outside.reason, "image_border_collision")
        self.assertTrue(outside.border_violation)

    def test_configured_body_is_also_checked(self):
        obstacle = rectangle(110, 75, 130, 85)
        self.assertTrue(self.check(others=[obstacle]).safe)
        model = self.model(
            body_enabled=True, body_width_mm=10, body_length_mm=5,
            body_longitudinal_offset_mm=-20,
        )
        result = self.check(model, others=[obstacle])
        self.assertEqual(result.reason, "other_object_collision")
        self.assertEqual(len(result.footprint_polygons_px), 3)

    def test_overlapping_neighbor_masks_form_union_without_holes(self):
        masks = [rectangle(10, 10, 20, 20), rectangle(15, 10, 25, 20)]
        mask = self.model()._other_objects_mask(masks, (300, 300, 3))
        self.assertEqual(mask[15, 17], 255)
        self.assertEqual(np.count_nonzero(mask), 16 * 11)

    def test_overlapping_footprints_form_union_without_holes(self):
        footprints = [rectangle(10, 10, 20, 20), rectangle(15, 10, 25, 20)]
        mask = self.model()._polygons_to_mask(footprints, (300, 300, 3))
        self.assertEqual(mask[15, 17], 255)
        self.assertEqual(np.count_nonzero(mask), 16 * 11)

    def test_overlap_of_body_and_finger_does_not_hide_obstacle(self):
        model = self.model(
            body_enabled=True, body_width_mm=40, body_length_mm=5,
            body_longitudinal_offset_mm=0,
        )
        result = self.check(model, others=[rectangle(90, 117, 95, 123)])
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "other_object_collision")
        self.assertGreater(result.object_collision_pixels, 4)

    def test_cached_context_does_not_recompute_distance_transform(self):
        model = self.model()
        others = [rectangle(190, 100, 210, 140)]
        context = model.prepare_context(other_polygons=others, image_shape=(300, 300, 3))
        with mock.patch(
            "vision.grasp.gripper_footprint.cv2.distanceTransform",
            side_effect=AssertionError("candidate recomputed distance transform"),
        ):
            result = self.check(model, others=others, context=context)
        self.assertTrue(result.safe, result)

    def test_roi_matches_full_image_raster_and_clearance(self):
        model = self.model(enforce_image_border=False)
        cases = [
            (self.target, (120.5, 120.5), [rectangle(200, 90, 220, 145)], angle, 40.0)
            for angle in (0.0, 13.0, 45.0, 90.0, 137.0)
        ]
        # Also cover a projected footprint partially clipped by the image.
        cases.append((rectangle(-10, 100, 30, 140), (10.0, 120.0),
                      [rectangle(60, 100, 80, 140)], 0.0, 28.0))
        # An obstacle inside the ROI verifies collision counts, not just clearance.
        cases.append((self.target, (120.0, 120.0),
                      [rectangle(140, 100, 180, 140)], 0.0, 28.0))
        for target, center, others, angle, opening in cases:
            with self.subTest(angle=angle, center=center, opening=opening):
                context = model.prepare_context(other_polygons=others, image_shape=(300, 300, 3))
                with mock.patch.object(
                    model, "_polygons_to_mask", side_effect=AssertionError("full-frame candidate mask"),
                ):
                    result = self.check(
                        model, target=target, grasp_pixel=center, others=others,
                        yaw_deg=angle, opening_mm=opening, context=context,
                    )
                full_mask = model._polygons_to_mask(result.footprint_polygons_px, (300, 300, 3))
                expected_pixels = int(np.count_nonzero(cv2.bitwise_and(full_mask, context.other_mask)))
                self.assertEqual(result.object_collision_pixels, expected_pixels)
                self.assertEqual(result.safe, expected_pixels <= model.object_overlap_tolerance_px)
                if result.safe:
                    expected_clearance = model._clearance_from_context(full_mask, context)
                    self.assertAlmostEqual(result.clearance_px, expected_clearance, places=6)

    def test_no_obstacle_clearance_uses_original_image_diagonal(self):
        result = self.check(self.model(enforce_image_border=False))
        self.assertTrue(result.safe, result)
        self.assertAlmostEqual(result.clearance_px, np.hypot(300, 300), places=6)
        self.assertAlmostEqual(result.clearance_mm, np.hypot(300, 300) * 0.5, places=6)

    def test_invalid_queried_distance_fails_closed(self):
        model = self.model()
        others = [rectangle(190, 100, 210, 140)]
        context = model.prepare_context(other_polygons=others, image_shape=(300, 300, 3))
        baseline = self.check(model, others=others, context=context)
        mask = model._polygons_to_mask(baseline.footprint_polygons_px, (300, 300, 3))
        y, x = np.argwhere(mask != 0)[0]
        for value in (np.nan, np.inf, -1.0):
            with self.subTest(value=value):
                context.distance_to_obstacle[y, x] = value
                result = self.check(model, others=others, context=context)
                self.assertFalse(result.safe)
                self.assertEqual(result.reason, "invalid_collision_context")

    def test_unqueried_distance_does_not_scan_full_frame(self):
        model = self.model()
        others = [rectangle(190, 100, 210, 140)]
        context = model.prepare_context(other_polygons=others, image_shape=(300, 300, 3))
        expected = self.check(model, others=others, context=context)
        context.distance_to_obstacle[0, 0] = np.nan
        actual = self.check(model, others=others, context=context)
        self.assertTrue(actual.safe, actual)
        self.assertEqual(actual.clearance_px, expected.clearance_px)

    def test_distance_shape_and_dtype_still_validated(self):
        model = self.model()
        others = [rectangle(190, 100, 210, 140)]
        context = model.prepare_context(other_polygons=others, image_shape=(300, 300, 3))
        for field in (np.ones((300, 300), dtype=np.int32), np.ones((1, 1), dtype=np.float32)):
            with self.subTest(shape=field.shape, dtype=field.dtype):
                context.distance_to_obstacle = field
                result = self.check(model, others=others, context=context)
                self.assertFalse(result.safe)
                self.assertEqual(result.reason, "invalid_collision_context")

    def test_overlap_tolerance_counts_pixels_not_linear_distance(self):
        # This tiny positive-area polygon rasterizes to exactly one obstacle pixel.
        obstacle = rectangle(143.1, 119.1, 143.3, 119.3)
        allowed = self.check(self.model(object_overlap_tolerance_px=1), others=[obstacle])
        blocked = self.check(self.model(object_overlap_tolerance_px=0), others=[obstacle])
        self.assertTrue(allowed.safe, allowed)
        self.assertEqual(allowed.object_collision_pixels, 1)
        self.assertEqual(allowed.clearance_mm, 0.0)
        self.assertEqual(blocked.reason, "other_object_collision")

    def test_invalid_geometry_fails_closed(self):
        for overrides in (
            {"target_polygon_robot_mm": None},
            {"target_polygon_robot_mm": np.full((4, 2), np.nan)},
            {"target_polygon_robot_mm": np.zeros((4, 2))},
            {"other_polygons": [np.zeros((2, 2))]},
            {"opening_mm": None}, {"opening_mm": np.nan}, {"opening_mm": 0.0},
            {"yaw_deg": np.nan}, {"image_shape": (0, 300, 3)},
        ):
            with self.subTest(overrides=overrides):
                result = self.check(**overrides)
                self.assertFalse(result.safe)
                self.assertEqual(result.reason, "invalid_nominal_geometry")

    def test_target_outside_preopen_fails_closed(self):
        result = self.check(opening_mm=10.0)
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "target_outside_preopen")

    def test_missing_or_nonfinite_projection_fails_closed(self):
        model = self.model()
        for projection in (np.empty((0, 2)), np.full((4, 2), np.nan)):
            with self.subTest(projection=projection), mock.patch.object(
                model, "_robot_to_pixels", return_value=projection,
            ):
                result = self.check(model)
                self.assertFalse(result.safe)
                self.assertEqual(result.reason, "nominal_projection_failed")
        with mock.patch.object(model, "_robot_to_pixels", side_effect=RuntimeError("bad transform")):
            self.assertEqual(self.check(model).reason, "nominal_projection_failed")

    def test_missing_distance_field_with_obstacles_fails_closed(self):
        model = self.model()
        others = [rectangle(190, 100, 210, 140)]
        context = model.prepare_context(other_polygons=others, image_shape=(300, 300, 3))
        context.distance_to_obstacle = None
        result = self.check(model, others=others, context=context)
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "invalid_collision_context")


if __name__ == "__main__":
    unittest.main()
