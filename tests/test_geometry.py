"""Geometry and distance-estimation tests."""

from __future__ import annotations

import pytest

from src.config import CameraConfig
from src.geometry import (
    BoundingBox,
    Detection,
    Zone,
    degrees_off_axis,
    estimate_distance,
    estimate_focal_length,
    non_max_suppression,
    time_to_contact,
)


class TestBoundingBox:
    def test_rejects_degenerate_box(self):
        with pytest.raises(ValueError):
            BoundingBox(10, 10, 10, 20)
        with pytest.raises(ValueError):
            BoundingBox(10, 10, 20, 10)

    def test_dimensions(self):
        box = BoundingBox(10, 20, 110, 320)
        assert box.width == 100
        assert box.height == 300
        assert box.area == 30_000
        assert box.center_x == 60.0

    def test_iou_identical_is_one(self):
        box = BoundingBox(0, 0, 10, 10)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_disjoint_is_zero(self):
        assert BoundingBox(0, 0, 10, 10).iou(BoundingBox(50, 50, 60, 60)) == 0.0

    def test_iou_half_overlap(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 0, 15, 10)
        # intersection 50, union 150
        assert a.iou(b) == pytest.approx(50 / 150)


class TestDistanceEstimation:
    def test_pinhole_formula(self):
        # f=554, H=1.70, h=300 -> 3.139m
        assert estimate_distance(300, 1.70, 554.0) == pytest.approx(3.1393, abs=1e-3)

    def test_distance_halves_when_box_doubles(self):
        near = estimate_distance(600, 1.70, 554.0)
        far = estimate_distance(300, 1.70, 554.0)
        assert far == pytest.approx(near * 2)

    def test_calibration_round_trips(self):
        """Calibrating at a known distance must reproduce that distance."""
        focal = estimate_focal_length(312, 1.70, 3.0)
        assert estimate_distance(312, 1.70, focal) == pytest.approx(3.0)

    @pytest.mark.parametrize(
        "box_h,real_h,focal",
        [(0, 1.7, 554.0), (-5, 1.7, 554.0), (300, 0, 554.0), (300, 1.7, 0)],
    )
    def test_rejects_nonpositive_inputs(self, box_h, real_h, focal):
        with pytest.raises(ValueError):
            estimate_distance(box_h, real_h, focal)

    def test_focal_scales_with_frame_width(self):
        """A focal length calibrated at 640px must be rescaled at 1280px."""
        camera = CameraConfig(width=640, focal_length_px=554.0)
        assert camera.scaled_focal(1280) == pytest.approx(1108.0)
        assert camera.scaled_focal(640) == pytest.approx(554.0)


class TestZones:
    @pytest.mark.parametrize(
        "center_x,expected",
        [(50, Zone.LEFT), (320, Zone.CENTER), (600, Zone.RIGHT)],
    )
    def test_zone_assignment(self, center_x, expected):
        half = 40
        box = BoundingBox(int(center_x - half), 100, int(center_x + half), 400)
        assert Detection("person", 0.9, box).zone(640) is expected

    def test_center_band_is_widened(self):
        """A 0.4 centre fraction puts the boundary at 30%/70% of the width."""
        det = lambda cx: Detection(  # noqa: E731
            "person", 0.9, BoundingBox(cx - 5, 100, cx + 5, 400)
        )
        assert det(205).zone(640) is Zone.CENTER   # 32% -> centre
        assert det(185).zone(640) is Zone.LEFT     # 29% -> left

    def test_rejects_bad_frame_width(self):
        det = Detection("person", 0.9, BoundingBox(0, 0, 10, 10))
        with pytest.raises(ValueError):
            det.zone(0)


class TestTimeToContact:
    def test_closing_obstacle(self):
        # 3.0m -> 2.0m in 0.5s = 2 m/s closing; 2.0m / 2.0 = 1.0s
        assert time_to_contact(2.0, 3.0, 0.5) == pytest.approx(1.0)

    def test_static_obstacle_has_no_ttc(self):
        assert time_to_contact(2.0, 2.0, 0.5) is None

    def test_receding_obstacle_has_no_ttc(self):
        assert time_to_contact(3.0, 2.0, 0.5) is None

    def test_first_observation_has_no_ttc(self):
        assert time_to_contact(2.0, None, 0.5) is None

    def test_zero_dt_has_no_ttc(self):
        assert time_to_contact(2.0, 3.0, 0.0) is None


class TestNonMaxSuppression:
    def test_keeps_most_confident_of_overlapping_pair(self):
        strong = Detection("person", 0.9, BoundingBox(0, 0, 100, 100))
        weak = Detection("person", 0.5, BoundingBox(5, 5, 105, 105))
        kept = non_max_suppression([weak, strong], 0.45)
        assert kept == [strong]

    def test_keeps_disjoint_detections(self):
        a = Detection("person", 0.9, BoundingBox(0, 0, 50, 50))
        b = Detection("chair", 0.8, BoundingBox(200, 200, 250, 250))
        assert len(non_max_suppression([a, b], 0.45)) == 2

    def test_empty_input(self):
        assert non_max_suppression([], 0.45) == []


class TestBearing:
    def test_center_is_zero_degrees(self):
        assert degrees_off_axis(320, 640) == pytest.approx(0.0)

    def test_left_is_negative_right_is_positive(self):
        assert degrees_off_axis(100, 640) < 0
        assert degrees_off_axis(540, 640) > 0
