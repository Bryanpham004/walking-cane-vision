"""Alert-policy tests.

These are the behavioural tests that matter: the policy's whole job is to stay
quiet, and silence is easy to get wrong in both directions.
"""

from __future__ import annotations

import pytest

from src.alerts import AlertPolicy, classify_urgency, describe
from src.config import AlertConfig
from src.geometry import BoundingBox, Detection, Urgency, Zone

FRAME_WIDTH = 640


def detection(distance_m: float, center_x: int = 320, label: str = "person"):
    """A calibrated detection at a given distance and horizontal position."""
    return Detection(
        label=label,
        confidence=0.9,
        box=BoundingBox(center_x - 40, 100, center_x + 40, 400),
        distance_m=distance_m,
        distance_is_calibrated=True,
    )


@pytest.fixture
def config():
    # confirm_frames=1 keeps most tests to a single frame; the confirmation
    # behaviour itself is tested explicitly below.
    return AlertConfig(confirm_frames=1, cooldown_s=1.6)


class TestUrgencyClassification:
    def test_very_close_is_critical(self, config):
        assert classify_urgency(0.8, None, config) is Urgency.CRITICAL

    def test_mid_range_is_warn(self, config):
        assert classify_urgency(2.0, None, config) is Urgency.WARN

    def test_far_is_info(self, config):
        assert classify_urgency(3.5, None, config) is Urgency.INFO

    def test_fast_approach_escalates_distant_obstacle(self, config):
        """A far obstacle closing fast is critical despite its distance."""
        assert classify_urgency(3.5, ttc_s=1.0, config=config) is Urgency.CRITICAL

    def test_slow_approach_does_not_escalate(self, config):
        assert classify_urgency(3.5, ttc_s=9.0, config=config) is Urgency.INFO


class TestConfirmation:
    def test_single_frame_detection_is_suppressed(self):
        policy = AlertPolicy(AlertConfig(confirm_frames=2))
        assert policy.update([detection(2.0)], FRAME_WIDTH, 0.0) == []

    def test_alerts_once_confirmed(self):
        policy = AlertPolicy(AlertConfig(confirm_frames=2))
        policy.update([detection(2.0)], FRAME_WIDTH, 0.0)
        assert len(policy.update([detection(2.0)], FRAME_WIDTH, 0.1)) == 1

    def test_disappearing_obstacle_must_be_reconfirmed(self):
        policy = AlertPolicy(AlertConfig(confirm_frames=2))
        policy.update([detection(2.0)], FRAME_WIDTH, 0.0)
        policy.update([], FRAME_WIDTH, 0.1)  # zone clears
        assert policy.update([detection(2.0)], FRAME_WIDTH, 0.2) == []


class TestCooldown:
    def test_static_obstacle_alerts_once_not_every_frame(self, config):
        policy = AlertPolicy(config)
        emitted = sum(
            len(policy.update([detection(2.0)], FRAME_WIDTH, i * 0.066))
            for i in range(15)  # one second at 15fps
        )
        assert emitted == 1

    def test_alerts_again_after_cooldown_expires(self, config):
        policy = AlertPolicy(config)
        assert len(policy.update([detection(2.0)], FRAME_WIDTH, 0.0)) == 1
        assert len(policy.update([detection(2.0)], FRAME_WIDTH, 1.0)) == 0
        assert len(policy.update([detection(2.0)], FRAME_WIDTH, 1.7)) == 1

    def test_escalation_bypasses_cooldown(self, config):
        """A warning that becomes critical must not be silenced by cooldown."""
        policy = AlertPolicy(config)
        first = policy.update([detection(2.0)], FRAME_WIDTH, 0.0)
        assert first[0].urgency is Urgency.WARN
        second = policy.update([detection(0.7)], FRAME_WIDTH, 0.2)
        assert len(second) == 1
        assert second[0].urgency is Urgency.CRITICAL

    def test_de_escalation_does_not_bypass_cooldown(self, config):
        policy = AlertPolicy(config)
        policy.update([detection(0.7)], FRAME_WIDTH, 0.0)
        assert policy.update([detection(2.0)], FRAME_WIDTH, 0.2) == []


class TestRangeGate:
    def test_obstacle_beyond_max_range_is_ignored(self, config):
        assert policy_update(config, detection(9.0)) == []

    def test_obstacle_at_edge_of_range_is_reported(self, config):
        assert len(policy_update(config, detection(3.9))) == 1

    def test_detection_without_distance_is_ignored(self, config):
        no_distance = Detection("person", 0.9, BoundingBox(280, 100, 360, 400))
        assert policy_update(config, no_distance) == []


class TestZoneHandling:
    def test_one_alert_per_zone_for_nearest_obstacle(self, config):
        policy = AlertPolicy(config)
        alerts = policy.update(
            [detection(2.0, center_x=320), detection(1.2, center_x=330)],
            FRAME_WIDTH,
            0.0,
        )
        assert len(alerts) == 1
        assert alerts[0].distance_m == 1.2

    def test_separate_zones_alert_independently(self, config):
        policy = AlertPolicy(config)
        alerts = policy.update(
            [detection(2.0, center_x=60), detection(2.0, center_x=580)],
            FRAME_WIDTH,
            0.0,
        )
        assert {a.zone for a in alerts} == {Zone.LEFT, Zone.RIGHT}

    def test_most_urgent_alert_is_ordered_first(self, config):
        policy = AlertPolicy(config)
        alerts = policy.update(
            [detection(3.5, center_x=60), detection(0.6, center_x=580)],
            FRAME_WIDTH,
            0.0,
        )
        assert alerts[0].urgency is Urgency.CRITICAL
        assert alerts[0].zone is Zone.RIGHT


class TestMessages:
    def test_center_obstacle_says_ahead(self):
        assert describe("chair", Zone.CENTER, 1.25, True) == "chair ahead, 1.2 metres"

    def test_side_obstacle_names_the_side(self):
        assert describe("person", Zone.LEFT, 2.0, True) == (
            "person to your left, 2.0 metres"
        )

    def test_unlabelled_detection_is_called_obstacle(self):
        assert describe("obstacle", Zone.CENTER, 1.0, True).startswith("obstacle")

    def test_labels_can_be_suppressed(self):
        assert describe("person", Zone.CENTER, 1.0, False).startswith("obstacle")

    def test_alert_carries_calibration_flag(self, config):
        uncalibrated = Detection(
            "obstacle", 0.7, BoundingBox(280, 100, 360, 400),
            distance_m=1.5, distance_is_calibrated=False,
        )
        alert = policy_update(config, uncalibrated)[0]
        assert alert.distance_is_calibrated is False


def policy_update(config: AlertConfig, det: Detection):
    """One detection through a fresh policy."""
    return AlertPolicy(config).update([det], FRAME_WIDTH, 0.0)
