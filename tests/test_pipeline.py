"""End-to-end pipeline tests, driven entirely by stubs.

No camera, no Arduino, no model - which is the point of the injected
collaborators in CaneVisionPipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.alerts import Alert
from src.audio import ConsoleAudio
from src.camera import ReplaySource
from src.config import AlertConfig, Settings
from src.geometry import BoundingBox, Detection, Urgency, Zone
from src.haptics import NullHaptics, encode_pulse
from src.pipeline import CaneVisionPipeline


class StaticDetector:
    """Returns the same detections for every frame."""

    def __init__(self, detections):
        self._detections = list(detections)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return list(self._detections)


class ApproachingDetector:
    """One obstacle closing steadily from ``start`` to ``end`` metres."""

    def __init__(self, frames: int, start: float = 4.0, end: float = 0.5):
        self._frames = frames
        self._start = start
        self._end = end
        self._i = 0

    def detect(self, frame):
        span = max(self._frames - 1, 1)
        distance = self._start - (self._start - self._end) * (self._i / span)
        self._i += 1
        return [
            Detection(
                "person", 0.92, BoundingBox(280, 100, 360, 400),
                distance_m=round(distance, 2), distance_is_calibrated=True,
            )
        ]


def blank_frames(n: int, width: int = 640, height: int = 480):
    return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(n)]


def build(detector, frames: int = 30, alerts: AlertConfig | None = None):
    haptics = NullHaptics()
    audio = ConsoleAudio()
    settings = Settings()
    if alerts is not None:
        from dataclasses import replace

        settings = replace(settings, alerts=alerts)
    pipeline = CaneVisionPipeline(
        ReplaySource(blank_frames(frames), fps=15.0),
        detector, haptics, audio, settings,
    )
    return pipeline, haptics, audio


class TestPipelineRun:
    def test_processes_every_frame(self):
        detector = StaticDetector([])
        pipeline, _, _ = build(detector, frames=20)
        stats = pipeline.run()
        assert detector.calls == 20
        assert stats.frames == 20

    def test_empty_scene_produces_no_alerts(self):
        pipeline, haptics, audio = build(StaticDetector([]), frames=20)
        stats = pipeline.run()
        assert stats.alerts == 0
        assert haptics.commands == ["S"]  # teardown stop only
        assert audio.spoken == []

    def test_approaching_obstacle_escalates_to_critical(self):
        pipeline, _, _ = build(ApproachingDetector(30), frames=30)
        alerts = list(pipeline.stream())
        assert alerts, "an approaching obstacle must raise at least one alert"
        assert alerts[-1].urgency is Urgency.CRITICAL

    def test_alert_rate_stays_usable(self):
        """The policy must not fire on every frame - that is the whole point."""
        pipeline, _, _ = build(ApproachingDetector(150), frames=150)
        stats = pipeline.run()
        assert stats.frames == 150
        # 150 frames at 15fps = 10s. More than ~2 alerts/sec is unusable.
        assert stats.alerts < 20, f"too chatty: {stats.alerts} alerts in 10s"

    def test_stats_are_recorded(self):
        pipeline, _, _ = build(ApproachingDetector(30), frames=30)
        stats = pipeline.run().as_dict()
        assert stats["frames"] == 30
        assert stats["detections"] == 30
        assert stats["fps"] == pytest.approx(15.0, abs=1.5)
        assert stats["alerts"] >= 1

    def test_max_frames_bounds_the_run(self):
        pipeline, _, _ = build(StaticDetector([]), frames=100)
        stats = pipeline.run(max_frames=10)
        assert stats.frames == 10

    def test_closes_hardware_on_completion(self):
        pipeline, haptics, audio = build(StaticDetector([]), frames=5)
        pipeline.run()
        assert haptics.closed
        assert audio.closed


class TestDispatchPolicy:
    def test_haptics_fire_for_every_alert(self):
        pipeline, haptics, _ = build(ApproachingDetector(30), frames=30)
        stats = pipeline.run()
        pulses = [c for c in haptics.commands if c.startswith("V")]
        assert len(pulses) == stats.alerts

    def test_info_alerts_stay_silent(self):
        """Routine distant obstacles buzz but do not speak."""
        far = Detection(
            "person", 0.9, BoundingBox(280, 100, 360, 400),
            distance_m=3.8, distance_is_calibrated=True,
        )
        pipeline, haptics, audio = build(
            StaticDetector([far]), frames=10, alerts=AlertConfig(confirm_frames=1)
        )
        pipeline.run()
        assert any(c.startswith("V") for c in haptics.commands)
        assert audio.spoken == []

    def test_critical_alerts_speak(self):
        near = Detection(
            "chair", 0.9, BoundingBox(280, 100, 360, 400),
            distance_m=0.7, distance_is_calibrated=True,
        )
        pipeline, _, audio = build(
            StaticDetector([near]), frames=5, alerts=AlertConfig(confirm_frames=1)
        )
        pipeline.run()
        assert audio.spoken and "chair" in audio.spoken[0]

    def test_unlabelled_warning_does_not_speak(self):
        """'obstacle somewhere' is not worth interrupting for below critical."""
        blob = Detection(
            "obstacle", 0.6, BoundingBox(280, 100, 360, 400),
            distance_m=2.0, distance_is_calibrated=False,
        )
        pipeline, _, audio = build(
            StaticDetector([blob]), frames=5, alerts=AlertConfig(confirm_frames=1)
        )
        pipeline.run()
        assert audio.spoken == []


class TestHapticEncoding:
    @pytest.mark.parametrize(
        "zone,motor", [(Zone.LEFT, 0), (Zone.CENTER, 1), (Zone.RIGHT, 2)]
    )
    def test_zone_maps_to_motor(self, zone, motor):
        from src.config import HapticConfig

        alert = Alert(Urgency.WARN, zone, "person", 2.0, "msg")
        assert encode_pulse(alert, HapticConfig()).startswith(f"V{motor},")

    def test_urgency_sets_intensity_and_duration(self):
        from src.config import HapticConfig

        config = HapticConfig()
        info = encode_pulse(Alert(Urgency.INFO, Zone.CENTER, "p", 3.0, "m"), config)
        crit = encode_pulse(
            Alert(Urgency.CRITICAL, Zone.CENTER, "p", 0.5, "m"), config
        )
        assert info == "V1,90,120\n"
        assert crit == "V1,255,320\n"
