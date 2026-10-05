"""The capture -> detect -> locate -> alert loop.

Every collaborator is injected, so the whole pipeline runs headless against a
ReplaySource, a stub detector, and recording sinks - no camera, no Arduino, no
speaker. That is what makes the behaviour in alerts.py testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from .alerts import Alert, AlertPolicy
from .audio import AudioSink
from .camera import CameraSource
from .config import Settings
from .detector import ObstacleDetector
from .geometry import Urgency
from .haptics import HapticSink


@dataclass
class RunStats:
    """What happened during a run - written to results/ for tuning."""

    frames: int = 0
    detections: int = 0
    alerts: int = 0
    alerts_by_urgency: dict[str, int] = field(default_factory=dict)
    first_t: Optional[float] = None
    last_t: Optional[float] = None

    @property
    def duration_s(self) -> float:
        if self.first_t is None or self.last_t is None:
            return 0.0
        return max(self.last_t - self.first_t, 0.0)

    @property
    def fps(self) -> float:
        return self.frames / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def alerts_per_minute(self) -> float:
        """The key usability number: how often the cane interrupts the user."""
        return (self.alerts / self.duration_s * 60.0) if self.duration_s > 0 else 0.0

    def record_frame(self, t: float, n_detections: int) -> None:
        self.frames += 1
        self.detections += n_detections
        if self.first_t is None:
            self.first_t = t
        self.last_t = t

    def record_alert(self, alert: Alert) -> None:
        self.alerts += 1
        key = alert.urgency.value
        self.alerts_by_urgency[key] = self.alerts_by_urgency.get(key, 0) + 1

    def as_dict(self) -> dict:
        return {
            "frames": self.frames,
            "detections": self.detections,
            "alerts": self.alerts,
            "alerts_by_urgency": dict(self.alerts_by_urgency),
            "duration_s": round(self.duration_s, 2),
            "fps": round(self.fps, 2),
            "alerts_per_minute": round(self.alerts_per_minute, 2),
        }


class CaneVisionPipeline:
    """Reads frames, decides what matters, drives the cane's outputs."""

    def __init__(
        self,
        camera: CameraSource,
        detector: ObstacleDetector,
        haptics: HapticSink,
        audio: AudioSink,
        settings: Optional[Settings] = None,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._haptics = haptics
        self._audio = audio
        self._settings = settings or Settings()
        self._policy = AlertPolicy(self._settings.alerts)
        self.stats = RunStats()

    def stream(self, max_frames: Optional[int] = None) -> Iterator[Alert]:
        """Yield alerts as they are decided, driving the sinks as a side effect.

        Yielding rather than returning lets a caller render a debug overlay or
        log without this class knowing about either.

        ``max_frames`` is enforced here rather than by the consumer, because a
        quiet scene yields nothing: a caller counting yielded alerts would
        never get the chance to stop.
        """
        for timestamp, frame in self._camera.frames():
            if max_frames is not None and self.stats.frames >= max_frames:
                return
            detections = self._detector.detect(frame)
            self.stats.record_frame(timestamp, len(detections))

            frame_width = frame.shape[1]
            for alert in self._policy.update(detections, frame_width, timestamp):
                self.stats.record_alert(alert)
                self._dispatch(alert)
                yield alert

    def run(self, max_frames: Optional[int] = None) -> RunStats:
        """Drain the stream. ``max_frames`` bounds a dry run."""
        try:
            for _ in self.stream(max_frames=max_frames):
                pass
        finally:
            self.close()
        return self.stats

    def _dispatch(self, alert: Alert) -> None:
        """Haptics always; speech only when it adds something.

        A buzz already conveys direction and urgency. Speech is reserved for
        naming an obstacle the user cannot infer, and is skipped for INFO so
        the cane stays quiet during routine walking.
        """
        self._haptics.pulse(alert)
        if alert.urgency is Urgency.INFO:
            return
        if alert.label == "obstacle" and alert.urgency is not Urgency.CRITICAL:
            return
        self._audio.announce(alert)

    def close(self) -> None:
        """Release hardware. Each close is independent so one failure cannot
        leave a motor running or a camera held open."""
        for closer in (self._haptics.stop, self._haptics.close,
                       self._audio.close, self._camera.close):
            try:
                closer()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
