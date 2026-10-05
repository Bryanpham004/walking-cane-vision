"""Typed configuration for the cane vision pipeline.

Every tunable lives here so thresholds can be calibrated per-user without
touching pipeline logic. Distances are metres, times are seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping


@dataclass(frozen=True)
class CameraConfig:
    """Capture settings and the intrinsics needed for distance estimation."""

    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 15
    # Horizontal focal length in pixels. Produced by scripts/calibrate_focal.py.
    # The default is a rough estimate for a 640px-wide frame at ~60 deg FOV and
    # MUST be calibrated before distance readings can be trusted.
    focal_length_px: float = 554.0
    calibrated: bool = False

    def scaled_focal(self, actual_width: int) -> float:
        """Focal length rescaled if frames arrive at a different width.

        Focal length in pixels is proportional to image width, so a value
        calibrated at 640px is wrong by the width ratio at 1280px.
        """
        if self.width <= 0:
            raise ValueError("camera width must be positive")
        return self.focal_length_px * (actual_width / self.width)


@dataclass(frozen=True)
class DetectionConfig:
    """Obstacle detector thresholds."""

    min_confidence: float = 0.45
    # Ignore blobs smaller than this fraction of frame area — usually noise.
    min_area_fraction: float = 0.004
    # Frame-difference threshold for the model-free fallback detector.
    motion_threshold: int = 24
    nms_iou: float = 0.45


@dataclass(frozen=True)
class AlertConfig:
    """Alert policy: when to speak, when to buzz, and how often."""

    # Obstacles beyond this are ignored entirely.
    max_range_m: float = 4.0
    # Inside this distance an obstacle is urgent regardless of closing speed.
    critical_distance_m: float = 1.0
    warn_distance_m: float = 2.5
    # Seconds before predicted contact that counts as urgent.
    critical_ttc_s: float = 1.5
    # Minimum gap between two alerts describing the same zone, to stop spam.
    cooldown_s: float = 1.6
    # An obstacle must be seen this many consecutive frames before alerting,
    # which suppresses single-frame false positives.
    confirm_frames: int = 2
    speak_labels: bool = True


@dataclass(frozen=True)
class HapticConfig:
    """Arduino serial link driving the vibration motors."""

    port: str = "/dev/ttyACM0"
    baud: int = 115200
    timeout_s: float = 0.2
    # Duty cycle (0-255) per urgency level.
    intensity: Mapping[str, int] = field(
        default_factory=lambda: {"info": 90, "warn": 160, "critical": 255}
    )
    pulse_ms: Mapping[str, int] = field(
        default_factory=lambda: {"info": 120, "warn": 200, "critical": 320}
    )


@dataclass(frozen=True)
class Settings:
    """Root configuration object passed through the pipeline."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    haptics: HapticConfig = field(default_factory=HapticConfig)

    def with_focal_length(self, focal_px: float) -> "Settings":
        """Return a copy carrying a calibrated focal length."""
        if focal_px <= 0:
            raise ValueError("focal length must be positive")
        camera = replace(self.camera, focal_length_px=focal_px, calibrated=True)
        return replace(self, camera=camera)


# Approximate real-world heights (metres) used for monocular distance
# estimation. Only classes with a stable height are useful here.
KNOWN_HEIGHTS_M: Mapping[str, float] = {
    "person": 1.70,
    "chair": 0.90,
    "bench": 0.85,
    "bicycle": 1.05,
    "motorbike": 1.10,
    "car": 1.50,
    "bus": 3.00,
    "truck": 3.20,
    "pottedplant": 0.55,
    "diningtable": 0.75,
    "dog": 0.55,
    "cat": 0.30,
    "fire hydrant": 0.75,
    "stop sign": 2.10,
    "parking meter": 1.25,
}
