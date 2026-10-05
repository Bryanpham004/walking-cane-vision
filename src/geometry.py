"""Detection primitives, frame zones, and monocular distance estimation.

Distance from a single camera relies on the pinhole model: an object of known
real height H projects to h pixels at distance Z, with Z = f * H / h. That
makes the estimate only as good as the focal-length calibration and the
assumed height, so every result carries its own uncertainty flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence


class Zone(str, Enum):
    """Horizontal thirds of the frame, as spoken to the user."""

    LEFT = "left"
    CENTER = "ahead"
    RIGHT = "right"


class Urgency(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass(frozen=True)
class BoundingBox:
    """Pixel box in frame coordinates, (x1, y1) top-left inclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate box: {self}")

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over union, used for non-max suppression and tracking."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Detection:
    """One obstacle in one frame."""

    label: str
    confidence: float
    box: BoundingBox
    distance_m: Optional[float] = None
    # False when distance came from a frame-height heuristic rather than a
    # known object height — treat such readings as ordering hints only.
    distance_is_calibrated: bool = False

    def zone(self, frame_width: int, center_fraction: float = 0.4) -> Zone:
        """Which third of the frame the obstacle occupies.

        The centre band is widened by ``center_fraction`` because obstacles
        straight ahead matter far more than peripheral ones.
        """
        if frame_width <= 0:
            raise ValueError("frame width must be positive")
        half = center_fraction / 2.0
        rel = self.box.center_x / frame_width
        if rel < 0.5 - half:
            return Zone.LEFT
        if rel > 0.5 + half:
            return Zone.RIGHT
        return Zone.CENTER


def estimate_distance(
    box_height_px: int,
    real_height_m: float,
    focal_length_px: float,
) -> float:
    """Pinhole distance estimate in metres: Z = f * H / h."""
    if box_height_px <= 0:
        raise ValueError("box height must be positive")
    if real_height_m <= 0:
        raise ValueError("real height must be positive")
    if focal_length_px <= 0:
        raise ValueError("focal length must be positive")
    return (focal_length_px * real_height_m) / box_height_px


def estimate_focal_length(
    box_height_px: int,
    real_height_m: float,
    known_distance_m: float,
) -> float:
    """Invert the pinhole model to calibrate focal length: f = h * Z / H."""
    if known_distance_m <= 0:
        raise ValueError("known distance must be positive")
    if real_height_m <= 0:
        raise ValueError("real height must be positive")
    if box_height_px <= 0:
        raise ValueError("box height must be positive")
    return (box_height_px * known_distance_m) / real_height_m


def time_to_contact(
    distance_m: float,
    previous_distance_m: Optional[float],
    dt_s: float,
) -> Optional[float]:
    """Seconds until contact at the current closing speed.

    Returns None when the obstacle is static or receding, since a negative or
    infinite TTC carries no urgency.
    """
    if previous_distance_m is None or dt_s <= 0:
        return None
    closing_speed = (previous_distance_m - distance_m) / dt_s
    if closing_speed <= 1e-3:
        return None
    return distance_m / closing_speed


def non_max_suppression(
    detections: Sequence[Detection], iou_threshold: float
) -> list[Detection]:
    """Drop overlapping boxes, keeping the most confident of each cluster."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(det.box.iou(k.box) < iou_threshold for k in kept):
            kept.append(det)
    return kept


def horizon_distance_heuristic(
    box_bottom_y: int, frame_height: int, max_range_m: float
) -> float:
    """Fallback depth cue for unknown objects, from where they meet the ground.

    With a forward-facing camera, the lower in the frame an object's base sits,
    the nearer it is. This is a monotonic ordering cue, not a measurement, so
    callers must mark the result uncalibrated.
    """
    if frame_height <= 0:
        raise ValueError("frame height must be positive")
    rel = min(max(box_bottom_y / frame_height, 0.0), 1.0)
    # rel = 1.0 (bottom of frame) -> nearest; rel = 0.5 -> max range.
    return max_range_m * max(1.0 - rel, 0.0) * 2.0 if rel > 0.5 else max_range_m


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def degrees_off_axis(center_x: float, frame_width: int, hfov_deg: float = 60.0) -> float:
    """Bearing of an obstacle relative to straight ahead, for spoken guidance."""
    if frame_width <= 0:
        raise ValueError("frame width must be positive")
    rel = (center_x / frame_width) - 0.5
    return math.degrees(math.atan(2 * rel * math.tan(math.radians(hfov_deg / 2))))
