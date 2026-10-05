"""Obstacle detectors behind one interface.

Two implementations ship here:

* ``SsdObstacleDetector`` — MobileNet-SSD via onnxruntime. Labels obstacles,
  which unlocks known-height distance estimation. Needs a model file.
* ``MotionObstacleDetector`` — frame differencing plus contours, pure OpenCV.
  No model, no download, no GPU. It cannot label what it sees, so distances
  fall back to the ground-contact heuristic and are marked uncalibrated.

The fallback exists so the repo runs the moment it is cloned; the SSD path is
what you would actually walk around with.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import KNOWN_HEIGHTS_M, CameraConfig, DetectionConfig
from .geometry import (
    BoundingBox,
    Detection,
    estimate_distance,
    horizon_distance_heuristic,
    non_max_suppression,
)

# MobileNet-SSD (VOC) class order.
VOC_LABELS: tuple[str, ...] = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


@runtime_checkable
class ObstacleDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def _attach_distance(
    detections: Sequence[Detection],
    frame_shape: tuple[int, ...],
    camera: CameraConfig,
    max_range_m: float,
) -> list[Detection]:
    """Fill in ``distance_m`` using the best cue available per detection."""
    height, width = frame_shape[0], frame_shape[1]
    focal = camera.scaled_focal(width)
    out: list[Detection] = []
    for det in detections:
        known_height = KNOWN_HEIGHTS_M.get(det.label)
        if known_height is not None:
            distance = estimate_distance(det.box.height, known_height, focal)
            calibrated = camera.calibrated
        else:
            distance = horizon_distance_heuristic(det.box.y2, height, max_range_m)
            calibrated = False
        out.append(
            Detection(
                label=det.label,
                confidence=det.confidence,
                box=det.box,
                distance_m=round(distance, 2),
                distance_is_calibrated=calibrated,
            )
        )
    return out


class SsdObstacleDetector:
    """MobileNet-SSD object detection through onnxruntime."""

    INPUT_SIZE = (300, 300)
    MEAN = 127.5
    SCALE = 1 / 127.5

    def __init__(
        self,
        model_path: str,
        camera: CameraConfig,
        config: DetectionConfig,
        max_range_m: float = 4.0,
        labels: Sequence[str] = VOC_LABELS,
    ) -> None:
        self._camera = camera
        self._config = config
        self._max_range_m = max_range_m
        self._labels = tuple(labels)
        self._session = self._load(model_path)
        self._input_name = self._session.get_inputs()[0].name

    @staticmethod
    def _load(model_path: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "onnxruntime is required for SsdObstacleDetector; "
                "install it or use MotionObstacleDetector instead"
            ) from exc
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        resized = cv2.resize(frame, self.INPUT_SIZE)
        blob = (resized.astype(np.float32) - self.MEAN) * self.SCALE
        return np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        raw = self._session.run(None, {self._input_name: self._preprocess(frame)})[0]
        # SSD output is (1, 1, N, 7): [_, class_id, score, x1, y1, x2, y2]
        rows = np.reshape(raw, (-1, 7))
        min_area = self._config.min_area_fraction * width * height
        found: list[Detection] = []
        for row in rows:
            score = float(row[2])
            if score < self._config.min_confidence:
                continue
            class_id = int(row[1])
            label = (
                self._labels[class_id]
                if 0 <= class_id < len(self._labels)
                else f"class_{class_id}"
            )
            if label == "background":
                continue
            x1 = int(max(row[3] * width, 0))
            y1 = int(max(row[4] * height, 0))
            x2 = int(min(row[5] * width, width - 1))
            y2 = int(min(row[6] * height, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            box = BoundingBox(x1, y1, x2, y2)
            if box.area < min_area:
                continue
            found.append(Detection(label=label, confidence=score, box=box))
        kept = non_max_suppression(found, self._config.nms_iou)
        return _attach_distance(kept, frame.shape, self._camera, self._max_range_m)


class MotionObstacleDetector:
    """Model-free detector: background subtraction plus contour extraction.

    Good enough to prove the alert pipeline end to end on a laptop, and it is
    genuinely useful on the cane for the thing a sensor array misses most —
    something moving into your path. It cannot classify, so every distance it
    reports is an ordering cue rather than a measurement.
    """

    def __init__(
        self,
        config: DetectionConfig,
        max_range_m: float = 4.0,
        history: int = 200,
    ) -> None:
        self._config = config
        self._max_range_m = max_range_m
        self._history = history
        self._subtractor = None

    def _ensure_subtractor(self):
        if self._subtractor is None:
            import cv2

            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self._history,
                varThreshold=float(self._config.motion_threshold),
                detectShadows=False,
            )
        return self._subtractor

    def detect(self, frame: np.ndarray) -> list[Detection]:
        import cv2

        height, width = frame.shape[:2]
        mask = self._ensure_subtractor().apply(frame)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self._config.min_area_fraction * width * height
        found: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            # Confidence is a blob-size proxy, capped so it never outranks a
            # real classifier score.
            confidence = min(0.5 + (area / (width * height)), 0.95)
            found.append(
                Detection(
                    label="obstacle",
                    confidence=float(confidence),
                    box=BoundingBox(x, y, x + w, y + h),
                )
            )
        kept = non_max_suppression(found, self._config.nms_iou)
        return _attach_distance(kept, frame.shape, CameraConfig(), self._max_range_m)


def build_detector(
    camera: CameraConfig,
    config: DetectionConfig,
    model_path: Optional[str] = None,
    max_range_m: float = 4.0,
) -> ObstacleDetector:
    """Use the SSD model when one is supplied, else the model-free fallback."""
    if model_path:
        return SsdObstacleDetector(model_path, camera, config, max_range_m)
    return MotionObstacleDetector(config, max_range_m)
