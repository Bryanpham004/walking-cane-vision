"""Camera sources behind one interface.

The pipeline never imports OpenCV or picamera2 directly — it takes a
``CameraSource``. That keeps the hardware optional, lets tests drive the
pipeline from a list of arrays, and makes the Pi backend a drop-in swap.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import CameraConfig


@runtime_checkable
class CameraSource(Protocol):
    """Anything that can yield BGR frames."""

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (monotonic_timestamp, BGR frame) until exhausted or stopped."""

    def close(self) -> None: ...


class WebcamSource:
    """OpenCV VideoCapture backend — USB webcams and most laptop cameras."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture = None

    def _open(self):
        import cv2  # imported lazily so the module works without OpenCV

        capture = cv2.VideoCapture(self._config.index)
        if not capture.isOpened():
            raise RuntimeError(
                f"could not open camera index {self._config.index}; "
                "check it is connected and not in use by another process"
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, self._config.fps)
        return capture

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        if self._capture is None:
            self._capture = self._open()
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield time.monotonic(), frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class PiCameraSource:
    """Raspberry Pi camera module via picamera2.

    Preferred on the cane itself: lower latency and no USB bandwidth contention
    with the Arduino link.
    """

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._camera = None

    def _open(self):
        from picamera2 import Picamera2  # lazy: only present on a Pi

        camera = Picamera2()
        camera.configure(
            camera.create_preview_configuration(
                main={"size": (self._config.width, self._config.height),
                      "format": "RGB888"}
            )
        )
        camera.start()
        return camera

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        if self._camera is None:
            self._camera = self._open()
        while True:
            rgb = self._camera.capture_array()
            # picamera2 hands back RGB; the rest of the pipeline assumes BGR.
            yield time.monotonic(), rgb[:, :, ::-1].copy()

    def close(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera = None


class ReplaySource:
    """Fixed frames with synthetic timestamps — for tests and recorded clips."""

    def __init__(self, frames: Sequence[np.ndarray], fps: float = 15.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._frames = list(frames)
        self._interval = 1.0 / fps
        self.closed = False

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        for i, frame in enumerate(self._frames):
            yield i * self._interval, frame

    def close(self) -> None:
        self.closed = True


class VideoFileSource:
    """Read a recorded video file, for offline threshold tuning."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._capture = None

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        import cv2

        self._capture = cv2.VideoCapture(self._path)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video file: {self._path}")
        fps = self._capture.get(cv2.CAP_PROP_FPS) or 15.0
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield index / fps, frame
            index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def open_camera(config: CameraConfig, prefer_picamera: Optional[bool] = None) -> CameraSource:
    """Pick the best available backend.

    Defaults to picamera2 when it imports (i.e. on the cane) and falls back to
    a USB/laptop webcam everywhere else, so the same command works on both.
    """
    if prefer_picamera is None:
        try:
            import picamera2  # noqa: F401

            prefer_picamera = True
        except ImportError:
            prefer_picamera = False
    return PiCameraSource(config) if prefer_picamera else WebcamSource(config)
