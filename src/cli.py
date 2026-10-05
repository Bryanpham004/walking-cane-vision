"""Command-line entry point.

    python -m src.cli run --dry-run            # laptop, no hardware
    python -m src.cli run --model models/ssd.onnx --speak
    python -m src.cli run --video data/walk.mp4 --stats results/walk.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .audio import ConsoleAudio, build_audio
from .camera import ReplaySource, VideoFileSource, open_camera
from .config import Settings
from .detector import build_detector
from .haptics import NullHaptics, SerialHaptics
from .pipeline import CaneVisionPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cane-vision",
        description="Camera-based obstacle detection for an assistive walking cane.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the detection loop")
    run.add_argument("--model", help="path to a MobileNet-SSD ONNX model")
    run.add_argument("--video", help="read a recorded video instead of a camera")
    run.add_argument("--camera-index", type=int, default=0)
    run.add_argument("--width", type=int, default=640)
    run.add_argument("--height", type=int, default=480)
    run.add_argument("--fps", type=int, default=15)
    run.add_argument(
        "--focal-px",
        type=float,
        help="calibrated focal length in pixels (see scripts/calibrate_focal.py)",
    )
    run.add_argument("--serial-port", default="/dev/ttyACM0")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="no Arduino, no speech: print alerts to the console",
    )
    run.add_argument("--speak", action="store_true", help="enable offline TTS")
    run.add_argument("--max-frames", type=int, help="stop after N frames")
    run.add_argument("--stats", help="write run statistics to this JSON path")

    selftest = sub.add_parser(
        "selftest", help="run the pipeline on synthetic frames, no hardware needed"
    )
    selftest.add_argument("--frames", type=int, default=40)
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    from dataclasses import replace

    settings = Settings()
    camera = replace(
        settings.camera,
        index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    settings = replace(settings, camera=camera)
    if args.focal_px:
        settings = settings.with_focal_length(args.focal_px)
    return settings


def _run(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)

    if not settings.camera.calibrated and args.model:
        print(
            "warning: focal length is not calibrated, so distances are rough "
            "estimates. Run scripts/calibrate_focal.py and pass --focal-px.",
            file=sys.stderr,
        )

    camera = VideoFileSource(args.video) if args.video else open_camera(settings.camera)
    detector = build_detector(
        settings.camera,
        settings.detection,
        model_path=args.model,
        max_range_m=settings.alerts.max_range_m,
    )
    haptics = (
        NullHaptics(settings.haptics)
        if args.dry_run
        else SerialHaptics(
            __import__("dataclasses").replace(settings.haptics, port=args.serial_port)
        )
    )
    audio = ConsoleAudio() if args.dry_run else build_audio(args.speak)

    pipeline = CaneVisionPipeline(camera, detector, haptics, audio, settings)
    stats = pipeline.run(max_frames=args.max_frames)

    payload = stats.as_dict()
    print(json.dumps(payload, indent=2))
    if args.stats:
        path = Path(args.stats)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


def _selftest(args: argparse.Namespace) -> int:
    """Prove the whole loop works with zero hardware and zero model."""
    import numpy as np

    from .geometry import BoundingBox, Detection

    class _ApproachingStub:
        """Synthetic obstacle closing from 4.0m to 0.5m."""

        def __init__(self, frames: int) -> None:
            self._frames = frames
            self._i = 0

        def detect(self, frame):
            distance = 4.0 - (3.5 * self._i / max(self._frames - 1, 1))
            self._i += 1
            return [
                Detection(
                    label="person",
                    confidence=0.92,
                    box=BoundingBox(280, 100, 360, 400),
                    distance_m=round(distance, 2),
                    distance_is_calibrated=True,
                )
            ]

    blank = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(args.frames)]
    haptics = NullHaptics()
    audio = ConsoleAudio()
    pipeline = CaneVisionPipeline(
        ReplaySource(blank, fps=15.0), _ApproachingStub(args.frames), haptics, audio
    )
    stats = pipeline.run()

    print()
    print(json.dumps(stats.as_dict(), indent=2))
    print(f"\nhaptic commands sent: {len(haptics.commands)}")
    for command in haptics.commands:
        print(f"  {command}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "selftest":
        return _selftest(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
