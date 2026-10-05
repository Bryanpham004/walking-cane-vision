#!/usr/bin/env python3
"""Measure your camera's focal length in pixels.

Every distance the cane reports depends on this one number, so it is worth the
two minutes. Uncalibrated, the default is a generic 60-degree-FOV guess and
readings can be off by 30% or more.

Procedure:
  1. Stand a known object a measured distance from the camera. A person at
     exactly 3.00 m works well; so does a 0.90 m chair at 2.00 m.
  2. Run this script and drag a box around the object's full height.
  3. Press ENTER to accept, 'r' to redraw, 'q' to quit.
  4. Pass the printed value to the pipeline: --focal-px <value>

Usage:
  python scripts/calibrate_focal.py --distance 3.0 --height 1.70
  python scripts/calibrate_focal.py --distance 3.0 --height 1.70 --box-height 312
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import estimate_distance, estimate_focal_length  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--distance", type=float, required=True,
        help="measured distance from camera to object, in metres",
    )
    parser.add_argument(
        "--height", type=float, required=True,
        help="true height of the object, in metres",
    )
    parser.add_argument(
        "--box-height", type=int,
        help="skip the camera entirely and use this pixel height",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height-px", type=int, default=480)
    return parser.parse_args()


def _report(box_height_px: int, real_height_m: float, distance_m: float) -> float:
    focal = estimate_focal_length(box_height_px, real_height_m, distance_m)
    print(f"\n  box height      : {box_height_px} px")
    print(f"  object height   : {real_height_m} m")
    print(f"  known distance  : {distance_m} m")
    print(f"\n  focal length    : {focal:.1f} px")
    # Sanity check: feeding the focal length back must reproduce the distance.
    check = estimate_distance(box_height_px, real_height_m, focal)
    print(f"  round-trip check: {check:.2f} m (should equal {distance_m} m)")
    print(f"\nUse it:  python -m src.cli run --focal-px {focal:.1f}\n")
    return focal


def _measure_interactively(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError:
        print(
            "OpenCV is needed for interactive calibration. Either install it\n"
            "(pip install opencv-python) or measure the pixel height yourself\n"
            "and pass --box-height.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        print(f"could not open camera {args.camera_index}", file=sys.stderr)
        raise SystemExit(1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height_px)

    print("Position the object, then press SPACE to freeze the frame.")
    frozen = None
    try:
        while frozen is None:
            ok, frame = capture.read()
            if not ok:
                print("camera read failed", file=sys.stderr)
                raise SystemExit(1)
            cv2.putText(
                frame, "SPACE = freeze, q = quit", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("calibrate", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                frozen = frame.copy()
            elif key in (ord("q"), 27):
                raise SystemExit(0)

        print("Drag a box around the object's FULL height, then press ENTER.")
        x, y, w, h = cv2.selectROI("calibrate", frozen, showCrosshair=True)
        if h <= 0:
            print("no box selected", file=sys.stderr)
            raise SystemExit(1)
        return int(h)
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> int:
    args = _parse_args()
    if args.distance <= 0 or args.height <= 0:
        print("distance and height must both be positive", file=sys.stderr)
        return 2

    box_height = args.box_height or _measure_interactively(args)
    _report(box_height, args.height, args.distance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
