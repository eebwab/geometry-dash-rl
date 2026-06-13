"""Interactive helper to find BlueStacks capture region and restart button."""

from __future__ import annotations

import argparse
import time

import cv2
import mss

from config import Config
from vision import VisionPipeline


def preview_capture(config: Config, duration_s: float = 10.0) -> None:
    """Show live preprocessed frames and print FPS."""
    pipeline = VisionPipeline(config.capture, config.vision)
    print(
        f"Capturing region: left={config.capture.left}, top={config.capture.top}, "
        f"width={config.capture.width}, height={config.capture.height}"
    )
    print(f"Preview running for {duration_s:.0f}s — press 'q' to quit early.")

    start = time.perf_counter()
    frames = 0

    try:
        while time.perf_counter() - start < duration_s:
            raw = pipeline.capture_raw()
            processed = pipeline.preprocess(raw)
            death = pipeline.detect_death(raw)

            display = cv2.resize(processed, (420, 420), interpolation=cv2.INTER_NEAREST)
            label = "DEATH" if death else "ALIVE"
            cv2.putText(
                display,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                255,
                2,
            )
            cv2.imshow("Geometry Dash RL — Preprocessed", display)

            frames += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        elapsed = time.perf_counter() - start
        fps = frames / elapsed if elapsed > 0 else 0.0
        print(f"Captured {frames} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
        pipeline.close()
        cv2.destroyAllWindows()


def pick_region() -> None:
    """Print instructions for finding monitor coordinates on Windows."""
    print(
        """
To find your BlueStacks capture region on Windows:

1. Open BlueStacks in windowed mode (not fullscreen).
2. Run this from PowerShell while hovering over the top-left corner of the game area:
     Add-Type -AssemblyName System.Windows.Forms
     [System.Windows.Forms.Cursor]::Position

3. Note X (left) and Y (top), then measure width/height of the game viewport.
4. Pass values to train.py:
     python train.py --left 100 --top 50 --width 1280 --height 720

Adjust restart button in config.py (control.restart_button_x/y) relative to the capture region.
"""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate capture region and preview vision pipeline")
    parser.add_argument("--duration", type=float, default=10.0, help="Preview duration in seconds")
    parser.add_argument("--left", type=int, default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--instructions", action="store_true", help="Print setup instructions only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.instructions:
        pick_region()
        return

    config = Config()
    cap = config.capture
    if args.left is not None:
        cap.left = args.left
    if args.top is not None:
        cap.top = args.top
    if args.width is not None:
        cap.width = args.width
    if args.height is not None:
        cap.height = args.height

    preview_capture(config, duration_s=args.duration)


if __name__ == "__main__":
    main()
