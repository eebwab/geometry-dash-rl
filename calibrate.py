"""Interactive helper to find BlueStacks capture region and restart button."""

from __future__ import annotations

import argparse
import time

import cv2
import mss

from config import Config
from vision import VisionPipeline


def preview_capture(config: Config, duration_s: float = 30.0) -> None:
    """Show live preprocessed frames with stillness-based death diagnostic."""
    import sys
    import numpy as np

    pipeline = VisionPipeline(config.capture, config.vision)
    still_threshold = config.vision.death_still_threshold
    still_frames_needed = config.vision.death_still_frames
    print(
        f"Capturing region: left={config.capture.left}, top={config.capture.top}, "
        f"width={config.capture.width}, height={config.capture.height}"
    )
    print(
        f"Death trigger: frame_diff < {still_threshold:.1f}  "
        f"for {still_frames_needed} consecutive frames"
    )
    print(f"Preview running for {duration_s:.0f}s — press 'q' to quit early.\n")

    prev_gray = None
    still_count = 0
    start = time.perf_counter()
    frames = 0

    try:
        while time.perf_counter() - start < duration_s:
            raw = pipeline.capture_raw()
            processed = pipeline.preprocess(raw)

            gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
            if prev_gray is not None and prev_gray.shape == gray.shape:
                diff = float(np.mean(np.abs(gray.astype("int16") - prev_gray.astype("int16"))))
                still_count = still_count + 1 if diff < still_threshold else 0
            else:
                diff = 999.0
                still_count = 0
            prev_gray = gray.copy()

            death = still_count >= still_frames_needed
            display = cv2.resize(processed, (420, 420), interpolation=cv2.INTER_NEAREST)
            label = f"DEATH  diff={diff:.2f}" if death else f"ALIVE  diff={diff:.2f}"
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            cv2.imshow("Geometry Dash RL — Preprocessed", display)

            sys.stdout.write(
                f"\r  frame_diff={diff:6.2f}  still_streak={still_count}  "
                f"{'*** DEATH ***' if death else 'alive         '}"
            )
            sys.stdout.flush()

            frames += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print()
        elapsed = time.perf_counter() - start
        fps = frames / elapsed if elapsed > 0 else 0.0
        print(f"Captured {frames} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
        pipeline.close()
        cv2.destroyAllWindows()


def pick_region() -> None:
    """Print platform-specific instructions for finding monitor coordinates."""
    import platform
    if platform.system() == "Darwin":
        print(
            """
To find your BlueStacks capture region on macOS, run:

    python calibrate.py --locate

That starts a live tracker — move your mouse to each corner of the BlueStacks
game area and note the (x, y) values printed. Then:

    Top-left corner  → that is  --left  and  --top
    Bottom-right corner → subtract top-left to get  --width  and  --height

Example:
    Top-left:     x=200, y=50
    Bottom-right: x=1480, y=770
    → --left 200 --top 50 --width 1280 --height 720

Then run:
    python calibrate.py --left 200 --top 50 --width 1280 --height 720

Adjust restart button coords in config.py (ControlConfig.restart_button_x/y)
relative to the top-left of the capture region.
"""
        )
    else:
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


def locate_mouse() -> None:
    """Print live mouse coordinates until the user presses Ctrl-C."""
    from pynput.mouse import Controller as MouseController

    mouse = MouseController()
    print("Move your mouse around — coordinates print live. Press Ctrl-C to stop.\n")
    try:
        import sys
        while True:
            x, y = mouse.position
            sys.stdout.write(f"\r  x={x:<6}  y={y:<6}  ")
            sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate capture region and preview vision pipeline")
    parser.add_argument("--duration", type=float, default=10.0, help="Preview duration in seconds")
    parser.add_argument("--left", type=int, default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--instructions", action="store_true", help="Print setup instructions only")
    parser.add_argument("--locate", action="store_true", help="Print live mouse coordinates (macOS/Linux)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.instructions:
        pick_region()
        return

    if args.locate:
        locate_mouse()
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
