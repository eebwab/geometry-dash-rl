"""Ultra-fast screen capture and image processing for Geometry Dash."""

from __future__ import annotations

import logging
from collections import deque

import cv2
import mss
import numpy as np

from config import CaptureConfig, VisionConfig

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Captures the BlueStacks region and produces stacked 84x84 frames."""

    def __init__(
        self,
        capture: CaptureConfig,
        vision: VisionConfig,
    ) -> None:
        self._capture = capture
        self._vision = vision
        self._sct = mss.mss()
        self._monitor = capture.monitor

        self._frame_size = vision.frame_size
        self._stack_depth = vision.stack_depth
        self._threshold = vision.threshold

        # Reusable buffers to avoid per-frame allocations.
        self._raw_bgra: np.ndarray | None = None
        self._gray_full: np.ndarray | None = None
        self._processed: np.ndarray = np.zeros(
            (self._frame_size, self._frame_size),
            dtype=np.uint8,
        )

        blank = np.zeros((self._frame_size, self._frame_size), dtype=np.uint8)
        self._frame_stack: deque[np.ndarray] = deque(
            [blank.copy() for _ in range(self._stack_depth)],
            maxlen=self._stack_depth,
        )

        self._death_template: np.ndarray | None = None
        if vision.death_template_path:
            template = cv2.imread(vision.death_template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self._death_template = template
                logger.info("Loaded death template from %s", vision.death_template_path)
            else:
                logger.warning(
                    "Death template not found at %s; using flash detection only",
                    vision.death_template_path,
                )

    def capture_raw(self) -> np.ndarray:
        """Capture BGRA frame from the configured monitor region.

        Returns:
            (Height, Width, 4) uint8 BGRA array.
        """
        shot = self._sct.grab(self._monitor)
        # mss returns a ctypes buffer; copy once into a contiguous numpy array.
        frame = np.asarray(shot, dtype=np.uint8)
        self._raw_bgra = frame
        return frame

    def preprocess(self, frame_bgra: np.ndarray | None = None) -> np.ndarray:
        """Convert raw capture to a single thresholded 84x84 grayscale frame.

        Shape flow:
            (H, W, 4) BGRA -> (H, W) gray -> (84, 84) resized -> (84, 84) binary
        """
        if frame_bgra is None:
            frame_bgra = self.capture_raw()

        if self._gray_full is None or self._gray_full.shape[:2] != frame_bgra.shape[:2]:
            self._gray_full = np.empty(frame_bgra.shape[:2], dtype=np.uint8)

        # BGRA -> grayscale in-place buffer.
        cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2GRAY, dst=self._gray_full)

        cv2.resize(
            self._gray_full,
            (self._frame_size, self._frame_size),
            dst=self._processed,
            interpolation=cv2.INTER_AREA,
        )

        # Single-threshold binarization keeps replay buffer memory small.
        cv2.threshold(
            self._processed,
            self._threshold,
            255,
            cv2.THRESH_BINARY,
            dst=self._processed,
        )
        return self._processed

    def push_frame(self, frame: np.ndarray | None = None) -> np.ndarray:
        """Preprocess (if needed), push into stack, return stacked observation.

        Returns:
            (stack_depth, 84, 84) uint8 — channels-first stack for the RL agent.
        """
        if frame is None:
            frame = self.preprocess()
        else:
            # Caller supplied a preprocessed (84, 84) frame.
            assert frame.shape == (self._frame_size, self._frame_size)

        self._frame_stack.append(frame)

        # Stack along channel axis: (4, 84, 84).
        return np.stack(list(self._frame_stack), axis=0)

    def reset_stack(self, frame: np.ndarray | None = None) -> np.ndarray:
        """Clear frame history and seed with identical frames (cold start)."""
        if frame is None:
            frame = self.preprocess()

        self._frame_stack.clear()
        for _ in range(self._stack_depth):
            self._frame_stack.append(frame.copy())

        return np.stack(list(self._frame_stack), axis=0)

    def detect_death(
        self,
        frame_bgra: np.ndarray | None = None,
        vision: VisionConfig | None = None,
    ) -> bool:
        """Detect death via screen flash and/or template matching."""
        vision = vision or self._vision

        if frame_bgra is None:
            frame_bgra = self._raw_bgra if self._raw_bgra is not None else self.capture_raw()

        if self._detect_death_flash(frame_bgra, vision):
            logger.info("Game Over Detected (flash)")
            return True

        if self._death_template is not None and self._detect_death_template(frame_bgra, vision):
            logger.info("Game Over Detected (template)")
            return True

        return False

    def _detect_death_flash(self, frame_bgra: np.ndarray, vision: VisionConfig) -> bool:
        """Bright-pixel ratio in center region indicates the death flash."""
        h, w = frame_bgra.shape[:2]
        y0, y1 = int(h * 0.25), int(h * 0.75)
        x0, x1 = int(w * 0.25), int(w * 0.75)
        region = frame_bgra[y0:y1, x0:x1, :3]

        # Use max channel brightness per pixel.
        brightness = region.max(axis=2)
        bright_ratio = (brightness >= vision.death_flash_brightness).mean()
        return bright_ratio >= vision.death_flash_pixel_ratio

    def _detect_death_template(self, frame_bgra: np.ndarray, vision: VisionConfig) -> bool:
        assert self._death_template is not None

        gray = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2GRAY)
        result = cv2.matchTemplate(gray, self._death_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= vision.death_template_threshold

    def close(self) -> None:
        self._sct.close()
