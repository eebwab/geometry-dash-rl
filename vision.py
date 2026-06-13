"""Ultra-fast screen capture and image processing for Geometry Dash."""

from __future__ import annotations

import logging
from collections import deque

import cv2
import mss
import numpy as np

from config import CaptureConfig, GameMode, ModeConfig, VisionConfig

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Captures the BlueStacks region and produces stacked 84x84 frames."""

    def __init__(
        self,
        capture: CaptureConfig,
        vision: VisionConfig,
        mode: ModeConfig | None = None,
    ) -> None:
        self._capture = capture
        self._vision = vision
        self._mode = mode or ModeConfig()
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

        # Stillness detection state.
        self._prev_gray: np.ndarray | None = None
        self._still_count: int = 0
        self._grace_remaining: int = 0  # frames after reset where death is suppressed

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

    def wait_for_motion(
        self,
        motion_threshold: float = 3.0,
        min_motion_frames: int = 3,
        timeout_frames: int = 300,
        sleep_s: float = 0.05,
    ) -> bool:
        """Block until the level is visibly scrolling (frames are changing).

        Polls pairs of consecutive frames and waits until `min_motion_frames`
        in a row all exceed `motion_threshold`. Called after click_restart() so
        the episode never starts on a static loading / attempt-counter screen.

        Returns True if motion detected, False if timed out.
        """
        import time as _time
        consecutive = 0
        prev = None
        for _ in range(timeout_frames):
            raw = self.capture_raw()
            gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
            if prev is not None:
                diff = float(
                    np.mean(np.abs(gray.astype(np.int16) - prev.astype(np.int16)))
                )
                if diff >= motion_threshold:
                    consecutive += 1
                    if consecutive >= min_motion_frames:
                        logger.info("Level motion detected — starting episode")
                        self.reset_death_state(grace_frames=0)
                        return True
                else:
                    consecutive = 0
            prev = gray
            _time.sleep(sleep_s)

        logger.warning("wait_for_motion timed out — starting episode anyway")
        self.reset_death_state(grace_frames=30)
        return False
        """Estimate level completion as a fraction in [0, 1].

        Scans a single row near the top of the raw frame for the rightmost
        bright pixel — that pixel's x-position divided by frame width gives
        the progress fraction. The GD progress bar fills left-to-right.

        Returns:
            Progress fraction in [0.0, 1.0]. Returns 0.0 if nothing detected.
        """
        if frame_bgra is None:
            frame_bgra = self._raw_bgra if self._raw_bgra is not None else self.capture_raw()

        row_idx = min(self._mode.progress_bar_row, frame_bgra.shape[0] - 1)
        row = frame_bgra[row_idx, :, :3]                    # (W, 3) BGR
        brightness = row.max(axis=1)                         # (W,) max channel per pixel
        bright_cols = np.where(brightness >= self._mode.progress_brightness)[0]

        if len(bright_cols) == 0:
            return 0.0

        return float(bright_cols[-1]) / float(frame_bgra.shape[1] - 1)

    def detect_game_mode(
        self,
        frame_bgra: np.ndarray | None = None,
        step: int | None = None,
    ) -> GameMode:
        """Return the current game mode.

        Uses step-count boundaries by default (reliable for a deterministic
        level). Falls back to progress-bar scanning if use_progress_bar=True.
        """
        if self._mode.use_progress_bar:
            progress = self.detect_progress(frame_bgra)
            for start, end in self._mode.ship_ranges:
                if start <= progress < end:
                    return GameMode.SHIP
            for start, end in self._mode.ball_ranges:
                if start <= progress < end:
                    return GameMode.BALL
            return GameMode.CUBE

        # Step-count based detection (default).
        if step is None:
            return GameMode.CUBE

        for start, end in self._mode.ship_step_ranges:
            if start <= step < end:
                return GameMode.SHIP

        for start, end in self._mode.ball_step_ranges:
            if start <= step < end:
                return GameMode.BALL

        return GameMode.CUBE

    def detect_death(
        self,
        frame_bgra: np.ndarray | None = None,
        vision: VisionConfig | None = None,
    ) -> bool:
        """Detect death via frame stillness (primary) or flash / template (fallback)."""
        vision = vision or self._vision

        if frame_bgra is None:
            frame_bgra = self._raw_bgra if self._raw_bgra is not None else self.capture_raw()

        if self._detect_death_stillness(frame_bgra, vision):
            logger.info("Game Over Detected (stillness)")
            return True

        if self._detect_death_flash(frame_bgra, vision):
            logger.info("Game Over Detected (flash)")
            return True

        if self._death_template is not None and self._detect_death_template(frame_bgra, vision):
            logger.info("Game Over Detected (template)")
            return True

        return False

    def reset_death_state(self, grace_frames: int = 60) -> None:
        """Clear stillness counters on env reset.

        grace_frames: steps to suppress death detection after reset so the
        level has time to load before stillness can trigger.
        """
        self._prev_gray = None
        self._still_count = 0
        self._grace_remaining = grace_frames

    def _detect_death_stillness(self, frame_bgra: np.ndarray, vision: VisionConfig) -> bool:
        """Return True once N consecutive frames differ by less than the still threshold.

        Suppressed for the first grace_frames steps after a reset so loading
        screens don't trigger a false death.
        """
        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            self._prev_gray = None
            self._still_count = 0
            return False

        gray = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2GRAY)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray.copy()
            self._still_count = 0
            return False

        diff = float(np.mean(np.abs(gray.astype(np.int16) - self._prev_gray.astype(np.int16))))
        self._prev_gray = gray.copy()

        if diff < vision.death_still_threshold:
            self._still_count += 1
        else:
            self._still_count = 0

        return self._still_count >= vision.death_still_frames

    def _detect_death_flash(self, frame_bgra: np.ndarray, vision: VisionConfig) -> bool:
        """Bright-pixel ratio in center region indicates the death flash."""
        h, w = frame_bgra.shape[:2]
        y0, y1 = int(h * 0.25), int(h * 0.75)
        x0, x1 = int(w * 0.25), int(w * 0.75)
        region = frame_bgra[y0:y1, x0:x1, :3]

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
