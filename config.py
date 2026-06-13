"""Runtime configuration for capture region, rewards, and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GameMode(Enum):
    CUBE = "cube"
    SHIP = "ship"
    BALL = "ball"
    UFO = "ufo"


@dataclass
class CaptureConfig:
    """BlueStacks window bounding box in screen coordinates (pixels)."""

    left: int = 260
    top: int = 191
    width: int = 960
    height: int = 537

    @property
    def monitor(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class VisionConfig:
    """Image processing and death-detection settings."""

    frame_size: int = 84
    stack_depth: int = 4
    threshold: int = 127

    # --- Stillness-based death detection ---
    death_still_threshold: float = 20.0
    # 20 frames @ 30fps = 0.67s — real deaths stay frozen indefinitely,
    # safe to require more frames to eliminate false positives on low-motion sections.
    death_still_frames: int = 20

    # --- Flash-based detection (fallback, effectively disabled) ---
    death_flash_brightness: int = 200
    death_flash_pixel_ratio: float = 0.95
    death_template_path: str | None = None
    death_template_threshold: float = 0.75


@dataclass
class ModeConfig:
    """Game mode detection for Stereo Madness.

    Primary method: step count (reliable, no visual dependency).
    The level is deterministic and plays at a fixed internal speed, so step
    boundaries are stable once calibrated.

    Stereo Madness segment map (user-verified % → steps at ~30 steps/sec):
        0 – 29%   cube
        29 – 48%  ship  ← ship_step_ranges[0]
        48 – 85%  cube / ball / ufo
        85 – 100% ship  ← ship_step_ranges[1]

    Default step values assume ~30 game-steps per second. Run calibrate.py
    --mode-calibrate while watching the level to tune these if needed.
    """

    # How long to hold space for one ship-mode action step (seconds).
    ship_hold_s: float = 0.055

    # Step-count boundaries for ship mode: (start_step, end_step).
    # Calibrated for Stereo Madness at ~30 steps/sec (user-verified).
    ship_step_ranges: tuple[tuple[int, int], ...] = ((680, 1120), (2120, 99999))
    # Step-count boundaries for ball mode (tap mechanic — same as cube).
    ball_step_ranges: tuple[tuple[int, int], ...] = ((1120, 1400),)

    # --- Progress bar (kept for calibrate.py display only) ---
    progress_bar_row: int = 12
    progress_brightness: int = 160
    # Fraction boundaries — only used if use_progress_bar is True.
    use_progress_bar: bool = False
    ship_ranges: tuple[tuple[float, float], ...] = ((0.29, 0.48), (0.85, 1.00))
    ball_ranges: tuple[tuple[float, float], ...] = ((0.48, 0.60),)


@dataclass
class RewardConfig:
    survive_reward: float = 1.0
    death_penalty: float = -100.0


@dataclass
class ControlConfig:
    jump_key: str = "space"
    window_app_name: str = "BlueStacks"
    restart_button_x: int = 302
    restart_button_y: int = 458
    restart_click_delay_s: float = 0.5
    post_reset_warmup_frames: int = 30


@dataclass
class TrainConfig:
    total_timesteps: int = 500_000
    checkpoint_freq: int = 10_000
    log_dir: str = "runs"
    model_dir: str = "models"
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    batch_size: int = 32
    gamma: float = 0.99
    train_freq: int = 4
    target_update_interval: int = 1000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    seed: int = 42


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    mode: ModeConfig = field(default_factory=ModeConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
