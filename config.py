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
    death_still_threshold: float = 1.5
    death_still_frames: int = 8

    # --- Flash-based detection (fallback, effectively disabled) ---
    death_flash_brightness: int = 200
    death_flash_pixel_ratio: float = 0.95
    death_template_path: str | None = None
    death_template_threshold: float = 0.75


@dataclass
class ModeConfig:
    """Game mode detection via progress bar scanning.

    Stereo Madness segment map (approximate progress fractions):
        0.00 – 0.18  cube
        0.18 – 0.47  ship   ← hold space to fly up
        0.47 – 0.55  ball
        0.55 – 0.60  ufo    (treated as cube — tap mechanic)
        0.60 – 0.73  cube
        0.73 – 0.85  ship
        0.85 – 1.00  cube
    """

    # Row (pixels from top of raw capture frame) where progress bar lives.
    progress_bar_row: int = 12
    # Min brightness to count a pixel as "filled" in the progress bar.
    progress_brightness: int = 160

    # How long to hold space for one ship-mode action step (seconds).
    ship_hold_s: float = 0.055

    # Ship ranges as (start_frac, end_frac) pairs — Stereo Madness verified.
    ship_ranges: tuple[tuple[float, float], ...] = ((0.29, 0.48), (0.85, 1.00))
    # Ball ranges — gravity-flip: tap mechanic same as cube.
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
