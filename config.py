"""Runtime configuration for capture region, rewards, and training."""

from dataclasses import dataclass, field


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
    # Binary threshold applied after grayscale resize (0-255).
    threshold: int = 127
    # --- Stillness-based death detection (primary, works with auto-retry off) ---
    # Mean absolute pixel difference below this value = frames are frozen = dead.
    death_still_threshold: float = 1.5
    # How many consecutive still frames required to confirm death (avoids brief pauses).
    death_still_frames: int = 8

    # --- Flash-based detection (fallback, disabled by default) ---
    death_flash_brightness: int = 200
    death_flash_pixel_ratio: float = 0.95   # effectively disabled (set < 1.0 to re-enable)
    # Optional template path for "Game Over" text (cv2.matchTemplate).
    death_template_path: str | None = None
    death_template_threshold: float = 0.75


@dataclass
class RewardConfig:
    survive_reward: float = 1.0
    death_penalty: float = -100.0


@dataclass
class ControlConfig:
    jump_key: str = "space"
    # Screen coordinates of the in-game restart button (relative to capture region).
    restart_button_x: int = 640
    restart_button_y: int = 400
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
    reward: RewardConfig = field(default_factory=RewardConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
