"""Gymnasium environment bridging vision pipeline and game controls."""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config, ControlConfig, RewardConfig
from controls import GameControls
from vision import VisionPipeline

logger = logging.getLogger(__name__)


class GeometryDashEnv(gym.Env):
    """Raw-pixel Geometry Dash environment for Stereo Madness."""

    metadata = {"render_modes": []}

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config or Config()
        self.vision = VisionPipeline(self.config.capture, self.config.vision)
        self.controls = GameControls(
            self.config.control,
            capture_offset=(self.config.capture.left, self.config.capture.top),
        )

        stack = self.config.vision.stack_depth
        size = self.config.vision.frame_size

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(stack, size, size),
            dtype=np.uint8,
        )

        self._reward_cfg: RewardConfig = self.config.reward
        self._control_cfg: ControlConfig = self.config.control
        self._alive = False
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.controls.click_restart()

        # Warmup frames let the level load before the agent acts.
        obs = None
        for _ in range(self._control_cfg.post_reset_warmup_frames):
            obs = self.vision.reset_stack()

        assert obs is not None
        self._alive = True
        self._step_count = 0
        logger.info("Environment reset complete")
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.controls.do_action(int(action))

        raw = self.vision.capture_raw()
        terminated = self.vision.detect_death(raw)
        obs = self.vision.push_frame(self.vision.preprocess(raw))

        if terminated:
            reward = self._reward_cfg.death_penalty
            self._alive = False
        else:
            reward = self._reward_cfg.survive_reward

        self._step_count += 1
        truncated = False
        info: dict[str, Any] = {"step": self._step_count, "alive": self._alive}

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.vision.close()
        super().close()
