"""Gymnasium environment bridging vision pipeline and game controls."""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config, ControlConfig, GameMode, RewardConfig
from controls import GameControls
from vision import VisionPipeline

logger = logging.getLogger(__name__)


class GeometryDashEnv(gym.Env):
    """Raw-pixel Geometry Dash environment for Stereo Madness."""

    metadata = {"render_modes": []}

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config or Config()
        self.vision = VisionPipeline(
            self.config.capture,
            self.config.vision,
            self.config.mode,
        )
        self.controls = GameControls(
            self.config.control,
            mode_config=self.config.mode,
            capture_offset=(self.config.capture.left, self.config.capture.top),
            window_app_name=self.config.control.window_app_name,
        )

        stack = self.config.vision.stack_depth
        size = self.config.vision.frame_size

        # 3 actions: 0 = noop, 1 = tap (cube/ball), 2 = hold (ship thrust)
        self.action_space = spaces.Discrete(3)
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
        self._episode_reward = 0.0
        self._current_mode: GameMode = GameMode.CUBE
        self._best_step: int = 0  # personal best across all episodes

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.controls.click_restart()
        self.vision.reset_death_state()

        obs = None
        for _ in range(self._control_cfg.post_reset_warmup_frames):
            obs = self.vision.reset_stack()

        assert obs is not None
        self._alive = True
        self._step_count = 0
        self._episode_reward = 0.0
        self._current_mode = GameMode.CUBE
        logger.info("Environment reset complete")
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw = self.vision.capture_raw()

        # Detect mode before acting — pass current step for step-count method.
        self._current_mode = self.vision.detect_game_mode(raw, step=self._step_count)
        self.controls.do_action(int(action), self._current_mode)

        terminated = self.vision.detect_death(raw)
        obs = self.vision.push_frame(self.vision.preprocess(raw))

        if terminated:
            reward = self._reward_cfg.death_penalty
            self._alive = False
        else:
            reward = self._reward_cfg.survive_reward
            # Small bonus for reaching a new personal best step — encourages
            # the agent to push further rather than playing it safe early.
            if self._step_count > self._best_step:
                reward += 0.5
                self._best_step = self._step_count

        self._step_count += 1
        self._episode_reward += reward
        truncated = False
        info: dict[str, Any] = {
            "step": self._step_count,
            "alive": self._alive,
            "mode": self._current_mode.value,
        }

        if terminated:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._step_count,
            }
            logger.info(
                "Episode done — steps: %d  total reward: %.1f  last mode: %s",
                self._step_count,
                self._episode_reward,
                self._current_mode.value,
            )

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.vision.close()
        super().close()
