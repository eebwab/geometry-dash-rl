"""Low-latency keyboard and mouse actuation via pydirectinput."""

from __future__ import annotations

import logging
import time

import pydirectinput

from config import ControlConfig

logger = logging.getLogger(__name__)

# Disable pydirectinput's built-in pause for minimum latency.
pydirectinput.PAUSE = 0


class GameControls:
    """Emulates physical keystrokes and clicks for BlueStacks."""

    def __init__(self, config: ControlConfig, capture_offset: tuple[int, int] = (0, 0)) -> None:
        self._config = config
        self._offset_x, self._offset_y = capture_offset

    def jump(self) -> None:
        pydirectinput.press(self._config.jump_key)

    def click_restart(self) -> None:
        """Click the in-game restart button using screen-absolute coordinates."""
        x = self._offset_x + self._config.restart_button_x
        y = self._offset_y + self._config.restart_button_y
        pydirectinput.click(x, y)
        logger.info("Clicked restart at (%d, %d)", x, y)
        time.sleep(self._config.restart_click_delay_s)

    def do_action(self, action: int) -> None:
        """Execute RL action: 0 = noop, 1 = jump."""
        if action == 1:
            self.jump()
