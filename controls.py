"""Low-latency keyboard and mouse actuation via pynput (macOS/Linux/Windows)."""

from __future__ import annotations

import logging
import time

from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

from config import ControlConfig

logger = logging.getLogger(__name__)

# Map common key name strings to pynput Key objects where needed.
_SPECIAL_KEYS: dict[str, Key] = {
    "space": Key.space,
    "enter": Key.enter,
    "up": Key.up,
    "left": Key.left,
    "right": Key.right,
}


class GameControls:
    """Emulates physical keystrokes and mouse clicks for BlueStacks on macOS.

    Requires macOS Accessibility permission for the running process
    (Terminal, Cursor, or whichever app launches training).
    System Settings → Privacy & Security → Accessibility → enable your terminal.
    """

    def __init__(self, config: ControlConfig, capture_offset: tuple[int, int] = (0, 0)) -> None:
        self._config = config
        self._offset_x, self._offset_y = capture_offset
        self._keyboard = KeyboardController()
        self._mouse = MouseController()
        self._jump_key = _SPECIAL_KEYS.get(config.jump_key, KeyCode.from_char(config.jump_key))

    def jump(self) -> None:
        """Tap the jump key — press and immediately release for minimum hold time."""
        self._keyboard.press(self._jump_key)
        self._keyboard.release(self._jump_key)

    def click_restart(self) -> None:
        """Click the in-game restart button at screen-absolute coordinates."""
        x = self._offset_x + self._config.restart_button_x
        y = self._offset_y + self._config.restart_button_y
        self._mouse.position = (x, y)
        self._mouse.click(Button.left, 1)
        logger.info("Clicked restart at (%d, %d)", x, y)
        time.sleep(self._config.restart_click_delay_s)

    def do_action(self, action: int) -> None:
        """Execute RL action: 0 = noop, 1 = jump."""
        if action == 1:
            self.jump()
