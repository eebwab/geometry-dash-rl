"""Low-latency keyboard and mouse actuation via pynput (macOS/Linux/Windows)."""

from __future__ import annotations

import logging
import platform
import subprocess
import time

from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

from config import ControlConfig

logger = logging.getLogger(__name__)

_SPECIAL_KEYS: dict[str, Key] = {
    "space": Key.space,
    "enter": Key.enter,
    "up": Key.up,
    "left": Key.left,
    "right": Key.right,
}


def _focus_app(app_name: str) -> None:
    """Bring an application to the foreground (macOS only, best-effort)."""
    if platform.system() != "Darwin":
        return
    try:
        subprocess.run(
            ["osascript", "-e", f'tell application "{app_name}" to activate'],
            check=False,
            capture_output=True,
            timeout=2.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not focus %s: %s", app_name, exc)


class GameControls:
    """Emulates physical keystrokes and mouse clicks for BlueStacks on macOS.

    Requires macOS Accessibility permission for the running process.
    System Settings → Privacy & Security → Accessibility → enable your terminal.
    """

    def __init__(
        self,
        config: ControlConfig,
        capture_offset: tuple[int, int] = (0, 0),
        window_app_name: str = "BlueStacks",
    ) -> None:
        self._config = config
        self._offset_x, self._offset_y = capture_offset
        self._keyboard = KeyboardController()
        self._mouse = MouseController()
        self._jump_key = _SPECIAL_KEYS.get(config.jump_key, KeyCode.from_char(config.jump_key))
        self._app_name = window_app_name

    def focus(self) -> None:
        """Bring the game window to the foreground before sending input."""
        _focus_app(self._app_name)
        time.sleep(0.15)

    def jump(self) -> None:
        """Tap the jump key — press and immediately release."""
        self._keyboard.press(self._jump_key)
        self._keyboard.release(self._jump_key)

    def click_restart(self) -> None:
        """Focus the game window, then click the in-game restart button."""
        self.focus()
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
