"""The only module that touches the real mouse and keyboard."""

import sys

try:
    import pyautogui
except ImportError:
    raise ImportError("Не установлена библиотека pyautogui. Выполни: pip install pyautogui") from None


class EmergencyStop(Exception):
    """Raised when the pyautogui failsafe triggers."""


def default_units_per_notch():
    if sys.platform.startswith("win"):
        return 120
    if sys.platform == "darwin":
        return 3
    return 1


class Controller:
    def __init__(self, failsafe=True, pause=0.0, edge_margin=2, min_move_px=2,
                 scroll_units_per_notch="auto", scroll_min_chunk="auto"):
        pyautogui.FAILSAFE = bool(failsafe)
        pyautogui.PAUSE = float(pause)

        self.screen_w, self.screen_h = pyautogui.size()
        self.enabled = False
        self._edge_margin = edge_margin
        self._min_move = min_move_px

        if scroll_units_per_notch == "auto":
            scroll_units_per_notch = default_units_per_notch()
        self._units_per_notch = float(scroll_units_per_notch)
        if scroll_min_chunk == "auto":
            scroll_min_chunk = max(1, int(self._units_per_notch // 3))
        self._scroll_min_chunk = max(1, int(scroll_min_chunk))
        self._scroll_accum = 0.0

        self._down = set()
        self._keys_down = set()

    @property
    def screen_size(self):
        return self.screen_w, self.screen_h

    @property
    def buttons_down(self):
        return set(self._down)

    @property
    def keys_down(self):
        return set(self._keys_down)

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if self.enabled and not enabled:
            self.release_all()
        self.enabled = enabled
        return enabled

    def toggle(self):
        return self.set_enabled(not self.enabled)

    @staticmethod
    def _call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except pyautogui.FailSafeException as error:
            raise EmergencyStop("Сработала аварийная защита (курсор в углу экрана).") from error

    def move_to(self, x, y):
        if not self.enabled:
            return False

        margin = self._edge_margin
        x = int(round(min(max(x, margin), self.screen_w - 1 - margin)))
        y = int(round(min(max(y, margin), self.screen_h - 1 - margin)))

        current_x, current_y = pyautogui.position()
        if abs(x - current_x) < self._min_move and abs(y - current_y) < self._min_move:
            return False

        self._call(pyautogui.moveTo, x, y)
        return True

    def position(self):
        x, y = pyautogui.position()
        return int(x), int(y)

    def move_exact(self, x, y):
        if not self.enabled:
            return False
        margin = self._edge_margin
        x = int(round(min(max(x, margin), self.screen_w - 1 - margin)))
        y = int(round(min(max(y, margin), self.screen_h - 1 - margin)))
        self._call(pyautogui.moveTo, x, y)
        return True

    def click(self, button="left"):
        if not self.enabled:
            return False
        self._call(pyautogui.click, button=button)
        return True

    def mouse_down(self, button="left"):
        if not self.enabled or button in self._down:
            return False
        self._call(pyautogui.mouseDown, button=button)
        self._down.add(button)
        return True

    def mouse_up(self, button="left"):
        if button not in self._down:
            return False
        self._release([button])
        return True

    def release_all(self):
        self._scroll_accum = 0.0
        if self._down or self._keys_down:
            self._release(list(self._down), list(self._keys_down))

    def _release(self, buttons, keys=()):
        for button in buttons:
            self._down.discard(button)
        for key in keys:
            self._keys_down.discard(key)
        # Failsafe checks fire before the call, so it must be bypassed to avoid stuck buttons.
        previous = pyautogui.FAILSAFE
        pyautogui.FAILSAFE = False
        try:
            for button in buttons:
                pyautogui.mouseUp(button=button)
            for key in keys:
                pyautogui.keyUp(key)
        finally:
            pyautogui.FAILSAFE = previous

    def press_key(self, key):
        if not self.enabled:
            return False
        self._call(pyautogui.press, key)
        return True

    def key_down(self, key):
        if not self.enabled or key in self._keys_down:
            return False
        self._call(pyautogui.keyDown, key)
        self._keys_down.add(key)
        return True

    def key_up(self, key):
        if key not in self._keys_down:
            return False
        self._release([], [key])
        return True

    def scroll(self, notches):
        """Scrolls the wheel. Positive values scroll up. Returns system units sent."""
        if not self.enabled or notches == 0:
            self._scroll_accum = 0.0
            return 0

        self._scroll_accum += notches * self._units_per_notch
        if abs(self._scroll_accum) < self._scroll_min_chunk:
            return 0

        amount = int(self._scroll_accum)
        self._scroll_accum -= amount
        if amount == 0:
            return 0
        self._call(pyautogui.scroll, amount)
        return amount
