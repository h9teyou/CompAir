"""Game mode: gestures press keys."""

from core.features import FIST
from core.state_machine import FROZEN, PINCH

NOTICE_SECONDS = 0.5


class GameMode:
    def __init__(self, cfg, controller):
        self.controller = controller
        self.jump_key = cfg["jump_key"]
        self.duck_key = cfg["duck_key"]
        self._prev_state = None
        self._notice = None

    def reset(self):
        self.controller.key_up(self.duck_key)
        self._prev_state = None
        self._notice = None

    def update(self, result, now):
        state = result.out.state
        if state == PINCH and self._prev_state != PINCH:
            self.controller.press_key(self.jump_key)
            self._notice = ("JUMP", now + NOTICE_SECONDS)
        self._prev_state = state

        item = result.pointer_item
        if item is not None and item.feat.pose == FIST and state == FROZEN:
            self.controller.key_down(self.duck_key)
        else:
            self.controller.key_up(self.duck_key)

    def notice(self, now):
        return self._notice[0] if self._notice and now < self._notice[1] else None

    def hud_lines(self):
        return [("PINCH = jump   FIST = duck", (255, 255, 255))]
