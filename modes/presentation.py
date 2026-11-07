"""Presentation mode: swipe an open palm to change slides."""

from core.features import PALM
from modes.swipe import SwipeDetector

NOTICE_SECONDS = 0.8


class PresentationMode:
    def __init__(self, cfg, controller):
        self.controller = controller
        self.next_key = cfg["next_key"]
        self.prev_key = cfg["prev_key"]
        self.swipe = SwipeDetector(cfg)
        self._notice = None

    def reset(self):
        self.swipe.reset()
        self._notice = None

    def update(self, result, now):
        item = result.pointer_item
        direction = None
        if item is not None and item.feat.pose == PALM:
            x, y = item.feat.palm
            direction = self.swipe.update(now, x, y, item.feat.size)
        else:
            self.swipe.update(now, None, None, 1.0, active=False)
        if direction is None:
            return
        forward = direction == "right"
        self.controller.press_key(self.next_key if forward else self.prev_key)
        self._notice = ("NEXT >" if forward else "< PREV", now + NOTICE_SECONDS)

    def notice(self, now):
        return self._notice[0] if self._notice and now < self._notice[1] else None

    def hud_lines(self):
        return [("SWIPE open palm: left / right", (255, 255, 255))]
