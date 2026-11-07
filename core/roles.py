"""Assigns pointer and scroll roles when one or two hands are visible."""

from dataclasses import dataclass

from core.features import TWO, dist


@dataclass
class HandItem:
    hand: object
    feat: object


class RoleTracker:
    def __init__(self, scroll_enabled=True, pinch_block=0.5):
        self.scroll_enabled = scroll_enabled
        self.pinch_block = pinch_block
        self._last_scroll_palm = None
        self._last_pointer_palm = None

    def reset(self):
        self._last_scroll_palm = None
        self._last_pointer_palm = None

    @staticmethod
    def _closest(items, reference):
        if reference is None or len(items) == 1:
            return items[0]
        return min(items, key=lambda item: dist(item.feat.palm, reference))

    def assign(self, items):
        scroll_item = None
        if self.scroll_enabled:
            candidates = [
                item for item in items
                if item.feat.pose == TWO
                and min(item.feat.pinch_index, item.feat.pinch_middle) >= self.pinch_block
            ]
            if candidates:
                scroll_item = self._closest(candidates, self._last_scroll_palm)

        others = [item for item in items if item is not scroll_item]
        if others:
            pointer_item = self._closest(others, self._last_pointer_palm)
        else:
            pointer_item = scroll_item

        self._last_scroll_palm = scroll_item.feat.palm if scroll_item else None
        if pointer_item is not None:
            self._last_pointer_palm = pointer_item.feat.palm
        return pointer_item, scroll_item
