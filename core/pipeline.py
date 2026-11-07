"""Per frame gesture processing without any I/O."""

from dataclasses import dataclass
from typing import Optional

from core.features import analyze
from core.roles import HandItem, RoleTracker
from core.scroll import ScrollEngine, ScrollObservation
from core.smoothing import create_smoother
from core.state_machine import Output, PointerMachine, observation_from
from core.zone import Zone

POINTER_NAMES = {5: "KNUCKLE", 8: "FINGERTIP"}


@dataclass
class FrameResult:
    items: list
    pointer_item: Optional[HandItem]
    scroll_item: Optional[HandItem]
    out: Output
    notches: float


class GesturePipeline:
    def __init__(self, cfg, frame_size, screen_size):
        gestures, scroll, pointer = cfg["gestures"], cfg["scroll"], cfg["pointer"]
        self._gestures = gestures
        self.zone = Zone(cfg["zone"], frame_size, screen_size)
        self.smoother = create_smoother(cfg["smoothing"])
        self.machine = PointerMachine(gestures)
        self.scroller = ScrollEngine(scroll)
        self.roles = RoleTracker(scroll_enabled=scroll["enabled"])
        self._landmarks = [pointer["landmark"], pointer["alt_landmark"]]
        self._pointer_index = 0
        self._lost_reset = pointer["lost_reset_seconds"]
        self._last_seen = None

    @property
    def pointer_landmark(self):
        return self._landmarks[self._pointer_index]

    @property
    def pointer_name(self):
        return POINTER_NAMES.get(self.pointer_landmark, f"POINT {self.pointer_landmark}")

    def reset(self):
        self.machine.reset()
        self.scroller.reset()
        self.roles.reset()
        self._last_seen = None

    def toggle_pointer(self):
        self._pointer_index = (self._pointer_index + 1) % len(self._landmarks)
        self.smoother.reset()
        self.machine.reset()

    def update(self, hands, now):
        g = self._gestures
        items = [
            HandItem(hand, analyze(hand.points, g["extended_threshold"], g["not_hidden_threshold"]))
            for hand in hands
        ]
        pointer_item, scroll_item = self.roles.assign(items)

        if pointer_item is not None:
            point_x, point_y = pointer_item.hand.points[self.pointer_landmark]
            raw = self.smoother.update(*self.zone.to_screen(point_x, point_y), now)
            observation = observation_from(pointer_item.feat)
            self._last_seen = now
        else:
            raw = observation = None
            if self._last_seen is not None and now - self._last_seen > self._lost_reset:
                self.smoother.reset()
                self._last_seen = None
        out = self.machine.update(now, observation, raw)

        scroll_observation = None
        if scroll_item is not None:
            palm_x, palm_y = scroll_item.feat.palm
            scroll_observation = ScrollObservation(palm_x, palm_y, scroll_item.feat.size)
        notches = self.scroller.update(now, scroll_observation)

        return FrameResult(items, pointer_item, scroll_item, out, notches)
