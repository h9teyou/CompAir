"""Horizontal swipe detection."""

from collections import deque

DEFAULTS = {
    "window_seconds": 0.5,
    "min_distance": 1.8,
    "min_speed": 4.0,
    "axis_ratio": 2.0,
    "cooldown_seconds": 0.8,
    "opposite_block_seconds": 1.0,
}
MAX_GAP = 0.15


class SwipeDetector:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in DEFAULTS})
        self.reset()

    def reset(self):
        self._history = deque()
        self._cooldown_until = float("-inf")
        self._blocked = None
        self._blocked_until = float("-inf")

    def update(self, t, x, y, size, active=True):
        if not active or x is None:
            self._history.clear()
            return None
        cfg = self.cfg
        if self._history and t - self._history[-1][0] > MAX_GAP:
            self._history.clear()
        self._history.append((t, x / size, y / size))
        while self._history and t - self._history[0][0] > cfg["window_seconds"]:
            self._history.popleft()
        if t < self._cooldown_until or len(self._history) < 2:
            return None

        points = list(self._history)
        now_t, now_x, now_y = points[-1]
        for direction, pick in (("right", min), ("left", max)):
            extreme = pick(points, key=lambda p: p[1])
            dx = now_x - extreme[1] if direction == "right" else extreme[1] - now_x
            elapsed = now_t - extreme[0]
            if elapsed <= 0 or dx < cfg["min_distance"] or dx / elapsed < cfg["min_speed"]:
                continue
            if abs(now_y - extreme[2]) > dx / cfg["axis_ratio"]:
                continue
            if direction == self._blocked and t < self._blocked_until:
                self._history.clear()
                return None
            self._cooldown_until = t + cfg["cooldown_seconds"]
            self._blocked = "left" if direction == "right" else "right"
            self._blocked_until = t + cfg["opposite_block_seconds"]
            self._history.clear()
            return direction
        return None
