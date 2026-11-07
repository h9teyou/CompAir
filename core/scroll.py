"""Joystick style wheel scrolling driven by the two finger pose."""

from dataclasses import dataclass

DEFAULTS = {
    "enter_hold_seconds": 0.15,
    "exit_hold_seconds": 0.12,
    "deadzone": 0.20,
    "full_speed_offset": 1.00,
    "max_notches_per_second": 18.0,
    "response_power": 1.6,
    "offset_smoothing": 0.5,
    "invert": False,
}

MAX_DT = 0.1


@dataclass
class ScrollObservation:
    x: float
    y: float
    size: float


class ScrollEngine:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in DEFAULTS})
        self.reset()

    def reset(self):
        self.active = False
        self.anchor_x = 0.0
        self.anchor_y = 0.0
        self.anchor_size = 1.0
        self.offset = 0.0
        self.speed = 0.0
        self._enter_since = None
        self._gone_since = None
        self._last_t = None

    def update(self, t, obs):
        cfg = self.cfg

        if not self.active:
            self.speed = 0.0
            if obs is None:
                self._enter_since = None
                return 0.0
            if self._enter_since is None:
                self._enter_since = t
            if t - self._enter_since >= cfg["enter_hold_seconds"]:
                self.active = True
                self.anchor_x, self.anchor_y, self.anchor_size = obs.x, obs.y, max(obs.size, 1e-6)
                self.offset = 0.0
                self._gone_since = None
                self._last_t = t
            return 0.0

        if obs is None:
            if self._gone_since is None:
                self._gone_since = t
            if t - self._gone_since >= cfg["exit_hold_seconds"]:
                self.reset()
            else:
                self.speed = 0.0
                self._last_t = t
            return 0.0
        self._gone_since = None

        dt = min(max(t - self._last_t, 0.0), MAX_DT) if self._last_t is not None else 0.0
        self._last_t = t

        raw_offset = (self.anchor_y - obs.y) / self.anchor_size
        alpha = cfg["offset_smoothing"]
        self.offset += alpha * (raw_offset - self.offset)

        magnitude = abs(self.offset)
        beyond = max(0.0, magnitude - cfg["deadzone"])
        full = max(cfg["full_speed_offset"] - cfg["deadzone"], 1e-6)
        fraction = min(1.0, beyond / full) ** cfg["response_power"]
        direction = 1.0 if self.offset > 0 else -1.0
        speed = direction * fraction * cfg["max_notches_per_second"]
        if cfg["invert"]:
            speed = -speed
        self.speed = speed
        return speed * dt
