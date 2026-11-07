"""Hand gesture control switch that works without window focus."""

from core.features import FIST, PALM
from core.state_machine import DRAG, PINCH, RPINCH

DEFAULTS = {
    "enabled": True,
    "off_hold_seconds": 3.0,
    "on_hold_seconds": 2.0,
    "cooldown_seconds": 1.5,
}
TOLERANCE = 0.2


class ControlSwitch:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in DEFAULTS})
        self.reset()

    def reset(self):
        self._since = None
        self._lost_since = None
        self._cooldown_until = float("-inf")
        self.progress = 0.0
        self.target = None

    def update(self, t, items, state, control_on):
        cfg = self.cfg
        if not cfg["enabled"] or t < self._cooldown_until:
            self._clear()
            return None

        if control_on:
            active = state not in (PINCH, DRAG, RPINCH) and any(i.feat.pose == FIST for i in items)
            hold, event, target = cfg["off_hold_seconds"], "off", "OFF"
        else:
            active = sum(1 for i in items if i.feat.pose == PALM) >= 2
            hold, event, target = cfg["on_hold_seconds"], "on", "ON"

        if active:
            self._lost_since = None
            if self._since is None:
                self._since = t
        elif self._since is not None:
            if self._lost_since is None:
                self._lost_since = t
            if t - self._lost_since > TOLERANCE:
                self._clear()

        if self._since is None:
            self.progress, self.target = 0.0, None
            return None

        self.progress = min(1.0, (t - self._since) / hold)
        self.target = target
        if self.progress >= 1.0:
            self._clear()
            self._cooldown_until = t + cfg["cooldown_seconds"]
            return event
        return None

    def _clear(self):
        self._since = None
        self._lost_since = None
        self.progress = 0.0
        self.target = None
