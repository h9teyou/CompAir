"""Pointer state machine: click, drag, right click and pause."""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from core.features import FIST, TWO

IDLE = "IDLE"
FOLLOW = "FOLLOW"
FROZEN = "FROZEN"
PINCH = "PINCH"
DRAG = "DRAG"
RPINCH = "RPINCH"

LEFT = "left"
RIGHT = "right"

DEFAULTS = {
    "pinch_on": 0.30,
    "pinch_off": 0.45,
    "pinch_approach": 0.70,
    "pinch_confirm_frames": 2,
    "release_confirm_frames": 2,
    "pinch_ambiguity_margin": 0.06,
    "drag_hold_seconds": 0.35,
    "right_click_enabled": True,
    "right_click_max_seconds": 0.8,
    "fist_release_seconds": 0.25,
    "pose_hold_seconds": 0.12,
    "lost_grace_seconds": 0.30,
    "anchor_lookback_default": 0.15,
    "anchor_lookback_min": 0.05,
    "anchor_lookback_max": 0.35,
    "clutch_catchup": 0.6,
}


@dataclass
class Observation:
    pose: str
    pinch_index: float
    pinch_middle: float
    index_ok: bool
    middle_ok: bool


def observation_from(features):
    return Observation(
        pose=features.pose,
        pinch_index=features.pinch_index,
        pinch_middle=features.pinch_middle,
        index_ok=features.index_ok,
        middle_ok=features.middle_ok,
    )


@dataclass
class Output:
    cursor: Optional[tuple] = None
    press: Optional[str] = None
    release: Optional[str] = None
    click: Optional[str] = None
    state: str = IDLE
    jump: bool = False


class CursorHistory:
    def __init__(self, keep_seconds=1.5):
        self._keep = keep_seconds
        self._items = deque()

    def add(self, t, pos):
        if pos is None:
            return
        self._items.append((t, pos))
        while self._items and t - self._items[0][0] > self._keep:
            self._items.popleft()

    def at(self, t):
        for item_time, pos in reversed(self._items):
            if item_time <= t:
                return pos
        return self._items[0][1] if self._items else None

    def clear(self):
        self._items.clear()


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp(value, low, high):
    return max(low, min(high, value))


class PointerMachine:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in DEFAULTS})
        self.reset()

    def reset(self):
        self.state = IDLE
        self._cursor = None
        self._offset = (0.0, 0.0)
        self._prev_raw = None
        self._history = CursorHistory()
        self._last_seen = None
        self._last_apart_t = None
        self._pending_kind = None
        self._pinch_count = 0
        self._release_count = 0
        self._pinch_t0 = None
        self._hidden_since = None
        self._pose_group = "follow"
        self._pose_cand = None
        self._pose_cand_since = None

    @property
    def cursor(self):
        return self._cursor

    def update(self, t, obs, raw):
        out = Output(state=self.state)

        if obs is None or raw is None:
            self._on_absent(t, out)
            out.state = self.state
            return out

        self._last_seen = t
        moved = _dist(raw, self._prev_raw) if self._prev_raw is not None else 0.0
        self._prev_raw = raw

        if self.state == IDLE:
            self.state = FOLLOW
            self._offset = (0.0, 0.0)
            self._cursor = raw
            self._pose_group = "follow"
            self._pose_cand = None
            self._pending_kind = None
            self._pinch_count = 0
            self._release_count = 0
            self._last_apart_t = None
            out.jump = True

        if self.state in (FOLLOW, FROZEN):
            self._update_free(t, obs, raw, moved, out)
        elif self.state in (PINCH, RPINCH):
            self._update_pinch(t, obs, raw, moved, out)
        elif self.state == DRAG:
            self._update_drag(t, obs, raw, moved, out)

        self._history.add(t, self._cursor)
        out.state = self.state
        return out

    def _on_absent(self, t, out):
        if self.state == IDLE:
            return
        if self._last_seen is not None and t - self._last_seen < self.cfg["lost_grace_seconds"]:
            return
        if self.state == DRAG:
            out.release = LEFT
        self.state = IDLE
        self._prev_raw = None
        self._pending_kind = None
        self._pinch_count = 0
        self._release_count = 0

    def _update_free(self, t, obs, raw, moved, out):
        cfg = self.cfg

        apart = obs.pinch_index >= cfg["pinch_approach"] and (
            not cfg["right_click_enabled"] or obs.pinch_middle >= cfg["pinch_approach"]
        )
        if apart:
            self._last_apart_t = t

        kind = self._pinch_start_kind(obs)
        if kind is not None and kind == self._pending_kind:
            self._pinch_count += 1
        elif kind is not None:
            self._pending_kind = kind
            self._pinch_count = 1
        else:
            self._pending_kind = None
            self._pinch_count = 0

        if kind is not None and self._pinch_count >= cfg["pinch_confirm_frames"]:
            self._begin_pinch(t, kind, out)
            return

        self._update_pose_group(t, obs, raw, out)
        if self.state == FOLLOW:
            self._follow(raw, moved, out)

    def _pinch_start_kind(self, obs):
        cfg = self.cfg
        left = obs.index_ok and obs.pinch_index < cfg["pinch_on"]
        right = cfg["right_click_enabled"] and obs.middle_ok and obs.pinch_middle < cfg["pinch_on"]
        if left and right:
            margin = cfg["pinch_ambiguity_margin"]
            if obs.pinch_middle + margin < obs.pinch_index:
                return RIGHT
            if obs.pinch_index + margin < obs.pinch_middle:
                return LEFT
            return None
        if left:
            return LEFT
        if right:
            return RIGHT
        return None

    def _begin_pinch(self, t, kind, out):
        cfg = self.cfg
        if self._last_apart_t is None:
            lookback = cfg["anchor_lookback_default"]
        else:
            lookback = _clamp(t - self._last_apart_t, cfg["anchor_lookback_min"], cfg["anchor_lookback_max"])
        anchor = self._history.at(t - lookback) or self._cursor

        self._cursor = anchor
        out.cursor = anchor
        out.jump = True
        self.state = PINCH if kind == LEFT else RPINCH
        self._pinch_t0 = t
        self._release_count = 0
        self._hidden_since = None
        self._pending_kind = None
        self._pinch_count = 0

    def _update_pose_group(self, t, obs, raw, out):
        group = "freeze" if obs.pose in (FIST, TWO) else "follow"
        if group == self._pose_group:
            self._pose_cand = None
            return
        if self._pose_cand != group:
            self._pose_cand = group
            self._pose_cand_since = t
        if t - self._pose_cand_since < self.cfg["pose_hold_seconds"]:
            return

        since = self._pose_cand_since
        self._pose_group = group
        self._pose_cand = None
        if group == "freeze":
            pos = self._history.at(since) or self._cursor
            self._cursor = pos
            out.cursor = pos
            out.jump = True
            self.state = FROZEN
        else:
            self.state = FOLLOW
            self._resume(raw)

    def _update_pinch(self, t, obs, raw, moved, out):
        cfg = self.cfg
        if self._still_pinched(t, obs, LEFT if self.state == PINCH else RIGHT):
            self._release_count = 0
        else:
            self._release_count += 1

        if self._release_count >= cfg["release_confirm_frames"]:
            if self.state == PINCH:
                out.click = LEFT
            elif t - self._pinch_t0 <= cfg["right_click_max_seconds"]:
                out.click = RIGHT
            self._finish_pinch(raw, moved, out)
            return

        if self.state == PINCH and t - self._pinch_t0 >= cfg["drag_hold_seconds"]:
            out.press = LEFT
            self.state = DRAG
            self._resume(raw)
            self._follow(raw, moved, out)

    def _update_drag(self, t, obs, raw, moved, out):
        if self._still_pinched(t, obs, LEFT):
            self._release_count = 0
        else:
            self._release_count += 1

        if self._release_count >= self.cfg["release_confirm_frames"]:
            out.release = LEFT
            self._finish_pinch(raw, moved, out, resume=False)
            return
        self._follow(raw, moved, out)

    def _still_pinched(self, t, obs, kind):
        off = self.cfg["pinch_off"]
        if kind == RIGHT:
            return obs.pinch_middle < off and obs.middle_ok

        if obs.index_ok:
            self._hidden_since = None
        elif self._hidden_since is None:
            self._hidden_since = t
        hidden_too_long = (
            self._hidden_since is not None and t - self._hidden_since >= self.cfg["fist_release_seconds"]
        )
        return obs.pinch_index < off and not hidden_too_long

    def _finish_pinch(self, raw, moved, out, resume=True):
        self.state = FOLLOW
        self._pose_group = "follow"
        self._pose_cand = None
        self._release_count = 0
        if resume:
            self._resume(raw)
        self._follow(raw, moved, out)

    def _resume(self, raw):
        if self._cursor is None:
            self._offset = (0.0, 0.0)
        else:
            self._offset = (self._cursor[0] - raw[0], self._cursor[1] - raw[1])

    def _follow(self, raw, moved, out):
        ox, oy = self._offset
        magnitude = math.hypot(ox, oy)
        if magnitude > 0.0:
            new_magnitude = max(0.0, magnitude - self.cfg["clutch_catchup"] * moved)
            if new_magnitude < 0.5:
                ox, oy = 0.0, 0.0
            else:
                ox, oy = ox * new_magnitude / magnitude, oy * new_magnitude / magnitude
            self._offset = (ox, oy)
        self._cursor = (raw[0] + ox, raw[1] + oy)
        out.cursor = self._cursor
