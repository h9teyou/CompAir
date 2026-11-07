"""Cursor interpolation at a higher rate than the camera."""

import math
import threading
import time

LEAD_MIN_SPEED = 60.0
LEAD_RAMP = 120.0
VELOCITY_SMOOTHING = 0.5
MAX_VELOCITY_GAP = 0.15
MAX_STEP_DT = 0.1


def _clamp(value, low, high):
    return max(low, min(high, value))


class CursorGlider:
    def __init__(self, move_fn, position_fn, tau=0.025, lead=0.025, rate_hz=125.0,
                 deadzone_px=2.0, clock=time.perf_counter, sleep=time.sleep):
        self._move = move_fn
        self._position = position_fn
        self.tau = tau
        self.lead = lead
        self._period = 1.0 / rate_hz
        self._deadzone = deadzone_px
        self._clock = clock
        self._sleep = sleep

        self._lock = threading.Lock()
        self._pos = None
        self._target = None
        self._active = False
        self._last_step = None
        self._last_sent = None
        self._prev_target = None
        self._vel = (0.0, 0.0)

        self._thread = None
        self._running = False
        self.error = None

    def set_target(self, x, y, now=None):
        now = self._clock() if now is None else now
        with self._lock:
            vx, vy = self._update_velocity(now, x, y)
            speed = math.hypot(vx, vy)
            gate = _clamp((speed - LEAD_MIN_SPEED) / LEAD_RAMP, 0.0, 1.0)
            tx, ty = x + vx * self.lead * gate, y + vy * self.lead * gate

            if not self._active:
                actual = self._position()
                self._pos = (float(actual[0]), float(actual[1]))
                self._last_sent = (int(actual[0]), int(actual[1]))
                if math.hypot(tx - self._pos[0], ty - self._pos[1]) < self._deadzone:
                    return
                self._active = True
                self._last_step = now
            self._target = (tx, ty)

    def snap(self, x, y):
        with self._lock:
            self._prev_target = None
            self._vel = (0.0, 0.0)
            self._active = False
            self._pos = self._target = (float(x), float(y))
            self._send(self._pos)

    def flush(self):
        with self._lock:
            if self._active and self._target is not None:
                self._pos = self._target
                self._active = False
                self._send(self._pos)

    def clear(self):
        with self._lock:
            self._active = False
            self._target = None
            self._prev_target = None
            self._vel = (0.0, 0.0)

    def step(self, now=None):
        now = self._clock() if now is None else now
        with self._lock:
            if not self._active or self._target is None:
                return False
            dt = _clamp(now - self._last_step, 0.0, MAX_STEP_DT)
            self._last_step = now
            fraction = 1.0 if self.tau <= 0 else 1.0 - math.exp(-dt / self.tau)
            px, py = self._pos
            tx, ty = self._target
            px += (tx - px) * fraction
            py += (ty - py) * fraction
            if math.hypot(tx - px, ty - py) < 0.5:
                px, py = tx, ty
                self._active = False
            self._pos = (px, py)
            self._send(self._pos)
            return True

    def _update_velocity(self, now, x, y):
        prev = self._prev_target
        self._prev_target = (now, x, y)
        if prev is None or now - prev[0] > MAX_VELOCITY_GAP or now <= prev[0]:
            self._vel = (0.0, 0.0)
        else:
            dt = now - prev[0]
            raw_x, raw_y = (x - prev[1]) / dt, (y - prev[2]) / dt
            a = VELOCITY_SMOOTHING
            self._vel = (a * raw_x + (1 - a) * self._vel[0], a * raw_y + (1 - a) * self._vel[1])
        return self._vel

    def _send(self, pos):
        ix, iy = int(round(pos[0])), int(round(pos[1]))
        if (ix, iy) != self._last_sent:
            self._move(ix, iy)
            self._last_sent = (ix, iy)

    def start(self):
        if self._thread is not None:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="cursor-glider")
        self._thread.start()
        return self

    def _run(self):
        while self._running:
            began = self._clock()
            try:
                self.step(began)
            except Exception as error:
                self.error = error
                self._running = False
                return
            self._sleep(max(0.001, self._period - (self._clock() - began)))

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
