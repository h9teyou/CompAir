"""Cursor smoothing filters."""

import math

MIN_DT = 1.0 / 240
MAX_DT = 0.25


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class ExponentialSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self._x = None
        self._y = None

    def update(self, x, y, t=None):
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x = self.alpha * x + (1 - self.alpha) * self._x
            self._y = self.alpha * y + (1 - self.alpha) * self._y
        return self._x, self._y

    def reset(self):
        self._x = None
        self._y = None


class _OneEuro1D:
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._prev_raw = None
        self._x_hat = None
        self._dx_hat = 0.0

    def update(self, x, dt):
        if self._prev_raw is None:
            self._prev_raw = x
            self._x_hat = x
            return x

        dx = (x - self._prev_raw) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx_hat = a_d * dx + (1 - a_d) * self._dx_hat

        cutoff = self.min_cutoff + self.beta * abs(self._dx_hat)
        a = _alpha(cutoff, dt)
        self._x_hat = a * x + (1 - a) * self._x_hat

        self._prev_raw = x
        return self._x_hat


class OneEuroSmoother:
    def __init__(self, min_cutoff=1.0, beta=0.03, d_cutoff=1.0):
        self._fx = _OneEuro1D(min_cutoff, beta, d_cutoff)
        self._fy = _OneEuro1D(min_cutoff, beta, d_cutoff)
        self._last_t = None

    def update(self, x, y, t):
        if self._last_t is None:
            dt = 1.0 / 30
        else:
            dt = min(max(t - self._last_t, MIN_DT), MAX_DT)
        self._last_t = t
        return self._fx.update(x, dt), self._fy.update(y, dt)

    def reset(self):
        cfg = (self._fx.min_cutoff, self._fx.beta, self._fx.d_cutoff)
        self._fx = _OneEuro1D(*cfg)
        self._fy = _OneEuro1D(*cfg)
        self._last_t = None


def create_smoother(smoothing_cfg):
    method = smoothing_cfg.get("method", "one_euro")
    if method == "ema":
        return ExponentialSmoother(alpha=smoothing_cfg["ema_alpha"])
    if method == "one_euro":
        return OneEuroSmoother(
            min_cutoff=smoothing_cfg["one_euro_min_cutoff"],
            beta=smoothing_cfg["one_euro_beta"],
            d_cutoff=smoothing_cfg["one_euro_d_cutoff"],
        )
    raise ValueError(f'Неизвестный метод сглаживания "{method}". Допустимо: "ema" или "one_euro".')
