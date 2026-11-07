"""Maps a control zone inside the camera frame to the full screen."""


def _clamp(value, low, high):
    return max(low, min(high, value))


class Zone:
    def __init__(self, zone_cfg, frame_size, screen_size):
        self.frame_w, self.frame_h = frame_size
        self.screen_w, self.screen_h = screen_size

        x_min, y_min = zone_cfg["x_min"], zone_cfg["y_min"]
        x_max, y_max = zone_cfg["x_max"], zone_cfg["y_max"]
        if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
            raise ValueError(
                "Неверная зона в config.json: нужно 0 <= x_min < x_max <= 1 и 0 <= y_min < y_max <= 1."
            )

        if zone_cfg.get("lock_aspect", True):
            screen_aspect = self.screen_w / self.screen_h
            x_min, y_min, x_max, y_max = self._fit_aspect(
                x_min, y_min, x_max, y_max, self.frame_w, self.frame_h, screen_aspect
            )

        self.x_min, self.y_min, self.x_max, self.y_max = x_min, y_min, x_max, y_max

    @staticmethod
    def _fit_aspect(x_min, y_min, x_max, y_max, frame_w, frame_h, screen_aspect):
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        width_px = (x_max - x_min) * frame_w
        height_px = width_px / screen_aspect

        if height_px > frame_h:
            height_px = frame_h
            width_px = height_px * screen_aspect

        half_w = width_px / frame_w / 2
        half_h = height_px / frame_h / 2

        center_x = _clamp(center_x, half_w, 1 - half_w)
        center_y = _clamp(center_y, half_h, 1 - half_h)

        return center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h

    @property
    def bounds(self):
        return {"x_min": self.x_min, "y_min": self.y_min, "x_max": self.x_max, "y_max": self.y_max}

    def to_screen(self, x_px, y_px):
        u = (x_px / self.frame_w - self.x_min) / (self.x_max - self.x_min)
        v = (y_px / self.frame_h - self.y_min) / (self.y_max - self.y_min)
        u = _clamp(u, 0.0, 1.0)
        v = _clamp(v, 0.0, 1.0)
        return u * (self.screen_w - 1), v * (self.screen_h - 1)
