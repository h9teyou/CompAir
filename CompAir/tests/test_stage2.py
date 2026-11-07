"""Тесты зоны, сглаживания и контроллера."""

import importlib
import math
import os
import random
import statistics
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.smoothing import ExponentialSmoother, OneEuroSmoother, create_smoother
from core.zone import Zone

ZONE_CFG = {"x_min": 0.20, "y_min": 0.15, "x_max": 0.80, "y_max": 0.65, "lock_aspect": True}
FRAME = (640, 480)
SCREEN = (1920, 1080)


class ZoneTests(unittest.TestCase):
    def test_aspect_of_zone_matches_screen(self):
        zone = Zone(ZONE_CFG, FRAME, SCREEN)
        b = zone.bounds
        zone_aspect = ((b["x_max"] - b["x_min"]) * FRAME[0]) / ((b["y_max"] - b["y_min"]) * FRAME[1])
        self.assertAlmostEqual(zone_aspect, SCREEN[0] / SCREEN[1], places=6)

    def test_corners_and_center(self):
        zone = Zone(ZONE_CFG, FRAME, SCREEN)
        b = zone.bounds
        left, top = b["x_min"] * FRAME[0], b["y_min"] * FRAME[1]
        right, bottom = b["x_max"] * FRAME[0], b["y_max"] * FRAME[1]
        self.assertEqual(zone.to_screen(left, top), (0.0, 0.0))
        x, y = zone.to_screen(right, bottom)
        self.assertAlmostEqual(x, SCREEN[0] - 1)
        self.assertAlmostEqual(y, SCREEN[1] - 1)
        x, y = zone.to_screen((left + right) / 2, (top + bottom) / 2)
        self.assertAlmostEqual(x, (SCREEN[0] - 1) / 2)
        self.assertAlmostEqual(y, (SCREEN[1] - 1) / 2)

    def test_outside_zone_is_clamped_to_screen_edge(self):
        zone = Zone(ZONE_CFG, FRAME, SCREEN)
        self.assertEqual(zone.to_screen(0, 0), (0.0, 0.0))
        x, y = zone.to_screen(FRAME[0], FRAME[1])
        self.assertAlmostEqual(x, SCREEN[0] - 1)
        self.assertAlmostEqual(y, SCREEN[1] - 1)

    def test_without_lock_bounds_stay_as_in_config(self):
        cfg = dict(ZONE_CFG, lock_aspect=False)
        b = Zone(cfg, FRAME, SCREEN).bounds
        self.assertEqual((b["x_min"], b["y_min"], b["x_max"], b["y_max"]), (0.20, 0.15, 0.80, 0.65))

    def test_zone_too_tall_for_frame_is_shrunk_and_stays_inside(self):
        cfg = {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0, "lock_aspect": True}
        b = Zone(cfg, FRAME, (1000, 1000)).bounds  # квадратный экран
        for value in b.values():
            self.assertTrue(-1e-9 <= value <= 1 + 1e-9)
        width_px = (b["x_max"] - b["x_min"]) * FRAME[0]
        height_px = (b["y_max"] - b["y_min"]) * FRAME[1]
        self.assertAlmostEqual(width_px, height_px)

    def test_zone_near_frame_edge_is_shifted_inside(self):
        cfg = {"x_min": 0.0, "y_min": 0.0, "x_max": 0.6, "y_max": 0.1, "lock_aspect": True}
        b = Zone(cfg, FRAME, SCREEN).bounds
        for value in b.values():
            self.assertTrue(-1e-9 <= value <= 1 + 1e-9)

    def test_invalid_zone_raises(self):
        with self.assertRaises(ValueError):
            Zone({"x_min": 0.8, "y_min": 0.1, "x_max": 0.2, "y_max": 0.9}, FRAME, SCREEN)


class SmoothingTests(unittest.TestCase):
    def test_first_sample_is_returned_unchanged(self):
        self.assertEqual(OneEuroSmoother().update(100.0, 200.0, 0.0), (100.0, 200.0))
        self.assertEqual(ExponentialSmoother(0.5).update(100.0, 200.0), (100.0, 200.0))

    def test_constant_input_stays_constant(self):
        s = OneEuroSmoother()
        for i in range(50):
            x, y = s.update(500.0, 300.0, i / 30)
        self.assertAlmostEqual(x, 500.0)
        self.assertAlmostEqual(y, 300.0)

    def test_jitter_at_rest_is_reduced(self):
        random.seed(0)
        s = OneEuroSmoother()
        raw, out = [], []
        for i in range(300):
            z = 960 + random.gauss(0, 4)
            raw.append(z)
            out.append(s.update(z, 540, i / 30)[0])
        self.assertLess(statistics.pstdev(out[30:]), 0.6 * statistics.pstdev(raw[30:]))

    def test_fast_motion_does_not_lag_much(self):
        s = OneEuroSmoother()
        worst = 0.0
        for i in range(60):
            t = i / 30
            if t < 0.3:
                truth = 300.0
            elif t < 0.65:
                truth = 300 + 1200 * (0.5 - 0.5 * math.cos(math.pi * (t - 0.3) / 0.35))
            else:
                truth = 1500.0
            x, _ = s.update(truth, 0, t)
            if 0.3 <= t <= 0.65:
                worst = max(worst, abs(x - truth))
        self.assertLess(worst, 80)  # px, на экране шириной 1920

    def test_reset_forgets_the_past(self):
        s = OneEuroSmoother()
        for i in range(30):
            s.update(100.0, 100.0, i / 30)
        s.reset()
        self.assertEqual(s.update(1800.0, 900.0, 5.0), (1800.0, 900.0))

    def test_output_never_leaves_input_range(self):
        s = OneEuroSmoother()
        for i in range(100):
            x, y = s.update(0.0 if i % 2 else 1919.0, 0.0, i / 30)
            self.assertTrue(-1e-6 <= x <= 1919 + 1e-6)

    def test_factory(self):
        self.assertIsInstance(create_smoother({"method": "ema", "ema_alpha": 0.5}), ExponentialSmoother)
        with self.assertRaises(ValueError):
            create_smoother({"method": "nope"})


class FakePyAutoGUI(types.ModuleType):
    """Двойник pyautogui: запоминает движения и никого не трогает."""

    class FailSafeException(Exception):
        pass

    def __init__(self):
        super().__init__("pyautogui")
        self.FAILSAFE = True
        self.PAUSE = 0.1
        self.pos = (500, 500)
        self.moves = []
        self.raise_failsafe = False

    def size(self):
        return (1920, 1080)

    def position(self):
        return self.pos

    def moveTo(self, x, y):
        self.moves.append((x, y))
        self.pos = (x, y)
        if self.raise_failsafe:
            raise self.FailSafeException()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakePyAutoGUI()
        self._saved = sys.modules.get("pyautogui")
        sys.modules["pyautogui"] = self.fake
        import core.controller as controller_module
        self.module = importlib.reload(controller_module)

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("pyautogui", None)
        else:
            sys.modules["pyautogui"] = self._saved

    def make(self, **kwargs):
        return self.module.Controller(**kwargs)

    def test_settings_applied_and_disabled_by_default(self):
        c = self.make(failsafe=True, pause=0.0)
        self.assertEqual(self.fake.PAUSE, 0.0)
        self.assertTrue(self.fake.FAILSAFE)
        self.assertFalse(c.enabled)
        self.assertFalse(c.move_to(900, 500))
        self.assertEqual(self.fake.moves, [])

    def test_moves_when_enabled(self):
        c = self.make()
        c.toggle()
        self.assertTrue(c.move_to(900.4, 600.6))
        self.assertEqual(self.fake.moves, [(900, 601)])

    def test_clamped_away_from_corners_and_edges(self):
        c = self.make(edge_margin=2)
        c.toggle()
        c.move_to(-50, -50)
        c.move_to(99999, 99999)
        self.assertEqual(self.fake.moves, [(2, 2), (1917, 1077)])

    def test_tiny_moves_are_skipped(self):
        c = self.make(min_move_px=2)
        c.toggle()
        c.move_to(500, 500)               # курсор уже там
        c.move_to(501, 501)               # сдвиг 1 px
        self.assertEqual(self.fake.moves, [])
        self.assertTrue(c.move_to(503, 500))  # сдвиг 3 px
        self.assertEqual(self.fake.moves, [(503, 500)])

    def test_respects_real_mouse_moved_by_user(self):
        c = self.make()
        c.toggle()
        c.move_to(900, 500)
        self.fake.pos = (100, 100)        # пользователь отодвинул настоящую мышь
        self.assertTrue(c.move_to(900, 500))
        self.assertEqual(self.fake.moves[-1], (900, 500))

    def test_failsafe_becomes_emergency_stop(self):
        c = self.make()
        c.toggle()
        self.fake.raise_failsafe = True
        with self.assertRaises(self.module.EmergencyStop):
            c.move_to(900, 500)

    def test_toggle(self):
        c = self.make()
        self.assertTrue(c.toggle())
        self.assertFalse(c.toggle())


if __name__ == "__main__":
    unittest.main()
