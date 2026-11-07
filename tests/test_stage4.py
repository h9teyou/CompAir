"""Тесты курсора, настроек, таймера, конвейера, камеры и контроллера."""

import copy
import importlib
import os
import sys
import tempfile
import time
import types
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hand_model as hm
from core.config import DEFAULTS, ConfigError, _merge, load_config, validate
from core.features import FIST, POINT
from core.glider import CursorGlider
from core.hand_tracker import Hand
from core.pipeline import GesturePipeline
from core.state_machine import Observation, PointerMachine
from core.timing import enable_high_res_timer


class Screen:
    """Двойник экрана: помнит, куда ставили курсор."""

    def __init__(self):
        self.pos = (500, 500)
        self.moves = []

    def move(self, x, y):
        self.pos = (x, y)
        self.moves.append((x, y))


def make_glider(tau=0.025, lead=0.0):
    screen = Screen()
    return CursorGlider(screen.move, lambda: screen.pos, tau=tau, lead=lead, deadzone_px=2.0), screen


class GliderTests(unittest.TestCase):
    def run_steps(self, glider, start, n, dt=0.008):
        t = start
        for _ in range(n):
            t += dt
            glider.step(t)
        return t

    def test_moves_gradually_not_in_one_jump(self):
        g, screen = make_glider()
        g.set_target(600, 500, now=0.0)
        g.step(0.008)
        self.assertTrue(515 < screen.pos[0] < 545, f"первый шаг: x={screen.pos[0]}")

    def test_converges_exactly_and_never_overshoots_without_lead(self):
        g, screen = make_glider()
        g.set_target(600, 500, now=0.0)
        self.run_steps(g, 0.0, 80)
        self.assertEqual(screen.pos, (600, 500))
        xs = [m[0] for m in screen.moves]
        self.assertEqual(xs, sorted(xs))
        self.assertLessEqual(max(xs), 600)

    def test_many_small_steps_instead_of_one(self):
        g, screen = make_glider()
        g.set_target(800, 500, now=0.0)
        self.run_steps(g, 0.0, 60)
        self.assertGreater(len(screen.moves), 15)

    def test_zero_tau_is_instant(self):
        g, screen = make_glider(tau=0.0)
        g.set_target(700, 400, now=0.0)
        g.step(0.008)
        self.assertEqual(screen.pos, (700, 400))

    def test_snap_is_instant_and_stops_gliding(self):
        g, screen = make_glider()
        g.set_target(900, 500, now=0.0)
        g.snap(300, 300)
        self.assertEqual(screen.pos, (300, 300))
        count = len(screen.moves)
        self.run_steps(g, 0.0, 20)
        self.assertEqual(len(screen.moves), count)

    def test_flush_jumps_to_current_target(self):
        g, screen = make_glider()
        g.set_target(900, 500, now=0.0)
        g.step(0.008)
        g.flush()
        self.assertEqual(screen.pos, (900, 500))

    def test_tiny_target_changes_at_rest_are_ignored(self):
        g, screen = make_glider()
        g.set_target(501, 500, now=0.0)
        self.run_steps(g, 0.0, 10)
        self.assertEqual(screen.moves, [])

    def test_starts_from_the_real_cursor_after_the_user_moved_the_mouse(self):
        g, screen = make_glider()
        g.set_target(600, 500, now=0.0)
        self.run_steps(g, 0.0, 80)
        screen.pos = (100, 100)                      # пользователь сдвинул настоящую мышь
        g.set_target(120, 100, now=5.0)
        g.step(5.008)
        self.assertTrue(100 <= screen.pos[0] <= 120, f"курсор прыгнул к старой позиции: {screen.pos}")

    def test_lead_pushes_target_ahead_when_moving_fast(self):
        g, _ = make_glider(lead=0.025)
        x = 400.0
        for k in range(8):                           # 1500 px/с при 30 кадрах в секунду
            g.set_target(x, 500, now=k / 30)
            x += 50.0
        sample_x = x - 50.0
        self.assertAlmostEqual(g._target[0] - sample_x, 1500 * 0.025, delta=8)

    def test_no_lead_when_hand_is_almost_still(self):
        g, _ = make_glider(lead=0.025)
        g.set_target(400, 500, now=0.0)
        for k in range(1, 6):
            g.set_target(400 + k * 1.0, 500, now=k / 30)    # 30 px/с
        self.assertAlmostEqual(g._target[0], 405.0, places=6)

    def test_lead_is_forgotten_after_a_gap(self):
        g, _ = make_glider(lead=0.025)
        for k in range(6):
            g.set_target(400 + 50 * k, 500, now=k / 30)
        g.set_target(1000, 500, now=3.0)
        self.assertAlmostEqual(g._target[0], 1000.0, places=6)

    def test_clear_stops_motion(self):
        g, screen = make_glider()
        g.set_target(900, 500, now=0.0)
        g.clear()
        self.run_steps(g, 0.0, 10)
        self.assertEqual(screen.moves, [])

    def test_background_thread_reaches_target(self):
        g, screen = make_glider()
        g.start()
        try:
            g.set_target(640, 500)
            deadline = time.time() + 2.0
            while time.time() < deadline and screen.pos != (640, 500):
                time.sleep(0.01)
        finally:
            g.stop()
        self.assertEqual(screen.pos, (640, 500))

    def test_thread_reports_errors_instead_of_dying_silently(self):
        def broken(x, y):
            raise RuntimeError("boom")
        g = CursorGlider(broken, lambda: (500, 500), tau=0.0)
        g.start()
        try:
            g.set_target(900, 500)
            deadline = time.time() + 2.0
            while time.time() < deadline and g.error is None:
                time.sleep(0.01)
        finally:
            g.stop()
        self.assertIsInstance(g.error, RuntimeError)


class ConfigTests(unittest.TestCase):
    def merged(self, user):
        warnings = []
        return _merge(DEFAULTS, user, "", warnings.append), warnings

    def test_defaults_are_valid(self):
        validate(copy.deepcopy(DEFAULTS))

    def test_project_config_loads_without_warnings(self):
        warnings = []
        load_config(os.path.join(ROOT, "config.json"), warn=warnings.append)
        self.assertEqual(warnings, [])

    def test_missing_keys_are_filled_from_defaults(self):
        cfg, _ = self.merged({"pointer": {"landmark": 8}})
        self.assertEqual(cfg["pointer"]["landmark"], 8)
        self.assertEqual(cfg["pointer"]["alt_landmark"], DEFAULTS["pointer"]["alt_landmark"])
        self.assertIn("glide_enabled", cfg["smoothing"])

    def test_typo_gives_a_warning_and_is_ignored(self):
        cfg, warnings = self.merged({"pointer": {"landmrk": 8}})
        self.assertEqual(len(warnings), 1)
        self.assertIn("pointer.landmrk", warnings[0])
        self.assertEqual(cfg["pointer"]["landmark"], DEFAULTS["pointer"]["landmark"])

    def bad(self, user, fragment):
        cfg, _ = self.merged(user)
        with self.assertRaises(ConfigError) as ctx:
            validate(cfg)
        self.assertIn(fragment, str(ctx.exception))

    def test_helpful_errors(self):
        self.bad({"gestures": {"pinch_on": 0.6}}, "pinch_on")
        self.bad({"zone": {"x_min": 0.9}}, "zone")
        self.bad({"tracker": {"max_hands": 3}}, "tracker.max_hands")
        self.bad({"modes": {"default": "nope"}}, "modes.default")
        self.bad({"camera": {"fourcc": "ab"}}, "camera.fourcc")
        self.bad({"controls": {"quit_key": "q", "next_mode_key": "q"}}, "не должны повторяться")
        self.bad({"scroll": {"deadzone": 2.0}}, "scroll.deadzone")
        self.bad({"scroll": {"deadzone": True}}, "scroll.deadzone")

    def test_file_problems_are_explained(self):
        with self.assertRaises(ConfigError):
            load_config("/nonexistent/config.json")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{ not json")
        try:
            with self.assertRaises(ConfigError) as ctx:
                load_config(f.name)
            self.assertIn("строка", str(ctx.exception))
        finally:
            os.unlink(f.name)


class TimingTests(unittest.TestCase):
    def test_noop_outside_windows(self):
        with mock.patch.object(sys, "platform", "linux"):
            enable_high_res_timer()()

    def test_requests_and_restores_one_millisecond_on_windows(self):
        import ctypes
        calls = []
        winmm = SimpleNamespace(timeBeginPeriod=lambda ms: calls.append(("begin", ms)),
                                timeEndPeriod=lambda ms: calls.append(("end", ms)))
        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(ctypes, "windll", SimpleNamespace(winmm=winmm), create=True):
            restore = enable_high_res_timer()
            restore()
        self.assertEqual(calls, [("begin", 1), ("end", 1)])


def hand(points3d, center):
    pts = hm.project(points3d, hm.make_rng(0), tilt_deg=0, scale=100, roll_deg=0, center=center, noise=0)
    return Hand(points=pts, norm_points=[(x / 640, y / 480, 0) for x, y in pts], handedness="Right", score=0.9)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(os.path.join(ROOT, "config.json"), warn=lambda m: None)
        self.pipeline = GesturePipeline(self.cfg, (640, 480), (1920, 1080))
        self.t = 0.0
        self.log = []

    def feed(self, hands, n=1):
        for _ in range(n):
            self.t += 1 / 30
            self.log.append(self.pipeline.update(hands, self.t))

    def test_tap_gives_one_click_and_a_jump_only_at_the_right_moments(self):
        c = (320, 330)
        self.feed([hand(hm.pose_point(), c)], 10)
        self.feed([hand(hm.pose_pinch(0.3, "curled", (0.05, 0.02)), c)], 5)
        self.feed([hand(hm.pose_point(), c)], 5)
        clicks = [r.out.click for r in self.log if r.out.click]
        self.assertEqual(clicks, ["left"])
        jumps = [i for i, r in enumerate(self.log) if r.out.jump]
        self.assertEqual(jumps[0], 0, "при появлении руки курсор встаёт мгновенно")
        self.assertEqual(len(jumps), 2, f"скачки должны быть только при появлении руки и в начале щипка: {jumps}")

    def test_two_hands_one_points_other_scrolls(self):
        pointer = hand(hm.pose_point(), (200, 330))
        for _ in range(10):
            self.feed([pointer, hand(hm.pose_two(), (500, 330))])
        self.feed([pointer, hand(hm.pose_two(), (500, 250))], 20)
        total = sum(r.notches for r in self.log)
        self.assertGreater(total, 0.3)
        self.assertIsNotNone(self.log[-1].scroll_item)
        self.assertIsNot(self.log[-1].pointer_item, self.log[-1].scroll_item)

    def test_no_hands_is_safe(self):
        self.feed([], 5)
        self.assertIsNone(self.log[-1].pointer_item)
        self.assertEqual(self.log[-1].notches, 0.0)

    def test_toggle_pointer_switches_landmark_and_forgets_state(self):
        self.assertEqual(self.pipeline.pointer_name, "KNUCKLE")
        self.feed([hand(hm.pose_point(), (320, 330))], 5)
        self.pipeline.toggle_pointer()
        self.assertEqual(self.pipeline.pointer_name, "FINGERTIP")
        self.assertEqual(self.pipeline.machine.state, "IDLE")


class JumpFlagTests(unittest.TestCase):
    def test_jump_is_set_only_for_discontinuities(self):
        machine = PointerMachine()
        point = Observation(POINT, 1.2, 0.9, True, False)
        fist = Observation(FIST, 0.2, 0.2, False, False)
        t = 0.0
        outs = []
        for obs in [point] * 8 + [fist] * 8 + [point] * 5:
            t += 1 / 30
            outs.append(machine.update(t, obs, (500.0, 500.0)))
        jumps = [i for i, o in enumerate(outs) if o.jump]
        self.assertEqual(jumps[0], 0)                       # рука появилась
        self.assertEqual(len(jumps), 2)                     # и начало паузы (возврат к месту)
        self.assertTrue(all(not o.jump for o in outs[1:8]))  # обычное движение без скачков


class CameraTests(unittest.TestCase):
    def test_reports_real_fps_and_accepts_fps_and_format_settings(self):
        import cv2
        import numpy as np
        from core.camera import CameraStream
        path = os.path.join(tempfile.mkdtemp(), "t.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
        for i in range(60):
            writer.write(np.full((480, 640, 3), i * 4, np.uint8))
        writer.release()
        cam = CameraStream(source=path, width=640, height=480, fps=30, fourcc="MJPG").start()
        try:
            got = 0
            deadline = time.time() + 3
            while got < 5 and time.time() < deadline:
                got += cam.read() is not None
        finally:
            cam.stop()
        self.assertGreaterEqual(got, 5)
        self.assertGreater(cam.measured_fps, 0)


class FakePyAutoGUI(types.ModuleType):
    class FailSafeException(Exception):
        pass

    def __init__(self):
        super().__init__("pyautogui")
        self.FAILSAFE, self.PAUSE, self.pos, self.calls = True, 0.1, (500, 500), []

    def size(self): return (1920, 1080)
    def position(self): return self.pos
    def moveTo(self, x, y):
        if self.FAILSAFE and self.pos == (0, 0):
            raise self.FailSafeException()
        self.pos = (x, y); self.calls.append(("move", x, y))
    def scroll(self, amount): self.calls.append(("scroll", amount))


class ControllerStage4Tests(unittest.TestCase):
    def setUp(self):
        self.fake = FakePyAutoGUI()
        self._saved = sys.modules.get("pyautogui")
        sys.modules["pyautogui"] = self.fake
        import core.controller as module
        self.module = importlib.reload(module)

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("pyautogui", None)
        else:
            sys.modules["pyautogui"] = self._saved

    def test_move_exact_clamps_does_not_skip_small_moves_and_needs_enabled(self):
        c = self.module.Controller(edge_margin=2)
        self.assertFalse(c.move_exact(900, 500))
        c.enabled = True
        self.assertTrue(c.move_exact(501, 500))            # сдвиг на 1 px не отбрасывается
        c.move_exact(-10, 99999)
        self.assertEqual(self.fake.calls, [("move", 501, 500), ("move", 2, 1077)])
        self.assertEqual(c.position(), (2, 1077))

    def test_move_exact_raises_emergency_stop_in_corner(self):
        c = self.module.Controller()
        c.enabled = True
        self.fake.pos = (0, 0)
        with self.assertRaises(self.module.EmergencyStop):
            c.move_exact(900, 500)

    def test_custom_scroll_chunk_makes_scroll_finer(self):
        c = self.module.Controller(scroll_units_per_notch=120, scroll_min_chunk=10)
        c.enabled = True
        self.assertEqual(c.scroll(0.1), 12)                # 12 единиц: по умолчанию (порог 40) не ушло бы


if __name__ == "__main__":
    unittest.main()
