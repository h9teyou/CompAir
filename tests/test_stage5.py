"""Тесты взмахов, выключателя жестами, клавиатуры и режимов."""

import os
import sys
import time
import types
import unittest
import importlib
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hand_model as hm
from core.config import ConfigError, DEFAULTS, _merge, load_config, validate
from core.features import FIST, PALM, POINT
from core.hand_tracker import Hand
from core.pipeline import GesturePipeline
from core.state_machine import DRAG, FOLLOW, PINCH
from core.switch import ControlSwitch
from modes.game import GameMode
from modes.presentation import PresentationMode
from modes.swipe import SwipeDetector

DT = 1 / 30


def hand(points3d, center):
    pts = hm.project(points3d, hm.make_rng(0), tilt_deg=0, scale=100, roll_deg=0, center=center, noise=0)
    return Hand(points=pts, norm_points=[(x / 640, y / 480, 0) for x, y in pts], handedness="Right", score=0.9)


class Rig:
    """Настоящий конвейер на модели руки: даёт режимам такие же результаты, как в программе."""

    def __init__(self):
        self.cfg = load_config(os.path.join(ROOT, "config.json"), warn=lambda m: None)
        self.pipeline = GesturePipeline(self.cfg, (640, 480), (1920, 1080))
        self.t = 0.0

    def feed(self, hands, n=1):
        results = []
        for _ in range(n):
            self.t += DT
            result = self.pipeline.update(hands, self.t)
            result.t = self.t            # у каждого кадра своё время
            results.append(result)
        return results


class Keys:
    """Двойник контроллера: помнит нажатия."""

    def __init__(self):
        self.pressed, self.held, self.enabled = [], set(), True

    def press_key(self, key):
        self.pressed.append(key)
        return True

    def key_down(self, key):
        self.held.add(key)
        return True

    def key_up(self, key):
        self.held.discard(key)
        return True


class SwipeTests(unittest.TestCase):
    def swipe(self, det, x0, x1, frames=10, y0=300.0, y1=None, size=100.0, t0=0.0):
        y1 = y0 if y1 is None else y1
        found = []
        for i in range(frames + 1):
            f = i / frames
            r = det.update(t0 + i * DT, x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, size)
            if r:
                found.append(r)
        return found

    def test_fast_sideways_motion_is_a_swipe_in_the_right_direction(self):
        self.assertEqual(self.swipe(SwipeDetector(), 100, 500), ["right"])
        self.assertEqual(self.swipe(SwipeDetector(), 500, 100), ["left"])

    def test_slow_drift_small_and_vertical_motions_are_not_swipes(self):
        self.assertEqual(self.swipe(SwipeDetector(), 100, 500, frames=90), [])        # 3 секунды
        self.assertEqual(self.swipe(SwipeDetector(), 100, 200), [])                   # всего 1 размер руки
        self.assertEqual(self.swipe(SwipeDetector(), 100, 400, y0=100, y1=400), [])   # по диагонали

    def test_return_stroke_does_not_flip_back_but_a_later_swipe_does(self):
        det = SwipeDetector()
        self.assertEqual(self.swipe(det, 100, 500, t0=0.0), ["right"])
        self.assertEqual(self.swipe(det, 500, 100, t0=0.5), [], "возвращение руки не должно листать назад")
        self.assertEqual(self.swipe(det, 500, 100, t0=3.0), ["left"])

    def test_return_stroke_is_blocked_even_after_the_plain_cooldown_is_over(self):
        # Пауза между взмахами 0.8 с, блокировка обратного взмаха 1.0 с. Рука возвращается как раз между ними.
        det = SwipeDetector()
        self.assertEqual(self.swipe(det, 100, 500, t0=0.0), ["right"])
        self.assertEqual(self.swipe(det, 500, 100, t0=0.9), [])

    def test_cooldown_between_swipes(self):
        det = SwipeDetector()
        self.assertEqual(self.swipe(det, 100, 500, t0=0.0), ["right"])
        self.assertEqual(self.swipe(det, 100, 500, t0=0.4), [])
        self.assertEqual(self.swipe(det, 100, 500, t0=2.0), ["right"])

    def test_works_the_same_for_near_and_far_hands(self):
        self.assertEqual(self.swipe(SwipeDetector(), 100, 300, size=50.0), ["right"])     # далеко: 4 размера руки
        self.assertEqual(self.swipe(SwipeDetector(), 100, 1700, size=200.0), ["right"])   # близко: 8 размеров

    def test_not_active_clears_history(self):
        det = SwipeDetector()
        for i in range(6):
            det.update(i * DT, 100 + 60 * i, 300, 100.0)
        det.update(6 * DT, None, None, 1.0, active=False)
        self.assertIsNone(det.update(7 * DT, 460, 300, 100.0))


def person(pose):
    return SimpleNamespace(feat=SimpleNamespace(pose=pose))


class SwitchTests(unittest.TestCase):
    def run_for(self, switch, seconds, items, state=FOLLOW, control_on=True, t0=0.0):
        events, t = [], t0
        for _ in range(int(seconds / DT)):
            t += DT
            e = switch.update(t, items, state, control_on)
            if e:
                events.append(e)
        return events, t

    def test_holding_a_fist_turns_control_off_only_after_the_full_time(self):
        s = ControlSwitch()
        self.assertEqual(self.run_for(s, 2.5, [person(FIST)])[0], [])
        s = ControlSwitch()
        events, _ = self.run_for(s, 3.3, [person(FIST)])
        self.assertEqual(events, ["off"])

    def test_progress_is_visible_while_holding(self):
        s = ControlSwitch()
        self.run_for(s, 1.5, [person(FIST)])
        self.assertAlmostEqual(s.progress, 0.5, delta=0.1)
        self.assertEqual(s.target, "OFF")

    def test_releasing_the_fist_cancels(self):
        s = ControlSwitch()
        self.run_for(s, 2.0, [person(FIST)])
        _, t = self.run_for(s, 0.5, [person(POINT)], t0=2.0)
        self.assertEqual(s.progress, 0.0)
        self.assertEqual(self.run_for(s, 2.0, [person(FIST)], t0=t)[0], [])

    def test_a_dropped_frame_does_not_reset_the_timer(self):
        s = ControlSwitch()
        events, t = self.run_for(s, 1.5, [person(FIST)])
        _, t = self.run_for(s, 0.1, [], t0=t)
        events, _ = self.run_for(s, 1.8, [person(FIST)], t0=t)
        self.assertEqual(events, ["off"])

    def test_a_fist_during_a_pinch_or_drag_does_not_count(self):
        for state in (PINCH, DRAG):
            self.assertEqual(self.run_for(ControlSwitch(), 4.0, [person(FIST)], state=state)[0], [])

    def test_two_open_palms_turn_control_on_but_one_does_not(self):
        two = [person(PALM), person(PALM)]
        self.assertEqual(self.run_for(ControlSwitch(), 2.3, two, control_on=False)[0], ["on"])
        self.assertEqual(self.run_for(ControlSwitch(), 5.0, [person(PALM)], control_on=False)[0], [])

    def test_gestures_only_work_in_the_matching_state(self):
        self.assertEqual(self.run_for(ControlSwitch(), 4.0, [person(FIST)], control_on=False)[0], [])
        self.assertEqual(self.run_for(ControlSwitch(), 4.0, [person(PALM), person(PALM)], control_on=True)[0], [])

    def test_cooldown_prevents_flip_flop(self):
        s = ControlSwitch({"off_hold_seconds": 0.3, "on_hold_seconds": 0.3, "cooldown_seconds": 1.0})
        events, t = self.run_for(s, 0.5, [person(FIST)])
        self.assertEqual(events, ["off"])
        early, t = self.run_for(s, 0.6, [person(PALM), person(PALM)], control_on=False, t0=t)
        self.assertEqual(early, [])
        late, _ = self.run_for(s, 0.9, [person(PALM), person(PALM)], control_on=False, t0=t)
        self.assertEqual(late, ["on"])

    def test_can_be_disabled(self):
        self.assertEqual(self.run_for(ControlSwitch({"enabled": False}), 5.0, [person(FIST)])[0], [])


class ControllerKeysTests(unittest.TestCase):
    def setUp(self):
        class Fake(types.ModuleType):
            class FailSafeException(Exception):
                pass

            def __init__(s):
                super().__init__("pyautogui")
                s.FAILSAFE, s.PAUSE, s.pos, s.calls = True, 0.1, (500, 500), []

            def size(s): return (1920, 1080)
            def position(s): return s.pos
            def _chk(s):
                if s.FAILSAFE and s.pos == (0, 0):
                    raise s.FailSafeException()
            def press(s, k): s._chk(); s.calls.append(("press", k))
            def keyDown(s, k): s._chk(); s.calls.append(("down", k))
            def keyUp(s, k): s._chk(); s.calls.append(("up", k))
            def mouseUp(s, button="left"): s._chk(); s.calls.append(("mouseup", button))

        self.fake = Fake()
        self._saved = sys.modules.get("pyautogui")
        sys.modules["pyautogui"] = self.fake
        import core.controller as module
        self.module = importlib.reload(module)
        self.c = self.module.Controller()
        self.c.enabled = True

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("pyautogui", None)
        else:
            sys.modules["pyautogui"] = self._saved

    def test_press_and_hold(self):
        self.assertTrue(self.c.press_key("right"))
        self.assertTrue(self.c.key_down("down"))
        self.assertFalse(self.c.key_down("down"))
        self.assertEqual(self.c.keys_down, {"down"})
        self.assertTrue(self.c.key_up("down"))
        self.assertEqual(self.fake.calls, [("press", "right"), ("down", "down"), ("up", "down")])

    def test_nothing_happens_when_disabled(self):
        self.c.enabled = False
        self.assertFalse(self.c.press_key("right"))
        self.assertFalse(self.c.key_down("down"))
        self.assertEqual(self.fake.calls, [])

    def test_release_all_frees_keys_even_in_the_failsafe_corner(self):
        self.c.key_down("down")
        self.fake.pos = (0, 0)
        self.c.release_all()
        self.assertIn(("up", "down"), self.fake.calls)
        self.assertEqual(self.c.keys_down, set())
        self.assertTrue(self.fake.FAILSAFE)

    def test_turning_control_off_releases_keys(self):
        self.c.key_down("down")
        self.c.set_enabled(False)
        self.assertEqual(self.c.keys_down, set())


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.rig, self.keys = Rig(), Keys()
        self.mode = PresentationMode(self.rig.cfg["presentation"], self.keys)

    def run_motion(self, pose, x0, x1, frames=10, y=330):
        for i in range(frames + 1):
            result = self.rig.feed([hand(pose, (x0 + (x1 - x0) * i / frames, y))])[0]
            self.mode.update(result, self.rig.t)

    def test_open_palm_swipe_presses_next_then_previous(self):
        self.run_motion(hm.pose_open(), 150, 150, 5)
        self.run_motion(hm.pose_open(), 150, 520, 10)
        self.assertEqual(self.keys.pressed, ["right"])
        self.rig.feed([], 90)
        self.run_motion(hm.pose_open(), 520, 150, 10)
        self.assertEqual(self.keys.pressed, ["right", "left"])
        self.assertEqual(self.mode.notice(self.rig.t), "< PREV")

    def test_the_return_stroke_does_not_flip_back(self):
        self.run_motion(hm.pose_open(), 150, 520, 10)
        self.run_motion(hm.pose_open(), 520, 150, 10)
        self.assertEqual(self.keys.pressed, ["right"])

    def test_only_an_open_palm_counts(self):
        self.run_motion(hm.pose_point(), 150, 520, 10)
        self.run_motion(hm.pose_fist(), 150, 520, 10)
        self.assertEqual(self.keys.pressed, [])


class GameTests(unittest.TestCase):
    def setUp(self):
        self.rig, self.keys = Rig(), Keys()
        self.mode = GameMode(self.rig.cfg["game"], self.keys)

    def step(self, pose, n):
        for result in self.rig.feed([hand(pose, (320, 330))], n):
            self.mode.update(result, result.t)

    def test_pinch_jumps_once_at_the_start_of_the_pinch(self):
        self.step(hm.pose_point(), 8)
        self.step(hm.pose_pinch(0.3, "curled", (0.05, 0.02)), 12)
        self.assertEqual(self.keys.pressed, ["space"])
        self.step(hm.pose_point(), 5)
        self.step(hm.pose_pinch(0.3, "curled", (0.05, 0.02)), 4)
        self.assertEqual(self.keys.pressed, ["space", "space"])

    def test_fist_holds_duck_and_releasing_the_fist_releases_the_key(self):
        self.step(hm.pose_point(), 5)
        self.step(hm.pose_fist(), 10)
        self.assertEqual(self.keys.held, {"down"})
        self.step(hm.pose_point(), 5)
        self.assertEqual(self.keys.held, set())

    def test_reset_releases_the_key(self):
        self.step(hm.pose_fist(), 12)
        self.assertEqual(self.keys.held, {"down"})
        self.mode.reset()
        self.assertEqual(self.keys.held, set())


class NewConfigTests(unittest.TestCase):
    def bad(self, user, fragment):
        cfg = _merge(DEFAULTS, user, "", lambda m: None)
        with self.assertRaises(ConfigError) as ctx:
            validate(cfg)
        self.assertIn(fragment, str(ctx.exception))

    def test_defaults_valid_and_sections_present(self):
        cfg = validate(_merge(DEFAULTS, {}, "", lambda m: None))
        for section in ("safety_switch", "presentation", "game"):
            self.assertIn(section, cfg)

    def test_helpful_errors(self):
        self.bad({"presentation": {"next_key": ""}}, "presentation.next_key")
        self.bad({"safety_switch": {"off_hold_seconds": 0.0}}, "safety_switch.off_hold_seconds")
        self.bad({"controls": {"toggle_pointer_key": "q"}}, "не должны повторяться")

    def test_removed_draw_mode_is_rejected(self):
        self.bad({"modes": {"available": ["mouse", "draw"]}}, "modes.available")

    def test_mode_names_must_be_strings(self):
        self.bad({"modes": {"available": [1]}}, "modes.available")


class FeedbackTests(unittest.TestCase):
    def test_beep_plays_distinct_tones(self):
        import core.feedback as feedback
        played = []
        with mock.patch.object(feedback, "_play", lambda f, ms: played.append((f, ms))):
            feedback.beep("on")
            feedback.beep("off")
            time.sleep(0.1)
        self.assertEqual(sorted(played), sorted([feedback.TONES["on"], feedback.TONES["off"]]))
        self.assertNotEqual(feedback.TONES["on"], feedback.TONES["off"])


if __name__ == "__main__":
    unittest.main()
