"""Тесты автомата указателя, колёсика, ролей рук и контроллера."""

import importlib
import math
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import FIST, OTHER, POINT, TWO
from core.roles import HandItem, RoleTracker
from core.scroll import ScrollEngine, ScrollObservation
from core.state_machine import DRAG, FOLLOW, FROZEN, IDLE, PINCH, RPINCH, Observation, PointerMachine

DT = 1 / 30
A = (500.0, 500.0)


def point():   return Observation(POINT, 1.2, 0.9, True, False)
def apart():   return point()
def pinch():   return Observation(POINT, 0.10, 0.80, True, False)
def fist():    return Observation(FIST, 0.20, 0.20, False, False)
def two():     return Observation(TWO, 1.2, 1.3, True, True)
def rpinch():  return Observation(OTHER, 0.60, 0.10, True, True)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Sim:
    """Подаёт автомату кадры и запоминает всё, что он ответил."""

    def __init__(self, cfg=None):
        self.machine = PointerMachine(cfg)
        self.t = 0.0
        self.raw = A
        self.log = []

    def step(self, obs, raw=None, n=1):
        for _ in range(n):
            self.t += DT
            if raw is not None:
                self.raw = raw
            out = self.machine.update(self.t, obs, self.raw if obs is not None else None)
            self.log.append(out)
        return self.log[-n:]

    def events(self, name):
        return [getattr(o, name) for o in self.log if getattr(o, name)]


class PointerMachineTests(unittest.TestCase):
    def test_tap_is_exactly_one_left_click(self):
        s = Sim()
        s.step(point(), A, 10)
        s.step(pinch(), n=5)
        s.step(point(), n=5)
        self.assertEqual(s.events("click"), ["left"])
        self.assertEqual(s.events("press"), [])
        self.assertEqual(s.events("release"), [])
        self.assertEqual(s.machine.state, FOLLOW)

    def test_one_frame_glitch_does_not_click(self):
        s = Sim()
        s.step(point(), A, 5)
        s.step(pinch(), n=1)
        s.step(point(), n=5)
        self.assertEqual(s.events("click"), [])
        self.assertEqual(s.machine.state, FOLLOW)

    def test_click_lands_where_the_cursor_was_before_the_fingers_closed(self):
        s = Sim()
        s.step(point(), A, 10)
        # пальцы смыкаются, и рука при этом уезжает на 40, 80, 120 пикселей
        s.step(Observation(POINT, 0.50, 0.80, True, False), (A[0] + 40, A[1]))
        s.step(Observation(POINT, 0.20, 0.80, True, False), (A[0] + 80, A[1]))
        s.step(Observation(POINT, 0.10, 0.80, True, False), (A[0] + 120, A[1]))
        begin = [o for o in s.log if o.state == PINCH][0]
        self.assertLess(dist(begin.cursor, A), 1.0, f"клик ушёл на {dist(begin.cursor, A):.0f} px")

    def test_no_cursor_jump_after_click(self):
        s = Sim()
        s.step(point(), A, 10)
        s.step(pinch(), (A[0] + 80, A[1]), 5)       # рука уехала на 80 px
        outs = s.step(point(), n=3)                  # пальцы разошлись, рука всё ещё там
        click_frame = [o for o in outs if o.click][0]
        self.assertLess(dist(click_frame.cursor, A), 2.0)

    def test_long_pinch_is_drag_not_click(self):
        s = Sim()
        s.step(point(), A, 10)
        s.step(pinch(), A, 6)                        # 0.2 с: ещё ждём
        self.assertEqual(s.events("press"), [])
        self.assertTrue(all(o.cursor is None for o in s.log[-4:]), "пока ждём, курсор должен стоять")
        s.step(pinch(), A, 12)                       # перешли 0.35 с
        self.assertEqual(s.events("press"), ["left"])
        self.assertEqual(s.machine.state, DRAG)
        # в перетаскивании курсор следует за рукой без прыжка
        start = s.log[-1].cursor
        outs = s.step(pinch(), (A[0] + 100, A[1]), 1)
        self.assertAlmostEqual(outs[0].cursor[0] - start[0], 100, delta=1)
        s.step(point(), n=3)
        self.assertEqual(s.events("release"), ["left"])
        self.assertEqual(s.events("click"), [])

    def test_hysteresis_keeps_pinch_when_distance_wobbles_between_thresholds(self):
        s = Sim()
        s.step(point(), A, 5)
        s.step(pinch(), n=3)
        for i in range(10):
            ratio = 0.35 if i % 2 == 0 else 0.42       # между pinch_on=0.30 и pinch_off=0.45
            s.step(Observation(POINT, ratio, 0.8, True, False))
        self.assertIn(s.machine.state, (PINCH, DRAG))
        self.assertEqual(s.events("click") + s.events("release"), [])
        s.step(Observation(POINT, 0.5, 0.8, True, False), n=2)
        self.assertEqual(len(s.events("click") + s.events("release")), 1)

    def test_fist_never_clicks_and_freezes_cursor(self):
        s = Sim()
        s.step(point(), A, 10)
        for i in range(10):
            s.step(fist(), (A[0] + 40 * (i + 1), A[1]))   # рука убегает вправо
        self.assertEqual(s.events("click"), [])
        self.assertEqual(s.machine.state, FROZEN)
        self.assertLess(dist(s.machine.cursor, A), 50, "курсор должен остаться возле места, где была пауза")

    def test_resume_after_pause_does_not_teleport_and_converges(self):
        s = Sim()
        s.step(point(), A, 10)
        s.step(fist(), A, 8)
        frozen = s.machine.cursor
        b = (A[0] + 400, A[1])
        out = s.step(point(), b, 5)[-1]
        self.assertLess(dist(out.cursor, frozen), 5, "после паузы курсор не должен прыгать к руке")
        raw = b
        for _ in range(100):                          # рука едет ещё на 1000 px
            raw = (raw[0] + 10, raw[1])
            out = s.step(point(), raw)[0]
        self.assertLess(dist(out.cursor, raw), 5, "разница должна исчезнуть, пока рука двигается")

    def test_two_fingers_freezes_cursor_too(self):
        s = Sim()
        s.step(point(), A, 10)
        s.step(two(), A, 8)
        self.assertEqual(s.machine.state, FROZEN)

    def test_short_hand_loss_keeps_drag_but_long_loss_releases_button(self):
        s = Sim()
        s.step(point(), A, 5)
        s.step(pinch(), A, 25)
        self.assertEqual(s.machine.state, DRAG)
        s.step(None, n=3)                             # 0.1 с без руки
        self.assertEqual(s.events("release"), [])
        s.step(pinch(), A, 3)
        self.assertEqual(s.machine.state, DRAG)
        s.step(None, n=14)                            # дольше 0.3 с
        self.assertEqual(s.events("release"), ["left"])
        self.assertEqual(s.machine.state, IDLE)

    def test_right_click_on_middle_pinch(self):
        s = Sim()
        s.step(point(), A, 10)
        s.step(rpinch(), n=6)
        self.assertEqual(s.machine.state, RPINCH)
        s.step(point(), n=3)
        self.assertEqual(s.events("click"), ["right"])

    def test_long_right_pinch_is_not_a_click(self):
        s = Sim()
        s.step(point(), A, 10)
        s.step(rpinch(), n=40)                        # 1.3 с
        s.step(point(), n=3)
        self.assertEqual(s.events("click"), [])
        self.assertEqual(s.machine.state, FOLLOW)

    def test_right_click_can_be_disabled(self):
        s = Sim({"right_click_enabled": False})
        s.step(point(), A, 10)
        s.step(rpinch(), n=6)
        s.step(point(), n=3)
        self.assertEqual(s.events("click"), [])

    def test_ambiguous_pinch_does_nothing(self):
        s = Sim()
        s.step(point(), A, 5)
        s.step(Observation(OTHER, 0.20, 0.22, True, True), n=10)
        self.assertEqual(s.machine.state, FOLLOW)
        self.assertEqual(s.events("click"), [])

    def test_right_pinch_ends_when_hand_returns_to_pointing_with_thumb_next_to_curled_middle(self):
        # Реальная ситуация: в позе «указание» большой палец лежит рядом с согнутым средним,
        # расстояние между их кончиками маленькое, но это уже не щипок.
        s = Sim()
        s.step(point(), A, 10)
        s.step(rpinch(), n=6)
        s.step(Observation(POINT, 1.2, 0.15, True, False), n=3)
        self.assertEqual(s.events("click"), ["right"])
        self.assertEqual(s.machine.state, FOLLOW)

    def test_drag_ended_with_a_fist_is_released(self):
        s = Sim()
        s.step(point(), A, 5)
        s.step(pinch(), A, 25)
        self.assertEqual(s.machine.state, DRAG)
        s.step(fist(), A, 4)                         # 0.13 с: ещё держим (вдруг дрожание)
        self.assertEqual(s.events("release"), [])
        s.step(fist(), A, 6)                         # дольше 0.25 с: кулак, отпускаем
        self.assertEqual(s.events("release"), ["left"])

    def test_pinch_with_briefly_hidden_index_is_not_released(self):
        s = Sim()
        s.step(point(), A, 5)
        s.step(pinch(), A, 25)
        s.step(Observation(FIST, 0.15, 0.8, False, False), A, 3)   # на 0.1 с указательный «спрятался»
        s.step(pinch(), A, 3)
        self.assertEqual(s.events("release"), [])
        self.assertEqual(s.machine.state, DRAG)


class ScrollTests(unittest.TestCase):
    def setUp(self):
        self.t = 0.0
        self.engine = ScrollEngine()

    def feed(self, y, n=1, size=100.0, engine=None):
        engine = engine or self.engine
        total = 0.0
        for _ in range(n):
            self.t += DT
            obs = None if y is None else ScrollObservation(100.0, y, size)
            total += engine.update(self.t, obs)
        return total

    def activate(self, engine=None, size=100.0):
        self.feed(200.0, 8, size, engine)
        self.assertTrue((engine or self.engine).active)

    def test_needs_a_moment_before_it_turns_on(self):
        self.feed(200.0, 4)
        self.assertFalse(self.engine.active)
        self.feed(200.0, 4)
        self.assertTrue(self.engine.active)

    def test_deadzone_gives_no_scroll(self):
        self.activate()
        self.assertEqual(self.feed(200.0 - 10, 30), 0.0)
        self.assertEqual(self.feed(200.0 + 15, 30), 0.0)

    def test_hand_up_scrolls_up_and_hand_down_scrolls_down(self):
        self.activate()
        self.assertGreater(self.feed(140.0, 30), 0)
        self.feed(200.0, 30)
        self.assertLess(self.feed(260.0, 30), 0)

    def test_speed_grows_with_offset_and_saturates(self):
        speeds = []
        for dy in (35, 60, 90, 400):
            engine = ScrollEngine()
            self.activate(engine)
            self.feed(200.0 - dy, 40, engine=engine)
            speeds.append(engine.speed)
        self.assertEqual(sorted(speeds), speeds)
        self.assertAlmostEqual(speeds[-1], 18.0, places=3)
        self.assertAlmostEqual(speeds[1], 18.0 * (0.4 / 0.8) ** 1.6, delta=0.5)

    def test_invert_flips_direction(self):
        engine = ScrollEngine({"invert": True})
        self.activate(engine)
        self.assertLess(self.feed(140.0, 30, engine=engine), 0)

    def test_same_gesture_same_speed_for_near_and_far_hand(self):
        small, big = ScrollEngine(), ScrollEngine()
        self.activate(small, size=50.0)
        self.activate(big, size=200.0)
        self.feed(200.0 - 30, 40, size=50.0, engine=small)    # 0.6 размера руки
        self.feed(200.0 - 120, 40, size=200.0, engine=big)    # 0.6 размера руки
        self.assertAlmostEqual(small.speed, big.speed, places=3)

    def test_turns_off_after_pose_is_gone(self):
        self.activate()
        self.feed(None, 3)                    # 0.1 с: ещё включено
        self.assertTrue(self.engine.active)
        self.feed(None, 6)
        self.assertFalse(self.engine.active)

    def test_long_pause_between_frames_does_not_cause_a_jump(self):
        self.activate()
        self.feed(140.0, 30)
        self.t += 5.0
        notches = self.engine.update(self.t, ScrollObservation(100.0, 140.0, 100.0))
        self.assertLessEqual(abs(notches), 18.0 * 0.1 + 1e-9)


class RoleTests(unittest.TestCase):
    @staticmethod
    def item(pose, palm=(100.0, 100.0), pinch_index=1.2, pinch_middle=1.2):
        return HandItem(None, SimpleNamespace(pose=pose, palm=palm, pinch_index=pinch_index, pinch_middle=pinch_middle))

    def test_single_two_finger_hand_is_both(self):
        hand = self.item(TWO)
        pointer, scroll = RoleTracker().assign([hand])
        self.assertIs(pointer, hand)
        self.assertIs(scroll, hand)

    def test_single_pointing_hand_does_not_scroll(self):
        hand = self.item(POINT)
        pointer, scroll = RoleTracker().assign([hand])
        self.assertIs(pointer, hand)
        self.assertIsNone(scroll)

    def test_two_hands_split_roles(self):
        scroller, pointer_hand = self.item(TWO, (50, 50)), self.item(POINT, (300, 50))
        pointer, scroll = RoleTracker().assign([pointer_hand, scroller])
        self.assertIs(scroll, scroller)
        self.assertIs(pointer, pointer_hand)

    def test_pinching_hand_is_not_a_scroll_hand(self):
        hand = self.item(TWO, pinch_middle=0.1)
        _, scroll = RoleTracker().assign([hand])
        self.assertIsNone(scroll)

    def test_pointer_role_sticks_to_the_nearest_hand(self):
        roles = RoleTracker()
        left, right = self.item(POINT, (100, 100)), self.item(POINT, (500, 100))
        first, _ = roles.assign([left, right])
        left2, right2 = self.item(POINT, (105, 100)), self.item(POINT, (495, 100))
        second, _ = roles.assign([right2, left2])       # порядок в списке поменялся
        self.assertEqual(first.feat.palm[0] < 300, second.feat.palm[0] < 300)

    def test_scroll_can_be_disabled(self):
        _, scroll = RoleTracker(scroll_enabled=False).assign([self.item(TWO)])
        self.assertIsNone(scroll)


class FakePyAutoGUI(types.ModuleType):
    class FailSafeException(Exception):
        pass

    def __init__(self):
        super().__init__("pyautogui")
        self.FAILSAFE = True
        self.PAUSE = 0.1
        self.pos = (500, 500)
        self.calls = []

    def size(self): return (1920, 1080)
    def position(self): return self.pos

    def _check(self):
        if self.FAILSAFE and self.pos == (0, 0):
            raise self.FailSafeException()

    def moveTo(self, x, y): self._check(); self.pos = (x, y); self.calls.append(("move", x, y))
    def click(self, button="left"): self._check(); self.calls.append(("click", button))
    def mouseDown(self, button="left"): self._check(); self.calls.append(("down", button))
    def mouseUp(self, button="left"): self._check(); self.calls.append(("up", button))
    def scroll(self, amount): self._check(); self.calls.append(("scroll", amount))


class ControllerButtonAndScrollTests(unittest.TestCase):
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

    def make(self, **kw):
        c = self.module.Controller(**kw)
        c.enabled = True
        return c

    def test_buttons_are_tracked(self):
        c = self.make()
        self.assertTrue(c.mouse_down("left"))
        self.assertFalse(c.mouse_down("left"))            # второй раз не нажимаем
        self.assertEqual(c.buttons_down, {"left"})
        c.enabled = False                                 # отпустить можно и при выключенном управлении
        self.assertTrue(c.mouse_up("left"))
        self.assertEqual(c.buttons_down, set())
        self.assertEqual(self.fake.calls, [("down", "left"), ("up", "left")])

    def test_release_all_works_even_in_failsafe_corner(self):
        c = self.make()
        c.mouse_down("left")
        self.fake.pos = (0, 0)                            # без обхода защиты pyautogui кнопка осталась бы зажатой
        c.release_all()
        self.assertIn(("up", "left"), self.fake.calls)
        self.assertEqual(c.buttons_down, set())
        self.assertTrue(self.fake.FAILSAFE, "защита должна включиться обратно")

    def test_switching_control_off_releases_buttons(self):
        c = self.make()
        c.mouse_down("left")
        c.toggle()
        self.assertFalse(c.enabled)
        self.assertEqual(c.buttons_down, set())

    def test_click_and_disabled_state(self):
        c = self.make()
        self.assertTrue(c.click("right"))
        self.assertEqual(self.fake.calls[-1], ("click", "right"))
        c.enabled = False
        self.assertFalse(c.click("left"))
        self.assertFalse(c.mouse_down("left"))

    def test_scroll_on_windows_units(self):
        c = self.make(scroll_units_per_notch=120)
        self.assertEqual(c.scroll(0.2), 0)                # 24 единицы: меньше трети щелчка, копим
        self.assertEqual(c.scroll(0.2), 48)
        self.assertEqual(c.scroll(-1), -120)
        self.assertEqual([call for call in self.fake.calls if call[0] == "scroll"], [("scroll", 48), ("scroll", -120)])

    def test_zero_scroll_clears_leftover(self):
        c = self.make(scroll_units_per_notch=120)
        c.scroll(0.2)
        c.scroll(0)
        self.assertEqual(c.scroll(0.2), 0)

    def test_scroll_on_one_unit_per_notch_systems_keeps_remainder(self):
        c = self.make(scroll_units_per_notch=1)
        self.assertEqual(c.scroll(0.6), 0)
        self.assertEqual(c.scroll(0.6), 1)
        self.assertEqual(c.scroll(0.9), 1)                # остаток 0.2 + 0.9 = 1.1

    def test_scroll_does_nothing_when_disabled(self):
        c = self.make(scroll_units_per_notch=120)
        c.enabled = False
        self.assertEqual(c.scroll(5), 0)
        self.assertEqual(self.fake.calls, [])

    def test_default_units_by_platform(self):
        for platform, expected in (("win32", 120), ("darwin", 3), ("linux", 1)):
            with mock.patch.object(self.module.sys, "platform", platform):
                self.assertEqual(self.module.default_units_per_notch(), expected)


if __name__ == "__main__":
    unittest.main()
