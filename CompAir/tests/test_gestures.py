"""Тесты распознавания жестов на искусственной руке (tests/hand_model.py)."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hand_model as hm
from core import features as F
from core.state_machine import PointerMachine, LEFT, RIGHT

N = 400
NOISE = 0.015
TILT = 40.0


def gap(rng, magnitude):
    angle = rng.uniform(0, 2 * math.pi)
    length = rng.uniform(0.5 * magnitude, magnitude)
    return (length * math.cos(angle), length * math.sin(angle))


def samples(make, seed, n=N):
    rng = hm.make_rng(seed)
    return [F.analyze(hm.project(make(rng), rng, tilt_deg=TILT, noise=NOISE)) for _ in range(n)]


class PoseAccuracyTests(unittest.TestCase):
    def assert_pose(self, make, expected, seed):
        feats = samples(make, seed)
        correct = sum(1 for f in feats if f.pose == expected) / len(feats)
        self.assertGreaterEqual(correct, 0.99, f"{expected}: верно только {correct:.1%}")

    def test_open_palm(self):
        self.assert_pose(lambda rng: hm.pose_open(), F.PALM, 1)

    def test_point(self):
        self.assert_pose(lambda rng: hm.pose_point(thumb_open=rng.random() < 0.5), F.POINT, 2)

    def test_two_fingers(self):
        self.assert_pose(lambda rng: hm.pose_two(), F.TWO, 3)

    def test_fist(self):
        self.assert_pose(lambda rng: hm.pose_fist(), F.FIST, 4)


class PinchTests(unittest.TestCase):
    def setUp(self):
        self.machine = PointerMachine()

    def kinds(self, feats):
        from core.state_machine import observation_from
        return [self.machine._pinch_start_kind(observation_from(f)) for f in feats]

    def test_left_pinch_is_found_in_most_frames(self):
        for others in ("curled", "open"):
            feats = samples(lambda rng: hm.pose_pinch(rng.choice([0.15, 0.3, 0.45]), others, gap(rng, 0.15)), 10)
            kinds = self.kinds(feats)
            found = sum(1 for k in kinds if k == LEFT) / len(kinds)
            wrong = sum(1 for k in kinds if k == RIGHT) / len(kinds)
            self.assertGreaterEqual(found, 0.88, f"щипок ({others}) найден только в {found:.1%} кадров")
            self.assertLessEqual(wrong, 0.01, f"щипок ({others}) принят за правый в {wrong:.1%} кадров")

    def test_right_pinch_is_found_and_rarely_confused(self):
        feats = samples(lambda rng: hm.pose_middle_pinch(rng.choice([0.2, 0.35, 0.5]), gap(rng, 0.15)), 11)
        kinds = self.kinds(feats)
        found = sum(1 for k in kinds if k == RIGHT) / len(kinds)
        wrong = sum(1 for k in kinds if k == LEFT) / len(kinds)
        self.assertGreaterEqual(found, 0.80)
        self.assertLessEqual(wrong, 0.03, f"правый щипок принят за левый в {wrong:.1%} кадров")

    def test_no_false_pinch_in_other_poses(self):
        makers = {
            "ладонь": lambda rng: hm.pose_open(),
            "указание": lambda rng: hm.pose_point(thumb_open=rng.random() < 0.5),
            "два пальца": lambda rng: hm.pose_two(),
            "кулак": lambda rng: hm.pose_fist(),
        }
        for seed, (name, make) in enumerate(makers.items(), start=20):
            kinds = self.kinds(samples(make, seed, n=800))
            false_rate = sum(1 for k in kinds if k is not None) / len(kinds)
            self.assertEqual(false_rate, 0.0, f"ложный щипок в позе «{name}»: {false_rate:.2%}")

    def test_fist_never_counts_as_pinch_even_though_fingertips_touch(self):
        feats = samples(lambda rng: hm.pose_fist(), 30, n=800)
        close = [f for f in feats if f.pinch_index < 0.30]
        self.assertGreater(len(close), 50, "в кулаке кончики часто рядом, этот тест это и проверяет")
        self.assertTrue(all(not f.index_ok for f in close))


class InvarianceTests(unittest.TestCase):
    def test_features_do_not_depend_on_scale_and_roll(self):
        base = hm.pose_pinch(0.3, "curled", (0.05, 0.02))
        rng = hm.make_rng(0)
        reference = F.analyze(hm.project(base, rng, tilt_deg=0, scale=100, roll_deg=0, noise=0))
        for scale in (50, 100, 200):
            for roll in (-90, -30, 0, 45, 120, 180):
                rng = hm.make_rng(0)
                f = F.analyze(hm.project(base, rng, tilt_deg=0, scale=scale, roll_deg=roll, noise=0))
                self.assertAlmostEqual(f.pinch_index, reference.pinch_index, places=6)
                self.assertEqual(f.pose, reference.pose)

    def test_hand_size_never_zero(self):
        points = [(10.0, 10.0)] * 21
        self.assertGreater(F.hand_size(points), 0)


if __name__ == "__main__":
    unittest.main()
