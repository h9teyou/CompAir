"""Синтетическая рука для тестов без камеры.

Длина ладони принята за единицу. Порядок точек совпадает с MediaPipe.
"""

import math
import random

# Основание пальца (x, y) на ладони и длины трёх костей пальца.
FINGERS = {
    "index":  {"mcp": (-0.30, 0.95), "lens": (0.42, 0.24, 0.20), "spread": -4},
    "middle": {"mcp": (0.00, 1.00),  "lens": (0.47, 0.28, 0.21), "spread": 0},
    "ring":   {"mcp": (0.28, 0.93),  "lens": (0.43, 0.26, 0.20), "spread": 4},
    "pinky":  {"mcp": (0.54, 0.80),  "lens": (0.33, 0.19, 0.17), "spread": 9},
}
FINGER_ORDER = ("index", "middle", "ring", "pinky")

# Углы сгибания суставов при полностью согнутом пальце (кулак), градусы.
FIST_FLEX = (85.0, 100.0, 60.0)

THUMB_BASE = (-0.22, 0.25)
THUMB_TUCK_TIP = (-0.12, 0.62)     # кончик большого, прижатого к ладони
THUMB_OPEN_TIP = (-0.95, 0.85)     # кончик отведённого большого


def _finger_chain(mcp_xy, lens, spread_deg, curl):
    """Возвращает 4 точки пальца (основание, PIP, DIP, кончик) в 3D. curl: 0 = прямой, 1 = кулак."""
    a1, a2, a3 = (curl * f for f in FIST_FLEX)
    cumulative = (a1, a1 + a2, a1 + a2 + a3)
    spread = math.radians(spread_deg)
    x, y, z = mcp_xy[0], mcp_xy[1], 0.0
    points = [(x, y, z)]
    for length, flex_deg in zip(lens, cumulative):
        phi = math.radians(flex_deg)
        x += length * math.sin(spread) * math.cos(phi)
        y += length * math.cos(spread) * math.cos(phi)
        z += length * math.sin(phi)
        points.append((x, y, z))
    return points


def _thumb_chain(tip_xy, tip_z=0.0):
    """Большой палец: основание фиксировано, кончик в заданной точке, середина по прямой."""
    cmc = (THUMB_BASE[0], THUMB_BASE[1], 0.0)
    tip = (tip_xy[0], tip_xy[1], tip_z)
    mcp = tuple(c + 0.40 * (t - c) for c, t in zip(cmc, tip))
    ip = tuple(c + 0.72 * (t - c) for c, t in zip(cmc, tip))
    return [cmc, mcp, ip, tip]


def build_hand(curls, thumb_tip=THUMB_TUCK_TIP, thumb_z=0.0, gap=(0.0, 0.0)):
    """
    Собирает 21 точку в 3D.
    curls: словарь {"index": 0..1, "middle": ..., "ring": ..., "pinky": ...}
    thumb_tip: (x, y) кончика большого пальца. Можно передать "on_index" или "on_middle",
               тогда кончик большого ставится на кончик соответствующего пальца.
    gap: небольшой зазор (x, y) между кончиками при щипке. У настоящей руки кончики
         «касаются» подушечками, поэтому расстояние между точками никогда не равно нулю.
    """
    chains = {}
    for name in FINGER_ORDER:
        spec = FINGERS[name]
        chains[name] = _finger_chain(spec["mcp"], spec["lens"], spec["spread"], curls[name])

    if thumb_tip == "on_index":
        tip = chains["index"][3]
        thumb = _thumb_chain((tip[0] + gap[0], tip[1] + gap[1]), tip[2])
    elif thumb_tip == "on_middle":
        tip = chains["middle"][3]
        thumb = _thumb_chain((tip[0] + gap[0], tip[1] + gap[1]), tip[2])
    else:
        thumb = _thumb_chain(thumb_tip, thumb_z)

    points = [(0.0, 0.0, 0.0)] + thumb
    for name in FINGER_ORDER:
        points += chains[name]
    return points


# ---------------------------------------------------------------- позы

def pose_open():
    return build_hand({"index": 0.02, "middle": 0.0, "ring": 0.02, "pinky": 0.05}, thumb_tip=THUMB_OPEN_TIP)


def pose_point(thumb_open=False):
    curls = {"index": 0.02, "middle": 0.95, "ring": 0.95, "pinky": 0.95}
    return build_hand(curls, THUMB_OPEN_TIP if thumb_open else THUMB_TUCK_TIP)


def pose_two():
    curls = {"index": 0.02, "middle": 0.02, "ring": 0.95, "pinky": 0.95}
    return build_hand(curls, THUMB_TUCK_TIP)


def pose_fist():
    curls = {"index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}
    return build_hand(curls, THUMB_TUCK_TIP)


def pose_pinch(index_curl=0.35, others="curled", gap=(0.0, 0.0)):
    """Щипок: кончик большого на кончике указательного. Остальные пальцы согнуты или раскрыты."""
    other = 0.95 if others == "curled" else 0.05
    curls = {"index": index_curl, "middle": other, "ring": other, "pinky": other}
    return build_hand(curls, thumb_tip="on_index", gap=gap)


def pose_middle_pinch(middle_curl=0.35, gap=(0.0, 0.0)):
    """Щипок большим и средним (правый клик). Указательный вытянут."""
    curls = {"index": 0.02, "middle": middle_curl, "ring": 0.95, "pinky": 0.95}
    return build_hand(curls, thumb_tip="on_middle", gap=gap)


# ---------------------------------------------------------------- проекция

def _rot_x(p, a):
    x, y, z = p
    return (x, y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a))


def _rot_y(p, a):
    x, y, z = p
    return (x * math.cos(a) + z * math.sin(a), y, -x * math.sin(a) + z * math.cos(a))


def project(points3d, rng, tilt_deg=30.0, scale=None, roll_deg=None, center=(320.0, 240.0), noise=0.004):
    """
    Поворачивает руку (наклон вперёд/назад и вбок, поворот в плоскости кадра),
    масштабирует, добавляет шум и возвращает 21 точку в пикселях кадра (как hand.points).
    """
    pitch = math.radians(rng.uniform(-tilt_deg, tilt_deg))
    yaw = math.radians(rng.uniform(-tilt_deg, tilt_deg))
    roll = math.radians(rng.uniform(-35, 35) if roll_deg is None else roll_deg)
    size = rng.uniform(60, 170) if scale is None else scale  # длина ладони в пикселях

    pixels = []
    for p in points3d:
        p = _rot_x(p, pitch)
        p = _rot_y(p, yaw)
        x, y = p[0], p[1]
        # поворот в плоскости кадра
        xr = x * math.cos(roll) - y * math.sin(roll)
        yr = x * math.sin(roll) + y * math.cos(roll)
        # в кадре ось y направлена вниз
        px = center[0] + size * (xr + rng.gauss(0, noise))
        py = center[1] - size * (yr + rng.gauss(0, noise))
        pixels.append((px, py))
    return pixels


def make_rng(seed=0):
    return random.Random(seed)
