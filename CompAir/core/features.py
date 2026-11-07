"""Scale and rotation invariant hand features computed from MediaPipe landmarks."""

import math
from dataclasses import dataclass

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20

FINGER_POINTS = {
    "index": (INDEX_MCP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_TIP),
}

FIST = "FIST"
POINT = "POINT"
TWO = "TWO"
PALM = "PALM"
OTHER = "OTHER"

EXTENDED_THRESHOLD = 1.35
NOT_HIDDEN_THRESHOLD = 1.05


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_size(points):
    """Palm length in pixels. Never returns zero."""
    return max(dist(points[WRIST], points[MIDDLE_MCP]), 1e-6)


def palm_center(points):
    ids = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
    return (
        sum(points[i][0] for i in ids) / len(ids),
        sum(points[i][1] for i in ids) / len(ids),
    )


def finger_reach(points, mcp_id, tip_id):
    base = max(dist(points[WRIST], points[mcp_id]), 1e-6)
    return dist(points[WRIST], points[tip_id]) / base


def pinch_ratio(points, finger_tip_id):
    return dist(points[THUMB_TIP], points[finger_tip_id]) / hand_size(points)


def classify_pose(extended):
    index, middle = extended["index"], extended["middle"]
    ring, pinky = extended["ring"], extended["pinky"]
    count = sum(extended.values())

    if count == 0:
        return FIST
    if index and not middle and not ring and not pinky:
        return POINT
    if index and middle and not ring and not pinky:
        return TWO
    if count >= 3:
        return PALM
    return OTHER


@dataclass
class HandFeatures:
    size: float
    palm: tuple
    reach: dict
    extended: dict
    pose: str
    pinch_index: float
    pinch_middle: float
    index_ok: bool
    middle_ok: bool


def analyze(points, extended_threshold=EXTENDED_THRESHOLD, not_hidden_threshold=NOT_HIDDEN_THRESHOLD):
    reach = {name: finger_reach(points, mcp, tip) for name, (mcp, tip) in FINGER_POINTS.items()}
    extended = {name: value > extended_threshold for name, value in reach.items()}

    return HandFeatures(
        size=hand_size(points),
        palm=palm_center(points),
        reach=reach,
        extended=extended,
        pose=classify_pose(extended),
        pinch_index=pinch_ratio(points, INDEX_TIP),
        pinch_middle=pinch_ratio(points, MIDDLE_TIP),
        index_ok=reach["index"] > not_hidden_threshold,
        middle_ok=reach["middle"] > not_hidden_threshold,
    )
