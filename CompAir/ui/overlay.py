"""Camera frame overlay."""

import cv2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
FINGERTIPS = (4, 8, 12, 16, 20)

COLOR_BONE = (0, 200, 255)
COLOR_SCROLL_BONE = (0, 140, 255)
COLOR_JOINT = (0, 255, 120)
COLOR_INDEX_TIP = (0, 0, 255)
COLOR_THUMB_TIP = (255, 0, 255)
COLOR_ZONE = (255, 200, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_OK = (0, 255, 120)
COLOR_WARN = (0, 180, 255)
COLOR_PINCH = (0, 255, 0)
COLOR_RPINCH = (255, 0, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_zone(frame, zone):
    height, width = frame.shape[:2]
    x1, y1 = int(zone["x_min"] * width), int(zone["y_min"] * height)
    x2, y2 = int(zone["x_max"] * width), int(zone["y_max"] * height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ZONE, 2)
    cv2.putText(frame, "CONTROL ZONE", (x1 + 6, y2 - 8), FONT, 0.5, COLOR_ZONE, 1, cv2.LINE_AA)


def draw_hand(frame, hand, bone_color=COLOR_BONE):
    points = [(int(x), int(y)) for x, y in hand.points]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], bone_color, 2, cv2.LINE_AA)

    for index, point in enumerate(points):
        radius = 6 if index in FINGERTIPS else 3
        cv2.circle(frame, point, radius, COLOR_JOINT, -1, cv2.LINE_AA)

    cv2.circle(frame, points[8], 11, COLOR_INDEX_TIP, 2, cv2.LINE_AA)
    cv2.circle(frame, points[4], 11, COLOR_THUMB_TIP, 2, cv2.LINE_AA)


def draw_pinch_link(frame, hand, state):
    points = [(int(x), int(y)) for x, y in hand.points]
    if state in ("PINCH", "DRAG"):
        cv2.line(frame, points[4], points[8], COLOR_PINCH, 5, cv2.LINE_AA)
        label_at = points[8]
        cv2.putText(frame, "DRAG" if state == "DRAG" else "CLICK", (label_at[0] + 14, label_at[1] - 8),
                    FONT, 0.6, COLOR_PINCH, 2, cv2.LINE_AA)
    elif state == "RPINCH":
        cv2.line(frame, points[4], points[12], COLOR_RPINCH, 5, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT CLICK", (points[12][0] + 14, points[12][1] - 8),
                    FONT, 0.6, COLOR_RPINCH, 2, cv2.LINE_AA)


def draw_scroll(frame, scroller):
    if not scroller.active:
        return
    x = int(scroller.anchor_x)
    y = int(scroller.anchor_y)
    size = scroller.anchor_size
    dead = int(scroller.cfg["deadzone"] * size)
    half_width = 80

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - half_width, y - dead), (x + half_width, y + dead), COLOR_WARN, -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.line(frame, (x - half_width, y), (x + half_width, y), COLOR_WARN, 2, cv2.LINE_AA)

    tip_y = int(y - scroller.offset * size)
    color = COLOR_OK if abs(scroller.speed) > 0.01 else COLOR_TEXT
    cv2.arrowedLine(frame, (x, y), (x, tip_y), color, 3, cv2.LINE_AA, tipLength=0.25)

    if abs(scroller.speed) > 0.01:
        label = f"SCROLL {'UP' if scroller.speed > 0 else 'DOWN'} {abs(scroller.speed):.0f}/s"
    else:
        label = "SCROLL: move hand up/down"
    text_width = cv2.getTextSize(label, FONT, 0.55, 1)[0][0]
    label_x = x + half_width + 8
    if label_x + text_width > frame.shape[1] - 4:
        label_x = max(4, x - half_width - 8 - text_width)
    cv2.putText(frame, label, (label_x, y + 5), FONT, 0.55, color, 1, cv2.LINE_AA)


def draw_hud(frame, mode, fps=None, hand_found=False, control_on=None, cursor=None, extra=None):
    height, width = frame.shape[:2]

    lines = [(f"MODE: {mode.upper()}", COLOR_TEXT)]
    if fps is not None:
        lines.append((f"FPS: {fps:.0f}", COLOR_TEXT))
    if hand_found:
        lines.append(("HAND: tracking", COLOR_OK))
    else:
        lines.append(("HAND: show your hand", COLOR_WARN))
    if control_on is not None:
        if control_on:
            lines.append(("CONTROL: ON", COLOR_OK))
        else:
            lines.append(("CONTROL: OFF (press c)", COLOR_WARN))
        if cursor is not None:
            lines.append((f"CURSOR: {int(cursor[0])}, {int(cursor[1])}", COLOR_TEXT))
        else:
            lines.append(("CURSOR: -", COLOR_TEXT))
    if extra:
        lines.extend(extra)

    line_height = 21
    panel_height = 12 + line_height * len(lines)
    longest = max(cv2.getTextSize(text, FONT, 0.52, 1)[0][0] for text, _ in lines)
    panel_width = max(282, longest + 24)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 29
    for text, color in lines:
        cv2.putText(frame, text, (16, y), FONT, 0.52, color, 1, cv2.LINE_AA)
        y += line_height

    cv2.putText(
        frame, "q quit   m mode   c control   p pointer",
        (10, height - 12), FONT, 0.5, COLOR_TEXT, 1, cv2.LINE_AA,
    )


def draw_notice(frame, text, color=COLOR_OK):
    scale, thickness = 1.1, 3
    size = cv2.getTextSize(text, FONT, scale, thickness)[0]
    x = max(10, (frame.shape[1] - size[0]) // 2)
    cv2.putText(frame, text, (x + 2, 52), FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, 50), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_progress(frame, fraction, label):
    height, width = frame.shape[:2]
    bar_w, bar_h = 300, 16
    x, y = (width - bar_w) // 2, height - 50
    cv2.rectangle(frame, (x - 2, y - 2), (x + bar_w + 2, y + bar_h + 2), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y), (x + int(bar_w * fraction), y + bar_h), COLOR_WARN, -1)
    cv2.putText(frame, label, (x, y - 8), FONT, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
