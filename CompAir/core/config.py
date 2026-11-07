"""Configuration loading and validation."""

import copy
import json
from pathlib import Path

SUPPORTED_MODES = ("mouse", "presentation", "game")


class ConfigError(Exception):
    """Raised when config.json cannot be loaded or contains invalid values."""


DEFAULTS = {
    "camera": {"source": 0, "width": 640, "height": 480, "flip": True, "fps": 30, "fourcc": ""},
    "tracker": {
        "max_hands": 2, "model_path": "models/hand_landmarker.task",
        "min_detection_confidence": 0.7, "min_tracking_confidence": 0.6,
    },
    "pointer": {"landmark": 5, "alt_landmark": 8, "lost_reset_seconds": 0.3, "min_move_px": 2},
    "zone": {"x_min": 0.20, "y_min": 0.15, "x_max": 0.80, "y_max": 0.65, "lock_aspect": True},
    "smoothing": {
        "method": "one_euro", "ema_alpha": 0.5,
        "one_euro_min_cutoff": 1.0, "one_euro_beta": 0.03, "one_euro_d_cutoff": 1.0,
        "glide_enabled": True, "glide_tau_seconds": 0.025, "glide_lead_seconds": 0.025, "glide_rate_hz": 125,
    },
    "gestures": {
        "extended_threshold": 1.35, "not_hidden_threshold": 0.95,
        "pinch_on": 0.30, "pinch_off": 0.45, "pinch_approach": 0.70,
        "pinch_confirm_frames": 2, "release_confirm_frames": 2, "pinch_ambiguity_margin": 0.06,
        "drag_hold_seconds": 0.35, "right_click_enabled": True, "right_click_max_seconds": 0.8,
        "fist_release_seconds": 0.25, "pose_hold_seconds": 0.12, "lost_grace_seconds": 0.30,
        "anchor_lookback_default": 0.15, "anchor_lookback_min": 0.05, "anchor_lookback_max": 0.35,
        "clutch_catchup": 0.6,
    },
    "scroll": {
        "enabled": True, "enter_hold_seconds": 0.15, "exit_hold_seconds": 0.12, "deadzone": 0.20,
        "full_speed_offset": 1.00, "max_notches_per_second": 18, "response_power": 1.6,
        "offset_smoothing": 0.5, "invert": False, "units_per_notch": "auto", "min_chunk_units": "auto",
    },
    "modes": {"default": "mouse", "available": ["mouse", "presentation", "game"]},
    "controls": {
        "quit_key": "q", "next_mode_key": "m", "toggle_control_key": "c",
        "toggle_pointer_key": "p", "show_hud": True,
    },
    "safety_switch": {
        "enabled": True, "off_hold_seconds": 3.0, "on_hold_seconds": 2.0, "cooldown_seconds": 1.5, "beep": True,
    },
    "presentation": {
        "next_key": "right", "prev_key": "left", "window_seconds": 0.5, "min_distance": 1.8,
        "min_speed": 4.0, "axis_ratio": 2.0, "cooldown_seconds": 0.8, "opposite_block_seconds": 1.0,
    },
    "game": {"jump_key": "space", "duck_key": "down"},
    "safety": {
        "start_enabled": False, "pyautogui_failsafe": True, "pyautogui_pause": 0.0, "edge_margin_px": 2,
    },
    "debug": {"show_landmarks": True, "show_zone": True, "show_fps": True, "show_timing": True},
}


def _merge(defaults, user, path, warn):
    result = copy.deepcopy(defaults)
    for key, value in user.items():
        where = f"{path}.{key}" if path else key
        if key not in defaults:
            warn(f"В config.json неизвестный параметр «{where}»: возможно, опечатка. Он игнорируется.")
            continue
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"В config.json «{where}» должен быть разделом {{...}}.")
            result[key] = _merge(defaults[key], value, where, warn)
        else:
            result[key] = value
    return result


def _number(cfg, path, low, high, integer=False):
    section, key = path.split(".")
    value = cfg[section][key]
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high
    if ok and integer and int(value) != value:
        ok = False
    if not ok:
        kind = "целым числом" if integer else "числом"
        raise ConfigError(f"В config.json «{path}» должно быть {kind} от {low} до {high}, а сейчас: {value!r}.")


def _flag(cfg, path):
    section, key = path.split(".")
    if not isinstance(cfg[section][key], bool):
        raise ConfigError(f"В config.json «{path}» должно быть true или false, а сейчас: {cfg[section][key]!r}.")


def _key_name(cfg, path):
    section, key = path.split(".")
    value = cfg[section][key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"В config.json «{path}» должно быть названием клавиши в кавычках "
            f'(например, "right" или "space"), а сейчас: {value!r}.'
        )


def _auto_or_number(cfg, path, low, high):
    section, key = path.split(".")
    if cfg[section][key] != "auto":
        _number(cfg, path, low, high)


def validate(cfg):
    cam = cfg["camera"]
    if isinstance(cam["source"], bool) or not isinstance(cam["source"], (int, str)):
        raise ConfigError(
            "В config.json «camera.source» должен быть номером камеры (0, 1, ...) "
            "или адресом в кавычках."
        )
    _number(cfg, "camera.width", 64, 7680, integer=True)
    _number(cfg, "camera.height", 64, 4320, integer=True)
    _number(cfg, "camera.fps", 1, 240, integer=True)
    _flag(cfg, "camera.flip")
    if not isinstance(cam["fourcc"], str) or len(cam["fourcc"]) not in (0, 4):
        raise ConfigError(
            'В config.json «camera.fourcc» должен быть пустым ("") '
            'или из 4 букв, например "MJPG".'
        )

    _number(cfg, "tracker.max_hands", 1, 2, integer=True)
    _number(cfg, "tracker.min_detection_confidence", 0.0, 1.0)
    _number(cfg, "tracker.min_tracking_confidence", 0.0, 1.0)

    _number(cfg, "pointer.landmark", 0, 20, integer=True)
    _number(cfg, "pointer.alt_landmark", 0, 20, integer=True)
    _number(cfg, "pointer.lost_reset_seconds", 0.0, 10.0)
    _number(cfg, "pointer.min_move_px", 0.0, 50.0)

    z = cfg["zone"]
    for key in ("x_min", "y_min", "x_max", "y_max"):
        _number(cfg, f"zone.{key}", 0.0, 1.0)
    if not (z["x_min"] < z["x_max"] and z["y_min"] < z["y_max"]):
        raise ConfigError(
            "В config.json в разделе «zone» x_min должно быть меньше x_max, "
            "а y_min меньше y_max."
        )
    _flag(cfg, "zone.lock_aspect")

    s = cfg["smoothing"]
    if s["method"] not in ("ema", "one_euro"):
        raise ConfigError('В config.json «smoothing.method» должен быть "one_euro" или "ema".')
    _number(cfg, "smoothing.ema_alpha", 0.01, 1.0)
    _number(cfg, "smoothing.one_euro_min_cutoff", 0.01, 100.0)
    _number(cfg, "smoothing.one_euro_beta", 0.0, 10.0)
    _number(cfg, "smoothing.one_euro_d_cutoff", 0.01, 100.0)
    _flag(cfg, "smoothing.glide_enabled")
    _number(cfg, "smoothing.glide_tau_seconds", 0.0, 0.5)
    _number(cfg, "smoothing.glide_lead_seconds", 0.0, 0.2)
    _number(cfg, "smoothing.glide_rate_hz", 30, 500)

    g = cfg["gestures"]
    for key in ("extended_threshold", "not_hidden_threshold", "pinch_on", "pinch_off", "pinch_approach"):
        _number(cfg, f"gestures.{key}", 0.01, 5.0)
    if not g["pinch_on"] < g["pinch_off"]:
        raise ConfigError(
            "В config.json «gestures.pinch_on» должно быть меньше «gestures.pinch_off» "
            "(так работает гистерезис щипка)."
        )
    if g["pinch_approach"] <= g["pinch_on"]:
        raise ConfigError("В config.json «gestures.pinch_approach» должно быть больше «gestures.pinch_on».")
    _number(cfg, "gestures.pinch_confirm_frames", 1, 30, integer=True)
    _number(cfg, "gestures.release_confirm_frames", 1, 30, integer=True)
    for key in (
        "pinch_ambiguity_margin", "drag_hold_seconds", "right_click_max_seconds",
        "fist_release_seconds", "pose_hold_seconds", "lost_grace_seconds",
        "anchor_lookback_default", "anchor_lookback_min", "anchor_lookback_max", "clutch_catchup",
    ):
        _number(cfg, f"gestures.{key}", 0.0, 10.0)
    _flag(cfg, "gestures.right_click_enabled")

    sc = cfg["scroll"]
    _flag(cfg, "scroll.enabled")
    _flag(cfg, "scroll.invert")
    for key in ("enter_hold_seconds", "exit_hold_seconds"):
        _number(cfg, f"scroll.{key}", 0.0, 5.0)
    _number(cfg, "scroll.deadzone", 0.0, 3.0)
    _number(cfg, "scroll.full_speed_offset", 0.05, 5.0)
    if sc["deadzone"] >= sc["full_speed_offset"]:
        raise ConfigError("В config.json «scroll.deadzone» должно быть меньше «scroll.full_speed_offset».")
    _number(cfg, "scroll.max_notches_per_second", 0.1, 200.0)
    _number(cfg, "scroll.response_power", 0.2, 5.0)
    _number(cfg, "scroll.offset_smoothing", 0.01, 1.0)
    _auto_or_number(cfg, "scroll.units_per_notch", 1, 10000)
    _auto_or_number(cfg, "scroll.min_chunk_units", 1, 10000)

    modes = cfg["modes"]
    available = modes["available"]
    if not isinstance(available, list) or not available:
        raise ConfigError("В config.json «modes.available» должен быть непустым списком режимов.")
    if not all(isinstance(name, str) for name in available):
        raise ConfigError("В config.json «modes.available» должен содержать названия режимов в кавычках.")
    unknown = [name for name in available if name not in SUPPORTED_MODES]
    if unknown:
        raise ConfigError(
            f"В config.json «modes.available» содержит неизвестный режим: {', '.join(unknown)}. "
            f"Доступны: {', '.join(SUPPORTED_MODES)}."
        )
    if modes["default"] not in available:
        raise ConfigError("В config.json «modes.default» должен быть одним из «modes.available».")

    keys = [
        cfg["controls"][k]
        for k in ("quit_key", "next_mode_key", "toggle_control_key", "toggle_pointer_key")
    ]
    if not all(isinstance(k, str) and len(k) == 1 for k in keys):
        raise ConfigError("В config.json клавиши в разделе «controls» должны быть одной буквой или цифрой.")
    if len(set(keys)) != len(keys):
        raise ConfigError("В config.json клавиши в разделе «controls» не должны повторяться.")
    _flag(cfg, "controls.show_hud")

    for key in ("start_enabled", "pyautogui_failsafe"):
        _flag(cfg, f"safety.{key}")
    _number(cfg, "safety.pyautogui_pause", 0.0, 5.0)
    _number(cfg, "safety.edge_margin_px", 0, 100)

    _flag(cfg, "safety_switch.enabled")
    _flag(cfg, "safety_switch.beep")
    for key in ("off_hold_seconds", "on_hold_seconds", "cooldown_seconds"):
        _number(cfg, f"safety_switch.{key}", 0.2, 60.0)

    for key in ("next_key", "prev_key"):
        _key_name(cfg, f"presentation.{key}")
    for key in (
        "window_seconds", "min_distance", "min_speed", "axis_ratio",
        "cooldown_seconds", "opposite_block_seconds",
    ):
        _number(cfg, f"presentation.{key}", 0.05, 50.0)
    for key in ("jump_key", "duck_key"):
        _key_name(cfg, f"game.{key}")
    return cfg


def load_config(path, warn=print):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"Не найден файл настроек: {path}") from None
    try:
        user = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError(f"Ошибка в config.json (строка {error.lineno}): {error.msg}") from None
    if not isinstance(user, dict):
        raise ConfigError("config.json должен содержать объект {...}.")
    return validate(_merge(DEFAULTS, user, "", warn))
