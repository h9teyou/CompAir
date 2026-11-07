"""CompAir entry point."""

import sys
import time
from pathlib import Path

import cv2

from core.camera import CameraStream
from core.config import ConfigError, load_config
from core.controller import Controller, EmergencyStop
from core.feedback import beep
from core.glider import CursorGlider
from core.hand_tracker import HandTracker
from core.pipeline import GesturePipeline
from core.state_machine import DRAG, FROZEN, PINCH, RPINCH
from core.switch import ControlSwitch
from core.timing import enable_high_res_timer
from modes.game import GameMode
from modes.presentation import PresentationMode
from ui.overlay import (
    COLOR_BONE, COLOR_OK, COLOR_SCROLL_BONE, COLOR_TEXT, COLOR_WARN,
    draw_hand, draw_hud, draw_notice, draw_pinch_link, draw_progress, draw_scroll, draw_zone,
)

CONFIG_PATH = Path(__file__).parent / "config.json"
WINDOW_NAME = "CompAir"


class FpsCounter:
    """Smoothed frames per second counter."""

    def __init__(self, smoothing=0.9):
        self.fps = 0.0
        self._last = time.perf_counter()
        self._smoothing = smoothing

    def tick(self):
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0:
            current = 1.0 / dt
            if self.fps == 0.0:
                self.fps = current
            else:
                self.fps = self._smoothing * self.fps + (1 - self._smoothing) * current
        return self.fps


def apply_output(controller, glider, out):
    """Applies a pointer machine output to the real mouse."""
    if out.cursor is not None:
        if glider is None:
            controller.move_to(*out.cursor)
        elif out.jump:
            glider.snap(*out.cursor)
        else:
            glider.set_target(*out.cursor)
    if glider is not None and (out.press or out.click or out.release):
        glider.flush()
    if out.press:
        controller.mouse_down(out.press)
    if out.click:
        controller.click(out.click)
    if out.release:
        controller.mouse_up(out.release)


def main():
    try:
        cfg = load_config(CONFIG_PATH)
    except ConfigError as error:
        sys.exit(str(error))

    cam_cfg, trk_cfg, ptr_cfg = cfg["camera"], cfg["tracker"], cfg["pointer"]
    smo_cfg, scr_cfg = cfg["smoothing"], cfg["scroll"]
    dbg_cfg, ctl_cfg, saf_cfg = cfg["debug"], cfg["controls"], cfg["safety"]

    modes = cfg["modes"]["available"]
    mode_index = modes.index(cfg["modes"]["default"])

    quit_key = ord(ctl_cfg["quit_key"])
    next_mode_key = ord(ctl_cfg["next_mode_key"])
    toggle_control_key = ord(ctl_cfg["toggle_control_key"])
    toggle_pointer_key = ord(ctl_cfg["toggle_pointer_key"])

    camera = CameraStream(
        source=cam_cfg["source"],
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
        fourcc=cam_cfg["fourcc"] or None,
    )
    tracker = controller = glider = pipeline = None
    mode_objects = {}
    switch = ControlSwitch(dict(cfg["safety_switch"]))
    notice_text, notice_until = None, 0.0
    fps_counter = FpsCounter()
    infer_ms = 0.0
    restore_timer = enable_high_res_timer()

    try:
        tracker = HandTracker(
            model_path=trk_cfg["model_path"],
            max_hands=trk_cfg["max_hands"],
            min_detection_confidence=trk_cfg["min_detection_confidence"],
            min_tracking_confidence=trk_cfg["min_tracking_confidence"],
        )
        controller = Controller(
            failsafe=saf_cfg["pyautogui_failsafe"],
            pause=saf_cfg["pyautogui_pause"],
            edge_margin=saf_cfg["edge_margin_px"],
            min_move_px=ptr_cfg["min_move_px"],
            scroll_units_per_notch=scr_cfg["units_per_notch"],
            scroll_min_chunk=scr_cfg["min_chunk_units"],
        )
        controller.enabled = bool(saf_cfg["start_enabled"])
        if smo_cfg["glide_enabled"]:
            glider = CursorGlider(
                controller.move_exact, controller.position,
                tau=smo_cfg["glide_tau_seconds"], lead=smo_cfg["glide_lead_seconds"],
                rate_hz=smo_cfg["glide_rate_hz"], deadzone_px=ptr_cfg["min_move_px"],
            ).start()
        camera.start()

        while True:
            if glider is not None and glider.error is not None:
                raise glider.error

            frame = camera.read()
            if frame is None:
                if cv2.waitKey(10) & 0xFF == quit_key:
                    break
                continue

            if cam_cfg["flip"]:
                frame = cv2.flip(frame, 1)

            if pipeline is None:
                frame_h, frame_w = frame.shape[:2]
                pipeline = GesturePipeline(cfg, (frame_w, frame_h), controller.screen_size)
                mode_objects = {
                    "presentation": PresentationMode(cfg["presentation"], controller),
                    "game": GameMode(cfg["game"], controller),
                }

            started = time.perf_counter()
            hands = tracker.process_all(frame)
            infer_ms = 0.9 * infer_ms + 0.1 * (time.perf_counter() - started) * 1000

            now = time.perf_counter()
            result = pipeline.update(hands, now)
            out, pointer_item = result.out, result.pointer_item
            scroll_item = result.scroll_item
            current_mode = modes[mode_index]

            event = switch.update(now, result.items, out.state, controller.enabled)
            if event is not None:
                controller.set_enabled(event == "on")
                if glider is not None:
                    glider.clear()
                pipeline.reset()
                for mode_object in mode_objects.values():
                    mode_object.reset()
                notice_text = "CONTROL ON" if event == "on" else "CONTROL OFF"
                notice_until = now + 2.0
                if cfg["safety_switch"]["beep"]:
                    beep(event)

            mode_object = mode_objects.get(current_mode)
            if current_mode == "mouse" and controller.enabled:
                apply_output(controller, glider, out)
                controller.scroll(result.notches)
            elif mode_object is not None and controller.enabled:
                mode_object.update(result, now)
            elif controller.buttons_down or controller.keys_down:
                controller.release_all()

            fps = fps_counter.tick()
            if ctl_cfg["show_hud"]:
                pose = pointer_item.feat.pose if pointer_item is not None else "-"
                state_color = COLOR_OK if out.state in (PINCH, DRAG, RPINCH) else (
                    COLOR_WARN if out.state == FROZEN else COLOR_TEXT
                )
                extra = [(f"POSE: {pose}", COLOR_TEXT), (f"STATE: {out.state}", state_color)]
                if pointer_item is not None:
                    extra.append((
                        f"PINCH: {pointer_item.feat.pinch_index:.2f} / "
                        f"{pointer_item.feat.pinch_middle:.2f}",
                        COLOR_TEXT,
                    ))
                extra.append((f"POINTER: {pipeline.pointer_name}", COLOR_TEXT))
                if pipeline.scroller.active:
                    extra.append((f"SCROLL: {pipeline.scroller.speed:+.1f}/s", COLOR_OK))
                if dbg_cfg["show_timing"]:
                    extra.append((
                        f"CAM: {camera.measured_fps:.0f} fps  AI: {infer_ms:.0f} ms",
                        COLOR_TEXT,
                    ))
                if mode_object is not None and current_mode != "mouse":
                    extra.extend(mode_object.hud_lines())
                draw_hud(
                    frame,
                    mode=current_mode,
                    fps=fps if dbg_cfg["show_fps"] else None,
                    hand_found=bool(result.items),
                    control_on=controller.enabled and current_mode == "mouse",
                    cursor=pipeline.machine.cursor,
                    extra=extra,
                )
            if dbg_cfg["show_zone"]:
                draw_zone(frame, pipeline.zone.bounds)
            if dbg_cfg["show_landmarks"]:
                for item in result.items:
                    only_scroll = item is scroll_item and item is not pointer_item
                    draw_hand(frame, item.hand, COLOR_SCROLL_BONE if only_scroll else COLOR_BONE)
                if pointer_item is not None:
                    draw_pinch_link(frame, pointer_item.hand, out.state)
            draw_scroll(frame, pipeline.scroller)
            if switch.progress > 0:
                draw_progress(frame, switch.progress, f"HOLD: CONTROL {switch.target}")
            text = notice_text if now < notice_until else None
            if text is None and mode_object is not None and current_mode != "mouse":
                text = mode_object.notice(now)
            if text:
                draw_notice(frame, text, COLOR_WARN if "OFF" in text else COLOR_OK)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == quit_key:
                break
            if key == next_mode_key:
                mode_index = (mode_index + 1) % len(modes)
                if glider is not None:
                    glider.clear()
                controller.release_all()
                pipeline.reset()
                for other_mode in mode_objects.values():
                    other_mode.reset()
                switch.reset()
            if key == toggle_control_key:
                controller.toggle()
                if glider is not None:
                    glider.clear()
                pipeline.reset()
                for other_mode in mode_objects.values():
                    other_mode.reset()
            if key == toggle_pointer_key:
                if glider is not None:
                    glider.clear()
                controller.release_all()
                pipeline.toggle_pointer()

    except EmergencyStop:
        sys.exit("Аварийная остановка: курсор оказался в углу экрана (защита pyautogui).")
    except RuntimeError as error:
        sys.exit(f"Ошибка: {error}")
    except KeyboardInterrupt:
        pass
    finally:
        if glider is not None:
            glider.stop()
        if controller is not None:
            controller.release_all()
        camera.stop()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()
        restore_timer()


if __name__ == "__main__":
    main()
