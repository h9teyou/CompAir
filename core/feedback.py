"""Audible feedback for control state changes."""

import sys
import threading

TONES = {"on": (880, 130), "off": (440, 260), "action": (660, 70)}


def _play(frequency, milliseconds):
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(frequency, milliseconds)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


def beep(kind):
    frequency, milliseconds = TONES.get(kind, TONES["action"])
    threading.Thread(target=_play, args=(frequency, milliseconds), daemon=True).start()
