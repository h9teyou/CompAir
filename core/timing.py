"""Windows timer resolution helper."""

import sys


def enable_high_res_timer(milliseconds=1):
    """Requests a high resolution timer and returns a restore function."""
    if not sys.platform.startswith("win"):
        return lambda: None
    try:
        import ctypes
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod(milliseconds)
    except Exception:
        return lambda: None

    def restore():
        try:
            winmm.timeEndPeriod(milliseconds)
        except Exception:
            pass

    return restore
