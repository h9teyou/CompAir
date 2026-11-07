"""Frame capture with a background reader thread."""

import sys
import threading
import time

import cv2


class CameraStream:
    MAX_FAILED_READS = 100

    def __init__(self, source=0, width=640, height=480, fps=None, fourcc=None):
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._fourcc = fourcc
        self.measured_fps = 0.0

        self._cap = None
        self._thread = None
        self._running = False

        self._frame = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._failed = False

    def start(self):
        self._cap = self._open_capture()
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(
                f"Не удалось открыть камеру (source={self._source!r}). "
                "Проверь номер камеры в config.json, не занята ли она другой программой "
                "и выдано ли разрешение на доступ к камере."
            )

        if self._fourcc and isinstance(self._source, int):
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps:
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        return self

    def _open_capture(self):
        if isinstance(self._source, int) and sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(self._source)

    def _update_loop(self):
        failed_reads = 0
        last_frame_time = None
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= self.MAX_FAILED_READS:
                    self._failed = True
                    self._running = False
                    break
                time.sleep(0.02)
                continue

            failed_reads = 0
            now = time.perf_counter()
            if last_frame_time is not None and now > last_frame_time:
                current = 1.0 / (now - last_frame_time)
                self.measured_fps = (
                    current if self.measured_fps == 0.0 else 0.9 * self.measured_fps + 0.1 * current
                )
            last_frame_time = now
            with self._lock:
                self._frame = frame
            self._new_frame.set()

    def read(self, timeout=0.1):
        """Returns a copy of the newest frame or None if no frame arrived in time."""
        if self._failed:
            raise RuntimeError("Камера перестала отдавать кадры. Проверь подключение.")

        if not self._new_frame.wait(timeout):
            return None
        self._new_frame.clear()

        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
