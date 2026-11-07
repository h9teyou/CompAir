"""MediaPipe Hand Landmarker wrapper."""

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass
class Hand:
    points: list
    norm_points: list
    handedness: str
    score: float


def _ensure_model(model_path):
    path = Path(model_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists():
        return path

    print(f"Модель руки не найдена. Скачиваю в {path} (один раз, около 8 МБ)...")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        tmp_path.replace(path)
    except Exception as error:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            "Не удалось скачать модель руки. Скачай файл вручную по ссылке\n"
            f"  {MODEL_URL}\n"
            f"и положи его сюда: {path}"
        ) from error
    print("Модель скачана.")
    return path


class HandTracker:
    def __init__(
        self,
        model_path="models/hand_landmarker.task",
        max_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ):
        path = _ensure_model(model_path)

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        self._start_time = time.perf_counter()
        self._last_timestamp_ms = -1

    def process_all(self, frame_bgr):
        height, width = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands = []
        for index, landmarks in enumerate(result.hand_landmarks or []):
            points = [(lm.x * width, lm.y * height) for lm in landmarks]
            norm_points = [(lm.x, lm.y, lm.z) for lm in landmarks]

            handedness, score = "", 0.0
            if result.handedness and index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                handedness, score = category.category_name, category.score

            hands.append(Hand(points=points, norm_points=norm_points, handedness=handedness, score=score))
        return hands

    def close(self):
        self._landmarker.close()
