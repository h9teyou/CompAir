"""Manual wheel test that does not need a camera."""

import json
import time
from pathlib import Path

from core.controller import Controller, EmergencyStop

FPS = 30


def main():
    cfg = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
    scroll_cfg = cfg["scroll"]
    controller = Controller(scroll_units_per_notch=scroll_cfg["units_per_notch"])
    controller.enabled = True
    direction = -1 if scroll_cfg.get("invert") else 1

    print("Наведи курсор на длинную страницу. Старт через:")
    for second in range(5, 0, -1):
        print(f"  {second}...")
        time.sleep(1)

    try:
        print("3 щелчка ВНИЗ")
        for _ in range(3):
            controller.scroll(-1 * direction)
            time.sleep(0.5)
        time.sleep(1)
        print("3 щелчка ВВЕРХ")
        for _ in range(3):
            controller.scroll(1 * direction)
            time.sleep(0.5)
        time.sleep(1)

        speed = min(scroll_cfg["max_notches_per_second"], 10)
        print(f"Плавно ВНИЗ 2 секунды ({speed} щелчков в секунду)")
        for _ in range(2 * FPS):
            controller.scroll(-speed / FPS * direction)
            time.sleep(1 / FPS)
        controller.scroll(0)
        time.sleep(1)
        print("Плавно ВВЕРХ 2 секунды")
        for _ in range(2 * FPS):
            controller.scroll(speed / FPS * direction)
            time.sleep(1 / FPS)
        controller.scroll(0)
    except EmergencyStop:
        print("Остановлено защитой pyautogui.")
        return
    print("Готово. Если направление или скорость не подходят, настрой раздел scroll в config.json.")


if __name__ == "__main__":
    main()
