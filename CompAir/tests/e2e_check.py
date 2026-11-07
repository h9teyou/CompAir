"""Сквозная проверка main.py с подставными камерой, временем и мышью."""
import math, sys, types
from types import SimpleNamespace
import numpy as np, cv2
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import hand_model as hm

state = {"frame": -1}

class FakePAG(types.ModuleType):
    class FailSafeException(Exception): pass
    def __init__(s):
        super().__init__("pyautogui"); s.FAILSAFE=True; s.PAUSE=0.1; s.pos=(960,540); s.calls=[]
    def size(s): return (1920,1080)
    def position(s): return s.pos
    def _chk(s):
        if s.FAILSAFE and s.pos==(0,0): raise s.FailSafeException()
    def moveTo(s,x,y): s._chk(); s.pos=(x,y); s.calls.append((state["frame"],"move",x,y))
    def press(s,k): s._chk(); s.calls.append((state["frame"],"press",k))
    def keyDown(s,k): s._chk(); s.calls.append((state["frame"],"keydown",k))
    def keyUp(s,k): s._chk(); s.calls.append((state["frame"],"keyup",k))
    def click(s,button="left"): s._chk(); s.calls.append((state["frame"],"click",button,s.pos))
    def mouseDown(s,button="left"): s._chk(); s.calls.append((state["frame"],"down",button))
    def mouseUp(s,button="left"): s._chk(); s.calls.append((state["frame"],"up",button))
    def scroll(s,amount): s._chk(); s.calls.append((state["frame"],"scroll",amount))

def hand(points3d, center):
    from core.hand_tracker import Hand
    pts = hm.project(points3d, hm.make_rng(0), tilt_deg=0, scale=100, roll_deg=0, center=center, noise=0)
    return Hand(points=pts, norm_points=[(x/640,y/480,0) for x,y in pts], handedness="Right", score=.9)

C1 = (320, 330)
POINT = hm.pose_point(); PINCH = hm.pose_pinch(0.3, "curled", (0.05, 0.02)); TWO = hm.pose_two()
RPINCH = hm.pose_middle_pinch(0.35, (0.05, 0.02)); FIST = hm.pose_fist()

def schedule(i):
    if i < 10:  return [hand(POINT, C1)]
    if i < 15:  return [hand(PINCH, C1)]                               # короткий щипок: клик
    if i < 20:  return [hand(POINT, C1)]
    if i < 38:                                                          # долгий щипок: перетаскивание
        c = C1 if i < 30 else (C1[0] + 4 * (i - 30), C1[1])
        return [hand(PINCH, c)]
    if i < 43:  return [hand(POINT, (C1[0] + 28, C1[1]))]
    if i < 53:  return [hand(POINT, C1), hand(TWO, (500, 330))]         # две руки: «два пальца» держим
    if i < 71:  return [hand(POINT, C1), hand(TWO, (500, 260))]         # вторая рука выше нуля: колёсико вверх
    if i < 81:  return [hand(POINT, C1)]
    if i < 87:  return [hand(RPINCH, C1)]                               # правый клик
    if i < 91:  return [hand(POINT, C1)]
    if i < 101: return [hand(FIST, (C1[0] + 3 * (i - 91), C1[1]))]      # кулак: пауза
    if i < 111: return [hand(POINT, C1)]
    if i < 121: return [hand(TWO, (320, 330))]                          # одна рука: «два пальца», держим
    if i < 140: return [hand(TWO, (320, 400))]                          # ниже нуля: колёсико вниз
    return []

def run(sched, keys, n_frames, slam_mouse_at=None, glide=False, mode="mouse"):
    pag = FakePAG(); sys.modules["pyautogui"] = pag
    for m in [m for m in sys.modules if m in ("main", "core.controller")]: del sys.modules[m]
    import main
    clock = [1000.0]; state["frame"] = -1
    main.time = SimpleNamespace(perf_counter=lambda: clock[0])
    class Cam:
        def __init__(s,*a,**k): s.measured_fps = 0.0
        def start(s): return s
        def read(s, timeout=0.1):
            if state["frame"] + 1 >= n_frames: return None
            state["frame"] += 1; clock[0] += 1/30
            if slam_mouse_at is not None and state["frame"] == slam_mouse_at: pag.pos = (0, 0)
            return np.full((480,640,3), 50, np.uint8)
        def stop(s): pass
    class Trk:
        def __init__(s,*a,**k): pass
        def close(s): pass
        def process_all(s, f): return sched(state["frame"])
    main.CameraStream = Cam; main.HandTracker = Trk
    _orig_load = main.load_config
    def _load(path):
        c = _orig_load(path); c["scroll"]["units_per_notch"] = 120; c["smoothing"]["glide_enabled"] = glide; c["modes"]["default"] = mode; c["safety_switch"]["beep"] = False; return c
    main.load_config = _load
    calls = [0]
    def waitKey(d):
        calls[0] += 1; return keys.get(calls[0], 255)
    cv2.waitKey = waitKey; cv2.imshow = lambda w, f: None; cv2.destroyAllWindows = lambda: None
    msg = None
    try: main.main()
    except SystemExit as e: msg = str(e)
    return pag, msg

def between(pag, kind, a, b, extra=None):
    return [c for c in pag.calls if c[1] == kind and a <= c[0] < b and (extra is None or c[2] == extra)]

pag, msg = run(schedule, {2: ord('c'), 150: ord('q')}, 145)
print("выход без ошибок:", msg is None)
ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("  ОК   " if cond else "  НЕТ  ") + name)

check("пока управление выключено (кадры 0-1), мышь не тронута", not [c for c in pag.calls if c[0] < 2])
check("короткий щипок = ровно один левый клик", len(between(pag, "click", 10, 20, "left")) == 1)
check("щипок не нажимает кнопку (нет down до перетаскивания)", not between(pag, "down", 10, 20))
check("долгий щипок = down, потом up, без клика", len(between(pag, "down", 20, 43, "left")) == 1 and len(between(pag, "up", 20, 43, "left")) == 1 and not between(pag, "click", 20, 43))
d = between(pag, "down", 20, 43)[0][0]; u = between(pag, "up", 20, 43)[0][0]
check(f"down раньше up (кадры {d} и {u})", d < u)
moves_in_drag = [c for c in pag.calls if c[1] == "move" and d <= c[0] <= u]
check("в перетаскивании курсор следует за рукой вправо", len(moves_in_drag) >= 3 and moves_in_drag[-1][2] > moves_in_drag[0][2])
check("две руки: до удержания «двух пальцев» колёсико молчит", not between(pag, "scroll", 43, 49))
sc = between(pag, "scroll", 49, 81)
check(f"две руки: рука выше нуля крутит ВВЕРХ ({len(sc)} команд)", len(sc) >= 5 and all(c[2] > 0 for c in sc))
check("две руки: во время колёсика нет случайных кликов", not between(pag, "click", 43, 81) and not between(pag, "down", 43, 81))
check("правый щипок = ровно один правый клик", len(between(pag, "click", 81, 95, "right")) == 1 and not between(pag, "click", 81, 95, "left"))
check("кулак: нет кликов, нет кнопок", not between(pag, "click", 91, 101) and not between(pag, "down", 91, 101))
check("кулак: курсор стоит (нет заметных движений)", len(between(pag, "move", 97, 101)) == 0)
sc2 = between(pag, "scroll", 121, 145)
check(f"одна рука «два пальца» ниже нуля крутит ВНИЗ ({len(sc2)} команд)", len(sc2) >= 5 and all(c[2] < 0 for c in sc2))
downs = sum(1 for c in pag.calls if c[1] == "down"); ups = sum(1 for c in pag.calls if c[1] == "up")
check(f"нажатий и отпусканий поровну ({downs}/{ups})", downs == ups)

print("\n--- безопасность ---")
# 1) выход клавишей q прямо во время перетаскивания: кнопка должна быть отпущена
pag2, msg2 = run(lambda i: [hand(POINT, C1)] if i < 5 else [hand(PINCH, C1)], {2: ord('c'), 40: ord('q')}, 60)
downs = sum(1 for c in pag2.calls if c[1] == "down"); ups = sum(1 for c in pag2.calls if c[1] == "up")
check(f"выход во время перетаскивания отпускает кнопку (down={downs}, up={ups})", downs == 1 and ups == 1)
# 2) пользователь утащил настоящую мышь в угол во время перетаскивания: аварийный стоп + кнопка отпущена
pag3, msg3 = run(lambda i: [hand(POINT, C1)] if i < 5 else [hand(PINCH, (C1[0] + 5*(i-20), C1[1]))] , {2: ord('c')}, 80, slam_mouse_at=26)
downs = sum(1 for c in pag3.calls if c[1] == "down"); ups = sum(1 for c in pag3.calls if c[1] == "up")
check(f"аварийный угол остановил программу: «{(msg3 or '')[:30]}…»", msg3 and "Аварийная" in msg3)
check(f"…и кнопка не осталась зажатой (down={downs}, up={ups})", downs == ups)
# 3) руки пропали во время перетаскивания: кнопка отпускается
pag4, _ = run(lambda i: [hand(POINT, C1)] if i < 5 else ([hand(PINCH, C1)] if i < 30 else []), {2: ord('c'), 70: ord('q')}, 60)
downs = sum(1 for c in pag4.calls if c[1] == "down"); ups = sum(1 for c in pag4.calls if c[1] == "up")
rel = [c[0] for c in pag4.calls if c[1] == "up"]
check(f"руки пропали во время перетаскивания, кнопка отпущена в кадре {rel} (down={downs}, up={ups})", downs == 1 and ups == 1 and rel and rel[0] < 45)

print("\n--- плавный поток курсора включён ---")
from core.zone import Zone
expected = Zone({"x_min": .2, "y_min": .15, "x_max": .8, "y_max": .65, "lock_aspect": True}, (640, 480), (1920, 1080)).to_screen(*hand(POINT, C1).points[5])
pag5, msg5 = run(lambda i: [hand(POINT, C1)] if i < 10 or i >= 15 else [hand(PINCH, C1)], {2: ord('c'), 40: ord('q')}, 30, glide=True)
clicks = [c for c in pag5.calls if c[1] == "click"]
check(f"с потоком курсора: один левый клик ({len(clicks)})", len(clicks) == 1 and clicks[0][2] == "left")
err = math.hypot(clicks[0][3][0] - expected[0], clicks[0][3][1] - expected[1]) if clicks else 999
check(f"клик пришёлся в точку до щипка: ошибка {err:.1f} px", err <= 3)
check("выход без ошибок", msg5 is None)
# аварийный угол при включённом потоке курсора: ошибка из потока доходит до основной программы
pag6, msg6 = run(lambda i: [hand(POINT, (C1[0] + 6*i, C1[1]))], {2: ord('c')}, 60, slam_mouse_at=8, glide=True)
check(f"аварийный угол при потоке курсора: «{(msg6 or '')[:28]}…»", msg6 and "Аварийная" in msg6)

print("\n--- режимы и выключатель жестами ---")
OPEN = hm.pose_open()
def swipe_schedule(i):
    if i < 8: return [hand(OPEN, (150, 330))]
    if i < 18: return [hand(OPEN, (150 + 40 * (i - 8), 330))]
    return [hand(OPEN, (550, 330))]
pagp, msgp = run(swipe_schedule, {2: ord('c'), 40: ord('q')}, 38, mode="presentation")
check("презентация: взмах ладонью вправо = одно нажатие «right»", [c[2] for c in pagp.calls if c[1] == "press"] == ["right"])

def game_schedule(i):
    if i < 5: return [hand(POINT, C1)]
    if i < 12: return [hand(PINCH, C1)]
    if i < 20: return [hand(POINT, C1)]
    return [hand(FIST, C1)]
pagg, msgg = run(game_schedule, {2: ord('c'), 40: ord('q')}, 35, mode="game")
check("игра: щипок = пробел, один раз", [c[2] for c in pagg.calls if c[1] == "press"] == ["space"])
check("игра: кулак зажал «вниз», выход его отпустил", [c[1] for c in pagg.calls if c[1] in ("keydown", "keyup")] == ["keydown", "keyup"])

def fist_off(i):
    if i < 10: return [hand(POINT, C1)]
    if i < 110: return [hand(FIST, C1)]                  # 3.3 секунды кулака
    if i < 120: return [hand(POINT, C1)]
    if i < 126: return [hand(PINCH, C1)]
    return [hand(POINT, C1)]
pagf, msgf = run(fist_off, {2: ord('c'), 150: ord('q')}, 135)
check("кулак 3 секунды выключил управление: клика после него нет", not between(pagf, "click", 108, 135))

def palms_on(i):
    if i < 75: return [hand(OPEN, (200, 330)), hand(OPEN, (480, 330))]   # две ладони 2.5 секунды
    if i < 85: return [hand(POINT, C1)]
    if i < 91: return [hand(PINCH, C1)]
    return [hand(POINT, C1)]
pago, msgo = run(palms_on, {120: ord('q')}, 100)
check("две ладони включили управление (клавиша c не нажималась): клик прошёл", len(between(pago, "click", 85, 100, "left")) == 1)
check("…а до этого мышь не трогали", not [c for c in pago.calls if c[0] < 60 and c[1] in ("move", "click", "down")])

print("\nИТОГ:", "ВСЁ ПРОШЛО" if ok else "ЕСТЬ ПРОВАЛЫ")
