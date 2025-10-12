import cv2
import winsound
from collections import deque
from Detector import FingerCounter
from Calculator import gen_ops, is_valid_fingers, evaluate_answer
from Canvas import QuizCanvas

# Total soal
TOTAL_Q = 10  

# Fungsi mengambil nilai terbanyak dari buffer
def majority(q):
    c = {}; best_v, best_c = 0, -1
    for v in q:
        c[v] = c.get(v, 0) + 1
        if c[v] > best_c or (c[v] == best_c and v > best_v):
            best_v, best_c = v, c[v]
    return best_v

# Fungsi jumlah frame stabil yang dibutuhkan per nilai jari
def need_frames(v):
    if v in (1, 2): return 12
    if v in (3, 4, 5): return 10
    return 8

# Fungsi reset state permainan saat START
def make_start_handler(state, raw_buf):
    def _start():
        state["started"] = True
        state["finished"] = False
        state["qs"] = gen_ops(TOTAL_Q)
        state["idx"] = 0
        state["score"] = 0
        state["msg"] = ""
        state["await_release"] = False
        state["release_cnt"] = 0
        state["stable"] = -1
        state["cnt"] = 0
        raw_buf.clear()
    return _start

# Fungsi kamera, GUI, detektor, dan state
cap = cv2.VideoCapture(0)
gui = QuizCanvas()
gui.arrange_windows()
fc = FingerCounter(image_flipped=True)
state = {
    "started": False, "finished": False, "qs": [], "idx": 0, "score": 0,
    "stable": -1, "cnt": 0, "await_release": False, "release_cnt": 0, "msg": ""
}
raw_buf = deque(maxlen=7)
gui.set_start_callback(make_start_handler(state, raw_buf))

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    raw = fc.process(frame)
    raw_buf.append(raw)
    fingers = majority(raw_buf)
    cv2.putText(frame, f"Jari: {fingers}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    # Logika kuis ketika berjalan
    if state["started"] and not state["finished"]:
        if state["await_release"]:
            if fingers == 0:
                state["release_cnt"] += 1
                if state["release_cnt"] >= 5:
                    state["await_release"], state["release_cnt"] = False, 0
                    state["msg"] = ""
            else:
                state["release_cnt"] = 0
        else:
            val = fingers if is_valid_fingers(fingers) else -1
            if val == state["stable"] and val != -1:
                state["cnt"] += 1
            else:
                state["stable"], state["cnt"] = val, 1
            need = need_frames(state["stable"]) if state["stable"] != -1 else 9999
            if state["cnt"] >= need:
                ans = state["qs"][state["idx"]][1]
                if evaluate_answer(state["stable"], ans):
                    state["score"] += 1; state["msg"] = ""; state["idx"] += 1
                    if state["idx"] >= TOTAL_Q:
                        state["finished"] = True
                else:
                    state["msg"] = "Jawaban Salah! Turunkan Tangan Dan Coba Lagi."
                    try: winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                    except: pass
                state["stable"], state["cnt"] = -1, 0
                state["await_release"], state["release_cnt"] = True, 0

    # Tampilkan kamera dan canvas
    cv2.imshow("Camera", frame)
    cv2.imshow("Quiz", gui.draw(state["started"], state["finished"], state["qs"], state["idx"], state["score"], state["msg"]))

    # Keluar dengan ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
