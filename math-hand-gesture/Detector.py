import cv2, math
from collections import deque
try:
    import mediapipe as mp
except Exception as e:
    raise ImportError("Butuh mediapipe. Install: pip install mediapipe") from e

# Class hitung jumlah jari dan gambar overlay penanda
class FingerCounter:
    
    # Set parameter, buffer, dan inisialisasi MediaPipe Hands
    def __init__(self,
                 image_flipped=True,
                 smooth_window=7,
                 ema_alpha=0.35,
                 ang_open=55.0, ang_close=40.0,
                 k_dist_open=0.35, k_dist_close=0.28,
                 z_margin_open=0.015,
                 z_margin_close=-0.005,
                 fist_dist_mean=0.26,
                 fist_ang_mean=48.0,
                 fist_lock_frames=10):
        self.image_flipped = bool(image_flipped)
        self.buf = deque(maxlen=max(3, int(smooth_window)))
        self.alpha = float(ema_alpha)
        self.ang_open, self.ang_close = float(ang_open), float(ang_close)
        self.k_do, self.k_dc = float(k_dist_open), float(k_dist_close)
        self.z_open, self.z_close = float(z_margin_open), float(z_margin_close)
        self.fist_d, self.fist_a = float(fist_dist_mean), float(fist_ang_mean)
        self.fist_lock_frames = int(fist_lock_frames)
        self.lock0 = 0
        self.state = [False] * 5
        self.ema_ang_pip = [None] * 5; self.ema_ang_mcp = [None] * 5; self.ema_dist = [None] * 5; self.ema_z = [None] * 5
        self.mp_h = mp.solutions.hands
        self.mp_d = mp.solutions.drawing_utils; self.mp_s = mp.solutions.drawing_styles
        self.h = self.mp_h.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.hands = self.h

    # Proses hitung jari per frame dan terapkan lock untuk fist
    def process(self, frame):
        n = self._count_and_draw(frame)
        if self.lock0 > 0:
            n, self.lock0 = 0, self.lock0 - 1
        self.buf.append(max(0, min(5, n)))
        return self._majority(self.buf)

    # Pilih nilai mayoritas dari buffer
    def _majority(self, q):
        c = {}; b = (0, -1)
        for v in q:
            c[v] = c.get(v, 0) + 1
            if c[v] > b[1] or (c[v] == b[1] and v > b[0]):
                b = (v, c[v])
        return b[0]

    # Hitung sudut 3D ABC (derajat)
    def _angle3d(self, a, b, c):
        ax, ay, az = a; bx, by, bz = b; cx, cy, cz = c
        v1 = (ax - bx, ay - by, az - bz); v2 = (cx - bx, cy - by, cz - bz)
        n1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2); n2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        if n1 == 0 or n2 == 0:
            return 180.0
        cosv = max(-1, min(1, (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (n1 * n2)))
        return math.degrees(math.acos(cosv))

    # Exponential moving average sederhana
    def _ema(self, prev, x):
        if prev is None:
            return x
        a = self.alpha
        return a * x + (1 - a) * prev

    # Deteksi, fitur, keputusan buka/tutup, fist override, dan overlay
    def _count_and_draw(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.h.process(rgb)
        if not res.multi_hand_landmarks:
            self.state = [False] * 5
            return 0
        lm = res.multi_hand_landmarks[0]; hd = res.multi_handedness[0].classification[0].label
        h, w, _ = frame.shape
        pts2 = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
        pts3 = [(p.x, p.y, p.z) for p in lm.landmark]
        self.mp_d.draw_landmarks(frame, lm, self.mp_h.HAND_CONNECTIONS,
                                 self.mp_s.get_default_hand_landmarks_style(),
                                 self.mp_s.get_default_hand_connections_style())

        WRIST = 0; TIP = [4, 8, 12, 16, 20]; DIP = [3, 7, 11, 15, 19]; PIP = [2, 6, 10, 14, 18]; MCP = [1, 5, 9, 13, 17]
        scale = max(20.0, math.hypot(pts2[WRIST][0] - pts2[MCP[2]][0], pts2[WRIST][1] - pts2[MCP[2]][1]))
        is_right_in_img = (hd == "Right") ^ self.image_flipped
        palm_x = (pts2[WRIST][0] + sum(pts2[i][0] for i in MCP)) / 6.0; palm_y = (pts2[WRIST][1] + sum(pts2[i][1] for i in MCP)) / 6.0
        palm = (palm_x, palm_y)

        angP = [0] * 5; angM = [0] * 5; dist = [0] * 5; zdelta = [0] * 5
        for i in range(5):
            angP_raw = self._angle3d(pts3[DIP[i]], pts3[PIP[i]], pts3[MCP[i]])
            angM_raw = self._angle3d(pts3[PIP[i]], pts3[MCP[i]], pts3[WRIST])
            d_raw = math.hypot(pts2[TIP[i]][0] - palm[0], pts2[TIP[i]][1] - palm[1]) / scale
            z_raw = pts3[TIP[i]][2] - pts3[PIP[i]][2]
            self.ema_ang_pip[i] = self._ema(self.ema_ang_pip[i], angP_raw)
            self.ema_ang_mcp[i] = self._ema(self.ema_ang_mcp[i], angM_raw)
            self.ema_dist[i] = self._ema(self.ema_dist[i], d_raw)
            self.ema_z[i] = self._ema(self.ema_z[i], z_raw)
            angP[i] = self.ema_ang_pip[i]; angM[i] = self.ema_ang_mcp[i]; dist[i] = self.ema_dist[i]; zdelta[i] = self.ema_z[i]

        mean_d = sum(dist) / 5.0; mean_a = sum((angP[i] + angM[i]) * 0.5 for i in range(5)) / 5.0
        curled_z = sum(1 for i in range(5) if zdelta[i] > self.z_close)
        if mean_d < self.fist_d and mean_a < self.fist_a and curled_z >= 3:
            self.state = [False] * 5; self.lock0 = self.fist_lock_frames
            self._draw_overlay(frame, pts2, PIP, TIP, [False] * 5)
            cv2.putText(frame, "Fist", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
            return 0

        d_open = self.k_do; d_close = self.k_dc
        for i in range(5):
            thumb_ok = True
            if i == 0:
                thumb_ok = (pts2[TIP[0]][0] > pts2[MCP[0]][0]) if is_right_in_img else (pts2[TIP[0]][0] < pts2[MCP[0]][0])
            open_cond = (angP[i] > self.ang_open and angM[i] > self.ang_open and dist[i] > d_open and (zdelta[i] < self.z_open)) and (thumb_ok if i == 0 else True)
            close_cond = (angP[i] < self.ang_close or angM[i] < self.ang_close or dist[i] < d_close or (zdelta[i] > self.z_close) or (not thumb_ok if i == 0 else False))
            if not self.state[i]:
                self.state[i] = open_cond
            else:
                self.state[i] = not close_cond

        self._draw_overlay(frame, pts2, PIP, TIP, self.state)
        total = sum(1 for s in self.state if s)
        return min(5, max(0, total))

    # Gambar garis TIP–PIP dan titik penanda
    def _draw_overlay(self, frame, pts2, PIP, TIP, state):
        for i in range(5):
            color = (0, 200, 0) if state[i] else (0, 0, 255)
            cv2.line(frame, pts2[PIP[i]], pts2[TIP[i]], color, 3)
            cv2.circle(frame, pts2[TIP[i]], 6, color, -1)
            cv2.circle(frame, pts2[PIP[i]], 5, (200, 200, 200), -1)
