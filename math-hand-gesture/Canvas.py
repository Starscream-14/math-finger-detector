import cv2
import numpy as np
import textwrap

class QuizCanvas:
    # Set ukuran window, gap, dan callback
    def __init__(self, cam_size=(640, 480), quiz_size=(520, 320), gap=20):
        self.cam_w, self.cam_h = cam_size
        self.quiz_w, self.quiz_h = quiz_size
        self.gap = gap
        self.BTN = (0, 0, 0, 0)
        self.on_start = None
        cv2.namedWindow("Camera")
        cv2.namedWindow("Quiz")
        cv2.setMouseCallback("Quiz", self._on_mouse)

    # Atur letak window Camera dan Quiz
    def arrange_windows(self):
        try:
            import ctypes
            user32 = ctypes.windll.user32
            sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            sw, sh = 1366, 768
        try:
            cv2.resizeWindow("Camera", self.cam_w, self.cam_h)
            cv2.resizeWindow("Quiz", self.quiz_w, self.quiz_h)
        except Exception:
            pass
        total_w = self.cam_w + self.gap + self.quiz_w
        total_h = max(self.cam_h, self.quiz_h)
        x0 = max(0, (sw - total_w) // 2)
        y0 = max(0, (sh - total_h) // 2)
        cv2.moveWindow("Camera", x0, y0)
        cv2.moveWindow("Quiz", x0 + self.cam_w + self.gap, y0)

    # Set callback saat tombol START diklik
    def set_start_callback(self, fn):
        self.on_start = fn

    # Tulis teks panjang dengan word-wrap
    def _draw_wrapped(self, canvas, text, x, y, max_w, line_h=28, scale=0.7, color=(0, 0, 0), thick=2):
        cols = max(10, int((max_w / (14 * scale + 1))))
        for line in textwrap.wrap(text, width=cols):
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
            y += line_h
        return y

    # Gambar UI quiz (soal, skor, pesan, tombol)
    def draw(self, started, finished, qs, idx, score, msg):
        W, H = self.quiz_w, self.quiz_h
        canvas = np.full((H, W, 3), 255, np.uint8)
        y = 30
        if started and not finished:
            cv2.putText(canvas, f"Soal {idx+1}/{len(qs) if qs else 0}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2); y += 35
            if qs:
                y = self._draw_wrapped(canvas, qs[idx][0], 30, y, max_w=W-60, line_h=28, scale=0.8); y += 8
            cv2.putText(canvas, f"Skor: {score}/{len(qs) if qs else 0}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2); y += 28
        elif started and finished:
            cv2.putText(canvas, f"Skor: {score}/{len(qs) if qs else 0}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2); y += 35
            y = self._draw_wrapped(canvas, "Selesai! Klik START untuk bermain lagi.", 30, y, max_w=W-60, line_h=26, scale=0.75, color=(50,50,50), thick=2); y += 6
        if msg:
            y = self._draw_wrapped(canvas, msg, 30, y, max_w=W-60, line_h=26, scale=0.7, color=(40,40,40), thick=2); y += 6

        btn_w, btn_h = 170, 48
        x1 = (W - btn_w) // 2
        y1 = (H - btn_h) // 2
        x2, y2 = x1 + btn_w, y1 + btn_h
        self.BTN = (x1, y1, x2, y2)

        txt = "START" if not started or finished else "RUNNING"
        color = (0,180,0) if not started or finished else (150,150,150)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        (tsize, _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        tx = x1 + (btn_w - tsize[0]) // 2
        ty = y1 + (btn_h + tsize[1]) // 2
        cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        return canvas

    # Handler klik mouse di window Quiz
    def _on_mouse(self, evt, x, y, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN and self.on_start is not None:
            x1, y1, x2, y2 = self.BTN
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.on_start()
