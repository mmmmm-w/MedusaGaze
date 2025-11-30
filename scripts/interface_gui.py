import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch
import tkinter as tk
from PIL import Image, ImageTk

from openface.face_detection import FaceDetector
from openface.view_model import ViewPredictor
from view_data import compute_bbox_feat


def get_screen_resolution() -> Tuple[int, int]:
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


class ViewInterface:
    def __init__(self):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using device: {self.device}")

        self.face_model_path = ROOT / "weights" / "Alignment_RetinaFace.pth"
        self.view_model_path = ROOT / "weights" / "view_mtl.pth"

        if not self.face_model_path.exists():
            raise FileNotFoundError(f"Missing face model at {self.face_model_path}")
        if not self.view_model_path.exists():
            raise FileNotFoundError(f"Missing view model at {self.view_model_path}")

        self.face_detector = FaceDetector(model_path=str(self.face_model_path), device=self.device)
        self.view_model = ViewPredictor(model_path=str(self.view_model_path), device=self.device)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam. Check permissions / device index.")

        self.screen_w, self.screen_h = get_screen_resolution()
        self.window = tk.Tk()
        self.window.title("MedusaGaze - ViewPoint")
        self.window.attributes("-fullscreen", True)
        self.window.configure(bg="#0d1117")

        # Canvas spans the full screen for drawing points
        self.canvas = tk.Canvas(
            self.window, width=self.screen_w, height=self.screen_h, highlightthickness=0, bg="#0d1117"
        )
        self.canvas.pack(fill="both", expand=True)

        # Controls
        self.btn_frame = tk.Frame(self.window, bg="#0d1117")
        self.btn_frame.place(x=20, y=20)
        tk.Button(self.btn_frame, text="Generate target", command=self.generate_target).grid(row=0, column=0, padx=6)
        tk.Button(self.btn_frame, text="Clear target", command=self.clear_target).grid(row=0, column=1, padx=6)
        tk.Button(self.btn_frame, text="Quit", command=self.window.destroy).grid(row=0, column=2, padx=6)

        self.target_point: Optional[Tuple[int, int]] = None
        self.pred_point: Optional[Tuple[float, float]] = None
        self.smoothed_point: Optional[np.ndarray] = None
        self.alpha = 0.25

        # Keep references for Tk images
        self.webcam_imgtk = None

        self.last_fps_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        self.update_loop()
        self.window.mainloop()

    def generate_target(self):
        x = random.randint(int(self.screen_w * 0.05), int(self.screen_w * 0.95))
        y = random.randint(int(self.screen_h * 0.05), int(self.screen_h * 0.95))
        self.target_point = (x, y)

    def clear_target(self):
        self.target_point = None

    def draw_overlay(self):
        self.canvas.delete("all")

        # Draw target point if present
        if self.target_point:
            tx, ty = self.target_point
            self.canvas.create_oval(tx - 12, ty - 12, tx + 12, ty + 12, fill="#ff3b30", outline="")

        # Draw predicted point (cyan)
        if self.smoothed_point is not None:
            px = int(np.clip(self.smoothed_point[0], 0, 1) * self.screen_w)
            py = int(np.clip(self.smoothed_point[1], 0, 1) * self.screen_h)
            self.canvas.create_oval(px - 12, py - 12, px + 12, py + 12, fill="#00c8ff", outline="")

        # Draw webcam feed (mirrored) in top-right
        if self.webcam_imgtk is not None:
            self.canvas.create_image(self.screen_w - 10, 10, anchor="ne", image=self.webcam_imgtk)

        # FPS text
        self.canvas.create_text(
            20,
            self.screen_h - 20,
            text=f"FPS: {self.fps:.1f}",
            fill="#e8edf5",
            anchor="w",
            font=("Helvetica", 14, "bold"),
        )

    def update_prediction(self, frame: np.ndarray):
        face, dets = self.face_detector.get_face_from_image(frame)
        if face is None or dets is None or len(dets) == 0:
            self.smoothed_point = None
            return
        det = dets[0]
        bbox_feat = compute_bbox_feat(det, frame.shape)
        pred = self.view_model.predict(face, bbox_feat=bbox_feat)

        if self.smoothed_point is None:
            self.smoothed_point = pred
        else:
            self.smoothed_point = (1 - self.alpha) * self.smoothed_point + self.alpha * pred

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.window.after(30, self.update_loop)
            return

        self.update_prediction(frame)

        # Prepare mirrored webcam thumbnail
        mirrored = cv2.flip(frame, 1)
        thumb_w = 320
        thumb_h = int(mirrored.shape[0] * thumb_w / mirrored.shape[1])
        mirrored = cv2.resize(mirrored, (thumb_w, thumb_h))
        rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        self.webcam_imgtk = ImageTk.PhotoImage(image=im)

        # FPS
        self.frame_count += 1
        if self.frame_count >= 10:
            now = time.time()
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.last_fps_time = now
            self.frame_count = 0

        self.draw_overlay()
        self.window.after(10, self.update_loop)


if __name__ == "__main__":
    ViewInterface()
