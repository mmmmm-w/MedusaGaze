import argparse
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
from torchvision import transforms

from openface.face_detection import FaceDetector
from openface.view_au_model import ViewAUModel
from view_data import compute_bbox_feat


def get_screen_resolution() -> Tuple[int, int]:
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fullscreen GUI for view+AU inference with gesture triggers.")
    parser.add_argument("--view-ckpt", type=Path, default=ROOT / "weights" / "view_mtl.pth", help="Checkpoint for view head.")
    parser.add_argument("--mtl-ckpt", type=Path, default=ROOT / "weights" / "MTL_backbone.pth", help="Checkpoint for backbone+AU head.")
    parser.add_argument("--face-ckpt", type=Path, default=ROOT / "weights" / "Alignment_RetinaFace.pth", help="RetinaFace weights.")
    parser.add_argument("--spring-k", type=float, default=10.0, help="Spring stiffness toward predicted point.")
    parser.add_argument("--spring-damping", type=float, default=3.0, help="Damping coefficient; increase to reduce oscillation.")
    parser.add_argument("--jaw-thresh", type=float, default=0.6, help="Threshold for AU26 (jaw drop) toggle freeze.")
    parser.add_argument("--brow-raise-thresh", type=float, default=0.6, help="Threshold for brow raise left-click cue (AU1/2).")
    parser.add_argument("--brow-lower-thresh", type=float, default=0.6, help="Threshold for brow lower right-click cue (AU4).")
    parser.add_argument("--gesture-cooldown", type=float, default=0.8, help="Cooldown (s) between successive gesture actions to prevent repeats.")
    return parser.parse_args()


class ViewAUGUI:
    def __init__(self, args: argparse.Namespace):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using device: {self.device}")

        if not args.face_ckpt.exists():
            raise FileNotFoundError(f"Missing face model at {args.face_ckpt}")
        if not args.mtl_ckpt.exists():
            raise FileNotFoundError(f"Missing MTL checkpoint at {args.mtl_ckpt}")
        if not args.view_ckpt.exists():
            raise FileNotFoundError(f"Missing view checkpoint at {args.view_ckpt}")

        self.face_detector = FaceDetector(model_path=str(args.face_ckpt), device=self.device)
        self.model = ViewAUModel().to(self.device)
        self.model.load_from_checkpoints(str(args.mtl_ckpt), str(args.view_ckpt), device=self.device)
        self.model.eval()

        self.k = args.spring_k
        self.damping = args.spring_damping
        self.jaw_thresh = args.jaw_thresh
        self.brow_raise_thresh = args.brow_raise_thresh
        self.brow_lower_thresh = args.brow_lower_thresh
        self.gesture_cooldown = args.gesture_cooldown

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam. Check permissions / device index.")

        self.screen_w, self.screen_h = get_screen_resolution()
        self.window = tk.Tk()
        self.window.title("MedusaGaze - View + AU")
        self.window.attributes("-fullscreen", True)
        self.window.configure(bg="#0d1117")

        self.canvas = tk.Canvas(
            self.window, width=self.screen_w, height=self.screen_h, highlightthickness=0, bg="#0d1117"
        )
        self.canvas.pack(fill="both", expand=True)

        self.btn_frame = tk.Frame(self.window, bg="#0d1117")
        self.btn_frame.place(x=20, y=20)
        tk.Button(self.btn_frame, text="Quit", command=self.window.destroy).grid(row=0, column=0, padx=6)

        self.target_point: Optional[Tuple[int, int]] = None
        self.smoothed_point: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.frozen = False
        self.last_jaw_high = False
        self.last_brow_raise_active = False
        self.last_brow_lower_active = False
        self.last_gesture_time = 0.0
        self.flash_text: Optional[str] = None
        self.flash_until = 0.0
        self.latest_au: Optional[np.ndarray] = None

        self.webcam_imgtk = None
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.update_loop()
        self.window.mainloop()

    def set_flash(self, text: str, duration: float = 0.7):
        self.flash_text = text
        self.flash_until = time.time() + duration

    def update_overlay(self):
        self.canvas.delete("all")

        # Predicted point
        if self.smoothed_point is not None:
            px = int(np.clip(self.smoothed_point[0], 0, 1) * self.screen_w)
            py = int(np.clip(self.smoothed_point[1], 0, 1) * self.screen_h)
            self.canvas.create_oval(px - 16, py - 16, px + 16, py + 16, fill="#003f52", outline="")
            self.canvas.create_oval(px - 12, py - 12, px + 12, py + 12, fill="#00c8ff", outline="")

        # Webcam thumbnail top-right
        if self.webcam_imgtk is not None:
            self.canvas.create_image(self.screen_w - 10, 10, anchor="ne", image=self.webcam_imgtk)

        # Flash text for clicks
        if self.flash_text and time.time() < self.flash_until:
            self.canvas.create_text(
                self.screen_w // 2,
                80,
                text=self.flash_text,
                fill="#ffeb3b",
                font=("Helvetica", 28, "bold"),
            )

        # Frozen indicator
        if self.frozen:
            self.canvas.create_text(
                self.screen_w - 120,
                self.screen_h - 40,
                text="FROZEN",
                fill="#ff5252",
                font=("Helvetica", 16, "bold"),
            )

        # FPS
        self.canvas.create_text(
            20,
            self.screen_h - 20,
            text=f"FPS: {self.fps:.1f}",
            fill="#e8edf5",
            anchor="w",
            font=("Helvetica", 14, "bold"),
        )

        # AU distribution bars (center-left)
        if self.latest_au is not None:
            bar_w = 200
            bar_h = 12
            start_x = 80
            start_y = self.screen_h // 2 - (len(self.latest_au) * (bar_h + 10)) // 2
            for i, val in enumerate(self.latest_au):
                y = start_y + i * (bar_h + 10)
                self.canvas.create_text(
                    start_x,
                    y,
                    text=f"AU{i+1}" if i < 2 else f"AU{[4,6,9,12,25,26][i-2]}",
                    fill="#e8edf5",
                    anchor="e",
                    font=("Helvetica", 12),
                )
                w = int(np.clip(val, 0, 1) * bar_w)
                self.canvas.create_rectangle(start_x + 10, y - bar_h // 2, start_x + 10 + bar_w, y + bar_h // 2, fill="#1f2a3a", outline="#444")
                self.canvas.create_rectangle(start_x + 10, y - bar_h // 2, start_x + 10 + w, y + bar_h // 2, fill="#00e676", outline="")
                self.canvas.create_text(
                    start_x + 20 + bar_w,
                    y,
                    text=f"{val:.2f}",
                    fill="#9fb3c8",
                    anchor="w",
                    font=("Helvetica", 12),
                )

    def handle_aus(self, au_probs: np.ndarray):
        # AU indices: [AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26]
        # jaw = float(au_probs[7]) if len(au_probs) > 7 else 0.0
        jaw = float(au_probs[6]) if len(au_probs) > 7 else 0.0  #Try use lips instead
        jaw_high = jaw > self.jaw_thresh
        if jaw_high and not self.last_jaw_high:
            self.frozen = not self.frozen
        self.last_jaw_high = jaw_high

        brow_raise = max(float(au_probs[0]), float(au_probs[1])) if len(au_probs) > 1 else 0.0
        brow_raise_active = brow_raise > self.brow_raise_thresh
        now = time.time()
        if brow_raise_active and not self.last_brow_raise_active and (now - self.last_gesture_time) >= self.gesture_cooldown:
            self.set_flash("LEFT CLICK")
            self.last_gesture_time = now
        self.last_brow_raise_active = brow_raise_active

        brow_lower = float(au_probs[2]) if len(au_probs) > 2 else 0.0
        brow_lower_active = brow_lower > self.brow_lower_thresh
        if brow_lower_active and not self.last_brow_lower_active and (now - self.last_gesture_time) >= self.gesture_cooldown:
            self.set_flash("RIGHT CLICK")
            self.last_gesture_time = now
        self.last_brow_lower_active = brow_lower_active

    def update_prediction(self, frame: np.ndarray, dt: float):
        face, dets = self.face_detector.get_face_from_image(frame)
        if face is None or dets is None or len(dets) == 0:
            return
        det = dets[0]
        bbox_feat = compute_bbox_feat(det, frame.shape)
        face_tensor = self.to_tensor(face)
        bbox_tensor = torch.tensor(bbox_feat, dtype=torch.float32, device=self.device).view(1, -1)

        with torch.no_grad():
            view_out, au_out = self.model(face_tensor, bbox_tensor)
        view_pred = view_out.squeeze(0).cpu().numpy()
        # Use raw AU logits to match original MTL outputs
        au_raw = au_out.squeeze(0).cpu().numpy()
        self.latest_au = au_raw

        self.handle_aus(au_raw)

        if not self.frozen:
            p_target = view_pred
            if self.smoothed_point is None:
                self.smoothed_point = p_target
                self.velocity = np.zeros_like(p_target)
            else:
                v = self.velocity if self.velocity is not None else np.zeros_like(p_target)
                dt = min(dt, 0.05)  # clamp for stability
                displacement = p_target - self.smoothed_point
                acc = self.k * displacement - self.damping * v
                v = v + acc * dt
                self.smoothed_point = self.smoothed_point + v * dt
                self.velocity = v

    def to_tensor(self, face: np.ndarray) -> torch.Tensor:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
        return tensor

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.window.after(30, self.update_loop)
            return

        now = time.time()
        dt = now - self.last_time
        self.update_prediction(frame, dt)

        # Webcam thumbnail
        mirrored = cv2.flip(frame, 1)
        thumb_w = 320
        thumb_h = int(mirrored.shape[0] * thumb_w / mirrored.shape[1])
        mirrored = cv2.resize(mirrored, (thumb_w, thumb_h))
        rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        self.webcam_imgtk = ImageTk.PhotoImage(image=im)

        # FPS update
        self.frame_count += 1
        if self.frame_count >= 10:
            self.fps = self.frame_count / (now - self.last_time)
            self.last_time = now
            self.frame_count = 0

        self.update_overlay()
        self.window.after(10, self.update_loop)


if __name__ == "__main__":
    args = parse_args()
    ViewAUGUI(args)
