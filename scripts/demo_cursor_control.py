import argparse
import json
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
from online_calibration import CalibrationConfig, run_calibration

try:
    import pyautogui
except ImportError:
    pyautogui = None


def get_screen_resolution() -> Tuple[int, int]:
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live cursor control via gaze + AU gestures.")
    parser.add_argument("--view-ckpt", type=Path, default=ROOT / "weights" / "view_mtl.pth")
    parser.add_argument("--mtl-ckpt", type=Path, default=ROOT / "weights" / "MTL_backbone.pth")
    parser.add_argument("--face-ckpt", type=Path, default=ROOT / "weights" / "Alignment_RetinaFace.pth")
    parser.add_argument("--spring-k", type=float, default=10.0)
    parser.add_argument("--spring-damping", type=float, default=3.0)
    parser.add_argument("--jaw-thresh", type=float, default=0.6)
    parser.add_argument("--brow-raise-thresh", type=float, default=0.6)
    parser.add_argument("--brow-lower-thresh", type=float, default=0.6)
    parser.add_argument("--gesture-cooldown", type=float, default=0.8)
    parser.add_argument("--calibration", type=Path, default=ROOT / "demo" / "checkpoints" / "affine_calibration.json")
    parser.add_argument("--force-calibrate", action="store_true", help="Re-run calibration even if it exists.")
    parser.add_argument("--no-finetune-view", action="store_true", help="Skip online finetuning of the view head during calibration.")
    parser.add_argument("--finetune-steps", type=int, default=CalibrationConfig.FINETUNE_STEPS, help="Steps for online finetuning.")
    parser.add_argument("--finetune-lr", type=float, default=CalibrationConfig.FINETUNE_LR, help="Learning rate for online finetuning.")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration step and run raw model output.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class CursorDemo:
    def __init__(self, args: argparse.Namespace):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        if not args.face_ckpt.exists() or not args.mtl_ckpt.exists() or not args.view_ckpt.exists():
            raise FileNotFoundError("Missing checkpoints; ensure face/mtl/view ckpts exist.")

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
        self.dry_run = args.dry_run or (pyautogui is None)

        self.affine_matrix: Optional[np.ndarray] = None
        self.is_calibrated = False
        if args.calibration.exists():
            try:
                with open(args.calibration, "r") as f:
                    calib_data = json.load(f)
                self.affine_matrix = np.array(calib_data["matrix"])
                self.is_calibrated = True
                print(f"Loaded calibration from {args.calibration}")
            except Exception as exc:  # keep running even if calibration fails to load
                print(f"Warning: could not load calibration ({exc}); continuing uncalibrated.")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam.")

        self.screen_w, self.screen_h = get_screen_resolution()
        self.window = tk.Tk()
        self.window.title("Gaze Cursor Demo")
        win_w, win_h = 800, 600
        x_pos = int((self.screen_w - win_w) / 2)
        y_pos = int((self.screen_h - win_h) / 2)
        self.window.geometry(f"{win_w}x{win_h}+{x_pos}+{y_pos}")
        self.window.configure(bg="#0d1117")
        self.window.attributes("-topmost", True)

        self.canvas = tk.Canvas(self.window, width=800, height=600, highlightthickness=0, bg="#0d1117")
        self.canvas.pack(pady=10)

        self.targets = [
            {"x": 0.25, "y": 0.3, "r": 50, "color": "#ff5252", "hit": False},
            {"x": 0.5, "y": 0.6, "r": 50, "color": "#40c4ff", "hit": False},
            {"x": 0.75, "y": 0.4, "r": 50, "color": "#69f0ae", "hit": False},
        ]

        self.status = tk.Label(self.window, text="Initializing...", fg="#e8edf5", bg="#0d1117", font=("Helvetica", 12))
        self.status.pack(pady=4)

        btn_frame = tk.Frame(self.window, bg="#0d1117")
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text="Reset Targets", command=self.reset_targets).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Quit", command=self.window.destroy).pack(side="left", padx=6)

        self.smoothed_point: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.frozen = False
        self.last_jaw_high = False
        self.last_brow_raise_active = False
        self.last_brow_lower_active = False
        self.last_gesture_time = 0.0
        self.flash_text: Optional[str] = None
        self.flash_until = 0.0

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

    def reset_targets(self):
        for t in self.targets:
            t["hit"] = False

    def set_flash(self, text: str, duration: float = 0.7):
        self.flash_text = text
        self.flash_until = time.time() + duration

    def to_tensor(self, face: np.ndarray) -> torch.Tensor:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return self.transform(face_rgb).unsqueeze(0).to(self.device)

    def apply_calibration(self, view_pred: np.ndarray) -> np.ndarray:
        """Map raw gaze prediction using affine calibration if available."""
        if not self.is_calibrated or self.affine_matrix is None:
            return view_pred
        view_px = view_pred * np.array([self.screen_w, self.screen_h])
        view_px_h = np.append(view_px, 1.0)
        corrected_px = self.affine_matrix @ view_px_h
        corrected_norm = corrected_px[:2] / np.array([self.screen_w, self.screen_h])
        return corrected_norm

    def handle_aus(self, au_raw: np.ndarray, px_screen: int, py_screen: int):
        jaw = float(au_raw[7]) if len(au_raw) > 7 else 0.0
        jaw_high = jaw > self.jaw_thresh
        if jaw_high and not self.last_jaw_high:
            self.frozen = not self.frozen
        self.last_jaw_high = jaw_high

        now = time.time()
        brow_raise = max(float(au_raw[0]), float(au_raw[1])) if len(au_raw) > 1 else 0.0
        brow_raise_active = brow_raise > self.brow_raise_thresh
        if brow_raise_active and not self.last_brow_raise_active and (now - self.last_gesture_time) >= self.gesture_cooldown:
            self.set_flash("LEFT CLICK")
            self.last_gesture_time = now
            if not self.dry_run and pyautogui:
                pyautogui.click(button="left")
            self.remove_target_at(px_screen, py_screen)
        self.last_brow_raise_active = brow_raise_active

        brow_lower = float(au_raw[2]) if len(au_raw) > 2 else 0.0
        brow_lower_active = brow_lower > self.brow_lower_thresh
        if brow_lower_active and not self.last_brow_lower_active and (now - self.last_gesture_time) >= self.gesture_cooldown:
            self.set_flash("RIGHT CLICK")
            self.last_gesture_time = now
            if not self.dry_run and pyautogui:
                pyautogui.click(button="right")
            self.add_target_at(px_screen, py_screen)
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
        au_raw = au_out.squeeze(0).cpu().numpy()

        view_pred = self.apply_calibration(view_pred)

        if not self.frozen:
            p_target = view_pred
            if self.smoothed_point is None:
                self.smoothed_point = p_target
                self.velocity = np.zeros_like(p_target)
            else:
                v = self.velocity if self.velocity is not None else np.zeros_like(p_target)
                dt = min(dt, 0.05)
                disp = p_target - self.smoothed_point
                acc = self.k * disp - self.damping * v
                v = v + acc * dt
                self.smoothed_point = self.smoothed_point + v * dt
                self.velocity = v

        use_point = self.smoothed_point if self.smoothed_point is not None else view_pred
        px_screen = int(np.clip(use_point[0], 0, 1) * self.screen_w)
        py_screen = int(np.clip(use_point[1], 0, 1) * self.screen_h)

        self.handle_aus(au_raw, px_screen, py_screen)

        if not self.frozen and not self.dry_run and pyautogui:
            pyautogui.moveTo(px_screen, py_screen, duration=0)

    def map_to_canvas(self, px_screen: int, py_screen: int) -> Tuple[int, int]:
        try:
            cx_abs = self.canvas.winfo_rootx()
            cy_abs = self.canvas.winfo_rooty()
        except Exception:
            cx_abs, cy_abs = 0, 0
        px = int(px_screen - cx_abs)
        py = int(py_screen - cy_abs)
        px = int(np.clip(px, 0, self.canvas.winfo_width() - 1))
        py = int(np.clip(py, 0, self.canvas.winfo_height() - 1))
        return px, py

    def try_hit(self, px_screen: int, py_screen: int):
        px, py = self.map_to_canvas(px_screen, py_screen)
        for t in self.targets:
            if t["hit"]:
                continue
            tx = int(t["x"] * self.canvas.winfo_width())
            ty = int(t["y"] * self.canvas.winfo_height())
            r = t["r"]
            if (px - tx) ** 2 + (py - ty) ** 2 <= r * r:
                t["hit"] = True
                self.set_flash("HIT!")

    def remove_target_at(self, px_screen: int, py_screen: int):
        px, py = self.map_to_canvas(px_screen, py_screen)
        for t in self.targets:
            if t["hit"]:
                continue
            tx = int(t["x"] * self.canvas.winfo_width())
            ty = int(t["y"] * self.canvas.winfo_height())
            r = t["r"]
            if (px - tx) ** 2 + (py - ty) ** 2 <= r * r:
                t["hit"] = True
                self.set_flash("HIT!")
                break

    def add_target_at(self, px_screen: int, py_screen: int):
        px, py = self.map_to_canvas(px_screen, py_screen)
        # add normalized positions
        x_norm = np.clip(px / max(1, self.canvas.winfo_width()), 0, 1)
        y_norm = np.clip(py / max(1, self.canvas.winfo_height()), 0, 1)
        self.targets.append({"x": x_norm, "y": y_norm, "r": 50, "color": "#ffd740", "hit": False})

    def update_overlay(self):
        self.canvas.delete("all")

        point_xy = None
        if self.smoothed_point is not None:
            px_screen = int(np.clip(self.smoothed_point[0], 0, 1) * self.screen_w)
            py_screen = int(np.clip(self.smoothed_point[1], 0, 1) * self.screen_h)
            point_xy = self.map_to_canvas(px_screen, py_screen)

        if self.webcam_imgtk is not None:
            self.canvas.create_image(self.canvas.winfo_width() - 10, 10, anchor="ne", image=self.webcam_imgtk)

        for t in self.targets:
            if t["hit"]:
                continue
            tx = int(t["x"] * self.canvas.winfo_width())
            ty = int(t["y"] * self.canvas.winfo_height())
            r = t["r"]
            self.canvas.create_oval(tx - r, ty - r, tx + r, ty + r, fill=t["color"], outline="#000")

        # Draw the point last so it stays on top
        if point_xy is not None:
            px, py = point_xy
            self.canvas.create_oval(px - 10, py - 10, px + 10, py + 10, fill="#00c8ff", outline="")

        if self.flash_text and time.time() < self.flash_until:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                30,
                text=self.flash_text,
                fill="#ffeb3b",
                font=("Helvetica", 16, "bold"),
            )

        status = f"FPS: {self.fps:.1f} | {'FROZEN' if self.frozen else 'LIVE'}"
        status += " | CAL" if self.is_calibrated else " | UNC"
        if self.dry_run:
            status += " | DRY-RUN"
        self.status.config(text=status)

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.window.after(30, self.update_loop)
            return

        now = time.time()
        dt = now - self.last_time
        self.update_prediction(frame, dt)

        mirrored = cv2.flip(frame, 1)
        thumb_w = 200
        thumb_h = int(mirrored.shape[0] * thumb_w / mirrored.shape[1])
        mirrored = cv2.resize(mirrored, (thumb_w, thumb_h))
        rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        self.webcam_imgtk = ImageTk.PhotoImage(image=im)

        self.frame_count += 1
        if self.frame_count >= 10:
            self.fps = self.frame_count / (now - self.last_time)
            self.last_time = now
            self.frame_count = 0

        self.update_overlay()
        self.window.after(10, self.update_loop)

if __name__ == "__main__":
    args = parse_args()

    if not args.skip_calibration:
        config = CalibrationConfig()
        # Keep calibration aligned with the checkpoints the demo will use
        config.AFFINE_MATRIX_FILE = args.calibration
        config.BASE_VIEW_MODEL = args.view_ckpt
        config.FACE_MODEL = args.face_ckpt
        config.FINETUNE_VIEW = not args.no_finetune_view
        config.FINETUNE_STEPS = args.finetune_steps
        config.FINETUNE_LR = args.finetune_lr
        config.FINETUNED_VIEW_FILE = config.BASE_VIEW_MODEL.with_name(config.BASE_VIEW_MODEL.stem + "_finetuned.pth")
        if not config.AFFINE_MATRIX_FILE.exists() or args.force_calibrate:
            print("Running gaze calibration...")
            calib_data = run_calibration(
                config,
                force=args.force_calibrate,
                finetune_view=config.FINETUNE_VIEW,
            )
            if calib_data and calib_data.get("finetuned_view_ckpt"):
                args.view_ckpt = Path(calib_data["finetuned_view_ckpt"])
        else:
            print(f"Using existing calibration at {config.AFFINE_MATRIX_FILE}")
            try:
                with open(config.AFFINE_MATRIX_FILE, "r") as f:
                    existing_calib = json.load(f)
                ckpt_path = existing_calib.get("finetuned_view_ckpt")
                if ckpt_path and Path(ckpt_path).exists():
                    args.view_ckpt = Path(ckpt_path)
                    print(f"Using finetuned view head from calibration: {args.view_ckpt}")
            except Exception as exc:
                print(f"Warning: could not read finetuned view from calibration: {exc}")

    CursorDemo(args)
