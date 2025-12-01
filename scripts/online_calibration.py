#!/usr/bin/env python3
"""
Integrated Gaze Calibration and Demo System

This script:
1. Checks if calibration exists
2. Runs calibration if needed (15 seconds)
3. Launches demo with calibration applied

Usage: python gaze_demo.py
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
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
from openface.model.MTL import View_MTL
from view_data import compute_bbox_feat


# ============================================================================
# PART 1: CALIBRATION
# ============================================================================

class CalibrationConfig:
    """Configuration for affine calibration"""
    NUM_POINTS = 9
    MARGIN_RATIO = 0.1
    BASE_RADIUS = 15
    RADIUS_AMP = 10
    PULSE_FREQ = 1.5
    MOVE_TIME = 0.5
    CAPTURE_TIME = 0.8
    
    WEIGHTS_DIR = Path("demo/checkpoints")
    FACE_MODEL = Path("weights/Alignment_RetinaFace.pth")
    BASE_VIEW_MODEL = WEIGHTS_DIR / "view_mtl.pth"
    AFFINE_MATRIX_FILE = WEIGHTS_DIR / "affine_calibration.json"


def get_screen_resolution() -> Tuple[int, int]:
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def make_grid_points(num_points: int, margin_ratio: float, screen_w: int, screen_h: int):
    """Generate calibration grid"""
    grid_size = int(math.sqrt(num_points))
    x_margin = int(screen_w * margin_ratio)
    y_margin = int(screen_h * margin_ratio)
    
    x_positions = np.linspace(x_margin, screen_w - x_margin, grid_size)
    y_positions = np.linspace(y_margin, screen_h - y_margin, grid_size)
    
    points = []
    for y in y_positions:
        for x in x_positions:
            points.append((int(x), int(y)))
    return points


def draw_centered_text(img: np.ndarray, text: str, y: int, scale: float = 1.0, thickness: int = 2):
    """Draw centered text"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def make_instruction_screen(screen_w: int, screen_h: int) -> np.ndarray:
    """Create instruction screen"""
    img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    y = 150
    line_gap = 50
    
    draw_centered_text(img, "Quick Gaze Calibration", y, scale=1.6, thickness=3)
    y += line_gap * 2
    
    instructions = [
        "Calibrate gaze tracking for your setup.",
        "1. Sit comfortably in your normal position.",
        "2. Keep your head still during calibration.",
        "3. Follow each dot with your eyes.",
        "4. Takes only 15 seconds!",
        "",
        "Press SPACE to start, ESC to skip."
    ]
    
    for line in instructions:
        draw_centered_text(img, line, y, scale=0.8, thickness=2)
        y += line_gap
    
    return img


def run_calibration(config: CalibrationConfig, force: bool = False) -> Optional[dict]:
    """Run affine calibration if needed"""
    
    # Check if calibration already exists
    if config.AFFINE_MATRIX_FILE.exists() and not force:
        print(f"✓ Calibration already exists: {config.AFFINE_MATRIX_FILE}")
        return None
    
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    
    print("\n" + "="*60)
    print("CALIBRATION SETUP")
    print("="*60)
    print("Loading models...")
    
    # Load models
    face_detector = FaceDetector(model_path=str(config.FACE_MODEL), device=str(device))
    view_model = View_MTL(base_model_name="tf_efficientnet_b0_ns", pretrained=False)
    state = torch.load(config.BASE_VIEW_MODEL, map_location=device)
    view_model.load_state_dict(state, strict=False)
    view_model.to(device)
    view_model.eval()
    print("✓ Models loaded")
    
    # Get screen resolution
    screen_w, screen_h = get_screen_resolution()
    print(f"Screen: {screen_w}x{screen_h}")
    
    # Generate points
    points = make_grid_points(config.NUM_POINTS, config.MARGIN_RATIO, screen_w, screen_h)
    
    # Setup window
    cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Show instructions
    instr_img = make_instruction_screen(screen_w, screen_h)
    while True:
        cv2.imshow("calibration", instr_img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            print("Calibration skipped.")
            return None
        if key == 32:  # SPACE
            break
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cv2.destroyAllWindows()
        print("✗ Cannot open webcam")
        return None
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Collect data
    predicted_points = []
    ground_truth_points = []
    
    print(f"Collecting {len(points)} calibration points...")
    
    for idx, (tx, ty) in enumerate(points, 1):
        point_start = time.time()
        point_predictions = []
        
        while True:
            elapsed = time.time() - point_start
            if elapsed > config.MOVE_TIME + config.CAPTURE_TIME:
                break
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Create stimulus
            stim_img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            phase_angle = 2 * math.pi * config.PULSE_FREQ * elapsed
            radius = int(config.BASE_RADIUS + config.RADIUS_AMP * 0.5 * (1 + math.sin(phase_angle)))
            
            if elapsed < config.MOVE_TIME:
                color = (0, 165, 255)
                is_capturing = False
            else:
                color = (0, 0, 255)
                is_capturing = True
            
            cv2.circle(stim_img, (tx, ty), radius, color, -1)
            progress_text = f"Point {idx}/{len(points)}"
            draw_centered_text(stim_img, progress_text, 80, scale=1.0, thickness=2)
            cv2.imshow("calibration", stim_img)
            
            # Capture predictions
            if is_capturing:
                face, dets = face_detector.get_face_from_image(frame)
                if face is not None and dets is not None and len(dets) > 0:
                    det = dets[0]
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = det[:4]
                    cx = (x1 + x2) * 0.5 / w
                    cy = (y1 + y2) * 0.5 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    bbox_feat = [cx, cy, bw, bh]
                    
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_tensor = transform(face_rgb).unsqueeze(0).to(device)
                    bbox_tensor = torch.tensor(bbox_feat, dtype=torch.float32, device=device).view(1, -1)
                    
                    with torch.no_grad():
                        pred = view_model(face_tensor, bbox_tensor)
                        pred_np = pred.squeeze(0).cpu().numpy()
                        pred_px = pred_np * [screen_w, screen_h]
                        point_predictions.append(pred_px)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                print("Calibration cancelled.")
                return None
        
        if point_predictions:
            avg_pred = np.mean(point_predictions, axis=0)
            predicted_points.append(avg_pred)
            ground_truth_points.append([tx, ty])
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(predicted_points) < 4:
        print(f"✗ Only {len(predicted_points)} valid points. Need at least 4.")
        return None
    
    print(f"✓ Collected {len(predicted_points)} points")
    
    # Compute affine transformation
    predicted_points = np.array(predicted_points)
    ground_truth_points = np.array(ground_truth_points)
    
    matrix, inliers = cv2.estimateAffinePartial2D(predicted_points, ground_truth_points)
    if matrix is None:
        matrix, _, _, _ = np.linalg.lstsq(
            np.c_[predicted_points, np.ones(len(predicted_points))],
            ground_truth_points,
            rcond=None
        )
        matrix = matrix.T
    
    matrix_3x3 = np.vstack([matrix, [0, 0, 1]])
    
    # Evaluate
    def apply_transform(pts, mat):
        pts_h = np.c_[pts, np.ones(len(pts))]
        return (pts_h @ mat.T)[:, :2]
    
    corrected = apply_transform(predicted_points, matrix_3x3)
    errors_before = np.linalg.norm(predicted_points - ground_truth_points, axis=1)
    errors_after = np.linalg.norm(corrected - ground_truth_points, axis=1)
    
    mean_before = np.mean(errors_before)
    mean_after = np.mean(errors_after)
    improvement = ((mean_before - mean_after) / mean_before) * 100
    
    print(f"\nCalibration Results:")
    print(f"  Before: {mean_before:.1f} px")
    print(f"  After:  {mean_after:.1f} px")
    print(f"  Improvement: {improvement:.1f}%")
    
    # Save
    calibration_data = {
        "timestamp": datetime.now().isoformat(),
        "matrix": matrix_3x3.tolist(),
        "screen_resolution": [screen_w, screen_h],
        "metrics": {
            "mean_error_before_px": float(mean_before),
            "mean_error_after_px": float(mean_after),
            "improvement_pct": float(improvement),
            "num_points": len(predicted_points)
        }
    }
    
    config.AFFINE_MATRIX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.AFFINE_MATRIX_FILE, "w") as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"\n✓ Calibration saved to: {config.AFFINE_MATRIX_FILE}")
    return calibration_data


# ============================================================================
# PART 2: DEMO WITH CALIBRATION
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated gaze calibration and demo.")
    parser.add_argument("--view-ckpt", type=Path, default=ROOT / "demo" / "checkpoints" / "view_mtl.pth")
    parser.add_argument("--mtl-ckpt", type=Path, default=ROOT / "demo" / "checkpoints" / "MTL_backbone.pth")
    parser.add_argument("--face-ckpt", type=Path, default=ROOT / "weights" / "Alignment_RetinaFace.pth")
    parser.add_argument("--calibration", type=Path, default=ROOT / "demo" / "checkpoints" / "affine_calibration.json")
    parser.add_argument("--spring-k", type=float, default=12.0)
    parser.add_argument("--spring-damping", type=float, default=4.0)
    parser.add_argument("--jaw-thresh", type=float, default=0.6)
    parser.add_argument("--brow-raise-thresh", type=float, default=0.6)
    parser.add_argument("--brow-lower-thresh", type=float, default=0.6)
    parser.add_argument("--gesture-cooldown", type=float, default=0.8)
    parser.add_argument("--force-calibrate", action="store_true", help="Force recalibration even if exists")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration check")
    return parser.parse_args()


class ViewAUGUI:
    def __init__(self, args: argparse.Namespace):
        self.device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
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

        # Load calibration
        self.affine_matrix = None
        self.is_calibrated = False
        if args.calibration.exists():
            try:
                with open(args.calibration, "r") as f:
                    calib_data = json.load(f)
                self.affine_matrix = np.array(calib_data["matrix"])
                self.is_calibrated = True
                improvement = calib_data["metrics"]["improvement_pct"]
                print(f"✓ Calibration loaded (improvement: {improvement:.1f}%)")
            except Exception as e:
                print(f"⚠ Could not load calibration: {e}")

        self.k = args.spring_k
        self.damping = args.spring_damping
        self.jaw_thresh = args.jaw_thresh
        self.brow_raise_thresh = args.brow_raise_thresh
        self.brow_lower_thresh = args.brow_lower_thresh
        self.gesture_cooldown = args.gesture_cooldown

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self.screen_w, self.screen_h = get_screen_resolution()
        self.window = tk.Tk()
        self.window.title("MedusaGaze - View + AU")
        self.window.attributes("-fullscreen", True)
        self.window.configure(bg="#0d1117")

        self.canvas = tk.Canvas(
            self.window, width=self.screen_w, height=self.screen_h, 
            highlightthickness=0, bg="#0d1117"
        )
        self.canvas.pack(fill="both", expand=True)

        self.btn_frame = tk.Frame(self.window, bg="#0d1117")
        self.btn_frame.place(x=20, y=20)
        tk.Button(self.btn_frame, text="Quit", command=self.window.destroy).grid(row=0, column=0, padx=6)

        self.target_point = None
        self.smoothed_point = None
        self.velocity = None
        self.frozen = False
        self.last_jaw_high = False
        self.last_brow_raise_active = False
        self.last_brow_lower_active = False
        self.last_gesture_time = 0.0
        self.flash_text = None
        self.flash_until = 0.0
        self.latest_au = None

        self.webcam_imgtk = None
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.update_loop()
        self.window.mainloop()

    def apply_calibration(self, view_pred: np.ndarray) -> np.ndarray:
        """Apply affine calibration"""
        if not self.is_calibrated or self.affine_matrix is None:
            return view_pred
        
        view_px = view_pred * np.array([self.screen_w, self.screen_h])
        view_px_h = np.append(view_px, 1.0)
        corrected_px = self.affine_matrix @ view_px_h
        corrected_norm = corrected_px[:2] / np.array([self.screen_w, self.screen_h])
        return corrected_norm

    def set_flash(self, text: str, duration: float = 0.7):
        self.flash_text = text
        self.flash_until = time.time() + duration

    def update_overlay(self):
        self.canvas.delete("all")

        if self.smoothed_point is not None:
            px = int(np.clip(self.smoothed_point[0], 0, 1) * self.screen_w)
            py = int(np.clip(self.smoothed_point[1], 0, 1) * self.screen_h)
            self.canvas.create_oval(px - 16, py - 16, px + 16, py + 16, fill="#003f52", outline="")
            self.canvas.create_oval(px - 12, py - 12, px + 12, py + 12, fill="#00c8ff", outline="")

        if self.webcam_imgtk is not None:
            self.canvas.create_image(self.screen_w - 10, 10, anchor="ne", image=self.webcam_imgtk)

        if self.flash_text and time.time() < self.flash_until:
            self.canvas.create_text(
                self.screen_w // 2, 80, text=self.flash_text,
                fill="#ffeb3b", font=("Helvetica", 28, "bold"),
            )

        # Calibration status
        status_y = 10 + 180 + 10
        if self.is_calibrated:
            status_text = "CALIBRATED"
            status_color = "#00e676"
        else:
            status_text = "UNCALIBRATED"
            status_color = "#ff9800"
        
        self.canvas.create_text(
            self.screen_w - 120, status_y, text=status_text,
            fill=status_color, font=("Helvetica", 12, "bold"),
        )

        if self.frozen:
            self.canvas.create_text(
                self.screen_w - 120, self.screen_h - 40, text="FROZEN",
                fill="#ff5252", font=("Helvetica", 16, "bold"),
            )

        self.canvas.create_text(
            20, self.screen_h - 20, text=f"FPS: {self.fps:.1f}",
            fill="#e8edf5", anchor="w", font=("Helvetica", 14, "bold"),
        )

        if self.latest_au is not None:
            bar_w = 200
            bar_h = 12
            start_x = 80
            start_y = self.screen_h // 2 - (len(self.latest_au) * (bar_h + 10)) // 2
            for i, val in enumerate(self.latest_au):
                y = start_y + i * (bar_h + 10)
                self.canvas.create_text(
                    start_x, y,
                    text=f"AU{i+1}" if i < 2 else f"AU{[4,6,9,12,25,26][i-2]}",
                    fill="#e8edf5", anchor="e", font=("Helvetica", 12),
                )
                w = int(np.clip(val, 0, 1) * bar_w)
                self.canvas.create_rectangle(start_x + 10, y - bar_h // 2, start_x + 10 + bar_w, y + bar_h // 2, fill="#1f2a3a", outline="#444")
                self.canvas.create_rectangle(start_x + 10, y - bar_h // 2, start_x + 10 + w, y + bar_h // 2, fill="#00e676", outline="")
                self.canvas.create_text(
                    start_x + 20 + bar_w, y, text=f"{val:.2f}",
                    fill="#9fb3c8", anchor="w", font=("Helvetica", 12),
                )

    def handle_aus(self, au_probs: np.ndarray):
        jaw = float(au_probs[6]) if len(au_probs) > 7 else 0.0
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
        view_pred = self.apply_calibration(view_pred)
        
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
                dt = min(dt, 0.05)
                displacement = p_target - self.smoothed_point
                acc = self.k * displacement - self.damping * v
                v = v + acc * dt
                self.smoothed_point = self.smoothed_point + v * dt
                self.velocity = v

    def to_tensor(self, face: np.ndarray) -> torch.Tensor:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return self.transform(face_rgb).unsqueeze(0).to(self.device)

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.window.after(30, self.update_loop)
            return

        now = time.time()
        dt = now - self.last_time
        self.update_prediction(frame, dt)

        mirrored = cv2.flip(frame, 1)
        thumb_w = 320
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


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    print("="*60)
    print("INTEGRATED GAZE CALIBRATION & DEMO")
    print("="*60)
    
    # Check if calibration needed
    if not args.skip_calibration:
        config = CalibrationConfig()
        
        if not config.AFFINE_MATRIX_FILE.exists() or args.force_calibrate:
            print("\nCalibration needed...")
            run_calibration(config, force=args.force_calibrate)
            print("\n" + "="*60)
        else:
            print(f"\n✓ Using existing calibration: {config.AFFINE_MATRIX_FILE}")
    
    # Launch demo
    print("\nLaunching demo...")
    print("="*60 + "\n")
    
    ViewAUGUI(args)