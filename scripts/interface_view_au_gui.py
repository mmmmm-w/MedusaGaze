import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tkinter as tk
from PIL import Image, ImageTk
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from openface.face_detection import FaceDetector
from openface.view_au_model import ViewAUModel
from openface.operation_model import FaceStatusModel 
from view_data import compute_bbox_feat


def get_screen_resolution() -> Tuple[int, int]:
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI: Gaze Tracking + Trained Action Model.")
    
    parser.add_argument("--view-ckpt", type=Path, default=ROOT / "weights" / "view_mtl.pth", help="Checkpoint for view head.")
    parser.add_argument("--mtl-ckpt", type=Path, default=ROOT / "weights" / "MTL_backbone.pth", help="Checkpoint for backbone.")
    parser.add_argument("--face-ckpt", type=Path, default=ROOT / "weights" / "Alignment_RetinaFace.pth", help="RetinaFace weights.")
    
    parser.add_argument("--op-ckpt", type=Path, default=ROOT / "weights" / "trained_models" / "last_operation_model.pth", help="Path to your trained FaceStatusModel.")

    parser.add_argument("--spring-k", type=float, default=10.0, help="Spring stiffness.")
    parser.add_argument("--spring-damping", type=float, default=3.0, help="Damping coefficient.")
    parser.add_argument("--conf-thresh", type=float, default=0.7, help="Confidence threshold to trigger action.")
    parser.add_argument("--gesture-cooldown", type=float, default=1.0, help="Seconds between clicks.")
    
    return parser.parse_args()


class ViewActionGUI:
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # --- Load Face Detector ---
        if not args.face_ckpt.exists():
            raise FileNotFoundError(f"Missing face model at {args.face_ckpt}")
        self.face_detector = FaceDetector(model_path=str(args.face_ckpt), device=str(self.device))

        # --- Load Gaze Model (ViewAUModel) ---
        print("Loading Gaze Model...")
        self.gaze_model = ViewAUModel().to(self.device)
        if args.mtl_ckpt.exists() and args.view_ckpt.exists():
            self.gaze_model.load_from_checkpoints(str(args.mtl_ckpt), str(args.view_ckpt), device=self.device)
        else:
            print("Warning: Gaze checkpoints not found. Gaze will be random.")
        self.gaze_model.eval()

        # --- Load Action Model (FaceStatusModel) ---
        print(f"Loading Action Model from {args.op_ckpt}...")
        self.action_model = FaceStatusModel(num_classes=4).to(self.device)
        if args.op_ckpt.exists():
            state_dict = torch.load(args.op_ckpt, map_location=self.device)
            self.action_model.load_state_dict(state_dict)
        else:
            print("Warning: Operation model weights not found!")
        self.action_model.eval()

        # --- Params ---
        self.k = args.spring_k
        self.damping = args.spring_damping
        self.conf_thresh = args.conf_thresh
        self.gesture_cooldown = args.gesture_cooldown

        # --- Webcam & UI ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.screen_w, self.screen_h = get_screen_resolution()
        self.window = tk.Tk()
        self.window.title("MedusaGaze - Gaze + Action")
        self.window.attributes("-fullscreen", True)
        self.window.configure(bg="#0d1117")

        self.canvas = tk.Canvas(self.window, width=self.screen_w, height=self.screen_h, highlightthickness=0, bg="#0d1117")
        self.canvas.pack(fill="both", expand=True)

        tk.Button(self.window, text="Quit", command=self.window.destroy, bg="white").place(x=20, y=20)

        # State Variables
        self.target_point = None
        self.smoothed_point = None
        self.velocity = None
        self.frozen = False
        
        self.last_action_time = 0.0
        self.last_open_state = False # For toggle logic

        self.flash_text = None
        self.flash_until = 0.0
        
        self.current_action_name = "Stable"
        self.current_conf = 0.0
        self.action_probs = [0.0]*4

        self.webcam_imgtk = None
        self.last_time = time.time()

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.update_loop()
        self.window.mainloop()

    def set_flash(self, text: str, duration: float = 0.8):
        self.flash_text = text
        self.flash_until = time.time() + duration

    def handle_action_logic(self, pred_idx: int, conf: float):
        """
        0: Open Mouth -> Toggle Freeze
        1: Left Wink  -> Left Click
        2: Right Wink -> Right Click
        3: Stable     -> Nothing
        """
        now = time.time()
        
        if conf < self.conf_thresh:
            return

        is_open = (pred_idx == 0)
        if is_open and not self.last_open_state:
            # Toggle Freeze
            self.frozen = not self.frozen
            state_text = "FROZEN" if self.frozen else "UNFROZEN"
            self.set_flash(state_text, 1.0)
        self.last_open_state = is_open

        # Cooldown check for clicks
        if (now - self.last_action_time) < self.gesture_cooldown:
            return

        # 1. Left Wink (Class 1) -> Left Click
        if pred_idx == 1:
            self.set_flash("LEFT CLICK")
            self.last_action_time = now

        # 2. Right Wink (Class 2) -> Right Click
        elif pred_idx == 2:
            self.set_flash("RIGHT CLICK")
            self.last_action_time = now
            # pyautogui.click(button='right')

    def update_prediction(self, frame, dt):
        # Face Detection
        face, dets = self.face_detector.get_face_from_image(frame)
        if face is None or len(dets) == 0:
            return
        
        # Prepare Tensors
        face_tensor = self.to_tensor(face)
        bbox_feat = compute_bbox_feat(dets[0], frame.shape)
        bbox_tensor = torch.tensor(bbox_feat, dtype=torch.float32, device=self.device).view(1, -1)

        # --- Inference ---
        with torch.no_grad():
            # 1. Gaze Prediction (using old model)
            # ViewAUModel returns (view, au), we only want view
            view_out, _ = self.gaze_model(face_tensor, bbox_tensor)
            
            # 2. Action Prediction (using new model)
            action_logits = self.action_model(face_tensor)
            action_probs = F.softmax(action_logits, dim=1)
            
        # Process Results
        view_pred = view_out.squeeze(0).cpu().numpy()
        probs_np = action_probs.squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs_np))
        conf = float(probs_np[pred_idx])

        # Store for UI
        self.action_probs = probs_np
        self.current_conf = conf
        # Updated Class Names for Display
        class_names = ["OPEN MOUTH", "LEFT WINK", "RIGHT WINK", "STABLE"]
        self.current_action_name = class_names[pred_idx]

        # Logic Trigger
        self.handle_action_logic(pred_idx, conf)

        # Cursor Smoothing
        if not self.frozen:
            p_target = view_pred
            if self.smoothed_point is None:
                self.smoothed_point = p_target
                self.velocity = np.zeros_like(p_target)
            else:
                v = self.velocity
                dt = min(dt, 0.05)
                displacement = p_target - self.smoothed_point
                acc = self.k * displacement - self.damping * v
                v += acc * dt
                self.smoothed_point += v * dt
                self.velocity = v

    def to_tensor(self, face_bgr):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(face_rgb).unsqueeze(0).to(self.device)

    def update_overlay(self):
        self.canvas.delete("all")

        # 1. Draw Gaze Point
        if self.smoothed_point is not None:
            px = int(np.clip(self.smoothed_point[0], 0, 1) * self.screen_w)
            py = int(np.clip(self.smoothed_point[1], 0, 1) * self.screen_h)
            
            color = "#ff5252" if self.frozen else "#00c8ff"
            self.canvas.create_oval(px-15, py-15, px+15, py+15, fill=color, outline="white", width=2)

        # 2. Webcam Thumbnail
        if self.webcam_imgtk:
            self.canvas.create_image(self.screen_w - 20, 20, anchor="ne", image=self.webcam_imgtk)

        # 3. Flash Message
        if self.flash_text and time.time() < self.flash_until:
            self.canvas.create_text(self.screen_w//2, 100, text=self.flash_text, 
                                    fill="#ffff00", font=("Helvetica", 40, "bold"))

        # 4. Action Status Panel (Left Side)
        self.draw_status_panel()

    def draw_status_panel(self):
        base_x = 50
        base_y = self.screen_h // 2
        bar_w = 150
        bar_h = 15
        
        # Current Action Text
        color_map = ["#ff4081", "#40c4ff", "#40c4ff", "#b0bec5"] # Red for Open, Blue for Left/Right, Grey for Stable
        pred_idx = int(np.argmax(self.action_probs))
        
        self.canvas.create_text(base_x, base_y - 40, text="CURRENT ACTION", fill="white", anchor="w", font=("Helvetica", 10))
        self.canvas.create_text(base_x, base_y, text=self.current_action_name, 
                                fill=color_map[pred_idx], anchor="w", font=("Helvetica", 24, "bold"))

        # Probability Bars
        labels = ["OPEN", "L-WINK", "R-WINK", "STABLE"]
        for i, prob in enumerate(self.action_probs):
            y = base_y + 60 + i * 30
            
            # Label
            self.canvas.create_text(base_x, y, text=labels[i], fill="#cfd8dc", anchor="w", font=("Helvetica", 10))
            
            # Bar Background
            self.canvas.create_rectangle(base_x + 60, y-7, base_x + 60 + bar_w, y+7, fill="#263238", outline="")
            
            # Active Bar
            fill_color = "#00e676" if i == pred_idx else "#546e7a"
            w = int(prob * bar_w)
            self.canvas.create_rectangle(base_x + 60, y-7, base_x + 60 + w, y+7, fill=fill_color, outline="")
            
            # Value
            self.canvas.create_text(base_x + 60 + bar_w + 10, y, text=f"{prob:.2f}", fill="#cfd8dc", anchor="w")

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.window.after(30, self.update_loop)
            return

        now = time.time()
        self.update_prediction(frame, now - self.last_time)
        self.last_time = now

        # Thumbnail
        frame_disp = cv2.flip(frame, 1)
        # Draw face box on thumbnail
        # (Optional: you can draw detection box here if you want)
        
        thumb = cv2.resize(frame_disp, (320, 180))
        img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)))
        self.webcam_imgtk = img

        self.update_overlay()
        self.window.after(10, self.update_loop)


if __name__ == "__main__":
    args = parse_args()
    ViewActionGUI(args)