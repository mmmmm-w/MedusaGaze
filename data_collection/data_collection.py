import cv2
import numpy as np
import time
import math
import csv
import os
from datetime import datetime
import tkinter as tk

# ==============================
# Screen resolution
# ==============================

def get_screen_resolution():
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h

SCREEN_W, SCREEN_H = get_screen_resolution()
print(f"Screen resolution: {SCREEN_W}x{SCREEN_H}")

# ==============================
# CONFIG
# ==============================

# Stimulus parameters
BASE_RADIUS = 15          # base radius of the dot (pixels)
RADIUS_AMP = 10           # "pumping" amplitude (pixels)
PULSE_FREQ = 1.5          # pulses per second

MOVE_TIME    = 1.0        # time for user to move eyes to dot (seconds)
CAPTURE_TIME = 1.5        # time we actually save frames (seconds)

# Camera config
CAMERA_INDEX = 0          # 0 for default webcam

# Output paths
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"data/gaze_data_{SESSION_ID}"
IMG_DIR = os.path.join(OUT_DIR, "images")
META_PATH = os.path.join(OUT_DIR, "labels.csv")

os.makedirs(IMG_DIR, exist_ok=True)

# ==============================
# Calibration points
# ==============================

def make_grid_points(cols=5, rows=3, margin_ratio=0.12):
    """Make a grid of screen points, avoiding edges a bit."""
    x_margin = int(SCREEN_W * margin_ratio)
    y_margin = int(SCREEN_H * margin_ratio)

    xs = np.linspace(x_margin, SCREEN_W - x_margin, cols).astype(int)
    ys = np.linspace(y_margin, SCREEN_H - y_margin, rows).astype(int)

    points = [(int(x), int(y)) for y in ys for x in xs]
    np.random.shuffle(points)
    return points

target_points = make_grid_points(cols=5, rows=3, margin_ratio=0.12)

# ==============================
# Helper drawing functions
# ==============================

def draw_centered_text(img, text, y, scale=1.0, thickness=2):
    """Draw text centered horizontally at given y."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def make_instruction_screen():
    img = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    y = 150
    line_gap = 50

    draw_centered_text(img, "Gaze Calibration", y, scale=1.6, thickness=3)
    y += line_gap * 2

    instructions = [
        "We are going to collect data to map your eye gaze to screen positions.",
        "1. Sit comfortably, keep your head as steady as you reasonably can.",
        "2. For each red dot, move your eyes to the dot and then hold your gaze.",
        "3. Try not to move your head too much while the dot is on the screen.",
        "",
        "After you press SPACE, the system may ask for CAMERA PERMISSION.",
        "Please click 'Allow' so we can record your face.",
        "",
        "Press SPACE to open the camera and start setup.",
        "Press ESC at any time to quit."
    ]

    for line in instructions:
        draw_centered_text(img, line, y, scale=0.8, thickness=2)
        y += line_gap

    # Fake "button" prompt
    draw_centered_text(img, "[ PRESS SPACE TO CONTINUE ]", SCREEN_H - 100, scale=1.0, thickness=2)

    return img

def make_preview_overlay(frame):
    """Overlay instructions over the camera preview."""
    overlay = frame.copy()
    alpha = 0.7
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    draw_centered_text(frame, "Camera preview - adjust your position & lighting", 40, scale=0.8, thickness=2)
    draw_centered_text(frame, "Look at the screen, not the camera.", 80, scale=0.8, thickness=2)
    draw_centered_text(frame, "Press SPACE to start calibration, ESC to quit.", 120, scale=0.8, thickness=2)
    return frame

# ==============================
# Main
# ==============================

def main():
    # Create window
    cv2.namedWindow("gaze_calib", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("gaze_calib", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 1) Instruction screen
    instr_img = make_instruction_screen()

    while True:
        cv2.imshow("gaze_calib", instr_img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            print("User cancelled at instruction screen.")
            return
        if key == 32:  # SPACE
            break

    # 2) Open camera (this is where OS may ask for permission)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cv2.destroyAllWindows()
        raise RuntimeError("Cannot open webcam. Check permissions / device index.")

    # 3) Camera preview / pose adjustment
    print("Camera opened. Showing preview…")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to read frame in preview.")
            continue
        preview = make_preview_overlay(frame)
        cv2.imshow("gaze_calib", preview)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            print("User cancelled at preview stage.")
            return
        if key == 32:  # SPACE starts calibration
            break

    # 4) Start recording loop
    meta_file = open(META_PATH, "w", newline="")
    csv_writer = csv.writer(meta_file)
    csv_writer.writerow(["filename", "tx", "ty", "x_norm", "y_norm", "timestamp", "phase"])

    print("Starting calibration over", len(target_points), "points.")
    print("Data directory:", OUT_DIR)

    try:
        point_index = 0
        total_points = len(target_points)

        for (tx, ty) in target_points:
            point_index += 1
            point_start = time.time()

            while True:
                t_now = time.time()
                elapsed = t_now - point_start

                if elapsed > MOVE_TIME + CAPTURE_TIME:
                    break

                ret, frame = cap.read()
                if not ret:
                    print("Warning: failed to read frame.")
                    continue

                # Background for stimulus (full-screen black)
                stim_img = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

                # Pulsing radius
                phase_angle = 2 * math.pi * PULSE_FREQ * elapsed
                radius = int(BASE_RADIUS + RADIUS_AMP * 0.5 * (1 + math.sin(phase_angle)))

                # Dot color: orange-ish while moving, red while recording
                if elapsed < MOVE_TIME:
                    color = (0, 165, 255)  # BGR (orange)
                    phase_label = "move"
                else:
                    color = (0, 0, 255)    # BGR (red)
                    phase_label = "capture"

                # Draw dot
                cv2.circle(stim_img, (tx, ty), radius, color, -1)

                # Progress / info text
                # draw_centered_text(
                #     stim_img,
                #     f"Point {point_index}/{total_points}",
                #     80,
                #     scale=1.0,
                #     thickness=2
                # )

                # if phase_label == "move":
                #     draw_centered_text(stim_img, "Move your EYES to the dot", 140, scale=0.9, thickness=2)
                # else:
                #     draw_centered_text(stim_img, "Hold your gaze on the dot", 140, scale=0.9, thickness=2)

                # draw_centered_text(
                #     stim_img,
                #     "Press ESC to quit",
                #     SCREEN_H - 80,
                #     scale=0.8,
                #     thickness=2
                # )

                # Show stimulus
                cv2.imshow("gaze_calib", stim_img)

                # Save frames only in capture phase
                if phase_label == "capture":
                    ts_ms = int(t_now * 1000)
                    img_name = f"{SESSION_ID}_{tx}_{ty}_{ts_ms}.jpg"
                    img_path = os.path.join(IMG_DIR, img_name)
                    cv2.imwrite(img_path, frame)

                    x_norm = tx / SCREEN_W
                    y_norm = ty / SCREEN_H
                    csv_writer.writerow([img_name, tx, ty, x_norm, y_norm, t_now, phase_label])

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    raise KeyboardInterrupt

        print("Calibration finished. Data saved in:", OUT_DIR)

    except KeyboardInterrupt:
        print("User interrupted calibration early.")

    finally:
        meta_file.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()