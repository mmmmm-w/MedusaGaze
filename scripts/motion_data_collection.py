import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
import tkinter as tk

# ==============================
# CONFIGURATION
# ==============================

# Time settings (Seconds)
PREP_DURATION = 3.0    # Time to get ready before each task
TASK_DURATION = 5.0    # Recording time for each task

# Output paths
GRANDPARENT_DIR = "data"
PARENT_DIR = "OperationData"
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# data/OperationData/data_collection_{TIMESTAMP}
OUT_DIR = os.path.join(GRANDPARENT_DIR, PARENT_DIR, f"data_collection_{SESSION_ID}")
IMG_DIR = os.path.join(OUT_DIR, "images")
META_PATH = os.path.join(OUT_DIR, "labels.csv")

os.makedirs(IMG_DIR, exist_ok=True)

# Define the Sequence of Tasks
# class_id: 0=Open, 1=Left Wink, 2=Right Wink, 3=Stable
TASKS_SEQUENCE = [
    # 1. Main Classes
    {"class_id": 0, "desc": "ACTION: OPEN MOUTH",          "sub_tag": "open_mouth"},
    {"class_id": 1, "desc": "ACTION: LEFT WINK (Close Left Eye)",  "sub_tag": "wink_left"},
    {"class_id": 2, "desc": "ACTION: RIGHT WINK (Close Right Eye)", "sub_tag": "wink_right"},
    
    # 2. Stable Variations (All class_id = 3)
    # Removed "Slight Tilt" tasks as requested
    {"class_id": 3, "desc": "STABLE: Normal Blinking",     "sub_tag": "stable_blinking"},
    {"class_id": 3, "desc": "STABLE: Random Expressions",  "sub_tag": "stable_random"},
    {"class_id": 3, "desc": "STABLE: Neutral / Staring",   "sub_tag": "stable_neutral"},
]

# ==============================
# HELPER FUNCTIONS
# ==============================

def get_screen_resolution():
    try:
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except:
        return 1920, 1080

SCREEN_W, SCREEN_H = get_screen_resolution()

def draw_centered_text(img, text, y, scale=1.0, thickness=2, color=(255, 255, 255), bg_color=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - tw) // 2
    
    if bg_color:
        cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 10), bg_color, -1)
        
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def make_ui_frame(frame, state, task_info, timer_val):
    """
    state: 'PREP' or 'RECORD'
    timer_val: remaining seconds
    """
    display = frame.copy()
    
    # Top Bar
    cv2.rectangle(display, (0, 0), (SCREEN_W, 120), (0, 0, 0), -1)
    
    desc = task_info["desc"]
    
    if state == "PREP":
        # Yellow Preparation Screen
        draw_centered_text(display, f"GET READY: {desc}", 60, scale=1.2, thickness=3, color=(0, 255, 255))
        draw_centered_text(display, f"Starts in: {timer_val:.1f}s", 100, scale=0.8, color=(200, 200, 200))
        
        # Central Countdown
        draw_centered_text(display, f"{int(timer_val)+1}", SCREEN_H//2, scale=5.0, thickness=10, color=(0, 255, 255))
        
    elif state == "RECORD":
        # Red Recording Screen
        draw_centered_text(display, f"RECORDING: {desc}", 60, scale=1.2, thickness=3, color=(0, 0, 255))
        draw_centered_text(display, f"Time left: {timer_val:.1f}s", 100, scale=0.8, color=(200, 200, 200))
        
        # Red Border
        cv2.rectangle(display, (0,0), (SCREEN_W, SCREEN_H), (0,0,255), 20)
        # Rec Circle
        cv2.circle(display, (50, 60), 20, (0, 0, 255), -1)
        
    return display

# ==============================
# MAIN LOOP
# ==============================

def main():
    # 1. Setup Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("Auto Collector", cv2.WINDOW_NORMAL)

    # 2. Setup CSV
    meta_file = open(META_PATH, "w", newline="")
    csv_writer = csv.writer(meta_file)
    csv_writer.writerow(["filename", "class_id", "sub_tag", "timestamp"])

    print(f"Session: {SESSION_ID}")
    print("Press ESC to quit early.")

    # 3. Intro Screen
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        display = frame.copy()
        draw_centered_text(display, "AUTOMATED DATA COLLECTION", 200, 1.5, 3, (0,255,0), (0,0,0))
        draw_centered_text(display, "Press [SPACE] to Start Sequence", 300, 1.0, 2, (255,255,255), (0,0,0))
        
        cv2.imshow("Auto Collector", display)
        if cv2.waitKey(1) & 0xFF == 32: break

    # 4. Task Loop
    total_images = 0
    
    for i, task in enumerate(TASKS_SEQUENCE):
        print(f"Starting Task {i+1}/{len(TASKS_SEQUENCE)}: {task['desc']}")
        
        # --- PHASE 1: PREP (Countdown) ---
        start_prep = time.time()
        while True:
            now = time.time()
            elapsed = now - start_prep
            remaining = PREP_DURATION - elapsed
            
            if remaining <= 0: break
            
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            display = make_ui_frame(frame, "PREP", task, remaining)
            cv2.imshow("Auto Collector", display)
            
            if cv2.waitKey(1) & 0xFF == 27: 
                cap.release(); return # Quit
        
        # --- PHASE 2: RECORDING ---
        start_task = time.time()
        while True:
            now = time.time()
            elapsed = now - start_task
            remaining = TASK_DURATION - elapsed
            
            if remaining <= 0: break
            
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # Save Logic
            ts_str = datetime.now().strftime("%H%M%S_%f")
            sub_tag = task["sub_tag"]
            class_id = task["class_id"]
            
            img_name = f"{sub_tag}_{ts_str}.jpg"
            save_path = os.path.join(IMG_DIR, img_name)
            
            cv2.imwrite(save_path, frame) # Save raw frame
            csv_writer.writerow([img_name, class_id, sub_tag, now])
            total_images += 1
            
            # UI
            display = make_ui_frame(frame, "RECORD", task, remaining)
            cv2.imshow("Auto Collector", display)
            
            if cv2.waitKey(1) & 0xFF == 27: 
                cap.release(); return

    # 5. Finish
    print(f"Done! Collected {total_images} images.")
    meta_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()