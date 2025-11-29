import time
from pathlib import Path

import cv2
import numpy as np
import torch

from openface.face_detection import FaceDetector
from openface.view_model import ViewPredictor


def compute_bbox_feat(det, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = det[:4]
    cx = (x1 + x2) * 0.5 / w
    cy = (y1 + y2) * 0.5 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return np.array([cx, cy, bw, bh], dtype=np.float32)


def get_screen_resolution():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    face_model_path = "./weights/Alignment_RetinaFace.pth"
    view_model_path = "./weights/view_mtl.pth"

    if not Path(view_model_path).exists():
        raise FileNotFoundError(f"Missing view model weights at {view_model_path}. Train with train_view_model.py first.")

    face_detector = FaceDetector(model_path=face_model_path, device=device)
    view_model = ViewPredictor(model_path=view_model_path, device=device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check permissions / device index.")

    screen_w, screen_h = get_screen_resolution()

    fps_time = time.time()
    frame_count = 0
    fps = 0.0
    smoothed_point = None
    alpha = 0.2  # smoothing factor for screen point

    window_name = "MedusaGaze ViewPoint (cv2)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                continue

            # Keep model input on original frame; create mirrored view for display only
            face, dets = face_detector.get_face_from_image(frame)
            display_frame = cv2.flip(frame, 1)

            # Resize mirrored frame to screen size for correct physical overlay coordinates
            display_resized = cv2.resize(display_frame, (screen_w, screen_h))
            overlay = display_resized.copy()
            view_point = None

            if dets is not None and len(dets) > 0 and face is not None:
                det = dets[0]
                x1, y1, x2, y2 = det[:4].astype(int)

                bbox_feat = compute_bbox_feat(det, frame.shape)
                view_point = view_model.predict(face, bbox_feat=bbox_feat)

                # Smooth the screen point to reduce jitter
                if smoothed_point is None:
                    smoothed_point = view_point
                else:
                    smoothed_point = (1 - alpha) * smoothed_point + alpha * view_point

                # Map detection box to screen coords (resize + mirror)
                scale_x = screen_w / frame.shape[1]
                scale_y = screen_h / frame.shape[0]
                x1s = int(x1 * scale_x)
                x2s = int(x2 * scale_x)
                y1s = int(y1 * scale_y)
                y2s = int(y2 * scale_y)
                mirrored_x1 = screen_w - x2s
                mirrored_x2 = screen_w - x1s
                cv2.rectangle(overlay, (mirrored_x1, y1s), (mirrored_x2, y2s), (0, 230, 120), 2)

                if smoothed_point is not None:
                    # Map normalized screen point to physical screen coords (no mirroring for model output)
                    px = int(np.clip(smoothed_point[0], 0, 1) * screen_w)
                    py = int(np.clip(smoothed_point[1], 0, 1) * screen_h)
                    cv2.circle(overlay, (px, py), 10, (0, 200, 255), -1)
            else:
                smoothed_point = None
                cv2.putText(
                    overlay,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            frame_count += 1
            if frame_count >= 10:
                now = time.time()
                fps = frame_count / (now - fps_time)
                fps_time = now
                frame_count = 0

            cv2.putText(
                overlay,
                f"FPS: {fps:.1f}",
                (20, overlay.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
