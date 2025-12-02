import cv2
import os
import glob
import pandas as pd
from tqdm import tqdm
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from openface.face_detection import FaceDetector

DATA_ROOT = "data/OperationData"
OUTPUT_ROOT = "data/OperationData_Crops"
FACE_MODEL_PATH = "weights/Alignment_RetinaFace.pth"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    detector = FaceDetector(model_path=FACE_MODEL_PATH, device=device)
    
    session_dirs = glob.glob(os.path.join(DATA_ROOT, "data_collection_*"))
    print(f"Found {len(session_dirs)} sessions.")

    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir)
        csv_path = os.path.join(session_dir, "labels.csv")
        
        if not os.path.exists(csv_path):
            continue

        out_session_dir = os.path.join(OUTPUT_ROOT, session_name)
        out_img_dir = os.path.join(out_session_dir, "images")
        os.makedirs(out_img_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        valid_rows = []

        print(f"Processing {session_name}...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_name = row['filename']
            img_path = os.path.join(session_dir, "images", img_name)
            
            if not os.path.exists(img_path):
                continue
                
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            frame_raw = cv2.flip(frame, 1)

            face_crop, _ = detector.get_face_from_image(frame_raw)

            if face_crop is not None:
                save_path = os.path.join(out_img_dir, img_name)
                cv2.imwrite(save_path, face_crop)
                
                valid_rows.append(row)
            else:
                pass

        if valid_rows:
            new_df = pd.DataFrame(valid_rows)
            new_df.to_csv(os.path.join(out_session_dir, "labels.csv"), index=False)

    print("\nPre-processing Done!")
    print(f"Cropped data saved to: {OUTPUT_ROOT}")
    print("Now go update your training script to point to this folder.")

if __name__ == "__main__":
    main()