import argparse
import sys
from pathlib import Path
from typing import Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from tqdm import tqdm

from openface.face_detection import FaceDetector
from view_data import collect_samples, compute_bbox_feat, save_cached_face_bbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-cache face crops and bbox feats for faster training.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory containing gaze_data_* folders.")
    parser.add_argument("--face-model", type=Path, default=Path("weights/Alignment_RetinaFace.pth"), help="RetinaFace weights path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.face_model.exists():
        raise FileNotFoundError(f"Face model not found at {args.face_model}")

    device = (
        "mps"
        if False  # disable mps in this script to avoid possible mismatch
        else "cuda"
        if False
        else "cpu"
    )
    # Use CPU by default for compatibility; small overhead is fine for preprocessing.
    face_detector = FaceDetector(model_path=str(args.face_model), device=device)

    samples = collect_samples(args.data_root)
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError(f"No samples under {args.data_root}")

    print(f"Processing {len(samples)} samples...")
    cached = 0
    written = 0

    for img_path, _ in tqdm(samples, desc="Caching"):
        face_path = img_path
        img = cv2.imread(str(face_path))
        if img is None:
            continue

        # If already cached, skip
        from view_data import load_cached_face_bbox  # local import to avoid circular refs

        if load_cached_face_bbox(img_path) is not None:
            cached += 1
            continue

        face, dets = face_detector.get_face(str(img_path))
        if face is None or dets is None or len(dets) == 0:
            continue
        det = dets[0]
        bbox_feat = compute_bbox_feat(det, img.shape)
        save_cached_face_bbox(img_path, face, bbox_feat)
        written += 1

    print(f"Existing cache found: {cached} | Newly cached: {written}")


if __name__ == "__main__":
    main()
