import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch

from openface.face_detection import FaceDetector
from openface.view_model import ViewPredictor
from view_data import collect_samples, verify_and_cache_faces


def sample_items(items: List[Tuple[Path, Tuple[float, float]]], k: int) -> List[Tuple[Path, Tuple[float, float]]]:
    if k >= len(items):
        return items
    return random.sample(items, k)


def draw_points(img: np.ndarray, label: Tuple[float, float], pred: Tuple[float, float], bbox=None) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()

    lx = int(np.clip(label[0], 0, 1) * w)
    ly = int(np.clip(label[1], 0, 1) * h)
    px = int(np.clip(pred[0], 0, 1) * w)
    py = int(np.clip(pred[1], 0, 1) * h)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 120), 2)

    cv2.circle(out, (lx, ly), 10, (0, 200, 255), -1)  # label in orange
    cv2.circle(out, (px, py), 10, (255, 50, 180), -1)  # prediction in magenta
    cv2.line(out, (lx, ly), (px, py), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "label", (lx + 12, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "pred", (px + 12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 180), 2, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual sanity-check for view_mtl predictions.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root with gaze_data_* folders.")
    parser.add_argument("--face-model", type=Path, default=Path("weights/Alignment_RetinaFace.pth"))
    parser.add_argument("--model-path", type=Path, default=Path("weights/view_mtl.pth"))
    parser.add_argument("--num-samples", type=int, default=8, help="How many examples to visualize.")
    parser.add_argument("--out-dir", type=Path, default=Path("eval_outputs"), help="Directory to save visualizations.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/mps).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    device = args.device
    if device is None:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if not args.model_path.exists():
        raise FileNotFoundError(f"view model not found at {args.model_path}")
    if not args.face_model.exists():
        raise FileNotFoundError(f"face model not found at {args.face_model}")

    print(f"Using device: {device}")
    face_detector = FaceDetector(model_path=str(args.face_model), device=device)
    predictor = ViewPredictor(model_path=str(args.model_path), device=device)

    samples = collect_samples(args.data_root)
    if not samples:
        raise RuntimeError(f"No samples under {args.data_root}")

    # Sample before caching to avoid preloading the entire dataset
    chosen = sample_items(samples, args.num_samples)
    chosen, cache = verify_and_cache_faces(face_detector, chosen)
    if not chosen:
        raise RuntimeError("No valid samples after face verification.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(chosen)} examples to {args.out_dir} ...")

    for img_path, target in chosen:
        face, bbox_feat = cache[img_path]
        pred = predictor.predict(face, bbox_feat=bbox_feat)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Get bbox for drawing if available
        dets = face_detector.get_face(str(img_path))[1]
        bbox = dets[0][:4] if dets is not None and len(dets) > 0 else None

        overlay = draw_points(img, target, pred, bbox=bbox)
        out_path = args.out_dir / f"{img_path.stem}_vis.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
