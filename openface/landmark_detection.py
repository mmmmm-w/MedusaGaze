import numpy as np
import argparse
from typing import List

from openface.STAR.demo import Alignment

class LandmarkDetector:
    def __init__(self, model_path: str, device: str = "cpu", device_ids: List[int] = [-1]):
        if device == "cpu":
            device_id = "cpu"
        elif device == "mps":
            device_id = "mps"
        elif device == "cuda":
            if not device_ids or device_ids[0] < 0:
                raise ValueError("When using 'cuda', provide at least one valid device ID.")
            device_id = f"cuda:{device_ids[0]}"
        else:
            raise ValueError(f"Invalid device type '{device}'. Use 'cpu' or 'cuda'.")

        # Prepare configuration
        config = {
            "config_name": "alignment",
            "device_id": device_id,
        }
        args = argparse.Namespace(**config)

        # Initialize alignment model
        self.alignment = Alignment(
            args, model_path, dl_framework="pytorch", device_ids=device_ids
        )


    def detect_landmarks(self, image: np.ndarray, dets: np.ndarray, confidence_threshold: float = 0.5):
        results = []
        for det in dets:
            x1, y1, x2, y2 = det[:4].astype(int)
            conf = det[4]

            if conf < confidence_threshold:
                continue

            # Center and scale for alignment
            center_w = (x2 + x1) / 2
            center_h = (y2 + y1) / 2
            scale = min(x2 - x1, y2 - y1) / 200 * 1.05

            # Landmark detection
            landmarks_pv = self.alignment.analyze(image, float(scale), float(center_w), float(center_h))
            results.append(landmarks_pv)

        return results

