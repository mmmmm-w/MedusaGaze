import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from openface.face_detection import FaceDetector


def collect_samples(data_root: Path) -> List[Tuple[Path, Tuple[float, float]]]:
    """Gather (image_path, (x_norm, y_norm)) pairs from all labels.csv files."""
    samples: List[Tuple[Path, Tuple[float, float]]] = []
    for csv_path in sorted(data_root.rglob("labels.csv")):
        img_dir = csv_path.parent / "images"
        if not img_dir.exists():
            continue
        df = pd.read_csv(csv_path)
        for row in df.itertuples():
            img_path = img_dir / row.filename
            if not img_path.exists():
                continue
            samples.append((img_path, (float(row.x_norm), float(row.y_norm))))
    return samples


def compute_bbox_feat(det: np.ndarray, frame_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = det[:4]
    cx = (x1 + x2) * 0.5 / w
    cy = (y1 + y2) * 0.5 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return np.array([cx, cy, bw, bh], dtype=np.float32)


def verify_and_cache_faces(
    face_detector: FaceDetector,
    samples: Sequence[Tuple[Path, Tuple[float, float]]],
) -> Tuple[List[Tuple[Path, Tuple[float, float]]], Dict[Path, Tuple[np.ndarray, np.ndarray]]]:
    """Detect once to filter bad samples and store (face, bbox_feat) in a cache."""
    cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}
    valid: List[Tuple[Path, Tuple[float, float]]] = []

    for img_path, target in tqdm(samples, desc="Verifying and caching faces"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        face, dets = face_detector.get_face(str(img_path))
        if face is None or dets is None or len(dets) == 0:
            continue
        det = dets[0]
        bbox_feat = compute_bbox_feat(det, img.shape)
        cache[img_path] = (face, bbox_feat)
        valid.append((img_path, target))

    return valid, cache


class ViewDataset(Dataset):
    """Dataset that serves face crops + bbox features + targets."""

    def __init__(
        self,
        samples: Sequence[Tuple[Path, Tuple[float, float]]],
        face_detector: FaceDetector,
        transform: transforms.Compose,
        cache: Dict[Path, Tuple[np.ndarray, np.ndarray]],
    ):
        self.samples = list(samples)
        self.face_detector = face_detector
        self.transform = transform
        self.cache = cache

    def __len__(self) -> int:
        return len(self.samples)

    def _get_face_and_feat(self, img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if img_path in self.cache:
            return self.cache[img_path]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Cannot read image {img_path}")
        face, dets = self.face_detector.get_face(str(img_path))
        if face is None or dets is None or len(dets) == 0:
            raise RuntimeError(f"Face not detected for {img_path}")
        bbox_feat = compute_bbox_feat(dets[0], img.shape)
        self.cache[img_path] = (face, bbox_feat)
        return face, bbox_feat

    def __getitem__(self, idx: int):
        img_path, target = self.samples[idx]
        face, bbox_feat = self._get_face_and_feat(img_path)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = self.transform(face_rgb)
        bbox_tensor = torch.tensor(bbox_feat, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return tensor, bbox_tensor, target_tensor
