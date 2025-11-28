import argparse
import os
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import csv
import logging
import sys
import datetime
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from openface.face_detection import FaceDetector
from openface.model.MTL import View_MTL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_samples(data_root: Path) -> List[Tuple[Path, Tuple[float, float]]]:
    """Collect (image_path, (x_norm, y_norm)) pairs from all labels.csv files."""
    samples: List[Tuple[Path, Tuple[float, float]]] = []
    csv_files = sorted(data_root.rglob("labels.csv"))
    for csv_path in csv_files:
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


class ViewDataset(Dataset):
    """Dataset that detects and caches faces once, then trains the view model."""

    def __init__(
        self,
        samples: Sequence[Tuple[Path, Tuple[float, float]]],
        face_detector: FaceDetector,
        transform: transforms.Compose,
        cache: Dict[Path, Tuple[np.ndarray, np.ndarray]],
        pos_feature_mode: str = "basic",
    ):
        self.samples = list(samples)
        self.face_detector = face_detector
        self.transform = transform
        self.cache = cache
        self.pos_feature_mode = pos_feature_mode

    def __len__(self) -> int:
        return len(self.samples)

    def _get_face_and_feat(self, img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if img_path in self.cache:
            return self.cache[img_path]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Cannot read image {img_path}")
        h, w = img.shape[:2]
        face, dets = self.face_detector.get_face(str(img_path))
        if face is None or dets is None or len(dets) == 0:
            raise RuntimeError(f"Face not detected for {img_path}")
        det = dets[0]
        x1, y1, x2, y2 = det[:4]
        cx = (x1 + x2) * 0.5 / w
        cy = (y1 + y2) * 0.5 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        bbox_feat = np.array([cx, cy, bw, bh], dtype=np.float32)
        self.cache[img_path] = (face, bbox_feat)
        return face, bbox_feat

    def __getitem__(self, idx: int):
        img_path, target = self.samples[idx]
        face, bbox_feat = self._get_face_and_feat(img_path)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = self.transform(face_rgb)

        # Expand positional features if requested
        if self.pos_feature_mode == "basic":
            feats = bbox_feat
        elif self.pos_feature_mode == "area":
            bw, bh = float(bbox_feat[2]), float(bbox_feat[3])
            area = bw * bh
            feats = np.concatenate([bbox_feat, np.array([area], dtype=np.float32)], axis=0)
        elif self.pos_feature_mode == "aspect":
            bw, bh = float(bbox_feat[2]), float(bbox_feat[3])
            aspect = bw / (bh + 1e-6)
            feats = np.concatenate([bbox_feat, np.array([aspect], dtype=np.float32)], axis=0)
        elif self.pos_feature_mode == "both":
            bw, bh = float(bbox_feat[2]), float(bbox_feat[3])
            area = bw * bh
            aspect = bw / (bh + 1e-6)
            feats = np.concatenate([bbox_feat, np.array([area, aspect], dtype=np.float32)], axis=0)
        else:
            feats = bbox_feat

        bbox_tensor = torch.tensor(feats, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return tensor, bbox_tensor, target_tensor


def verify_and_cache_faces(
    face_detector: FaceDetector,
    samples: Sequence[Tuple[Path, Tuple[float, float]]],
) -> Tuple[List[Tuple[Path, Tuple[float, float]]], Dict[Path, Tuple[np.ndarray, np.ndarray]]]:
    """Run face detection once to drop invalid samples and warm the cache."""
    cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}
    valid: List[Tuple[Path, Tuple[float, float]]] = []

    for img_path, target in tqdm(samples, desc="Verifying faces"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        face, dets = face_detector.get_face(str(img_path))
        if face is None or dets is None or len(dets) == 0:
            continue
        det = dets[0]
        x1, y1, x2, y2 = det[:4]
        cx = (x1 + x2) * 0.5 / w
        cy = (y1 + y2) * 0.5 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        bbox_feat = np.array([cx, cy, bw, bh], dtype=np.float32)
        cache[img_path] = (face, bbox_feat)
        valid.append((img_path, target))

    return valid, cache


def compute_edge_proximity(x: float, y: float) -> float:
    """[0,1]: 0=center, 1=edge/corner; Chebyshev distance from center normalized."""
    dx = abs(x - 0.5)
    dy = abs(y - 0.5)
    return float(min(1.0, max(dx, dy) * 2.0))


def compute_sample_weights_for_edges(
    samples: Sequence[Tuple[Path, Tuple[float, float]]],
    mode: str = "linear",
    gamma: float = 1.5,
    min_weight: float = 1.0,
) -> List[float]:
    """
    Assign higher weights to samples nearer edges.
    mode: 'linear' -> 1 + p
          'quadratic' -> 1 + p^2
          'power' -> 1 + p^gamma
    """
    weights: List[float] = []
    for _, (x, y) in samples:
        p = compute_edge_proximity(float(x), float(y))
        if mode == "linear":
            w = 1.0 + p
        elif mode == "quadratic":
            w = 1.0 + p * p
        elif mode == "power":
            w = 1.0 + (p ** gamma)
        else:
            w = 1.0
        w = max(min_weight, float(w))
        weights.append(w)
    return weights


def train_one_epoch(
    model: View_MTL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    # keep backbone frozen (including BN stats)
    model.base_model.eval()
    total_loss = 0.0
    total_samples = 0

    for faces, bboxes, targets in tqdm(loader, desc="Train", leave=False):
        faces = faces.to(device)
        bboxes = bboxes.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(faces, bboxes)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * faces.size(0)
        total_samples += faces.size(0)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: View_MTL,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_samples = 0

    for faces, bboxes, targets in tqdm(loader, desc="Val", leave=False):
        faces = faces.to(device)
        bboxes = bboxes.to(device)
        targets = targets.to(device)
        preds = model(faces, bboxes)

        loss = criterion(preds, targets)
        l1 = torch.nn.functional.l1_loss(preds, targets, reduction="sum")

        total_loss += loss.item() * faces.size(0)
        total_l1 += l1.item()
        total_samples += faces.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_l1 = total_l1 / max(total_samples, 1)
    return avg_loss, avg_l1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the view_mtl model on collected gaze data.")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"), help="Root directory to store experiment runs.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory containing gaze_data_* folders.")
    parser.add_argument("--face-model", type=Path, default=Path("weights/Alignment_RetinaFace.pth"), help="RetinaFace weights path.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 to keep face cache shared.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick experiments.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("weights/view_mtl.pth"))
    parser.add_argument("--base-model", type=str, default="tf_efficientnet_b0_ns", help="Backbone name for View_MTL.")
     # Edge-weighted sampling
    parser.add_argument("--edge-weighting", type=str, default="none", choices=["none", "linear", "quadratic", "power"], help="Weighted sampler favoring edges.")
    parser.add_argument("--edge-gamma", type=float, default=1.5, help="Gamma for 'power' mode.")
    # Loss options
    parser.add_argument("--loss-type", type=str, default="mse", choices=["mse", "l1", "huber"], help="Training loss for view regression.")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Delta for SmoothL1Loss (Huber).")
    # Positional feature options
    parser.add_argument("--pos-features", type=str, default="basic", choices=["basic", "area", "aspect", "both"], help="Extra positional features to fuse.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # ---------------------------------------------
    # Create timestamped experiment directory
    # ---------------------------------------------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = args.experiments_dir / ts
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    cfg_path = exp_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, f, indent=2)

    # Setup logging to both stdout and file
    log_path = exp_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("train_view_model")

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")

    samples = collect_samples(args.data_root)
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError(f"No gaze samples found under {args.data_root}")
    logger.info(f"Found {len(samples)} labeled frames.")

    face_detector = FaceDetector(model_path=str(args.face_model), device=str(device))
    samples, cache = verify_and_cache_faces(face_detector, samples)
    if not samples:
        raise RuntimeError("No valid samples after face verification.")
    logger.info(f"{len(samples)} samples kept after face detection.")

    val_size = max(1, int(len(samples) * args.val_split))
    train_size = len(samples) - val_size
    train_split, val_split = random_split(samples, [train_size, val_size])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = ViewDataset(train_split, face_detector, transform, cache, pos_feature_mode=args.pos_features)
    val_ds = ViewDataset(val_split, face_detector, transform, cache, pos_feature_mode=args.pos_features)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Determine positional feature dimension for model
    if args.pos_features == "basic":
        pos_dim = 4
    elif args.pos_features in ("area", "aspect"):
        pos_dim = 5
    elif args.pos_features == "both":
        pos_dim = 6
    else:
        pos_dim = 4

    model = View_MTL(base_model_name=args.base_model, pretrained=False, pos_dim=pos_dim).to(device)

    for param in model.base_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Loss selection
    if args.loss_type == "mse":
        criterion = nn.MSELoss()
    elif args.loss_type == "l1":
        criterion = nn.L1Loss()
    elif args.loss_type == "huber":
        criterion = nn.SmoothL1Loss(beta=args.huber_delta)
    else:
        criterion = nn.MSELoss()

    best_val = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Optionally replace train_loader with a weighted sampler
    if args.edge_weighting != "none":
        train_indices = list(train_split.indices)
        train_samples = [samples[i] for i in train_indices]
        weights = compute_sample_weights_for_edges(train_samples, mode=args.edge_weighting, gamma=args.edge_gamma, min_weight=1.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler,
        )

    # Prepare metrics CSV
    metrics_path = exp_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_l1"])

    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        # Keep backbone frozen (including BN stats) during head training
        model.base_model.eval()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_l1 = evaluate(model, val_loader, criterion, device)
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val L1: {val_l1:.4f}")

        # Append metrics
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_l1:.6f}"])

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.output)
            # Also save a copy under the experiment directory
            torch.save(model.state_dict(), exp_dir / "model_best.pth")
            logger.info(f"Saved best model to {args.output} and {exp_dir / 'model_best.pth'}")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
