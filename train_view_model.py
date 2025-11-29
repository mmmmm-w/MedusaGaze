import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from openface.face_detection import FaceDetector
from openface.model.MTL import View_MTL
from view_data import ViewDataset, collect_samples, verify_and_cache_faces
from view_utils import (
    TrainConfig,
    create_experiment_dir,
    plot_losses,
    save_config,
    set_seed,
)


def train_one_epoch(
    model: View_MTL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    model.base_model.eval()  # keep backbone frozen / BN stable
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
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory containing gaze_data_* folders.")
    parser.add_argument("--face-model", type=Path, default=Path("weights/Alignment_RetinaFace.pth"), help="RetinaFace weights path.")
    parser.add_argument("--backbone-weights", type=Path, default=Path("weights/MTL_backbone.pth"), help="Pretrained backbone weights to load before training heads.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 to keep face cache shared.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick experiments.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("weights/view_mtl.pth"))
    parser.add_argument("--base-model", type=str, default="tf_efficientnet_b0_ns", help="Backbone name for View_MTL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    if not args.face_model.exists():
        raise FileNotFoundError(f"Face model not found at {args.face_model}")

    samples = collect_samples(args.data_root)
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError(f"No gaze samples found under {args.data_root}")
    print(f"Found {len(samples)} labeled frames.")

    face_detector = FaceDetector(model_path=str(args.face_model), device=str(device))
    samples, cache = verify_and_cache_faces(face_detector, samples)
    if not samples:
        raise RuntimeError("No valid samples after face verification.")
    print(f"{len(samples)} samples kept after face detection.")

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

    train_ds = ViewDataset(train_split, face_detector, transform, cache)
    val_ds = ViewDataset(val_split, face_detector, transform, cache)

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

    model = View_MTL(base_model_name=args.base_model, pretrained=False).to(device)

    # Load backbone weights (pretrained) and freeze backbone params
    if args.backbone_weights and args.backbone_weights.exists():
        print(f"Loading backbone weights from {args.backbone_weights}")
        state = torch.load(args.backbone_weights, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded with missing keys: {missing}, unexpected keys: {unexpected}")
    else:
        print("No backbone weights found; using randomly initialized backbone.")

    for param in model.base_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    criterion = nn.MSELoss()

    exp_dir = create_experiment_dir(Path("experiments"))
    output_path = exp_dir / "view_mtl_best.pth"
    history = []
    best_val = float("inf")

    config = TrainConfig(
        data_root=str(args.data_root),
        face_model=str(args.face_model),
        backbone_weights=str(args.backbone_weights),
        base_model=args.base_model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        num_workers=args.num_workers,
        max_samples=args.max_samples if args.max_samples is not None else -1,
        seed=args.seed,
        output=str(args.output),
    )
    save_config(config, exp_dir)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_l1 = evaluate(model, val_loader, criterion, device)
        print(f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val L1: {val_l1:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_l1": val_l1})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"Saved best model to {output_path}")

    # Save loss curve and history
    plot_losses(history, exp_dir / "loss_curve.png")
    torch.save(history, exp_dir / "history.pt")

    # Copy to requested output for downstream use
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if output_path != args.output:
        import shutil

        shutil.copyfile(output_path, args.output)
        print(f"Copied best model to {args.output}")

    print(f"Experiment artifacts saved in {exp_dir}")


if __name__ == "__main__":
    main()
