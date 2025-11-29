import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class TrainConfig:
    data_root: str
    face_model: str
    backbone_weights: str
    base_model: str
    batch_size: int
    epochs: int
    lr: float
    val_split: float
    num_workers: int
    max_samples: int
    seed: int
    output: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_experiment_dir(root: Path = Path("experiments")) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = root / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_config(config: TrainConfig, exp_dir: Path) -> None:
    cfg_path = exp_dir / "config.json"
    with cfg_path.open("w") as f:
        json.dump(asdict(config), f, indent=2)


def plot_losses(history: List[Dict[str, float]], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train, label="train MSE")
    plt.plot(epochs, val, label="val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
