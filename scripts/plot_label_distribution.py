import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

from view_data import collect_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot distribution of labeled screen points (x_norm, y_norm).")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory containing gaze_data_* folders.")
    parser.add_argument("--out", type=Path, default=Path("eval_outputs/label_distribution.png"), help="Path to save the plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(args.data_root)
    if not samples:
        raise RuntimeError(f"No labeled samples found under {args.data_root}")

    xs = [pt[1][0] for pt in samples]
    ys = [pt[1][1] for pt in samples]

    plt.figure(figsize=(6, 6))
    plt.hist2d(xs, ys, bins=40, range=[[0, 1], [0, 1]], cmap="magma")
    plt.colorbar(label="count")
    plt.xlabel("x_norm")
    plt.ylabel("y_norm")
    plt.title("Labeled point distribution")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print(f"Saved distribution plot to {args.out}")


if __name__ == "__main__":
    main()
