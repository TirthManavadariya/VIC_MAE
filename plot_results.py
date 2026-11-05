"""
Parse ViC-MAE training logs and generate training/validation curves.

Usage:
  python plot_results.py --log_dir output/finetune_50ep

The log_dir should contain log.txt (written by util/misc.py MetricLogger).
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_log(log_path: Path):
    """
    Parse log.txt written by image_finetune.py.
    Each line is a JSON object with keys:
      train_loss, train_lr, test_loss, test_acc1, test_acc5, epoch, n_parameters
    (from image_finetune.py lines 349-358)
    """
    epochs, train_loss, val_top1, val_top5 = [], [], [], []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "epoch" in d and "train_loss" in d:
                    epochs.append(d["epoch"])
                    train_loss.append(d["train_loss"])
                    val_top1.append(d.get("test_acc1"))
                    val_top5.append(d.get("test_acc5"))
            except json.JSONDecodeError:
                pass

    return epochs, train_loss, val_top1, val_top5


def plot(log_dir: str):
    log_dir = Path(log_dir)
    log_file = log_dir / "log.txt"

    if not log_file.exists():
        print(f"[ERROR] {log_file} not found. Available files:")
        for f in log_dir.iterdir():
            print(f"  {f.name}")
        return

    epochs, train_loss, val_top1, val_top5 = parse_log(log_file)

    if not epochs:
        print(f"[ERROR] Could not parse any epoch data from {log_file}")
        print("  Check the file format manually.")
        return

    print(f"Parsed {len(epochs)} epochs.")
    print(f"Final train loss:  {train_loss[-1]:.4f}")
    print(f"Final val top-1:   {val_top1[-1]:.2f}%")
    print(f"Final val top-5:   {val_top5[-1]:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ViC-MAE ViT-B/16 Finetuning on Tiny-ImageNet-200", fontsize=13, fontweight="bold")

    # --- Training Loss ---
    ax = axes[0]
    ax.plot(epochs, train_loss, color="#e55034", linewidth=1.8, label="Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Val Accuracy ---
    ax = axes[1]
    if any(v is not None for v in val_top1):
        ax.plot(epochs, val_top1, color="#3498db", linewidth=1.8, label="Top-1 Acc (%)")
    if any(v is not None for v in val_top5):
        ax.plot(epochs, val_top5, color="#2ecc71", linewidth=1.8, label="Top-5 Acc (%)", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    out_path = log_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[SAVED] {out_path}")

    # Also save CSV
    csv_path = log_dir / "training_log.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_top1,val_top5\n")
        for e, tl, v1, v5 in zip(epochs, train_loss, val_top1, val_top5):
            f.write(f"{e},{tl},{v1},{v5}\n")
    print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="output/finetune_50ep",
                        help="Directory containing log.txt")
    args = parser.parse_args()
    plot(args.log_dir)
