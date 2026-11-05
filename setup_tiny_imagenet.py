"""
Prepare Tiny-ImageNet-200 for use with torchvision.datasets.ImageFolder.

The zip unpacks to:
  tiny-imagenet-200/
    train/
      n01443537/
        images/  <-- images are nested one level deeper
    val/
      images/    <-- all val images flat, labels in val_annotations.txt
    wnids.txt
    words.txt

After this script:
  tiny-imagenet-200/
    train/
      n01443537/  <-- images directly here (no 'images/' subfolder)
    val/
      n01443537/  <-- per-class subfolders (ImageFolder compatible)
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ZIP_PATH = PROJECT_DIR / "tiny-imagenet-200.zip"
DATA_DIR = PROJECT_DIR / "tiny-imagenet-200"


def extract_zip():
    if DATA_DIR.exists():
        print(f"[INFO] {DATA_DIR} already exists, skipping extraction.")
        return
    print(f"[INFO] Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(PROJECT_DIR)
    print("[INFO] Extraction complete.")


def fix_train_structure():
    """Move images from train/nXXXX/images/*.JPEG -> train/nXXXX/*.JPEG"""
    train_dir = DATA_DIR / "train"
    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue
        images_subdir = class_dir / "images"
        if images_subdir.exists():
            for img in images_subdir.iterdir():
                shutil.move(str(img), str(class_dir / img.name))
            images_subdir.rmdir()
    print("[INFO] Train structure fixed.")


def fix_val_structure():
    """
    Reorganize val/ from flat structure into per-class subdirs.
    Uses val/val_annotations.txt which maps filename -> class_id.
    """
    val_dir = DATA_DIR / "val"
    annotations_file = val_dir / "val_annotations.txt"
    images_dir = val_dir / "images"

    if not images_dir.exists():
        print("[INFO] val/images/ not found — val may already be reorganized.")
        return

    # Parse annotations
    img_to_class = {}
    with open(annotations_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            img_to_class[parts[0]] = parts[1]  # filename -> class_id

    # Create per-class dirs and move images
    for img_name, class_id in img_to_class.items():
        class_dir = val_dir / class_id
        class_dir.mkdir(exist_ok=True)
        src = images_dir / img_name
        dst = class_dir / img_name
        if src.exists():
            shutil.move(str(src), str(dst))

    # Cleanup
    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()

    print(f"[INFO] Val structure fixed: {len(set(img_to_class.values()))} classes.")


def verify():
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
    val_classes = [d for d in val_dir.iterdir() if d.is_dir()]
    print(f"[VERIFY] Train classes: {len(train_classes)}, Val classes: {len(val_classes)}")
    # Count images
    train_imgs = sum(len(list(c.glob("*.JPEG"))) for c in train_classes)
    val_imgs = sum(len(list(c.glob("*.JPEG"))) for c in val_classes)
    print(f"[VERIFY] Train images: {train_imgs}, Val images: {val_imgs}")
    print("[VERIFY] Expected: 100,000 train / 10,000 val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and reformat Tiny-ImageNet-200 for PyTorch ImageFolder.")
    parser.add_argument("--project_dir", type=str, default=str(PROJECT_DIR), help="Path containing tiny-imagenet-200.zip")
    args = parser.parse_args()

    PROJECT_DIR = Path(args.project_dir).resolve()
    ZIP_PATH = PROJECT_DIR / "tiny-imagenet-200.zip"
    DATA_DIR = PROJECT_DIR / "tiny-imagenet-200"

    extract_zip()
    fix_train_structure()
    fix_val_structure()
    verify()
    print("\n[DONE] Tiny-ImageNet ready at:", DATA_DIR)
    print("Use --data_path", DATA_DIR, "in image_finetune.py")
